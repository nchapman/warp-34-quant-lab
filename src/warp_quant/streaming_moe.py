"""Streaming W4A16-HGPTQ + 4-bit Custom-LUT4 (W_kv-HGPTQ) PLE quantizer & probe for Qwen/Qwen3.8-Flash-Next.

Designed for a single NVIDIA H200 (141 GB VRAM):
  1. Vision tower (`model.visual.*`): W4A16-HGPTQ (`g64-h64` on K%64==0, channel+H16 on `linear_fc2`), calibrated on `vision-calib-v2` (`token_exp=0.25`).
  2. Layer-by-layer streaming (`model.language_model.layers.0..47`):
     - Peak GPU VRAM `< 25 GB` (loads one 5.15 GB decoder layer at a time).
     - Layer-1 PLE (`layers.1.ple.ple_embedding.ngram_embedding.shard_0..127`, 51.2B params / 102.4 GB BF16 -> 28.8 GB):
       4-bit `Custom-LUT4` (`g=32`, FP16 group scale) with exact per-head downstream Hessian
       `H_m = 0.5 * (norm(W_{k,m}^T W_{k,m}) + norm(W_{v,m}^T W_{v,m}))` Cholesky GPTQ + 1-pass Coordinate Descent.
     - Attention (`self_attn.{q,k,v,o}_proj`) and GDN (`linear_attn.{in_proj_qkv,in_proj_z,out_proj}`):
       W4A16-HGPTQ (`g64-h64`, static actorder).
     - 512 Routed Experts (`mlp.experts.0..511.{gate,up,down}_proj`):
       Batched W4A16 Cholesky GPTQ (`g64`, static actorder) using shared `[2560, 2560]` input Hessian for `gate_proj`/`up_proj`
       and batched `[512, 640, 640]` post-SwiGLU Hessians for `down_proj`.
  3. Native 4-stream Hyper-Connection MTP (`mtp.*`):
     W4A16-HGPTQ (`g64-h64`) on `mtp.layers.0.self_attn.*` + batched W4A16 GPTQ (`g64`) on `mtp.layers.0.mlp.experts.0..511.*`.
  4. Streaming layer-by-layer fidelity probe (`probe-v4`, `probe-guard`, `mtp_fidelity`, `vision_fidelity`, `vision_guard_fidelity`).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load as load_safetensors_bytes
from safetensors.torch import save_file
from transformers import AutoProcessor

from warp_quant.cachepath import resolve_snapshot
from warp_quant.hadamard_gptq import (
    BASE_IGNORE,
    MTP_HAD_TARGET_RE,
    MTP_TARGET_RE,
    TEXT_HAD_TARGET_RE,
    TEXT_TARGET_RE,
    VISION_FC2_TARGET_RE,
    VISION_GROUP_TARGET_RE,
    _assert_packed_alignment,
    _attach_expert_hadamard_hooks,
    _gptq_calibrate_vision,
    _hadamard_rotate_last_dim,
    _hadamard_rotate_weight,
    _is_power_of_two,
    _load_shard_map,
    _pack_w4_tensor,
)
from warp_quant.lablog import log_event, run_done, run_fail
from warp_quant.models import quant_recipe, resolve
from warp_quant.stamp import (
    build_generation_config,
    check_template_renders,
    ensure_license_files,
    ensure_processor_configs,
    stamp_release_template,
)

ROOT = Path(__file__).resolve().parents[2]

# Fitted 16-entry non-uniform codebook for 4-bit group-normalized (g=32) Gaussian PLE rows
PLE_CUSTOM_LUT4: list[float] = [
    -1.0000,
    -0.8128,
    -0.6546,
    -0.5151,
    -0.3918,
    -0.2808,
    -0.1791,
    -0.0834,
    0.0105,
    0.1048,
    0.2023,
    0.3060,
    0.4208,
    0.5524,
    0.7140,
    0.9487,
]

QWEN4_EXP_EXTRA_IGNORE: list[str] = [
    r"re:.*hyper_connection.*",
    r"re:.*self_attn\.indexer\..*",
    r"re:.*ple\.(key_proj|value_proj)$",
    r"re:^mtp\.fc_.*",
]

QWEN4_EXP_IGNORE: list[str] = list(BASE_IGNORE) + QWEN4_EXP_EXTRA_IGNORE


def _is_qwen35_moe_text_config(text_config) -> bool:
    return getattr(text_config, "model_type", "") == "qwen3_5_moe_text"


def _ensure_qwen4_exp_device_kernels(text_config, device: torch.device) -> None:
    """Configure Qwen4Exp or Qwen3_5Moe GDN and MoE kernels for CUDA or CPU execution."""
    if _is_qwen35_moe_text_config(text_config):
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
    else:
        import transformers.models.qwen4_exp.modeling_qwen4_exp as qm

    if device.type == "cpu":
        qm.causal_conv1d_fn = getattr(qm.causal_conv1d_fn, "__wrapped__", qm.causal_conv1d_fn)
        qm.torch_chunk_gated_delta_rule = getattr(
            qm.torch_chunk_gated_delta_rule, "__wrapped__", qm.torch_chunk_gated_delta_rule
        )
        text_config._experts_implementation = "eager"
    else:
        text_config._experts_implementation = "grouped_mm"
    text_config._attn_implementation = "sdpa"


def compute_ple_head_hessians(
    key_proj_weight: torch.Tensor,
    value_proj_weight: torch.Tensor,
    num_heads: int,
    dampening_frac: float = 0.01,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute exact per-head [D_h, D_h] downstream Hessians H_m and upper inverse Cholesky factors U_m."""
    embed_dim = key_proj_weight.shape[1]
    if embed_dim % num_heads != 0:
        raise ValueError(f"ple_embed_dim={embed_dim} not divisible by num_heads={num_heads}")
    head_dim = embed_dim // num_heads
    device = key_proj_weight.device
    wk = key_proj_weight.detach().float()
    wv = value_proj_weight.detach().float()

    h_list: list[torch.Tensor] = []
    u_list: list[torch.Tensor] = []
    eye = torch.eye(head_dim, device=device, dtype=torch.float32)

    for m in range(num_heads):
        c0, c1 = m * head_dim, (m + 1) * head_dim
        wk_m = wk[:, c0:c1]
        wv_m = wv[:, c0:c1]
        hk = (wk_m.T @ wk_m) / max(wk_m.shape[0], 1)
        hv = (wv_m.T @ wv_m) / max(wv_m.shape[0], 1)
        hk_norm = hk / torch.diag(hk).mean().clamp(min=1e-12)
        hv_norm = hv / torch.diag(hv).mean().clamp(min=1e-12)
        h_m = 0.5 * (hk_norm + hv_norm)
        damp = max(dampening_frac * torch.diag(h_m).mean().item(), 1e-6)
        l_m = torch.linalg.cholesky(h_m + damp * eye)
        h_inv = torch.cholesky_inverse(l_m)
        u_m = torch.linalg.cholesky(h_inv, upper=True)
        h_list.append(h_m)
        u_list.append(u_m)

    return h_list, u_list


@torch.no_grad()
def quantize_ple_batch_gptq_lut4(
    w_batch: torch.Tensor,
    h_mat: torch.Tensor,
    u_mat: torch.Tensor,
    lut: torch.Tensor | None = None,
    group_size: int = 32,
    cd_passes: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D PLE slice [B, D_h] using Custom-LUT4 + W_kv Cholesky GPTQ + Coordinate Descent.

    Returns:
      q_deq: [B, D_h] float32 dequantized rows
      packed_u8: [B, D_h // 2] uint8 packed 4-bit indices (low nibble = col 2j, high nibble = col 2j+1)
      scales_fp16: [B, D_h // group_size] float16 group scales
    """
    device = w_batch.device
    if lut is None:
        lut = torch.tensor(PLE_CUSTOM_LUT4, device=device, dtype=torch.float32)
    else:
        lut = lut.to(device=device, dtype=torch.float32)
    midpoints = 0.5 * (lut[:-1] + lut[1:])

    b_rows, d_h = w_batch.shape
    eff_group_size = min(group_size, d_h)
    if d_h % eff_group_size != 0 or d_h % 2 != 0:
        raise ValueError(f"Invalid PLE head_dim={d_h} for group_size={eff_group_size}")
    num_groups = d_h // eff_group_size

    w_f32 = w_batch.detach().float()
    w_work = w_f32.clone()
    q_out = torch.empty_like(w_work)
    idx_out = torch.empty((b_rows, d_h), dtype=torch.uint8, device=device)
    scales_fp16 = torch.empty((b_rows, num_groups), dtype=torch.float16, device=device)

    u_diag = torch.diag(u_mat)
    h_diag = torch.diag(h_mat)

    for g in range(num_groups):
        c0, c1 = g * eff_group_size, (g + 1) * eff_group_size
        w_g = w_work[:, c0:c1]
        s_fp16 = (w_g.abs().amax(dim=1) / 1.0).clamp(min=1e-7).to(torch.float16)
        scales_fp16[:, g] = s_fp16
        s = s_fp16.float()
        inv_s = 1.0 / s
        err_g = torch.empty((b_rows, eff_group_size), dtype=torch.float32, device=device)
        for c in range(c0, c1):
            lc = c - c0
            w_c = w_g[:, lc]
            idx_c = torch.bucketize(w_c * inv_s, midpoints)
            q_c = lut[idx_c] * s
            q_out[:, c] = q_c
            idx_out[:, c] = idx_c.to(torch.uint8)
            err_c = (w_c - q_c) / u_diag[c]
            err_g[:, lc] = err_c
            if lc + 1 < eff_group_size:
                w_g[:, lc + 1 :] -= err_c.unsqueeze(1) * u_mat[c, c + 1 : c1].unsqueeze(0)
        if c1 < d_h:
            w_work[:, c1:] -= err_g @ u_mat[c0:c1, c1:]

    for _ in range(cd_passes):
        r_mat = (q_out - w_f32) @ h_mat
        for g in range(num_groups):
            c0, c1 = g * eff_group_size, (g + 1) * eff_group_size
            s = scales_fp16[:, g].float()
            inv_s = 1.0 / s
            q_g_old = q_out[:, c0:c1].clone()
            r_g = r_mat[:, c0:c1]
            for c in range(c0, c1):
                lc = c - c0
                q_old = q_out[:, c]
                target = q_old - r_g[:, lc] / h_diag[c]
                idx_new = torch.bucketize(target * inv_s, midpoints)
                q_new = lut[idx_new] * s
                dq = q_new - q_old
                q_out[:, c] = q_new
                idx_out[:, c] = idx_new.to(torch.uint8)
                if lc + 1 < eff_group_size:
                    r_g[:, lc + 1 :] += dq.unsqueeze(1) * h_mat[c, c + 1 : c1].unsqueeze(0)
            if c1 < d_h:
                dq_g = q_out[:, c0:c1] - q_g_old
                r_mat[:, c1:] += dq_g @ h_mat[c0:c1, c1:]

    packed_u8 = (idx_out[:, 0::2] & 0x0F) | ((idx_out[:, 1::2] & 0x0F) << 4)
    return q_out, packed_u8, scales_fp16


@torch.no_grad()
def dequantize_ple_rows_lut4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_lut: torch.Tensor,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize arbitrary-shaped [..., D_h // 2] uint8 packed PLE rows back to [..., D_h]."""
    low = (weight_packed & 0x0F).long()
    high = ((weight_packed >> 4) & 0x0F).long()
    idx = torch.stack([low, high], dim=-1).reshape(*weight_packed.shape[:-1], -1)
    d_h = idx.shape[-1]
    eff_group_size = min(group_size, d_h)
    num_groups = d_h // eff_group_size
    lut_dev = weight_lut.to(device=weight_packed.device, dtype=torch.float32)
    vals = lut_dev[idx].view(*weight_packed.shape[:-1], num_groups, eff_group_size)
    deq = (vals * weight_scale.float().unsqueeze(-1)).reshape(*weight_packed.shape[:-1], d_h)
    return deq.to(out_dtype)


def pack_ple_page_slotted_4k(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Pack [N, 80] uint8 + [N, 5] float16 (90 B/row) into 4 KiB pages (45 rows/page, 0% page straddles)."""
    if weight_packed.ndim != 2 or weight_scale.ndim != 2 or weight_packed.shape[0] != weight_scale.shape[0]:
        raise ValueError("weight_packed and weight_scale must be 2D tensors with matching row count")
    n_rows = weight_packed.shape[0]
    scale_u8 = weight_scale.contiguous().view(torch.uint8)
    row_bytes = torch.cat([weight_packed.contiguous(), scale_u8], dim=1)
    rec_size = row_bytes.shape[1]
    rows_per_page = 4096 // rec_size
    if rows_per_page < 1:
        raise ValueError(f"Row record size {rec_size} exceeds 4096 B page")
    n_pages = math.ceil(n_rows / rows_per_page)
    padded_rows = n_pages * rows_per_page
    if padded_rows > n_rows:
        pad = torch.zeros((padded_rows - n_rows, rec_size), dtype=torch.uint8, device=row_bytes.device)
        row_bytes = torch.cat([row_bytes, pad], dim=0)
    pages = torch.zeros((n_pages, 4096), dtype=torch.uint8, device=row_bytes.device)
    pages[:, : rows_per_page * rec_size] = row_bytes.view(n_pages, rows_per_page * rec_size)
    return pages


def unpack_ple_page_slotted_4k(
    pages: torch.Tensor,
    num_rows: int,
    head_dim: int = 160,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack 4 KiB page-slotted uint8 pages back into (weight_packed [N, D_h//2], weight_scale [N, D_h//G])."""
    eff_group_size = min(group_size, head_dim)
    packed_cols = head_dim // 2
    scale_bytes = (head_dim // eff_group_size) * 2
    rec_size = packed_cols + scale_bytes
    rows_per_page = 4096 // rec_size
    flat_rows = pages[:, : rows_per_page * rec_size].reshape(-1, rec_size)[:num_rows]
    weight_packed = flat_rows[:, :packed_cols].contiguous()
    weight_scale = flat_rows[:, packed_cols:].contiguous().view(torch.float16)
    return weight_packed, weight_scale


def _shift_right_ignore_eos(token_ids: torch.Tensor, shift: int, eos_token_id: int) -> torch.Tensor:
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    segment_start = previous_eos + 1
    position_in_segment = positions.unsqueeze(0) - segment_start
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


@torch.no_grad()
def compute_ple_ngram_ids(
    input_ids: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Compute [B, S, ngram_heads] global PLE row indices matching Qwen4ExpTextNGramEmbedding.forward."""
    input_ids = input_ids.long()
    context_len = ngram_size - 1
    previous_context = input_ids.new_full((input_ids.shape[0], context_len), eos_token_id)
    token_history = torch.cat([previous_context, input_ids], dim=-1)
    shifted_tokens = [
        _shift_right_ignore_eos(token_history, shift, eos_token_id) for shift in range(ngram_size)
    ]

    lm = layer_multipliers.to(input_ids.device)
    vs = ngram_heads_vocab_sizes.to(input_ids.device)
    offs = ngram_heads_offsets.to(input_ids.device)

    blocks = []
    for ngram in range(2, ngram_size + 1):
        start_idx = (ngram - 2) * heads_per_ngram
        end_idx = start_idx + heads_per_ngram
        mixed_ids = shifted_tokens[0] * lm[0]
        for position in range(1, ngram):
            mixed_ids = torch.bitwise_xor(mixed_ids, shifted_tokens[position] * lm[position])
        head_vocab_sizes = vs[start_idx:end_idx]
        head_offsets = offs[start_idx:end_idx]
        ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
        blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

    return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]


def _ple_head_segments_for_shard(
    row_start: int,
    row_end: int,
    ngram_heads_offsets: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
) -> list[tuple[int, int, int]]:
    """Return contiguous (local_r0, local_r1, head_idx) segments for global rows [row_start, row_end)."""
    offs = ngram_heads_offsets.tolist()
    sizes = ngram_heads_vocab_sizes.tolist()
    num_heads = len(offs)
    segments: list[tuple[int, int, int]] = []
    for m in range(num_heads):
        h_start = int(offs[m])
        h_end = int(offs[m] + sizes[m]) if m + 1 < num_heads else max(int(offs[m] + sizes[m]), row_end)
        s0 = max(row_start, h_start)
        s1 = min(row_end, h_end)
        if s0 < s1:
            segments.append((s0 - row_start, s1 - row_start, m))
    if not segments:
        segments.append((0, row_end - row_start, num_heads - 1))
    return segments


def _discover_ple_shards(
    shard_map: dict[str, str],
    ckpt_dir: Path,
    layer_idx: int,
) -> list[tuple[int, str, str]]:
    """Return sorted [(shard_id, key_prefix, filename), ...] for PLE ngram_embedding shards."""
    ple_base = f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
    shard_pat = re.compile(rf"^{re.escape(ple_base)}\.shard_(\d+)\.(weight|weight_packed)$")
    found: dict[int, tuple[str, str]] = {}
    if shard_map:
        for k, fname in shard_map.items():
            m = shard_pat.match(k)
            if m:
                s_id = int(m.group(1))
                found[s_id] = (f"{ple_base}.shard_{s_id}", fname)
        if not found:
            for suffix in ("weight", "weight_packed"):
                k = f"{ple_base}.{suffix}"
                if k in shard_map:
                    found[0] = (ple_base, shard_map[k])
                    break
    else:
        for sp in sorted(ckpt_dir.glob("*.safetensors")):
            with safe_open(str(sp), framework="pt", device="cpu") as f:
                for k in f.keys():
                    m = shard_pat.match(k)
                    if m:
                        s_id = int(m.group(1))
                        found[s_id] = (f"{ple_base}.shard_{s_id}", sp.name)
                    elif k in (f"{ple_base}.weight", f"{ple_base}.weight_packed"):
                        found[0] = (ple_base, sp.name)
    return [(s_id, found[s_id][0], found[s_id][1]) for s_id in sorted(found)]


@torch.no_grad()
def stream_quantize_ple_shards(
    base_dir: Path,
    shard_map: dict[str, str],
    output_dir: Path,
    layer_idx: int,
    ple_layer: nn.Module,
    calib_ids_list: list[torch.Tensor],
    *,
    device: torch.device,
    ple_group_size: int = 32,
    dampening_frac: float = 0.01,
    cd_passes: int = 1,
    batch_rows: int = 2_500_100,
    shards_per_file: int = 16,
) -> tuple[list[torch.Tensor], dict[str, str], int, int]:
    """Stream through all PLE shards, quantize with Custom-LUT4 + W_kv-HGPTQ, save, and gather quantized embeds."""
    ple_emb = ple_layer.ple_embedding
    ngram_heads = ple_emb.ngram_heads
    eos_token_id = ple_emb.eos_token_id

    if ple_group_size > 0:
        h_list, u_list = compute_ple_head_hessians(
            ple_layer.key_proj.weight,
            ple_layer.value_proj.weight,
            num_heads=ngram_heads,
            dampening_frac=dampening_frac,
        )
        lut_tensor = torch.tensor(PLE_CUSTOM_LUT4, device=device, dtype=torch.float32)
        lut_cpu = lut_tensor.cpu()
    else:
        h_list, u_list = [], []
        lut_tensor = torch.empty(0, device=device, dtype=torch.float32)
        lut_cpu = lut_tensor.cpu()

    # Precompute ngram_ids [1, S_i, ngram_heads] for every calibration sequence and flatten for O(1) vectorized gather
    head_dim = ple_layer.key_proj.weight.shape[1] // ngram_heads
    seq_shapes: list[tuple[int, int]] = []
    flat_n_ids_parts: list[torch.Tensor] = []

    for ids in calib_ids_list:
        seq_shapes.append((ids.shape[0], ids.shape[1]))
        n_ids = compute_ple_ngram_ids(
            ids,
            ple_emb.layer_multipliers,
            ple_emb.ngram_heads_vocab_sizes,
            ple_emb.ngram_heads_offsets,
            ple_emb.ngram_size,
            ple_emb.heads_per_ngram,
            eos_token_id,
        ).cpu()
        flat_n_ids_parts.append(n_ids.reshape(-1))

    all_ngram_ids = torch.cat(flat_n_ids_parts, dim=0)
    del flat_n_ids_parts
    all_quant_embeds = torch.zeros((all_ngram_ids.numel(), head_dim), dtype=torch.bfloat16)

    ple_shards = _discover_ple_shards(shard_map, base_dir, layer_idx)
    if not ple_shards:
        raise RuntimeError(f"No PLE shards found for layer {layer_idx} in {base_dir}")

    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    current_bucket: dict[str, torch.Tensor] = {}
    bucket_idx = 0
    global_row_cursor = 0
    t_ple0 = time.time()

    def _read_ple_shard(item: tuple[int, str, str]) -> torch.Tensor:
        _sid, k_pref, s_fname = item
        with safe_open(str(base_dir / s_fname), framework="pt", device="cpu") as f:
            return f.get_tensor(f"{k_pref}.weight")

    with ThreadPoolExecutor(max_workers=2) as ple_io_pool:
        next_read_fut: Future | None = ple_io_pool.submit(_read_ple_shard, ple_shards[0])
        pending_save_fut: Future | None = None

        for idx_in_list, (shard_id, _key_prefix, _shard_fname) in enumerate(ple_shards):
            w_shard_cpu = next_read_fut.result()
            if idx_in_list + 1 < len(ple_shards):
                next_read_fut = ple_io_pool.submit(_read_ple_shard, ple_shards[idx_in_list + 1])
            else:
                next_read_fut = None
            n_rows, d_h = w_shard_cpu.shape
            row_start = global_row_cursor
            row_end = row_start + n_rows
            global_row_cursor = row_end

            out_prefix = (
                f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding.shard_{shard_id}"
            )
            hit_mask = (all_ngram_ids >= row_start) & (all_ngram_ids < row_end)

            if ple_group_size <= 0:
                w_bf16 = w_shard_cpu.to(torch.bfloat16)
                del w_shard_cpu
                if hit_mask.any():
                    local_rows = (all_ngram_ids[hit_mask] - row_start).long()
                    all_quant_embeds[hit_mask] = w_bf16[local_rows]
                shard_tensors = {f"{out_prefix}.weight": w_bf16}
            else:
                eff_group_size = min(ple_group_size, d_h)
                num_groups = d_h // eff_group_size
                packed_shard = torch.empty((n_rows, d_h // 2), dtype=torch.uint8)
                scales_shard = torch.empty((n_rows, num_groups), dtype=torch.float16)

                segments = _ple_head_segments_for_shard(
                    row_start, row_end, ple_emb.ngram_heads_offsets, ple_emb.ngram_heads_vocab_sizes
                )

                for seg_r0, seg_r1, head_m in segments:
                    h_m = h_list[head_m]
                    u_m = u_list[head_m]
                    for b0 in range(seg_r0, seg_r1, batch_rows):
                        b1 = min(b0 + batch_rows, seg_r1)
                        w_batch = w_shard_cpu[b0:b1].to(device=device, dtype=torch.float32)
                        q_deq, p_u8, s_fp16 = quantize_ple_batch_gptq_lut4(
                            w_batch,
                            h_m,
                            u_m,
                            lut=lut_tensor,
                            group_size=eff_group_size,
                            cd_passes=cd_passes,
                        )
                        packed_shard[b0:b1].copy_(p_u8.cpu())
                        scales_shard[b0:b1].copy_(s_fp16.cpu())
                        del w_batch, q_deq, p_u8, s_fp16

                del w_shard_cpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                if hit_mask.any():
                    local_rows = (all_ngram_ids[hit_mask] - row_start).long()
                    all_quant_embeds[hit_mask] = dequantize_ple_rows_lut4(
                        packed_shard[local_rows],
                        scales_shard[local_rows],
                        lut_cpu,
                        group_size=eff_group_size,
                        out_dtype=torch.bfloat16,
                    )

                shard_tensors = {
                    f"{out_prefix}.weight_packed": packed_shard,
                    f"{out_prefix}.weight_scale": scales_shard,
                    f"{out_prefix}.weight_lut": lut_cpu.clone(),
                    f"{out_prefix}.weight_shape": torch.tensor([n_rows, d_h], dtype=torch.int64),
                }
            out_shard_fname = f"model-ple-{bucket_idx:03d}.safetensors"
            for tk, tv in shard_tensors.items():
                current_bucket[tk] = tv
                new_weight_map[tk] = out_shard_fname
                total_bytes += tv.numel() * tv.element_size()

            if (idx_in_list + 1) % shards_per_file == 0 or (idx_in_list + 1) == len(ple_shards):
                if pending_save_fut is not None:
                    pending_save_fut.result()
                bucket_to_save = dict(current_bucket)
                current_bucket.clear()
                save_path_str = str(output_dir / out_shard_fname)
                pending_save_fut = ple_io_pool.submit(save_file, bucket_to_save, save_path_str)
                log_event(
                    "QWEN4_EXP_PLE_SHARD_PROGRESS",
                    completed_shards=idx_in_list + 1,
                    total_shards=len(ple_shards),
                    out_file=out_shard_fname,
                    elapsed_s=round(time.time() - t_ple0, 2),
                )
                bucket_idx += 1

        if pending_save_fut is not None:
            pending_save_fut.result()

    del all_ngram_ids
    flat_embeds: list[torch.Tensor] = []
    cur = 0
    for bsz, slen in seq_shapes:
        n_el = bsz * slen * ngram_heads
        flat_embeds.append(all_quant_embeds[cur : cur + n_el].view(bsz, slen, ngram_heads * head_dim).clone())
        cur += n_el
    del all_quant_embeds
    return flat_embeds, new_weight_map, total_bytes, len(ple_shards)


@torch.no_grad()
def stream_gather_ple_embeds(
    ckpt_dir: Path,
    shard_map: dict[str, str],
    ple_layer: nn.Module,
    layer_idx: int,
    input_ids_list: list[torch.Tensor],
    *,
    ple_group_size: int = 32,
) -> list[torch.Tensor]:
    """Stream through PLE shards in parallel and gather [B, S, ple_embed_dim] embeddings for input_ids_list."""
    ple_emb = ple_layer.ple_embedding
    ngram_heads = ple_emb.ngram_heads
    head_dim = ple_layer.key_proj.weight.shape[1] // ngram_heads
    eos_token_id = ple_emb.eos_token_id
    dev = ple_layer.key_proj.weight.device

    seq_shapes = [(ids.shape[0], ids.shape[1]) for ids in input_ids_list]
    if all(b == 1 for b, _ in seq_shapes) and len(input_ids_list) > 1:
        max_len = max(s for _, s in seq_shapes)
        padded = torch.full((len(input_ids_list), max_len), eos_token_id, dtype=torch.long, device=dev)
        for i, ids in enumerate(input_ids_list):
            padded[i, : ids.shape[1]] = ids[0].to(dev)
        n_ids_batch = compute_ple_ngram_ids(
            padded,
            ple_emb.layer_multipliers,
            ple_emb.ngram_heads_vocab_sizes,
            ple_emb.ngram_heads_offsets,
            ple_emb.ngram_size,
            ple_emb.heads_per_ngram,
            eos_token_id,
        )
        all_ngram_ids_dev = torch.cat(
            [n_ids_batch[i, : seq_shapes[i][1]].reshape(-1) for i in range(len(input_ids_list))],
            dim=0,
        )
        del padded, n_ids_batch
    else:
        flat_n_ids_parts: list[torch.Tensor] = []
        for ids in input_ids_list:
            n_ids = compute_ple_ngram_ids(
                ids.to(dev),
                ple_emb.layer_multipliers,
                ple_emb.ngram_heads_vocab_sizes,
                ple_emb.ngram_heads_offsets,
                ple_emb.ngram_size,
                ple_emb.heads_per_ngram,
                eos_token_id,
            )
            flat_n_ids_parts.append(n_ids.reshape(-1))
        all_ngram_ids_dev = torch.cat(flat_n_ids_parts, dim=0)
        del flat_n_ids_parts

    all_embeds = torch.zeros((all_ngram_ids_dev.numel(), head_dim), dtype=torch.bfloat16)

    ple_shards = _discover_ple_shards(shard_map, ckpt_dir, layer_idx)
    by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for sid, kpref, sfname in ple_shards:
        by_file[sfname].append((sid, kpref))

    def _read_file_meta(sfname: str) -> list[tuple[int, str, str, int, int, bool]]:
        out = []
        with safe_open(str(ckpt_dir / sfname), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for sid, kpref in by_file[sfname]:
                if f"{kpref}.weight_packed" in keys:
                    wsh = f.get_tensor(f"{kpref}.weight_shape").tolist()
                    out.append((sid, kpref, sfname, int(wsh[0]), int(wsh[1]), True))
                else:
                    sh = f.get_slice(f"{kpref}.weight").get_shape()
                    out.append((sid, kpref, sfname, int(sh[0]), int(sh[1]), False))
        return out

    with ThreadPoolExecutor(max_workers=min(8, max(len(by_file), 1))) as pool:
        meta_lists = list(pool.map(_read_file_meta, by_file.keys()))
    meta = sorted((x for sub in meta_lists for x in sub), key=lambda x: x[0])

    file_tasks: dict[str, list[tuple[str, int, bool, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    global_row_cursor = 0
    for _sid, kpref, sfname, n_rows, d_h, is_q in meta:
        row_start = global_row_cursor
        row_end = row_start + n_rows
        global_row_cursor = row_end
        hit_idx = torch.nonzero(
            (all_ngram_ids_dev >= row_start) & (all_ngram_ids_dev < row_end), as_tuple=False
        ).squeeze(-1)
        if hit_idx.numel() > 0:
            local_rows = (all_ngram_ids_dev[hit_idx] - row_start).long().cpu()
            file_tasks[sfname].append((kpref, d_h, is_q, hit_idx.cpu(), local_rows))

    del all_ngram_ids_dev

    def _gather_file_raw(sfname: str):
        res = []
        with safe_open(str(ckpt_dir / sfname), framework="pt", device="cpu") as f:
            for kpref, d_h, is_q, hit_idx_cpu, local_rows in file_tasks[sfname]:
                if is_q:
                    wp = f.get_tensor(f"{kpref}.weight_packed")[local_rows]
                    ws = f.get_tensor(f"{kpref}.weight_scale")[local_rows]
                    wlut = f.get_tensor(f"{kpref}.weight_lut")
                    res.append((hit_idx_cpu, d_h, True, (wp, ws, wlut)))
                else:
                    w = f.get_tensor(f"{kpref}.weight")[local_rows]
                    res.append((hit_idx_cpu, d_h, False, w))
        return res

    if file_tasks:
        with ThreadPoolExecutor(max_workers=min(8, len(file_tasks))) as pool:
            for file_res in pool.map(_gather_file_raw, file_tasks.keys()):
                for hit_idx_cpu, d_h, is_q, payload in file_res:
                    if is_q:
                        wp, ws, wlut = payload
                        eff_g = min(ple_group_size, d_h)
                        deq = dequantize_ple_rows_lut4(
                            wp.to(dev), ws.to(dev), wlut.to(dev), group_size=eff_g, out_dtype=torch.bfloat16
                        ).cpu()
                        all_embeds[hit_idx_cpu] = deq
                    else:
                        all_embeds[hit_idx_cpu] = payload.to(torch.bfloat16)

    embeds_list: list[torch.Tensor] = []
    cur = 0
    for bsz, slen in seq_shapes:
        n_el = bsz * slen * ngram_heads
        embeds_list.append(all_embeds[cur : cur + n_el].view(bsz, slen, ngram_heads * head_dim).clone())
        cur += n_el
    del all_embeds
    return embeds_list


@torch.no_grad()
def forward_ple_with_precomputed_embeds(
    ple_layer: nn.Module,
    hidden_states: torch.Tensor,
    embeddings: torch.Tensor,
    conv_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute Qwen4ExpTextPLELayer.forward given pre-gathered n-gram embeddings [B, S, ple_embed_dim]."""
    embeddings = embeddings.to(device=hidden_states.device, dtype=hidden_states.dtype)
    key_normed = ple_layer.norm_key(ple_layer.key_proj(embeddings)).unflatten(
        -1, (ple_layer.hc_count, ple_layer.hidden_size)
    )
    value = ple_layer.value_proj(embeddings)
    query_normed = ple_layer.norm_query(hidden_states).unflatten(
        -1, (ple_layer.hc_count, ple_layer.hidden_size)
    )
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(ple_layer.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = ple_layer.norm_conv(gated_value.flatten(-2))
    gated_value = gated_value.flatten(-2)
    if conv_mask is not None:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_mask_to_padding_states

        gated_value = apply_mask_to_padding_states(gated_value, conv_mask)
        gated_value_normed = apply_mask_to_padding_states(gated_value_normed, conv_mask)
    output = gated_value + ple_layer._short_conv(gated_value_normed, past_key_values=None)
    return output


def _build_lightweight_decoder_layer(text_config, layer_idx: int, device: torch.device) -> nn.Module:
    """Construct Qwen4ExpTextDecoderLayer or Qwen3_5MoeDecoderLayer on device without allocating the 102.4 GB PLE embedding table."""
    from llmcompressor.utils.dev import skip_weights_initialize

    cfg_copy = copy.deepcopy(text_config)
    _ensure_qwen4_exp_device_kernels(cfg_copy, device)

    if _is_qwen35_moe_text_config(cfg_copy):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

        with torch.device(device), skip_weights_initialize():
            layer = Qwen3_5MoeDecoderLayer(cfg_copy, layer_idx).to(dtype=torch.bfloat16).eval()
        layer.ple = None
        return layer

    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

    orig_base = getattr(cfg_copy, "ngram_vocab_size_base", 64)
    cfg_copy.ngram_vocab_size_base = 16
    cfg_copy.make_ngram_vocab_size_divisible_by = 4

    with torch.device(device), skip_weights_initialize():
        layer = Qwen4ExpTextDecoderLayer(cfg_copy, layer_idx).to(dtype=torch.bfloat16).eval()

    if layer.ple is not None:
        # Restore true metadata on ple_embedding and drop the dummy nn.Embedding parameter
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            _build_layer_multipliers,
            _find_nth_prime_after,
        )

        ple_emb = layer.ple.ple_embedding
        ple_emb.ngram_vocab_size_base = orig_base
        ple_emb.head_vocab_sizes = []
        ple_emb.head_offsets = []
        ple_emb.total_vocab_size = 0
        for head_idx in range(ple_emb.ngram_heads):
            global_head_idx = ple_emb.ple_layer_index * ple_emb.ngram_heads + head_idx
            size = _find_nth_prime_after(orig_base - 1, global_head_idx + 1)
            ple_emb.head_vocab_sizes.append(size)
            ple_emb.head_offsets.append(ple_emb.total_vocab_size)
            ple_emb.total_vocab_size += size
        ple_emb.layer_multipliers.data = _build_layer_multipliers(
            ple_emb.unigram_vocab_size, ple_emb.ngram_size, ple_emb.ple_layer_index, ple_emb.seed
        ).to(device)
        ple_emb.ngram_heads_vocab_sizes.data = torch.tensor(
            ple_emb.head_vocab_sizes, dtype=torch.long, device=device
        )
        ple_emb.ngram_heads_offsets.data = torch.tensor(ple_emb.head_offsets, dtype=torch.long, device=device)
        ple_emb.ngram_embedding = nn.Identity()

    return layer


def _prefetch_layer_raw_tensors(
    layer_prefix: str,
    ckpt_dir: Path,
    shard_map: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Read raw CPU tensors for a decoder layer from safetensors shards and pre-stack expert tensors."""
    prefix_dot = f"{layer_prefix}."
    ple_shard_prefix = f"{layer_prefix}.ple.ple_embedding.ngram_embedding"
    if shard_map:
        rel_shards = sorted(
            {
                ckpt_dir / fname
                for k, fname in shard_map.items()
                if k.startswith(prefix_dot) and not k.startswith(ple_shard_prefix)
            }
        )
    else:
        rel_shards = sorted(ckpt_dir.glob("*.safetensors"))

    raw_tensors: dict[str, torch.Tensor] = {}
    for sp in rel_shards:
        if len(rel_shards) == 1 and sp.name.startswith(("model-layer-", "model-mtp")):
            loaded = load_safetensors_bytes(sp.read_bytes())
            for k, v in loaded.items():
                if k.startswith(prefix_dot) and not k.startswith(ple_shard_prefix):
                    raw_tensors[k.removeprefix(prefix_dot)] = v
            del loaded
        else:
            with safe_open(str(sp), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith(prefix_dot) and not k.startswith(ple_shard_prefix):
                        raw_tensors[k.removeprefix(prefix_dot)] = f.get_tensor(k)

    # Pre-concatenate / pre-stack routed expert weights in CPU RAM so the main GPU thread does a single copy
    if "mlp.experts.0.gate_proj.weight_packed" in raw_tensors:
        num_exp = 0
        while f"mlp.experts.{num_exp}.gate_proj.weight_packed" in raw_tensors:
            num_exp += 1
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            wps = [raw_tensors.pop(f"mlp.experts.{e}.{proj_name}.weight_packed") for e in range(num_exp)]
            wss = [raw_tensors.pop(f"mlp.experts.{e}.{proj_name}.weight_scale") for e in range(num_exp)]
            for e in range(num_exp):
                raw_tensors.pop(f"mlp.experts.{e}.{proj_name}.weight_shape", None)
            raw_tensors[f"__cat_{proj_name}.weight_packed"] = torch.cat(wps, dim=0)
            raw_tensors[f"__cat_{proj_name}.weight_scale"] = torch.cat(wss, dim=0)
            del wps, wss
    elif "mlp.experts.0.gate_proj.weight" in raw_tensors:
        num_exp = 0
        while f"mlp.experts.{num_exp}.gate_proj.weight" in raw_tensors:
            num_exp += 1
        gws = [raw_tensors.pop(f"mlp.experts.{e}.gate_proj.weight") for e in range(num_exp)]
        uws = [raw_tensors.pop(f"mlp.experts.{e}.up_proj.weight") for e in range(num_exp)]
        dws = [raw_tensors.pop(f"mlp.experts.{e}.down_proj.weight") for e in range(num_exp)]
        raw_tensors["mlp.experts.gate_up_proj"] = torch.cat(
            [torch.stack(gws, dim=0), torch.stack(uws, dim=0)], dim=1
        )
        raw_tensors["mlp.experts.down_proj"] = torch.stack(dws, dim=0)
        del gws, uws, dws

    return raw_tensors


def _load_decoder_layer_weights(
    layer: nn.Module,
    layer_prefix: str,
    ckpt_dir: Path,
    shard_map: dict[str, str],
    device: torch.device,
    *,
    group_size: int = 64,
    hadamard_block_size: int = 0,
    prefetched_raw: dict[str, torch.Tensor] | None = None,
) -> None:
    """Load a BF16 or compressed-tensors W4A16 (+ online Hadamard) decoder layer from ckpt_dir."""
    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    raw_tensors = (
        prefetched_raw
        if prefetched_raw is not None
        else _prefetch_layer_raw_tensors(layer_prefix, ckpt_dir, shard_map)
    )

    packed_prefixes = sorted(
        k.removesuffix(".weight_packed")
        for k in raw_tensors
        if k.endswith(".weight_packed") and not k.startswith("__cat_")
    )
    packed_prefix_set = set(packed_prefixes)

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type="int", symmetric=True, strategy="group", group_size=group_size
        ),
    )

    experts = layer.mlp.experts
    num_exp = experts.num_experts
    inter = experts.intermediate_dim
    hidden = experts.hidden_dim

    # 1. Load routed experts (either pre-concatenated W4A16, packed W4A16, 2D per-expert BF16, or 3D fused BF16)
    if "__cat_gate_proj.weight_packed" in raw_tensors:
        for proj_name, out_dim, in_dim in (
            ("gate_proj", inter, hidden),
            ("up_proj", inter, hidden),
            ("down_proj", hidden, inter),
        ):
            wp_cat = raw_tensors.pop(f"__cat_{proj_name}.weight_packed").to(device)
            ws_cat = raw_tensors.pop(f"__cat_{proj_name}.weight_scale").to(device)
            deq = PackedQuantizationCompressor.decompress(
                {
                    "weight_packed": wp_cat,
                    "weight_scale": ws_cat,
                    "weight_shape": torch.tensor([num_exp * out_dim, in_dim]),
                },
                scheme,
            )["weight"].view(num_exp, out_dim, in_dim)
            if proj_name == "gate_proj":
                experts.gate_up_proj.data[:, :inter].copy_(deq)
            elif proj_name == "up_proj":
                experts.gate_up_proj.data[:, inter:].copy_(deq)
            else:
                experts.down_proj.data.copy_(deq)
            del wp_cat, ws_cat, deq
        if hadamard_block_size > 0:
            _attach_expert_hadamard_hooks(
                experts,
                min(hadamard_block_size, hidden & -hidden),
                min(hadamard_block_size, inter & -inter),
            )
    elif "mlp.experts.0.gate_proj" in packed_prefix_set:
        for proj_name, out_dim, in_dim in (
            ("gate_proj", inter, hidden),
            ("up_proj", inter, hidden),
            ("down_proj", hidden, inter),
        ):
            wps = [raw_tensors.pop(f"mlp.experts.{e}.{proj_name}.weight_packed") for e in range(num_exp)]
            wss = [raw_tensors.pop(f"mlp.experts.{e}.{proj_name}.weight_scale") for e in range(num_exp)]
            for e in range(num_exp):
                raw_tensors.pop(f"mlp.experts.{e}.{proj_name}.weight_shape", None)
            deq = PackedQuantizationCompressor.decompress(
                {
                    "weight_packed": torch.cat(wps, dim=0).to(device),
                    "weight_scale": torch.cat(wss, dim=0).to(device),
                    "weight_shape": torch.tensor([num_exp * out_dim, in_dim]),
                },
                scheme,
            )["weight"].view(num_exp, out_dim, in_dim)
            if proj_name == "gate_proj":
                experts.gate_up_proj.data[:, :inter].copy_(deq)
            elif proj_name == "up_proj":
                experts.gate_up_proj.data[:, inter:].copy_(deq)
            else:
                experts.down_proj.data.copy_(deq)
            del wps, wss, deq
        if hadamard_block_size > 0:
            _attach_expert_hadamard_hooks(
                experts,
                min(hadamard_block_size, hidden & -hidden),
                min(hadamard_block_size, inter & -inter),
            )
    elif "mlp.experts.0.gate_proj.weight" in raw_tensors:
        for e in range(num_exp):
            gw = raw_tensors.pop(f"mlp.experts.{e}.gate_proj.weight")
            uw = raw_tensors.pop(f"mlp.experts.{e}.up_proj.weight")
            dw = raw_tensors.pop(f"mlp.experts.{e}.down_proj.weight")
            experts.gate_up_proj.data[e, :inter].copy_(gw.to(device=device, dtype=torch.bfloat16))
            experts.gate_up_proj.data[e, inter:].copy_(uw.to(device=device, dtype=torch.bfloat16))
            experts.down_proj.data[e].copy_(dw.to(device=device, dtype=torch.bfloat16))
    elif "mlp.experts.gate_up_proj" in raw_tensors:
        experts.gate_up_proj.data.copy_(
            raw_tensors.pop("mlp.experts.gate_up_proj").to(device=device, dtype=torch.bfloat16)
        )
        experts.down_proj.data.copy_(
            raw_tensors.pop("mlp.experts.down_proj").to(device=device, dtype=torch.bfloat16)
        )
    else:
        raise RuntimeError(f"Missing routed expert weights for {layer_prefix}")

    # 2. Load any packed 2D linears (self_attn / linear_attn) and attach online Hadamard pre-hook if needed
    for p_rel in packed_prefixes:
        if p_rel.startswith("mlp.experts."):
            continue
        wp = raw_tensors.pop(f"{p_rel}.weight_packed").to(device)
        ws = raw_tensors.pop(f"{p_rel}.weight_scale").to(device)
        wsh = raw_tensors.pop(f"{p_rel}.weight_shape")
        deq = PackedQuantizationCompressor.decompress(
            {"weight_packed": wp, "weight_scale": ws, "weight_shape": wsh},
            scheme,
        )["weight"].to(torch.bfloat16)
        mod = layer.get_submodule(p_rel)
        mod.weight.data.copy_(deq)
        if hadamard_block_size > 0:
            k_in = int(deq.shape[1])
            b_size = min(hadamard_block_size, k_in & -k_in)

            def _make_had_hook(bs: int):
                def _pre_hook(_m, args):
                    x = _hadamard_rotate_last_dim(args[0], bs, compute_dtype=torch.float32)
                    return (x, *args[1:])

                return _pre_hook

            if b_size > 1:
                mod.register_forward_pre_hook(_make_had_hook(b_size))

    # 3. Copy remaining dense parameters and buffers
    param_map = dict(layer.named_parameters())
    buf_map = dict(layer.named_buffers())
    for k_rel, tensor in raw_tensors.items():
        if k_rel in param_map:
            param_map[k_rel].data.copy_(tensor.to(device=device, dtype=param_map[k_rel].dtype))
        elif k_rel in buf_map:
            buf_map[k_rel].data.copy_(tensor.to(device=device, dtype=buf_map[k_rel].dtype))
        else:
            raise RuntimeError(f"Unexpected tensor key in {layer_prefix}: {k_rel}")


@torch.no_grad()
def _gptq_quantize_and_pack_2d(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    num_samples: int,
    *,
    group_size: int,
    actorder: str | None,
    dampening_frac: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run Cholesky GPTQ on a 2D weight tensor [N, K], returning (dequantized_bf16_on_device, cpu_packed_dict)."""
    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from compressed_tensors.quantization.utils import calculate_qparams
    from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight

    w_f32 = weight.detach().float()
    n_out, k_in = w_f32.shape
    eff_group_size = min(group_size, k_in)
    norm_h = (hessian / max(num_samples, 1)).to(torch.float32)

    qargs = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=eff_group_size,
        actorder=actorder,
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=qargs)
    grouped = weight.detach().view(n_out, k_in // eff_group_size, eff_group_size)
    scale, zp = calculate_qparams(grouped.amin(dim=-1), grouped.amax(dim=-1), qargs)
    scale = scale.to(weight.dtype)

    q_w_batch, _losses, used_rtn = quantize_weight(
        weights=w_f32.unsqueeze(0),
        hessians=norm_h.unsqueeze(0).clone(),
        scale=scale.unsqueeze(0),
        zero_point=zp.unsqueeze(0),
        global_scale=None,
        quant_args=qargs,
        blocksize=128,
        percdamp=dampening_frac,
    )
    if used_rtn[0].item():
        raise RuntimeError("GPTQ fell back to RTN on 2D linear module")
    q_w = q_w_batch[0].to(weight.dtype)
    compressed = PackedQuantizationCompressor.compress(
        {"weight": q_w, "weight_scale": scale},
        scheme,
    )
    packed_cpu = {k: v.contiguous().cpu() for k, v in compressed.items()}
    return q_w, packed_cpu


@torch.no_grad()
def _gptq_quantize_and_pack_3d_experts(
    weights_3d: torch.Tensor,
    hessians: torch.Tensor,
    *,
    group_size: int,
    actorder: str | None,
    dampening_frac: float,
    expert_chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-GPTQ and batch-pack a 3D expert tensor [E, N_out, K_in].

    hessians can be either [1, K_in, K_in] (shared across all experts, e.g. gate_proj / up_proj)
    or [E, K_in, K_in] (per-expert, e.g. down_proj).
    Returns:
      q_w_3d: [E, N_out, K_in] bfloat16 on device
      packed_3d_cpu: [E, N_out, K_in // 8] int32 on CPU
      scales_3d_cpu: [E, N_out, K_in // group_size] bfloat16 on CPU
      shape_tensor_cpu: [2] int64 ([N_out, K_in])
    """
    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from compressed_tensors.quantization.utils import calculate_qparams
    from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight

    num_exp, n_out, k_in = weights_3d.shape
    eff_group_size = min(group_size, k_in)
    num_groups = k_in // eff_group_size

    qargs = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=eff_group_size,
        actorder=actorder,
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=qargs)

    q_w_3d = torch.empty_like(weights_3d)
    all_grouped = weights_3d.detach().view(num_exp, n_out, num_groups, eff_group_size)
    all_scale_3d, all_zp_3d = calculate_qparams(
        all_grouped.amin(dim=-1),
        all_grouped.amax(dim=-1),
        qargs,
    )
    all_scale_3d = all_scale_3d.to(weights_3d.dtype)
    shared_h = hessians.shape[0] == 1

    if shared_h:
        # Stack expert_chunk experts along N_out so Cholesky is computed only once per chunk
        for e0 in range(0, num_exp, expert_chunk):
            e1 = min(e0 + expert_chunk, num_exp)
            cur_e = e1 - e0
            w_chunk = weights_3d[e0:e1].reshape(1, cur_e * n_out, k_in).float()
            scale_chunk = all_scale_3d[e0:e1].reshape(1, cur_e * n_out, num_groups)
            zp_chunk = all_zp_3d[e0:e1].reshape(1, cur_e * n_out, num_groups)
            q_chunk, _losses, used_rtn = quantize_weight(
                weights=w_chunk,
                hessians=hessians.float().clone(),
                scale=scale_chunk,
                zero_point=zp_chunk,
                global_scale=None,
                quant_args=qargs,
                blocksize=128,
                percdamp=dampening_frac,
            )
            if used_rtn.any().item():
                raise RuntimeError(f"GPTQ fell back to RTN on shared-H experts [{e0}:{e1}]")
            q_w_3d[e0:e1].copy_(q_chunk[0].view(cur_e, n_out, k_in).to(weights_3d.dtype))
            del w_chunk, scale_chunk, zp_chunk, q_chunk
    else:
        for e0 in range(0, num_exp, expert_chunk):
            e1 = min(e0 + expert_chunk, num_exp)
            w_chunk = weights_3d[e0:e1].float()
            scale_chunk = all_scale_3d[e0:e1]
            zp_chunk = all_zp_3d[e0:e1]
            q_chunk, _losses, used_rtn = quantize_weight(
                weights=w_chunk,
                hessians=hessians[e0:e1].float().clone(),
                scale=scale_chunk,
                zero_point=zp_chunk,
                global_scale=None,
                quant_args=qargs,
                blocksize=128,
                percdamp=dampening_frac,
            )
            if used_rtn.any().item():
                raise RuntimeError(f"GPTQ fell back to RTN on per-expert H [{e0}:{e1}]")
            q_w_3d[e0:e1].copy_(q_chunk.to(weights_3d.dtype))
            del w_chunk, scale_chunk, zp_chunk, q_chunk

    # Batch-pack all experts in one call using the exact observer scales
    all_scale_2d = all_scale_3d.reshape(num_exp * n_out, num_groups)
    comp = PackedQuantizationCompressor.compress(
        {"weight": q_w_3d.view(num_exp * n_out, k_in), "weight_scale": all_scale_2d},
        scheme,
    )
    packed_3d_cpu = comp["weight_packed"].view(num_exp, n_out, -1).contiguous().cpu()
    scales_3d_cpu = all_scale_3d.contiguous().cpu()
    shape_tensor_cpu = torch.tensor([n_out, k_in], dtype=torch.int64)
    return q_w_3d, packed_3d_cpu, scales_3d_cpu, shape_tensor_cpu


@torch.no_grad()
def _compute_vectorized_causal_qsa_mask(
    indexer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Vectorized bit-exact equivalent of Qwen4ExpTextQSAIndexer.forward for causal attention (S > token_budget)."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    bsz, seq_len, _ = hidden_states.shape
    dev = hidden_states.device
    r_comp = indexer.compress_ratio
    d_idx = indexer.index_head_dim
    b_topk = indexer.block_topk

    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_len:, :], full_sin[:, -seq_len:, :]

    qk = indexer.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk,
        [indexer.index_n_heads * d_idx, indexer.index_kv_heads * d_idx],
        dim=-1,
    )
    q = indexer.q_layernorm(q.view(bsz, seq_len, -1, d_idx))
    q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
    raw_keys = token_k.view(bsz, seq_len, d_idx)

    total_blocks = seq_len // r_comp
    key_groups = raw_keys[:, : total_blocks * r_comp, :].view(bsz, total_blocks, r_comp, d_idx)
    pooled_keys = indexer.k_layernorm(key_groups.float().mean(dim=2).to(raw_keys.dtype))
    group_starts = torch.arange(total_blocks, device=dev) * r_comp
    block_key_states = apply_rotary_pos_emb(
        pooled_keys.unsqueeze(2),
        cos=full_cos.index_select(1, group_starts),
        sin=full_sin.index_select(1, group_starts),
        unsqueeze_dim=2,
    ).squeeze(2)

    # Queries with num_complete_blocks <= b_topk select all causal tokens; only q >= b_topk * r_comp - 1 need topk
    q_start = b_topk * r_comp - 1
    q_sub = q[:, q_start:, :, :]
    scores = torch.einsum("bshd,btd->bsht", q_sub.float(), block_key_states.float())
    scores = torch.relu(scores).sum(dim=2) / math.sqrt(d_idx)

    q_indices = torch.arange(q_start, seq_len, device=dev)
    num_comp = (q_indices + 1) // r_comp
    block_idx = torch.arange(total_blocks, device=dev).unsqueeze(0)
    valid_blocks = block_idx < num_comp.unsqueeze(1)
    scores = scores.masked_fill(~valid_blocks.unsqueeze(0), float("-inf"))
    top_blocks = scores.topk(b_topk, dim=-1).indices

    block_mask = torch.zeros((bsz, seq_len - q_start, total_blocks), dtype=torch.bool, device=dev)
    block_mask.scatter_(-1, top_blocks, True)
    token_mask_sub = (
        block_mask.unsqueeze(-1)
        .expand(-1, -1, -1, r_comp)
        .reshape(bsz, seq_len - q_start, total_blocks * r_comp)
    )
    if total_blocks * r_comp < seq_len:
        pad_tail = torch.zeros(
            (bsz, seq_len - q_start, seq_len - total_blocks * r_comp), dtype=torch.bool, device=dev
        )
        token_mask_sub = torch.cat([token_mask_sub, pad_tail], dim=-1)

    k_idx = torch.arange(seq_len, device=dev).unsqueeze(0)
    tail_mask = (k_idx >= (num_comp * r_comp).unsqueeze(1)) & (k_idx <= q_indices.unsqueeze(1))
    token_mask_sub = token_mask_sub | tail_mask.unsqueeze(0)

    fast_mask = torch.tril(torch.ones((bsz, 1, seq_len, seq_len), dtype=torch.bool, device=dev))
    fast_mask[:, 0, q_start:, :] = token_mask_sub
    return fast_mask


@torch.no_grad()
def _forward_qsa_attention_fast(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Qwen4ExpTextAttention or Qwen3_5MoeAttention; uses exact SDPA when S <= budget and vectorized causal indexer when S > budget."""
    seq_len = hidden_states.shape[1]
    if attention_mask is None:
        has_indexer = getattr(attn, "indexer", None) is not None
        if has_indexer:
            from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

            sdpa_mask = (
                None
                if seq_len <= attn.indexer.token_budget
                else _compute_vectorized_causal_qsa_mask(attn.indexer, hidden_states, position_embeddings)
            )
        else:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_rotary_pos_emb

            sdpa_mask = None
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query_states, gate = torch.chunk(
            attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=sdpa_mask,
            is_causal=(sdpa_mask is None),
            scale=attn.scaling,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return attn.o_proj(attn_output)

    out, _ = attn(hidden_states, position_embeddings, attention_mask=attention_mask, past_key_values=None)
    return out


@torch.no_grad()
def _quantize_decoder_layer_streaming(
    layer: nn.Module,
    layer_prefix: str,
    h_calib: list[torch.Tensor],
    pos_emb_list: list[tuple[torch.Tensor, torch.Tensor]],
    calib_ids_list: list[torch.Tensor],
    *,
    base_dir: Path,
    shard_map: dict[str, str],
    output_dir: Path,
    layer_idx: int,
    group_size: int,
    hadamard_block_size: int,
    actorder: str | None,
    dampening_frac: float,
    ple_group_size: int = 32,
    ple_cd_passes: int = 1,
    dense_expert_calib_seqs: int = 32,
    reuse_ple_dir: Path | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, str], int, dict[str, int]]:
    """Quantize a single Qwen4ExpTextDecoderLayer or Qwen3_5MoeDecoderLayer (or MTP layer) in-place, advance h_calib, and return shard dict."""
    device = next(layer.parameters()).device
    n_seqs = len(h_calib)
    counts = {"attn_linears": 0, "shared_experts": 0, "routed_experts": 0, "ple_shards": 0}
    ple_weight_map: dict[str, str] = {}
    ple_bytes = 0

    # ------------------------------------------------------------------
    # Stage A: Layer-1 PLE (runs FIRST in Qwen4ExpTextDecoderLayer.forward!)
    # ------------------------------------------------------------------
    keep_calib_on_device = h_calib[0].device == device
    if getattr(layer, "ple", None) is not None:
        reuse_smap: dict[str, str] = {}
        reuse_ple_shards: list[tuple[int, str, str]] = []
        reuse_ple_files: list[str] = []
        if reuse_ple_dir is not None and (reuse_ple_dir / "model.safetensors.index.json").exists():
            reuse_smap, _ = _load_shard_map(reuse_ple_dir)
            reuse_ple_shards = _discover_ple_shards(reuse_smap, reuse_ple_dir, layer_idx)
            reuse_ple_files = sorted({fname for _sid, _kpref, fname in reuse_ple_shards})
        can_reuse_ple = bool(reuse_ple_files) and all(
            (reuse_ple_dir / fname).exists() for fname in reuse_ple_files
        )
        if can_reuse_ple:
            import shutil

            ple_prefix = f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding."
            for sname in reuse_ple_files:
                src_p = reuse_ple_dir / sname
                dst_p = output_dir / sname
                if not dst_p.exists():
                    try:
                        os.link(src_p, dst_p)
                    except OSError:
                        shutil.copy2(src_p, dst_p)
                with safe_open(str(dst_p), framework="pt", device="cpu") as sf:
                    for k in sf.keys():
                        t = sf.get_tensor(k)
                        ple_bytes += t.numel() * t.element_size()
                        ple_weight_map[k] = sname
            for k, v in reuse_smap.items():
                if k.startswith(ple_prefix):
                    ple_weight_map[k] = v
            n_ple_shards = len(reuse_ple_shards)
            quant_ple_embeds = stream_gather_ple_embeds(
                output_dir,
                ple_weight_map,
                layer.ple,
                layer_idx,
                calib_ids_list,
                ple_group_size=ple_group_size,
            )
            log_event("QWEN4_EXP_PLE_REUSED", reuse_dir=str(reuse_ple_dir), shards=n_ple_shards)
        else:
            quant_ple_embeds, ple_weight_map, ple_bytes, n_ple_shards = stream_quantize_ple_shards(
                base_dir,
                shard_map,
                output_dir,
                layer_idx,
                layer.ple,
                calib_ids_list,
                device=device,
                ple_group_size=ple_group_size,
                dampening_frac=dampening_frac,
                cd_passes=ple_cd_passes,
            )
        counts["ple_shards"] = n_ple_shards
        for i in range(n_seqs):
            h_dev = h_calib[i].to(device)
            ple_out = forward_ple_with_precomputed_embeds(layer.ple, h_dev, quant_ple_embeds[i])
            res_ple = h_dev + ple_out
            h_calib[i] = res_ple if keep_calib_on_device else res_ple.cpu()
        del quant_ple_embeds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    layer_tensors: dict[str, torch.Tensor] = {}

    # Pack all calibration sequences into a single contiguous tensor H_all [N_tot, hc_dim]
    # slice-by-slice while freeing per-sequence buffers so 999x4096 never doubles H_all VRAM.
    # When H_all exceeds 24 GiB (e.g. Qwen3.8-Flash-Next hc_count=4 -> 66.6 GiB at 999x4096),
    # keep H_all in host RAM and stream tok_step=65536 chunks to GPU.
    seq_lens = [h.shape[1] for h in h_calib]
    n_tot = sum(seq_lens)
    sample_h = h_calib[0].squeeze(0)
    h_all_bytes = n_tot * sample_h.shape[-1] * sample_h.element_size()
    h_all_device = device if (device.type != "cuda" or h_all_bytes <= 24 * (1024**3)) else torch.device("cpu")
    H_all = torch.empty((n_tot, *sample_h.shape[1:]), dtype=sample_h.dtype, device=h_all_device)
    cur_off = 0
    for i, slen in enumerate(seq_lens):
        H_all[cur_off : cur_off + slen].copy_(h_calib[i].squeeze(0), non_blocking=True)
        h_calib[i] = torch.empty(0, device=h_calib[i].device, dtype=H_all.dtype)
        cur_off += slen
    has_hc = hasattr(layer, "attn_hyper_connection")
    hc_count = layer.attn_hyper_connection.hc_count if has_hc else 1
    tok_step = 65536

    # ------------------------------------------------------------------
    # Stage B: Attention / GDN Sub-block (W4A16-HGPTQ)
    # ------------------------------------------------------------------
    layer_kind = getattr(layer, "layer_type", getattr(layer, "block_type", ""))
    is_linear_attn = layer_kind == "linear_attention"
    attn_mod = layer.linear_attn if is_linear_attn else layer.self_attn
    in_proj_names = ("in_proj_qkv", "in_proj_z") if is_linear_attn else ("q_proj", "k_proj", "v_proj")
    out_proj_name = "out_proj" if is_linear_attn else "o_proj"
    out_proj_mod: nn.Linear = getattr(attn_mod, out_proj_name)

    h_dim = layer.attn_hyper_connection.hidden_size if has_hc else layer.hidden_size
    had_in = min(hadamard_block_size, h_dim & -h_dim) if hadamard_block_size > 0 else 0
    k_out = out_proj_mod.weight.shape[1]
    had_out = min(hadamard_block_size, k_out & -k_out) if hadamard_block_size > 0 else 0

    # B.1 Fused attn_hyper_connection / input_layernorm + shared input Hessian on device
    h_attn_in = torch.zeros((h_dim, h_dim), dtype=torch.float32, device=device)
    branch_in_all = torch.empty((n_tot, h_dim), dtype=H_all.dtype, device=device)
    inj_w_all = torch.empty((n_tot, hc_count), dtype=H_all.dtype, device=device) if has_hc else None

    for s0 in range(0, n_tot, tok_step):
        s1 = min(s0 + tok_step, n_tot)
        h_slice = H_all[s0:s1].to(device, non_blocking=True)
        if has_hc:
            b_in, hyp_in, iw = layer.attn_hyper_connection(h_slice.unsqueeze(0))
            b_in_2d = b_in.squeeze(0)
            branch_in_all[s0:s1] = b_in_2d
            H_all[s0:s1].copy_(hyp_in.squeeze(0), non_blocking=True)
            inj_w_all[s0:s1] = iw.squeeze(0)
            del b_in, hyp_in, iw
        else:
            b_in_2d = layer.input_layernorm(h_slice)
            branch_in_all[s0:s1] = b_in_2d
        if had_in > 0:
            x_2d = _hadamard_rotate_last_dim(b_in_2d, had_in, compute_dtype=torch.float32).float()
        else:
            x_2d = b_in_2d.float()
        h_attn_in.addmm_(x_2d.T, x_2d, beta=1.0, alpha=2.0)
        del h_slice, b_in_2d, x_2d

    # Quantize input projections and attach online Hadamard rotation hook
    in_hooks = []
    for pname in in_proj_names:
        pmod: nn.Linear = getattr(attn_mod, pname)
        if had_in > 0:
            pmod.weight.data = _hadamard_rotate_weight(pmod.weight.data, had_in)
        q_w, packed_dict = _gptq_quantize_and_pack_2d(
            pmod.weight.data,
            h_attn_in,
            n_tot,
            group_size=group_size,
            actorder=actorder,
            dampening_frac=dampening_frac,
        )
        pmod.weight.data.copy_(q_w)
        sub_prefix = f"{layer_prefix}.{'linear_attn' if is_linear_attn else 'self_attn'}.{pname}"
        for pk, pv in packed_dict.items():
            layer_tensors[f"{sub_prefix}.{pk}"] = pv
        counts["attn_linears"] += 1
        if had_in > 0:
            in_hooks.append(
                pmod.register_forward_pre_hook(
                    lambda _m, args, bs=had_in: (
                        _hadamard_rotate_last_dim(args[0], bs, compute_dtype=torch.float32),
                        *args[1:],
                    )
                )
            )
    del h_attn_in

    # B.2 Accumulate output projection Hessian (o_proj / out_proj) and cache pre_out_all directly on device
    h_attn_out = torch.zeros((k_out, k_out), dtype=torch.float32, device=device)
    pre_out_all = torch.empty((n_tot, k_out), dtype=H_all.dtype, device=device)
    cur_out_tok = 0

    def _out_hessian_hook(_m, args, _pre_out=pre_out_all, _h_out=h_attn_out):
        nonlocal cur_out_tok
        raw_in = args[0].reshape(-1, k_out)
        n_cur = raw_in.shape[0]
        if had_out > 0:
            rot_in = _hadamard_rotate_last_dim(raw_in, had_out, compute_dtype=torch.float32)
            _pre_out[cur_out_tok : cur_out_tok + n_cur] = rot_in.to(_pre_out.dtype)
            x_o = rot_in.float()
        else:
            _pre_out[cur_out_tok : cur_out_tok + n_cur] = raw_in
            x_o = raw_in.float()
        cur_out_tok += n_cur
        _h_out.addmm_(x_o.T, x_o, beta=1.0, alpha=2.0)
        return args

    out_hook = out_proj_mod.register_forward_pre_hook(_out_hessian_hook)
    try:
        seq_cur = 0
        for i, slen in enumerate(seq_lens):
            branch_in = branch_in_all[seq_cur : seq_cur + slen].unsqueeze(0)
            seq_cur += slen
            if is_linear_attn:
                _ = attn_mod(branch_in)
            else:
                pos_gpu = (pos_emb_list[i][0].to(device), pos_emb_list[i][1].to(device))
                _ = _forward_qsa_attention_fast(attn_mod, branch_in, pos_gpu)
    finally:
        out_hook.remove()
    del branch_in_all

    for h in in_hooks:
        h.remove()

    # Quantize o_proj / out_proj
    if had_out > 0:
        out_proj_mod.weight.data = _hadamard_rotate_weight(out_proj_mod.weight.data, had_out)
    q_w_out, packed_out_dict = _gptq_quantize_and_pack_2d(
        out_proj_mod.weight.data,
        h_attn_out,
        n_tot,
        group_size=group_size,
        actorder=actorder,
        dampening_frac=dampening_frac,
    )
    out_proj_mod.weight.data.copy_(q_w_out)
    out_prefix = f"{layer_prefix}.{'linear_attn' if is_linear_attn else 'self_attn'}.{out_proj_name}"
    for pk, pv in packed_out_dict.items():
        layer_tensors[f"{out_prefix}.{pk}"] = pv
    counts["attn_linears"] += 1
    del h_attn_out

    # ------------------------------------------------------------------
    # Stage C: Sparse MoE Sub-block (Shared Expert W4A16-HGPTQ + Batched Expert W4A16-HGPTQ)
    # ------------------------------------------------------------------
    experts = layer.mlp.experts
    shared_exp = layer.mlp.shared_expert
    num_exp = experts.num_experts
    inter = experts.intermediate_dim
    hidden = experts.hidden_dim
    shared_inter = shared_exp.down_proj.weight.shape[1]
    had_moe_in = min(hadamard_block_size, hidden & -hidden) if hadamard_block_size > 0 else 0
    had_moe_down = min(hadamard_block_size, inter & -inter) if hadamard_block_size > 0 else 0
    had_shared_down = min(hadamard_block_size, shared_inter & -shared_inter) if hadamard_block_size > 0 else 0

    # B.3 + C.1: Fused on-device out_proj + mlp_hyper_connection / post_attention_layernorm + rotated MoE input Hessian
    h_moe_in = torch.zeros((hidden, hidden), dtype=torch.float32, device=device)
    mlp_in_all = torch.empty((n_tot, hidden), dtype=H_all.dtype, device=device)

    for s0 in range(0, n_tot, tok_step):
        s1 = min(s0 + tok_step, n_tot)
        attn_out = F.linear(pre_out_all[s0:s1], out_proj_mod.weight, out_proj_mod.bias)
        h_slice = H_all[s0:s1].to(device, non_blocking=True)
        if has_hc:
            h_post_attn = h_slice + (attn_out.unsqueeze(-2) * inj_w_all[s0:s1].unsqueeze(-1)).flatten(-2)
            m_in, hyp_m, iw_m = layer.mlp_hyper_connection(h_post_attn.unsqueeze(0))
            x_raw = m_in.squeeze(0)
            mlp_in_all[s0:s1] = x_raw
            H_all[s0:s1].copy_(hyp_m.squeeze(0), non_blocking=True)
            inj_w_all[s0:s1] = iw_m.squeeze(0)
            del m_in, hyp_m, iw_m
        else:
            h_post_attn = h_slice + attn_out
            x_raw = layer.post_attention_layernorm(h_post_attn)
            mlp_in_all[s0:s1] = x_raw
            H_all[s0:s1].copy_(h_post_attn, non_blocking=True)

        if had_moe_in > 0:
            x_rot = _hadamard_rotate_last_dim(x_raw, had_moe_in, compute_dtype=torch.float32).float()
        else:
            x_rot = x_raw.float()
        h_moe_in.addmm_(x_rot.T, x_rot, beta=1.0, alpha=2.0)
        del h_slice, attn_out, h_post_attn, x_raw, x_rot
    del pre_out_all

    # Quantize shared_expert.{gate_proj, up_proj} with W4A16-HGPTQ
    shared_hooks = []
    for sp_name in ("gate_proj", "up_proj"):
        sp_mod: nn.Linear = getattr(shared_exp, sp_name)
        if had_moe_in > 0:
            sp_mod.weight.data = _hadamard_rotate_weight(sp_mod.weight.data, had_moe_in)
        q_sp, packed_sp = _gptq_quantize_and_pack_2d(
            sp_mod.weight.data,
            h_moe_in,
            n_tot,
            group_size=group_size,
            actorder=actorder,
            dampening_frac=dampening_frac,
        )
        sp_mod.weight.data.copy_(q_sp)
        for pk, pv in packed_sp.items():
            layer_tensors[f"{layer_prefix}.mlp.shared_expert.{sp_name}.{pk}"] = pv
        counts["shared_experts"] += 1
        if had_moe_in > 0:
            shared_hooks.append(
                sp_mod.register_forward_pre_hook(
                    lambda _m, args, bs=had_moe_in: (
                        _hadamard_rotate_last_dim(args[0], bs, compute_dtype=torch.float32),
                        *args[1:],
                    )
                )
            )

    # Accumulate post-SwiGLU Hessian for shared_expert.down_proj on device
    h_shared_down = torch.zeros((shared_inter, shared_inter), dtype=torch.float32, device=device)
    for s0 in range(0, n_tot, tok_step):
        s1 = min(s0 + tok_step, n_tot)
        x_raw = mlp_in_all[s0:s1]
        z_sh = shared_exp.act_fn(shared_exp.gate_proj(x_raw)) * shared_exp.up_proj(x_raw)
        if had_shared_down > 0:
            z_sh = _hadamard_rotate_last_dim(z_sh, had_shared_down, compute_dtype=torch.float32).float()
        else:
            z_sh = z_sh.float()
        h_shared_down.addmm_(z_sh.T, z_sh, beta=1.0, alpha=2.0)
        del z_sh

    sp_down: nn.Linear = shared_exp.down_proj
    if had_shared_down > 0:
        sp_down.weight.data = _hadamard_rotate_weight(sp_down.weight.data, had_shared_down)
    q_sp_down, packed_sp_down = _gptq_quantize_and_pack_2d(
        sp_down.weight.data,
        h_shared_down,
        n_tot,
        group_size=group_size,
        actorder=actorder,
        dampening_frac=dampening_frac,
    )
    sp_down.weight.data.copy_(q_sp_down)
    for pk, pv in packed_sp_down.items():
        layer_tensors[f"{layer_prefix}.mlp.shared_expert.down_proj.{pk}"] = pv
    counts["shared_experts"] += 1
    if had_shared_down > 0:
        shared_hooks.append(
            sp_down.register_forward_pre_hook(
                lambda _m, args, bs=had_shared_down: (
                    _hadamard_rotate_last_dim(args[0], bs, compute_dtype=torch.float32),
                    *args[1:],
                )
            )
        )
    del h_shared_down

    norm_h_moe_in = (h_moe_in / max(n_tot, 1)).unsqueeze(0)
    del h_moe_in

    # Pre-rotate experts.gate_up_proj [E, 2*inter, hidden] by H_{had_moe_in}^T and quantize gate_proj + up_proj
    if had_moe_in > 0:
        for e0 in range(0, num_exp, 64):
            e1 = min(e0 + 64, num_exp)
            experts.gate_up_proj.data[e0:e1].copy_(
                _hadamard_rotate_last_dim(
                    experts.gate_up_proj.data[e0:e1], had_moe_in, compute_dtype=torch.float64
                )
            )
    for proj_name, sl in (("gate_proj", slice(0, inter)), ("up_proj", slice(inter, 2 * inter))):
        q_3d, wp_3d, ws_3d, wsh_1d = _gptq_quantize_and_pack_3d_experts(
            experts.gate_up_proj.data[:, sl, :].contiguous(),
            norm_h_moe_in,
            group_size=group_size,
            actorder=actorder,
            dampening_frac=dampening_frac,
        )
        experts.gate_up_proj.data[:, sl, :].copy_(q_3d)
        for e in range(num_exp):
            epref = f"{layer_prefix}.mlp.experts.{e}.{proj_name}"
            layer_tensors[f"{epref}.weight_packed"] = wp_3d[e].clone()
            layer_tensors[f"{epref}.weight_scale"] = ws_3d[e].clone()
            layer_tensors[f"{epref}.weight_shape"] = wsh_1d.clone()
            counts["routed_experts"] += 1
        del q_3d, wp_3d, ws_3d
    del norm_h_moe_in

    # C.2 Accumulate [E, inter, inter] post-SwiGLU (H_{had_moe_down}-rotated) Hessians for down_proj on device
    h_moe_down = torch.zeros((num_exp, inter, inter), dtype=torch.float32, device=device)
    tok_counts_down = torch.zeros((num_exp, 1, 1), dtype=torch.float32, device=device)
    exp_bmm_chunk = min(64, num_exp)
    dense_toks = min(n_tot, max(dense_expert_calib_seqs, 1) * 256)

    if dense_toks > 0:
        x_d = (
            _hadamard_rotate_last_dim(mlp_in_all[:dense_toks], had_moe_in, compute_dtype=torch.float32)
            if had_moe_in > 0
            else mlp_in_all[:dense_toks]
        )
        for e0 in range(0, num_exp, exp_bmm_chunk):
            e1 = min(e0 + exp_bmm_chunk, num_exp)
            gu = torch.matmul(x_d.unsqueeze(0), experts.gate_up_proj.data[e0:e1].transpose(1, 2))
            z = experts.act_fn(gu[:, :, :inter]) * gu[:, :, inter:]
            if had_moe_down > 0:
                z = _hadamard_rotate_last_dim(z, had_moe_down, compute_dtype=torch.float32)
            z_f32 = z.float()
            h_moe_down[e0:e1].baddbmm_(z_f32.transpose(1, 2), z_f32, beta=1.0, alpha=2.0)
            tok_counts_down[e0:e1] += x_d.shape[0]
            del gu, z, z_f32
        del x_d

    routed_chunk_toks = 655360
    for c0 in range(dense_toks, n_tot, routed_chunk_toks):
        x_2d = mlp_in_all[c0 : min(c0 + routed_chunk_toks, n_tot)]
        n_b = x_2d.shape[0]
        _, _routing_w, selected_experts = layer.mlp.gate(x_2d)
        x_2d_rot = (
            _hadamard_rotate_last_dim(x_2d, had_moe_in, compute_dtype=torch.float32)
            if had_moe_in > 0
            else x_2d
        )
        flat_exp = selected_experts.reshape(-1)
        flat_tok = (
            torch.arange(n_b, device=device, dtype=torch.int64)
            .unsqueeze(1)
            .expand(-1, selected_experts.shape[1])
            .reshape(-1)
        )
        perm = torch.argsort(flat_exp)
        sorted_tok = flat_tok[perm]
        exp_counts = torch.bincount(flat_exp, minlength=num_exp + 1).cpu().tolist()
        cursor = 0
        for e_idx in range(num_exp):
            c = exp_counts[e_idx]
            if c == 0:
                continue
            tok_idx = sorted_tok[cursor : cursor + c]
            cursor += c
            cur_x = x_2d_rot.index_select(0, tok_idx)
            gate, up = F.linear(cur_x, experts.gate_up_proj.data[e_idx]).chunk(2, dim=-1)
            z = experts.act_fn(gate) * up
            if had_moe_down > 0:
                z = _hadamard_rotate_last_dim(z, had_moe_down, compute_dtype=torch.float32)
            z_f32 = z.float()
            h_moe_down[e_idx].addmm_(z_f32.T, z_f32, beta=1.0, alpha=2.0)
            tok_counts_down[e_idx] += c
        del selected_experts, x_2d_rot, flat_exp, flat_tok, perm, sorted_tok

    norm_h_moe_down = h_moe_down / tok_counts_down.clamp(min=1.0)
    del h_moe_down, tok_counts_down

    # Pre-rotate experts.down_proj [E, hidden, inter] by H_{had_moe_down}^T and quantize across all experts
    if had_moe_down > 0:
        for e0 in range(0, num_exp, 64):
            e1 = min(e0 + 64, num_exp)
            experts.down_proj.data[e0:e1].copy_(
                _hadamard_rotate_last_dim(
                    experts.down_proj.data[e0:e1], had_moe_down, compute_dtype=torch.float64
                )
            )
    q_down_3d, wp_down_3d, ws_down_3d, wsh_down_1d = _gptq_quantize_and_pack_3d_experts(
        experts.down_proj.data,
        norm_h_moe_down,
        group_size=group_size,
        actorder=actorder,
        dampening_frac=dampening_frac,
    )
    experts.down_proj.data.copy_(q_down_3d)
    for e in range(num_exp):
        epref = f"{layer_prefix}.mlp.experts.{e}.down_proj"
        layer_tensors[f"{epref}.weight_packed"] = wp_down_3d[e].clone()
        layer_tensors[f"{epref}.weight_scale"] = ws_down_3d[e].clone()
        layer_tensors[f"{epref}.weight_shape"] = wsh_down_1d.clone()
        counts["routed_experts"] += 1
    del norm_h_moe_down, q_down_3d, wp_down_3d, ws_down_3d

    # C.3 Advance H_all through the quantized MoE sub-block on device using grouped_mm + online Hadamard hooks
    exp_hooks = _attach_expert_hadamard_hooks(experts, had_moe_in, had_moe_down)
    for s0 in range(0, n_tot, tok_step):
        s1 = min(s0 + tok_step, n_tot)
        mlp_out = layer.mlp(mlp_in_all[s0:s1].unsqueeze(0)).squeeze(0)
        h_slice = H_all[s0:s1].to(device, non_blocking=True)
        if has_hc:
            h_upd = h_slice + (mlp_out.unsqueeze(-2) * inj_w_all[s0:s1].unsqueeze(-1)).flatten(-2)
        else:
            h_upd = h_slice + mlp_out
        H_all[s0:s1].copy_(h_upd, non_blocking=True)
        del mlp_out, h_slice, h_upd
    del mlp_in_all, inj_w_all
    for h in exp_hooks:
        h.remove()

    seq_cur = 0
    for i, slen in enumerate(seq_lens):
        h_seq = H_all[seq_cur : seq_cur + slen].unsqueeze(0)
        seq_cur += slen
        h_calib[i] = h_seq.to(device).clone() if keep_calib_on_device else h_seq.cpu().clone()
    del H_all

    for h in shared_hooks:
        h.remove()

    # ------------------------------------------------------------------
    # Collect all remaining unquantized (BF16 / I64) parameters & buffers
    # ------------------------------------------------------------------
    packed_mod_prefixes = {k[: -len(".weight_packed")] for k in layer_tensors if k.endswith(".weight_packed")}
    for name, param in layer.named_parameters():
        if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
            continue
        full_name = f"{layer_prefix}.{name}"
        if full_name.removesuffix(".weight") in packed_mod_prefixes:
            continue
        layer_tensors[full_name] = param.detach().cpu().contiguous()

    for name, buf in layer.named_buffers():
        if name.startswith("ple.ple_embedding.ngram_embedding"):
            continue
        # Only persist buffers that were in the checkpoint (PLE multipliers/offsets/vocab_sizes)
        if "ple.ple_embedding." in name:
            layer_tensors[f"{layer_prefix}.{name}"] = buf.detach().cpu().contiguous()

    layer_bytes = sum(t.numel() * t.element_size() for t in layer_tensors.values())
    return layer_tensors, ple_weight_map, layer_bytes + ple_bytes, counts


class Qwen4ExpMultiTokenPredictor(nn.Module):
    """4-stream Hyper-Connection Native MTP module for Qwen/Qwen3.8-Flash-Next."""

    def __init__(self, text_config, device: torch.device):
        super().__init__()
        from llmcompressor.utils.dev import skip_weights_initialize
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextDecoderLayer,
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextRMSNorm,
            Qwen4ExpTextRotaryEmbedding,
        )

        cfg = copy.deepcopy(text_config)
        _ensure_qwen4_exp_device_kernels(cfg, device)
        cfg.num_hidden_layers = 1
        cfg.layer_types = ["indexed_attention"]
        cfg.ple_layer_ids = []

        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.hc_count = cfg.hc_count
        hc_hidden_size = self.hidden_size * self.hc_count

        with torch.device(device), skip_weights_initialize():
            self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(self.hidden_size, eps=cfg.rms_norm_eps)
            self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(hc_hidden_size, eps=cfg.rms_norm_eps)
            self.fc_embedding = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.layers = nn.ModuleList([Qwen4ExpTextDecoderLayer(cfg, layer_idx=0)])
            self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
            self.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=cfg)
        self.to(device=device, dtype=torch.bfloat16).eval()

    @torch.no_grad()
    def prepare_layer_input(
        self,
        pre_mixer_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse 4-stream pre-mixer backbone state [B, S, hc_count*H] with next-token embeds [B, S, H]."""
        e_norm = self.pre_fc_norm_embedding(inputs_embeds)
        e_proj = self.fc_embedding(e_norm)
        h_norm = self.pre_fc_norm_hidden(pre_mixer_hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        h_proj = self.fc_hidden(h_norm)
        return (e_proj.unsqueeze(-2) + h_proj).flatten(-2)

    @torch.no_grad()
    def forward(
        self,
        pre_mixer_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device).view(1, 1, -1).expand(3, bsz, -1)
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        pos_emb = self.rotary_emb(inputs_embeds, position_ids)
        h = self.prepare_layer_input(pre_mixer_hidden_states, inputs_embeds)
        layer = self.layers[0]
        branch_in, hyper_in, inj_w = layer.attn_hyper_connection(h)
        attn_out = _forward_qsa_attention_fast(layer.self_attn, branch_in, pos_emb)
        h = hyper_in + (attn_out.unsqueeze(-2) * inj_w.unsqueeze(-1)).flatten(-2)
        mlp_in, hyper_in, inj_w = layer.mlp_hyper_connection(h)
        mlp_out = layer.mlp(mlp_in)
        h = hyper_in + (mlp_out.unsqueeze(-2) * inj_w.unsqueeze(-1)).flatten(-2)
        return self.hyper_connection_mixer(h)


class Qwen35MoeMultiTokenPredictor(nn.Module):
    """Single-stream Native MTP module for Qwen/Qwen3.6-35B-A3B (Qwen3_5Moe)."""

    def __init__(self, text_config, device: torch.device):
        super().__init__()
        from llmcompressor.utils.dev import skip_weights_initialize
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer,
            Qwen3_5MoeRMSNorm,
            Qwen3_5MoeTextRotaryEmbedding,
        )

        cfg = copy.deepcopy(text_config)
        _ensure_qwen4_exp_device_kernels(cfg, device)
        cfg.num_hidden_layers = 1
        cfg.layer_types = ["full_attention"]

        self.config = cfg
        self.hidden_size = cfg.hidden_size

        with torch.device(device), skip_weights_initialize():
            self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
            self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(self.hidden_size, eps=cfg.rms_norm_eps)
            self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(self.hidden_size, eps=cfg.rms_norm_eps)
            self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(cfg, layer_idx=0)])
            self.layers[0].ple = None
            self.norm = Qwen3_5MoeRMSNorm(self.hidden_size, eps=cfg.rms_norm_eps)
            self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=cfg)
        self.to(device=device, dtype=torch.bfloat16).eval()

    @torch.no_grad()
    def prepare_layer_input(
        self,
        post_norm_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse post-norm backbone state [B, S, H] with next-token embeds [B, S, H]."""
        return self.fc(
            torch.cat(
                [
                    self.pre_fc_norm_embedding(inputs_embeds),
                    self.pre_fc_norm_hidden(post_norm_hidden_states),
                ],
                dim=-1,
            )
        )

    @torch.no_grad()
    def forward(
        self,
        post_norm_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device).view(1, 1, -1).expand(3, bsz, -1)
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        pos_emb = self.rotary_emb(inputs_embeds, position_ids)
        h = self.prepare_layer_input(post_norm_hidden_states, inputs_embeds)
        layer = self.layers[0]
        b_in = layer.input_layernorm(h)
        attn_out = _forward_qsa_attention_fast(layer.self_attn, b_in, pos_emb)
        h = h + attn_out
        m_in = layer.post_attention_layernorm(h)
        m_out = layer.mlp(m_in)
        h = h + m_out
        return self.norm(h)


def _has_mtp_weights(ckpt_dir: Path, shard_map: dict[str, str]) -> bool:
    if shard_map:
        return any(k.startswith("mtp.") for k in shard_map)
    for sp in sorted(ckpt_dir.glob("*.safetensors")):
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            if any(k.startswith("mtp.") for k in f.keys()):
                return True
    return False


def _load_qwen4_exp_mtp_from_checkpoint(
    ckpt_dir: Path,
    text_config,
    device: torch.device,
) -> nn.Module:
    """Load BF16 or W4A16 (+ Hadamard) Qwen4ExpMultiTokenPredictor or Qwen35MoeMultiTokenPredictor from ckpt_dir."""
    shard_map, _ = _load_shard_map(ckpt_dir)
    raw_cfg = json.loads((ckpt_dir / "config.json").read_text())
    qcfg = raw_cfg.get("quantization_config", {})
    g_size = qcfg.get("config_groups", {}).get("group_0", {}).get("weights", {}).get("group_size", 64)
    tcfg = qcfg.get("transform_config", {}).get("config_groups", {})
    had_size = next(iter(tcfg.values()))["head_dim"] if tcfg else 0

    mtp = (
        Qwen35MoeMultiTokenPredictor(text_config, device)
        if _is_qwen35_moe_text_config(text_config)
        else Qwen4ExpMultiTokenPredictor(text_config, device)
    )
    _load_decoder_layer_weights(
        mtp.layers[0],
        "mtp.layers.0",
        ckpt_dir,
        shard_map,
        device,
        group_size=g_size,
        hadamard_block_size=had_size,
    )

    if shard_map:
        top_shards = sorted(
            {
                ckpt_dir / fname
                for k, fname in shard_map.items()
                if k.startswith("mtp.") and not k.startswith("mtp.layers.")
            }
        )
    else:
        top_shards = sorted(ckpt_dir.glob("*.safetensors"))

    top_tensors: dict[str, torch.Tensor] = {}
    for sp in top_shards:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("mtp.") and not k.startswith("mtp.layers."):
                    top_tensors[k.removeprefix("mtp.")] = f.get_tensor(k)

    param_map = dict(mtp.named_parameters())
    for k_rel, tensor in top_tensors.items():
        if k_rel not in param_map:
            raise RuntimeError(f"Unexpected top-level MTP tensor key: mtp.{k_rel}")
        param_map[k_rel].data.copy_(tensor.to(device=device, dtype=param_map[k_rel].dtype))

    return mtp


def _write_qwen4_exp_quant_config(
    save_dir: Path,
    base_cfg_dict: dict,
    *,
    group_size: int,
    actorder: str | None,
    hadamard_block_size: int,
    quantize_mtp: bool,
    quantize_vision: bool,
    vision_fc2_channel: bool,
    ple_group_size: int = 32,
    ple_cd_passes: int = 1,
) -> None:
    """Write complete config.json with compressed-tensors W4A16-HGPTQ + 4-bit Custom-LUT4 PLE metadata."""
    cfg = copy.deepcopy(base_cfg_dict)
    text_targets = [TEXT_TARGET_RE]
    if quantize_mtp:
        text_targets.append(MTP_TARGET_RE)
    if quantize_vision:
        if vision_fc2_channel:
            text_targets.append(VISION_GROUP_TARGET_RE)
        else:
            text_targets.extend([VISION_GROUP_TARGET_RE, VISION_FC2_TARGET_RE])

    base_weights = {
        "actorder": actorder,
        "block_structure": None,
        "dynamic": False,
        "group_size": group_size,
        "num_bits": 4,
        "observer": "memoryless_minmax",
        "observer_kwargs": {},
        "scale_dtype": None,
        "strategy": "group",
        "symmetric": True,
        "type": "int",
        "zp_dtype": None,
    }
    config_groups = {
        "group_0": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": text_targets,
            "weights": base_weights,
        }
    }
    if quantize_vision and vision_fc2_channel:
        fc2_weights = dict(base_weights, strategy="channel", group_size=None, actorder=None)
        config_groups["vision_fc2_channel"] = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": [VISION_FC2_TARGET_RE],
            "weights": fc2_weights,
        }

    is_qwen35_moe = base_cfg_dict.get("text_config", {}).get("model_type") == "qwen3_5_moe_text"
    ignore_list = list(BASE_IGNORE) if is_qwen35_moe else list(QWEN4_EXP_IGNORE)

    if hadamard_block_size > 0:
        main_had_targets = [TEXT_HAD_TARGET_RE]
        if quantize_mtp:
            main_had_targets.append(MTP_HAD_TARGET_RE)
        if quantize_vision:
            vision_inter = cfg.get("vision_config", {}).get("intermediate_size", 0)
            fc2_div = (vision_inter & -vision_inter) if vision_inter > 0 else hadamard_block_size
            if fc2_div < hadamard_block_size:
                main_had_targets.append(VISION_GROUP_TARGET_RE)
                fc2_had_size = fc2_div
            else:
                main_had_targets.extend([VISION_GROUP_TARGET_RE, VISION_FC2_TARGET_RE])
                fc2_had_size = None
        else:
            fc2_had_size = None

        t_groups = {
            "v": {
                "type": "hadamard",
                "head_dim": hadamard_block_size,
                "randomize": False,
                "requires_grad": False,
                "precision": "torch.float32",
                "apply": [
                    {
                        "targets": main_had_targets,
                        "location": "weight_input",
                        "inverse": True,
                        "ignore": list(ignore_list),
                    },
                    {
                        "targets": main_had_targets,
                        "location": "input",
                        "inverse": False,
                        "ignore": list(ignore_list),
                    },
                ],
            }
        }
        if fc2_had_size is not None and fc2_had_size > 1:
            t_groups["v_vision_fc2"] = {
                "type": "hadamard",
                "head_dim": fc2_had_size,
                "randomize": False,
                "requires_grad": False,
                "precision": "torch.float32",
                "apply": [
                    {
                        "targets": [VISION_FC2_TARGET_RE],
                        "location": "weight_input",
                        "inverse": True,
                        "ignore": [],
                    },
                    {
                        "targets": [VISION_FC2_TARGET_RE],
                        "location": "input",
                        "inverse": False,
                        "ignore": [],
                    },
                ],
            }
        transform_cfg = {"config_groups": t_groups}
    else:
        transform_cfg = {}

    qcfg_dict = {
        "config_groups": config_groups,
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": list(ignore_list),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": transform_cfg,
        "version": "0.14.0",
    }
    if not is_qwen35_moe and ple_group_size > 0:
        qcfg_dict["ple_quantization"] = {
            "cd_passes": ple_cd_passes,
            "format": "lut4-gptq",
            "group_size": ple_group_size,
            "hessian": "w_kv_exact",
            "lut": list(PLE_CUSTOM_LUT4),
            "num_bits": 4,
            "scale_dtype": "float16",
            "targets": ["model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_*"],
        }
    cfg["quantization_config"] = qcfg_dict
    (save_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def _load_top_level_module_tensors(
    ckpt_dir: Path,
    shard_map: dict[str, str],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if shard_map:
        rel_shards = sorted({ckpt_dir / fname for k, fname in shard_map.items() if k.startswith(prefix)})
    else:
        rel_shards = sorted(ckpt_dir.glob("*.safetensors"))
    out: dict[str, torch.Tensor] = {}
    for sp in rel_shards:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k] = f.get_tensor(k)
    return out


@torch.no_grad()
def run_quantize_qwen4_exp(args: argparse.Namespace) -> None:
    """Execute full-model streaming W4A16-HGPTQ + 4-bit Custom-LUT4 PLE quantization for Qwen4Exp."""
    from llmcompressor.utils.dev import skip_weights_initialize
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRotaryEmbedding,
        Qwen4ExpVisionModel,
    )

    from warp_quant.data import get_calibration_dataset

    if args.hadamard_block_size != 0 and not _is_power_of_two(args.hadamard_block_size):
        raise ValueError(f"--hadamard-block-size must be 0 or a power of 2, got {args.hadamard_block_size}")

    hub_id, info = resolve(args.model)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"--output-dir already exists: {output_dir}")

    if not Path(hub_id).exists():
        hub_id = resolve_snapshot(hub_id, revision=info.get("hub_revision"))
        if not list(Path(hub_id).glob("*.safetensors")):
            raise RuntimeError(f"Base snapshot {hub_id} has no .safetensors shards")

    base_dir = Path(hub_id)
    ensure_processor_configs(
        str(base_dir),
        info.get("processor_hub_fallback"),
        info.get("processor_hub_fallback_revision"),
    )
    ensure_license_files(None, str(base_dir), info.get("license_files"), dry_run=True)

    recipe_defaults = quant_recipe(info)
    calib = args.calib or recipe_defaults["calib"]
    num_samples = args.num_samples if args.num_samples is not None else recipe_defaults["num_samples"]
    max_seq_length = (
        args.max_seq_length if args.max_seq_length is not None else recipe_defaults["max_seq_length"]
    )
    window = args.window or recipe_defaults["window"]
    actorder = None if args.actorder == "none" else args.actorder
    ple_group_size = getattr(args, "ple_group_size", 32)
    ple_cd_passes = getattr(args, "ple_cd_passes", 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log_event(
        "QWEN4_EXP_QUANT_START",
        model=args.model,
        snapshot=str(base_dir),
        output_dir=str(output_dir),
        group_size=args.group_size,
        hadamard_block_size=args.hadamard_block_size,
        actorder=actorder,
        ple_group_size=ple_group_size,
        ple_cd_passes=ple_cd_passes,
        quantize_mtp=args.quantize_mtp,
        quantize_vision=args.quantize_vision,
        calib=calib,
        num_samples=num_samples,
        max_seq_length=max_seq_length,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    raw_base_cfg = json.loads((base_dir / "config.json").read_text())
    is_qwen35_moe = raw_base_cfg.get("text_config", {}).get("model_type") == "qwen3_5_moe_text"
    if is_qwen35_moe:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeRMSNorm,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextRotaryEmbedding as RotaryCls,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeVisionModel as VisionCls,
        )

        cfg = Qwen3_5MoeConfig.from_dict(raw_base_cfg)
    else:
        cfg = Qwen4ExpConfig.from_dict(raw_base_cfg)
        RotaryCls = Qwen4ExpTextRotaryEmbedding
        VisionCls = Qwen4ExpVisionModel

    _ensure_qwen4_exp_device_kernels(cfg.text_config, device)
    processor = AutoProcessor.from_pretrained(str(base_dir))
    shard_map, _ = _load_shard_map(base_dir)
    reuse_ple_vision_dir = (
        Path(args.reuse_ple_vision_from).resolve() if getattr(args, "reuse_ple_vision_from", None) else None
    )

    t0 = time.time()
    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    qwen35_shard1_tensors: dict[str, torch.Tensor] = {}
    qwen35_shard2_tensors: dict[str, torch.Tensor] = {}
    shard_counts = {
        "stripped_transform_weights": 0,
        "mtp_quantized": 0,
        "vision_quantized": 0,
        "vision_fc2_channel": 0,
        "ple_shards_quantized": 0,
        "text_attn_quantized": 0,
        "text_shared_experts_quantized": 0,
        "text_experts_quantized": 0,
    }

    # ------------------------------------------------------------------
    # Phase 1: Vision Tower W4A16-HGPTQ (if present)
    # ------------------------------------------------------------------
    vis_raw = _load_top_level_module_tensors(base_dir, shard_map, "model.visual.")
    if vis_raw:
        reused_qwen35_vis: dict[str, torch.Tensor] = {}
        if is_qwen35_moe and reuse_ple_vision_dir is not None and reuse_ple_vision_dir.exists():
            reuse_vis_smap, _ = _load_shard_map(reuse_ple_vision_dir)
            reused_qwen35_vis = _load_top_level_module_tensors(
                reuse_ple_vision_dir, reuse_vis_smap, "model.visual."
            )
        can_reuse_vis = reuse_ple_vision_dir is not None and (
            (not is_qwen35_moe and (reuse_ple_vision_dir / "model-vision.safetensors").exists())
            or (
                is_qwen35_moe
                and bool(reused_qwen35_vis)
                and (not args.quantize_vision or any(k.endswith(".weight_packed") for k in reused_qwen35_vis))
            )
        )
        if can_reuse_vis:
            if is_qwen35_moe:
                for tk, tv in reused_qwen35_vis.items():
                    qwen35_shard1_tensors[tk] = tv.contiguous()
                    if tk.endswith(".weight_packed"):
                        shard_counts["vision_quantized"] += 1
                    elif tk.endswith(".weight_scale") and tv.ndim == 2 and tv.shape[1] == 1:
                        shard_counts["vision_fc2_channel"] += 1
                del reused_qwen35_vis
            else:
                import shutil

                vis_shard_name = "model-vision.safetensors"
                src_vis = reuse_ple_vision_dir / vis_shard_name
                dst_vis = output_dir / vis_shard_name
                try:
                    os.link(src_vis, dst_vis)
                except OSError:
                    shutil.copy2(src_vis, dst_vis)
                with safe_open(str(dst_vis), framework="pt", device="cpu") as sf:
                    for tk in sf.keys():
                        tv = sf.get_tensor(tk)
                        new_weight_map[tk] = vis_shard_name
                        total_bytes += tv.numel() * tv.element_size()
                        if tk.endswith(".weight_packed"):
                            shard_counts["vision_quantized"] += 1
                        elif tk.endswith(".weight_scale") and tv.ndim == 2 and tv.shape[1] == 1:
                            shard_counts["vision_fc2_channel"] += 1
            log_event("QWEN4_EXP_VISION_REUSED", reuse_dir=str(reuse_ple_vision_dir))
            del vis_raw
        else:
            with torch.device(device), skip_weights_initialize():
                visual = VisionCls._from_config(cfg.vision_config, dtype=torch.bfloat16).eval()
            vis_sd = {
                k.removeprefix("model.visual."): v.to(device=device, dtype=torch.bfloat16)
                for k, v in vis_raw.items()
            }
            visual.load_state_dict(vis_sd, strict=True)

            vis_out_tensors: dict[str, torch.Tensor] = {}
            if args.quantize_vision:
                packed_vis: dict[str, dict[str, torch.Tensor]] = {}
                if args.component_quant_mode == "gptq":
                    if not args.vision_calib_dir:
                        raise ValueError(
                            "--vision-calib-dir is required when --quantize-vision is used with --component-quant-mode gptq"
                        )
                    vis_calib_path = Path(args.vision_calib_dir)
                    if not vis_calib_path.is_absolute():
                        vis_calib_path = ROOT / vis_calib_path
                    dummy_wrapper = nn.Module()
                    dummy_wrapper.model = nn.Module()
                    dummy_wrapper.model.visual = visual
                    dummy_wrapper.register_buffer("_dev_buf", torch.empty(1, device=device))
                    packed_vis = _gptq_calibrate_vision(
                        dummy_wrapper,
                        processor,
                        vis_calib_path,
                        group_size=args.group_size,
                        hadamard_block_size=args.hadamard_block_size,
                        actorder=actorder,
                        dampening_frac=args.dampening_frac,
                        max_calib_samples=args.vision_calib_samples,
                    )
                else:
                    for name, mod in visual.named_modules():
                        if not isinstance(mod, nn.Linear):
                            continue
                        w = mod.weight.data
                        k_in = w.shape[1]
                        strat = "group" if k_in % args.group_size == 0 else "channel"
                        g_size = args.group_size if strat == "group" else None
                        had_b = (
                            min(args.hadamard_block_size, k_in & -k_in) if args.hadamard_block_size > 0 else 0
                        )
                        if had_b > 0:
                            w = _hadamard_rotate_weight(w, had_b)
                        packed_vis[f"model.visual.{name}"] = _pack_w4_tensor(
                            w.cpu(), strategy=strat, group_size=g_size
                        )

                for k_full, t_cpu in vis_raw.items():
                    prefix = k_full.removesuffix(".weight") if k_full.endswith(".weight") else ""
                    if prefix in packed_vis:
                        if t_cpu.shape[1] % args.group_size != 0:
                            shard_counts["vision_fc2_channel"] += 1
                        for pk, pv in packed_vis[prefix].items():
                            vis_out_tensors[f"{prefix}.{pk}"] = pv.cpu().contiguous()
                        shard_counts["vision_quantized"] += 1
                    else:
                        vis_out_tensors[k_full] = t_cpu.contiguous()
            else:
                vis_out_tensors = {k: v.contiguous() for k, v in vis_raw.items()}

            if is_qwen35_moe:
                qwen35_shard1_tensors.update(vis_out_tensors)
            else:
                vis_shard_name = "model-vision.safetensors"
                save_file(vis_out_tensors, str(output_dir / vis_shard_name))
                for tk, tv in vis_out_tensors.items():
                    new_weight_map[tk] = vis_shard_name
                    total_bytes += tv.numel() * tv.element_size()
            del visual, vis_sd, vis_raw, vis_out_tensors
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: Initial Embeddings + Calibration Stream Setup
    # ------------------------------------------------------------------
    ds = get_calibration_dataset(processor, calib, num_samples, max_seq_length, window=window, seed=args.seed)
    calib_ids_list: list[torch.Tensor] = []
    for row in ds:
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        ids = ids[:, :max_seq_length]
        if ids.shape[1] >= 4:
            calib_ids_list.append(ids)
    if not calib_ids_list:
        raise RuntimeError("No valid calibration sequences with length >= 4")

    embed_raw = _load_top_level_module_tensors(base_dir, shard_map, "model.language_model.embed_tokens.")
    embed_w = embed_raw["model.language_model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    rotary = RotaryCls(config=cfg.text_config).to(device)

    h_calib: list[torch.Tensor] = []
    pos_emb_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    hc_count = 1 if is_qwen35_moe else cfg.text_config.hc_count
    est_tot_tokens = sum(t.shape[1] for t in calib_ids_list)
    est_h_bytes = est_tot_tokens * hc_count * cfg.text_config.hidden_size * 2
    keep_calib_on_device = (
        device.type == "cuda"
        and torch.cuda.get_device_properties(device).total_memory >= 100 * (1024**3)
        and est_h_bytes <= 24 * (1024**3)
    )
    calib_store_device = device if keep_calib_on_device else torch.device("cpu")

    for ids in calib_ids_list:
        ids_dev = ids.to(device)
        seq_len = ids_dev.shape[1]
        inputs_embeds = F.embedding(ids_dev, embed_w)
        pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, ids_dev.shape[0], -1)
        cos, sin = rotary(inputs_embeds, pos_ids)
        pos_emb_list.append((cos.to(calib_store_device), sin.to(calib_store_device)))
        h_init = inputs_embeds if is_qwen35_moe else inputs_embeds.repeat(1, 1, hc_count)
        h_calib.append(h_init.to(calib_store_device))
        del ids_dev, inputs_embeds, cos, sin

    if is_qwen35_moe:
        norm_raw = _load_top_level_module_tensors(base_dir, shard_map, "model.language_model.norm.")
        head_raw = _load_top_level_module_tensors(base_dir, shard_map, "lm_head.")
        qwen35_shard1_tensors.update(embed_raw)
        qwen35_shard1_tensors.update(norm_raw)
        qwen35_shard2_tensors.update(head_raw)
        del norm_raw, head_raw
    else:
        embed_head_shard = "model-embed-head.safetensors"
        if reuse_ple_vision_dir is not None and (reuse_ple_vision_dir / embed_head_shard).exists():
            import shutil

            src_eh = reuse_ple_vision_dir / embed_head_shard
            dst_eh = output_dir / embed_head_shard
            try:
                os.link(src_eh, dst_eh)
            except OSError:
                shutil.copy2(src_eh, dst_eh)
            with safe_open(str(dst_eh), framework="pt", device="cpu") as sf:
                for tk in sf.keys():
                    tv = sf.get_tensor(tk)
                    new_weight_map[tk] = embed_head_shard
                    total_bytes += tv.numel() * tv.element_size()
        else:
            mixer_raw = _load_top_level_module_tensors(
                base_dir, shard_map, "model.language_model.hyper_connection_mixer."
            )
            head_raw = _load_top_level_module_tensors(base_dir, shard_map, "lm_head.")
            embed_head_tensors = {**embed_raw, **mixer_raw, **head_raw}
            save_file(embed_head_tensors, str(output_dir / embed_head_shard))
            for tk, tv in embed_head_tensors.items():
                new_weight_map[tk] = embed_head_shard
                total_bytes += tv.numel() * tv.element_size()
            del embed_head_tensors, mixer_raw, head_raw

    # Keep embed_w on CPU if needed for MTP later; free from GPU now
    embed_w_cpu = embed_w.cpu()
    del embed_w, embed_raw
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 3: Layer-by-Layer Streaming W4A16-HGPTQ + Layer-1 PLE
    # ------------------------------------------------------------------
    num_layers = cfg.text_config.num_hidden_layers
    with ThreadPoolExecutor(max_workers=2) as layer_io_pool:
        next_layer_fut: Future | None = (
            layer_io_pool.submit(
                _prefetch_layer_raw_tensors, "model.language_model.layers.0", base_dir, shard_map
            )
            if num_layers > 0
            else None
        )
        pending_layer_save_fut: Future | None = None

        for l_idx in range(num_layers):
            t_layer = time.time()
            prefetched_raw = next_layer_fut.result() if next_layer_fut is not None else None
            if l_idx + 1 < num_layers:
                next_layer_fut = layer_io_pool.submit(
                    _prefetch_layer_raw_tensors,
                    f"model.language_model.layers.{l_idx + 1}",
                    base_dir,
                    shard_map,
                )
            else:
                next_layer_fut = None

            layer = _build_lightweight_decoder_layer(cfg.text_config, l_idx, device)
            layer_prefix = f"model.language_model.layers.{l_idx}"
            _load_decoder_layer_weights(
                layer, layer_prefix, base_dir, shard_map, device, prefetched_raw=prefetched_raw
            )
            del prefetched_raw

            layer_tensors, ple_wmap, l_bytes, l_counts = _quantize_decoder_layer_streaming(
                layer,
                layer_prefix,
                h_calib,
                pos_emb_list,
                calib_ids_list,
                base_dir=base_dir,
                shard_map=shard_map,
                output_dir=output_dir,
                layer_idx=l_idx,
                group_size=args.group_size,
                hadamard_block_size=args.hadamard_block_size,
                actorder=actorder,
                dampening_frac=args.dampening_frac,
                ple_group_size=ple_group_size,
                ple_cd_passes=ple_cd_passes,
                reuse_ple_dir=reuse_ple_vision_dir,
            )

            if is_qwen35_moe:
                qwen35_shard1_tensors.update(layer_tensors)
            else:
                layer_shard_name = f"model-layer-{l_idx:02d}.safetensors"
                for tk in layer_tensors:
                    new_weight_map[tk] = layer_shard_name
                new_weight_map.update(ple_wmap)
                total_bytes += l_bytes

                if pending_layer_save_fut is not None:
                    pending_layer_save_fut.result()
                pending_layer_save_fut = layer_io_pool.submit(
                    save_file, layer_tensors, str(output_dir / layer_shard_name)
                )

            shard_counts["ple_shards_quantized"] += l_counts["ple_shards"]
            shard_counts["text_attn_quantized"] += l_counts["attn_linears"]
            shard_counts["text_shared_experts_quantized"] += l_counts["shared_experts"]
            shard_counts["text_experts_quantized"] += l_counts["routed_experts"]

            del layer, layer_tensors
            if device.type == "cuda":
                torch.cuda.empty_cache()

            log_event(
                "QWEN4_EXP_LAYER_DONE",
                layer=l_idx,
                elapsed_s=round(time.time() - t_layer, 2),
                attn_linears=l_counts["attn_linears"],
                routed_experts=l_counts["routed_experts"],
                ple_shards=l_counts["ple_shards"],
            )

        if pending_layer_save_fut is not None:
            pending_layer_save_fut.result()

    if is_qwen35_moe:
        if qwen35_shard2_tensors:
            s1_name = "model-00001-of-00002.safetensors"
            s2_name = "model-00002-of-00002.safetensors"
            save_file(qwen35_shard1_tensors, str(output_dir / s1_name))
            save_file(qwen35_shard2_tensors, str(output_dir / s2_name))
            for tk, tv in qwen35_shard1_tensors.items():
                new_weight_map[tk] = s1_name
                total_bytes += tv.numel() * tv.element_size()
            for tk, tv in qwen35_shard2_tensors.items():
                new_weight_map[tk] = s2_name
                total_bytes += tv.numel() * tv.element_size()
        else:
            s1_name = "model.safetensors"
            save_file(qwen35_shard1_tensors, str(output_dir / s1_name))
            for tk, tv in qwen35_shard1_tensors.items():
                new_weight_map[tk] = s1_name
                total_bytes += tv.numel() * tv.element_size()
        del qwen35_shard1_tensors, qwen35_shard2_tensors

    # ------------------------------------------------------------------
    # Phase 4: Native MTP (if present)
    # ------------------------------------------------------------------
    has_mtp = _has_mtp_weights(base_dir, shard_map)
    if has_mtp:
        mtp_shard_name = "model_mtp.safetensors" if is_qwen35_moe else "model-mtp.safetensors"
        if args.quantize_mtp:
            mtp = _load_qwen4_exp_mtp_from_checkpoint(base_dir, cfg.text_config, device)
            embed_w_dev = embed_w_cpu.to(device)
            n_mtp = min(len(h_calib), args.component_calib_samples)
            h_mtp_calib: list[torch.Tensor] = []
            mtp_pos_list: list[tuple[torch.Tensor, torch.Tensor]] = []
            mtp_ids_list: list[torch.Tensor] = []

            final_norm_mtp = None
            if is_qwen35_moe:
                with torch.device(device), skip_weights_initialize():
                    final_norm_mtp = (
                        Qwen3_5MoeRMSNorm(cfg.text_config.hidden_size, eps=cfg.text_config.rms_norm_eps)
                        .to(dtype=torch.bfloat16)
                        .eval()
                    )
                norm_raw = _load_top_level_module_tensors(base_dir, shard_map, "model.language_model.norm.")
                final_norm_mtp.weight.data.copy_(
                    norm_raw["model.language_model.norm.weight"].to(device=device, dtype=torch.bfloat16)
                )
                del norm_raw

            for i in range(n_mtp):
                h_backbone = h_calib[i][:, :-1, :].to(device)
                if final_norm_mtp is not None:
                    h_backbone = final_norm_mtp(h_backbone)
                next_ids = calib_ids_list[i][:, 1:].to(device)
                next_embeds = F.embedding(next_ids, embed_w_dev)
                h_in = mtp.prepare_layer_input(h_backbone, next_embeds)
                seq_len_m = next_ids.shape[1]
                pos_ids_m = torch.arange(seq_len_m, device=device).view(1, 1, -1).expand(3, 1, -1)
                cos_m, sin_m = mtp.rotary_emb(next_embeds, pos_ids_m)
                h_mtp_calib.append(h_in.cpu())
                mtp_pos_list.append((cos_m.cpu(), sin_m.cpu()))
                mtp_ids_list.append(next_ids.cpu())
                del h_backbone, next_ids, next_embeds, h_in, cos_m, sin_m
            del embed_w_dev, final_norm_mtp

            if args.component_quant_mode == "gptq":
                mtp_layer_tensors, _, mtp_l_bytes, mtp_counts = _quantize_decoder_layer_streaming(
                    mtp.layers[0],
                    "mtp.layers.0",
                    h_mtp_calib,
                    mtp_pos_list,
                    mtp_ids_list,
                    base_dir=base_dir,
                    shard_map=shard_map,
                    output_dir=output_dir,
                    layer_idx=0,
                    group_size=args.group_size,
                    hadamard_block_size=args.hadamard_block_size,
                    actorder=actorder,
                    dampening_frac=args.dampening_frac,
                )
                shard_counts["mtp_quantized"] += (
                    mtp_counts["attn_linears"] + mtp_counts["shared_experts"] + mtp_counts["routed_experts"]
                )
            else:
                # RTN fallback for MTP if requested
                mtp_layer_tensors = {}
                mtp_l_bytes = 0
                for pname in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    w = getattr(mtp.layers[0].self_attn, pname).weight.data
                    k_in = w.shape[1]
                    had_b = min(args.hadamard_block_size, k_in & -k_in) if args.hadamard_block_size > 0 else 0
                    if had_b > 0:
                        w = _hadamard_rotate_weight(w, had_b)
                    packed = _pack_w4_tensor(w.cpu(), strategy="group", group_size=args.group_size)
                    for pk, pv in packed.items():
                        mtp_layer_tensors[f"mtp.layers.0.self_attn.{pname}.{pk}"] = pv.cpu().contiguous()
                    shard_counts["mtp_quantized"] += 1
                exps = mtp.layers[0].mlp.experts
                inter = exps.intermediate_dim
                for e in range(exps.num_experts):
                    for pname, w_slice in (
                        ("gate_proj", exps.gate_up_proj.data[e, :inter]),
                        ("up_proj", exps.gate_up_proj.data[e, inter:]),
                        ("down_proj", exps.down_proj.data[e]),
                    ):
                        k_in = w_slice.shape[1]
                        had_b = (
                            min(args.hadamard_block_size, k_in & -k_in) if args.hadamard_block_size > 0 else 0
                        )
                        if had_b > 0:
                            w_slice = _hadamard_rotate_weight(w_slice, had_b)
                        packed = _pack_w4_tensor(w_slice.cpu(), strategy="group", group_size=args.group_size)
                        for pk, pv in packed.items():
                            mtp_layer_tensors[f"mtp.layers.0.mlp.experts.{e}.{pname}.{pk}"] = (
                                pv.cpu().contiguous()
                            )
                        shard_counts["mtp_quantized"] += 1
                for sp_name in ("gate_proj", "up_proj", "down_proj"):
                    w_sp = getattr(mtp.layers[0].mlp.shared_expert, sp_name).weight.data
                    k_in = w_sp.shape[1]
                    had_b = min(args.hadamard_block_size, k_in & -k_in) if args.hadamard_block_size > 0 else 0
                    if had_b > 0:
                        w_sp = _hadamard_rotate_weight(w_sp, had_b)
                    packed = _pack_w4_tensor(w_sp.cpu(), strategy="group", group_size=args.group_size)
                    for pk, pv in packed.items():
                        mtp_layer_tensors[f"mtp.layers.0.mlp.shared_expert.{sp_name}.{pk}"] = (
                            pv.cpu().contiguous()
                        )
                    shard_counts["mtp_quantized"] += 1
                for name, param in mtp.layers[0].named_parameters():
                    if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                        continue
                    if any(name == f"self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
                        continue
                    if any(
                        name == f"mlp.shared_expert.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")
                    ):
                        continue
                    mtp_layer_tensors[f"mtp.layers.0.{name}"] = param.detach().cpu().contiguous()
                mtp_l_bytes = sum(t.numel() * t.element_size() for t in mtp_layer_tensors.values())

            # Add top-level mtp.* weights (fc_*, pre_fc_*, hyper_connection_mixer.* / norm.*)
            for name, param in mtp.named_parameters():
                if not name.startswith("layers."):
                    t_cpu = param.detach().cpu().contiguous()
                    mtp_layer_tensors[f"mtp.{name}"] = t_cpu
                    mtp_l_bytes += t_cpu.numel() * t_cpu.element_size()

            save_file(mtp_layer_tensors, str(output_dir / mtp_shard_name))
            for tk in mtp_layer_tensors:
                new_weight_map[tk] = mtp_shard_name
            total_bytes += mtp_l_bytes
            del mtp, mtp_layer_tensors, h_mtp_calib
        else:
            mtp_raw = _load_top_level_module_tensors(base_dir, shard_map, "mtp.")
            save_file(mtp_raw, str(output_dir / mtp_shard_name))
            for tk, tv in mtp_raw.items():
                new_weight_map[tk] = mtp_shard_name
                total_bytes += tv.numel() * tv.element_size()
            del mtp_raw
        if device.type == "cuda":
            torch.cuda.empty_cache()

    quant_wall_s = round(time.time() - t0, 2)

    # ------------------------------------------------------------------
    # Phase 5: Write index, config, processor, template, and receipt
    # ------------------------------------------------------------------
    index_payload = {
        "metadata": {"total_size": total_bytes},
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n"
    )

    _write_qwen4_exp_quant_config(
        output_dir,
        raw_base_cfg,
        group_size=args.group_size,
        actorder=actorder,
        hadamard_block_size=args.hadamard_block_size,
        quantize_mtp=args.quantize_mtp and has_mtp,
        quantize_vision=args.quantize_vision,
        vision_fc2_channel=shard_counts["vision_fc2_channel"] > 0,
        ple_group_size=ple_group_size,
        ple_cd_passes=ple_cd_passes,
    )

    processor.save_pretrained(str(output_dir))
    _assert_packed_alignment(output_dir)

    gc = build_generation_config(processor, info, str(base_dir), {})
    (output_dir / "generation_config.json").write_text(json.dumps(gc, indent=2) + "\n")
    tpl_sha = stamp_release_template(str(output_dir))
    check_template_renders(str(output_dir))
    ensure_license_files(str(output_dir), str(base_dir), info.get("license_files"))

    receipt = {
        "model": args.model,
        "base_snapshot": base_dir.name,
        "group_size": args.group_size,
        "hadamard_block_size": args.hadamard_block_size,
        "actorder": actorder,
        "dampening_frac": args.dampening_frac,
        "ple_group_size": ple_group_size,
        "ple_cd_passes": ple_cd_passes,
        "quantize_mtp": args.quantize_mtp and has_mtp,
        "quantize_vision": args.quantize_vision,
        "component_quant_mode": args.component_quant_mode,
        "shard_counts": shard_counts,
        "calib": calib,
        "num_samples": num_samples,
        "max_seq_length": max_seq_length,
        "quant_wall_s": quant_wall_s,
        "template_sha": tpl_sha,
    }
    (output_dir / "rot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    log_event("ROT_QUANT_DONE", **receipt)
    run_done(output_dir=str(output_dir))
    print(json.dumps(receipt, indent=2))


@torch.no_grad()
def _load_qwen4_exp_visual_from_checkpoint(
    ckpt_dir: Path,
    vision_config,
    device: torch.device,
) -> nn.Module:
    """Load BF16 or compressed-tensors W4A16 (+ online Hadamard) Qwen4ExpVisionModel or Qwen3_5MoeVisionModel from ckpt_dir."""
    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor.utils.dev import skip_weights_initialize

    shard_map, _ = _load_shard_map(ckpt_dir)
    raw_cfg = json.loads((ckpt_dir / "config.json").read_text())
    is_qwen35_moe = raw_cfg.get("text_config", {}).get("model_type") == "qwen3_5_moe_text"
    if is_qwen35_moe:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel as VisCls
    else:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel as VisCls

    qcfg = raw_cfg.get("quantization_config", {})
    g_size = qcfg.get("config_groups", {}).get("group_0", {}).get("weights", {}).get("group_size", 64)
    tcfg = qcfg.get("transform_config", {}).get("config_groups", {})
    had_size = tcfg.get("v", {}).get("head_dim", 0) if tcfg else 0

    vis_raw = _load_top_level_module_tensors(ckpt_dir, shard_map, "model.visual.")
    with torch.device(device), skip_weights_initialize():
        visual = VisCls._from_config(vision_config, dtype=torch.bfloat16).eval()

    rel_tensors = {k.removeprefix("model.visual."): v for k, v in vis_raw.items()}
    packed_prefixes = sorted(
        k.removesuffix(".weight_packed") for k in rel_tensors if k.endswith(".weight_packed")
    )

    group_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="group", group_size=g_size),
    )
    chan_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="channel"),
    )

    for p_rel in packed_prefixes:
        wp = rel_tensors.pop(f"{p_rel}.weight_packed").to(device)
        ws = rel_tensors.pop(f"{p_rel}.weight_scale").to(device)
        wsh = rel_tensors.pop(f"{p_rel}.weight_shape")
        scheme = chan_scheme if ws.shape[1] == 1 else group_scheme
        deq = PackedQuantizationCompressor.decompress(
            {"weight_packed": wp, "weight_scale": ws, "weight_shape": wsh},
            scheme,
        )["weight"].to(torch.bfloat16)
        mod = visual.get_submodule(p_rel)
        mod.weight.data.copy_(deq)
        if had_size > 0:
            k_in = int(deq.shape[1])
            b_size = min(had_size, k_in & -k_in)
            if b_size > 1:
                mod.register_forward_pre_hook(
                    lambda _m, args, bs=b_size: (
                        _hadamard_rotate_last_dim(args[0], bs, compute_dtype=torch.float32),
                        *args[1:],
                    )
                )

    param_map = dict(visual.named_parameters())
    buf_map = dict(visual.named_buffers())
    for k_rel, tensor in rel_tensors.items():
        if k_rel in param_map:
            param_map[k_rel].data.copy_(tensor.to(device=device, dtype=param_map[k_rel].dtype))
        elif k_rel in buf_map:
            buf_map[k_rel].data.copy_(tensor.to(device=device, dtype=buf_map[k_rel].dtype))
        else:
            raise RuntimeError(f"Unexpected visual tensor key: model.visual.{k_rel}")

    return visual


def _write_hot_expert_rankings(
    save_dir: Path,
    routing_stats: dict[str, dict[str, torch.Tensor]],
    *,
    source_desc: str,
) -> dict[str, list[int]]:
    """Write runtime/static_hot_cache_rankings.json, static_hot_cache_rankings.json, and runtime/expert_routing_profile.json."""
    rankings: dict[str, list[int]] = {}
    profile_layers: dict[str, dict] = {}
    cache_tiers = (16, 32, 64, 88, 128, 192, 256)

    for layer_key, st in routing_stats.items():
        counts = st["counts"].detach().cpu().to(torch.int64)
        weights = st["weights"].detach().cpu().to(torch.float64)
        num_exp = int(counts.numel())
        tot_assignments = max(int(counts.sum().item()), 1)
        tot_weight = max(float(weights.sum().item()), 1e-12)

        # Sort experts by (token_hit_count DESC, routing_weight_sum DESC, expert_id ASC)
        order = sorted(
            range(num_exp),
            key=lambda e: (-int(counts[e].item()), -float(weights[e].item()), e),
        )
        # Keep only active experts (count > 0) in static_hot_cache_rankings.json if any were hit, matching runtime format
        active_order = [e for e in order if int(counts[e].item()) > 0] or order
        if layer_key.isdigit():
            rankings[layer_key] = active_order

        counts_sorted = counts[order].cumsum(0)
        weights_sorted = weights[order].cumsum(0)
        hit_rates = {
            f"top_{k}": round(int(counts_sorted[min(k, num_exp) - 1].item()) / tot_assignments, 6)
            for k in cache_tiers
            if k <= num_exp
        }
        weight_mass = {
            f"top_{k}": round(float(weights_sorted[min(k, num_exp) - 1].item()) / tot_weight, 6)
            for k in cache_tiers
            if k <= num_exp
        }
        profile_layers[layer_key] = {
            "num_experts": num_exp,
            "active_experts": len(active_order),
            "total_expert_assignments": int(counts.sum().item()),
            "ranked_expert_ids": order,
            "active_ranked_expert_ids": active_order,
            "hit_rate_by_cache_size": hit_rates,
            "routing_mass_by_cache_size": weight_mass,
        }

    runtime_dir = save_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    rankings_text = json.dumps(rankings, indent=2) + "\n"
    (runtime_dir / "static_hot_cache_rankings.json").write_text(rankings_text)
    (save_dir / "static_hot_cache_rankings.json").write_text(rankings_text)

    profile_payload = {
        "source": source_desc,
        "num_target_layers": len(rankings),
        "layers": profile_layers,
    }
    (runtime_dir / "expert_routing_profile.json").write_text(json.dumps(profile_payload, indent=2) + "\n")
    return rankings


@torch.no_grad()
def _stream_forward_qwen4_exp_states(
    ckpt_dir: Path,
    text_config,
    input_ids_list: list[torch.Tensor],
    device: torch.device,
    *,
    inputs_embeds_override: list[torch.Tensor] | None = None,
    position_ids_override: list[torch.Tensor] | None = None,
    return_pre_mixer: bool = False,
    routing_stats_out: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None, nn.Linear]:
    """Stream through a BF16 or quantized Qwen4Exp / Qwen3_5Moe checkpoint layer-by-layer and return final states + lm_head."""
    from llmcompressor.utils.dev import skip_weights_initialize

    is_qwen35_moe = _is_qwen35_moe_text_config(text_config)
    if is_qwen35_moe:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeRMSNorm,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextRotaryEmbedding as RotaryCls,
        )
    else:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextGatedResidual,
        )
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextRotaryEmbedding as RotaryCls,
        )

    shard_map, _ = _load_shard_map(ckpt_dir)
    raw_cfg = json.loads((ckpt_dir / "config.json").read_text())
    qcfg = raw_cfg.get("quantization_config", {})
    g_size = qcfg.get("config_groups", {}).get("group_0", {}).get("weights", {}).get("group_size", 64)
    tcfg = qcfg.get("transform_config", {}).get("config_groups", {})
    had_size = tcfg.get("v", {}).get("head_dim", 0) if tcfg else 0
    ple_g_size = qcfg.get("ple_quantization", {}).get("group_size", 32)

    embed_raw = _load_top_level_module_tensors(ckpt_dir, shard_map, "model.language_model.embed_tokens.")
    embed_w = embed_raw["model.language_model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    rotary = RotaryCls(config=text_config).to(device)
    hc_count = 1 if is_qwen35_moe else text_config.hc_count

    h_list: list[torch.Tensor] = []
    pos_emb_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    keep_states_on_device = device.type == "cuda" and torch.cuda.get_device_properties(
        device
    ).total_memory >= 100 * (1024**3)
    state_store_device = device if keep_states_on_device else torch.device("cpu")

    for idx_s, ids in enumerate(input_ids_list):
        ids_dev = ids.to(device)
        if inputs_embeds_override is not None and inputs_embeds_override[idx_s] is not None:
            inp_emb = inputs_embeds_override[idx_s].to(device=device, dtype=torch.bfloat16)
        else:
            inp_emb = F.embedding(ids_dev, embed_w)
        seq_len = inp_emb.shape[1]
        if position_ids_override is not None and position_ids_override[idx_s] is not None:
            pos_ids = position_ids_override[idx_s].to(device)
        else:
            pos_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, inp_emb.shape[0], -1)
        cos, sin = rotary(inp_emb, pos_ids)
        pos_emb_list.append((cos.to(state_store_device), sin.to(state_store_device)))
        h_init = inp_emb if is_qwen35_moe else inp_emb.repeat(1, 1, hc_count)
        h_list.append(h_init.to(state_store_device))
        del ids_dev, inp_emb, cos, sin

    del embed_w, embed_raw
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seq_lens = [h.shape[1] for h in h_list]
    H_all = torch.cat([h.squeeze(0).to(device) for h in h_list], dim=0)
    del h_list
    n_tot = H_all.shape[0]
    h_dim = text_config.hidden_size
    tok_step = 65536
    t_stream0 = time.time()

    num_layers = text_config.num_hidden_layers
    with ThreadPoolExecutor(max_workers=1) as probe_io_pool:
        next_probe_fut: Future | None = (
            probe_io_pool.submit(
                _prefetch_layer_raw_tensors, "model.language_model.layers.0", ckpt_dir, shard_map
            )
            if num_layers > 0
            else None
        )

        for l_idx in range(num_layers):
            prefetched_raw = next_probe_fut.result() if next_probe_fut is not None else None
            if l_idx + 1 < num_layers:
                next_probe_fut = probe_io_pool.submit(
                    _prefetch_layer_raw_tensors,
                    f"model.language_model.layers.{l_idx + 1}",
                    ckpt_dir,
                    shard_map,
                )
            else:
                next_probe_fut = None

            layer = _build_lightweight_decoder_layer(text_config, l_idx, device)
            layer_prefix = f"model.language_model.layers.{l_idx}"
            _load_decoder_layer_weights(
                layer,
                layer_prefix,
                ckpt_dir,
                shard_map,
                device,
                group_size=g_size,
                hadamard_block_size=had_size,
                prefetched_raw=prefetched_raw,
            )
            del prefetched_raw

            if getattr(layer, "ple", None) is not None:
                ple_embeds = stream_gather_ple_embeds(
                    ckpt_dir, shard_map, layer.ple, l_idx, input_ids_list, ple_group_size=ple_g_size
                )
                seq_cur = 0
                for i, slen in enumerate(seq_lens):
                    h_seq = H_all[seq_cur : seq_cur + slen].unsqueeze(0)
                    ple_out = forward_ple_with_precomputed_embeds(layer.ple, h_seq, ple_embeds[i])
                    H_all[seq_cur : seq_cur + slen] = (h_seq + ple_out).squeeze(0)
                    seq_cur += slen
                del ple_embeds

            has_hc = hasattr(layer, "attn_hyper_connection")
            layer_kind = getattr(layer, "layer_type", getattr(layer, "block_type", ""))
            is_linear_attn = layer_kind == "linear_attention"
            branch_in_all = torch.empty((n_tot, h_dim), dtype=H_all.dtype, device=device)
            inj_w_all = torch.empty((n_tot, hc_count), dtype=H_all.dtype, device=device) if has_hc else None
            for s0 in range(0, n_tot, tok_step):
                s1 = min(s0 + tok_step, n_tot)
                if has_hc:
                    b_in, hyp_in, iw = layer.attn_hyper_connection(H_all[s0:s1].unsqueeze(0))
                    branch_in_all[s0:s1] = b_in.squeeze(0)
                    H_all[s0:s1] = hyp_in.squeeze(0)
                    inj_w_all[s0:s1] = iw.squeeze(0)
                    del b_in, hyp_in, iw
                else:
                    branch_in_all[s0:s1] = layer.input_layernorm(H_all[s0:s1])

            attn_out_all = torch.empty((n_tot, h_dim), dtype=H_all.dtype, device=device)
            seq_cur = 0
            for i, slen in enumerate(seq_lens):
                b_seq = branch_in_all[seq_cur : seq_cur + slen].unsqueeze(0)
                if is_linear_attn:
                    a_out = layer.linear_attn(b_seq)
                else:
                    pos_gpu = (pos_emb_list[i][0].to(device), pos_emb_list[i][1].to(device))
                    a_out = _forward_qsa_attention_fast(layer.self_attn, b_seq, pos_gpu)
                attn_out_all[seq_cur : seq_cur + slen] = a_out.squeeze(0)
                seq_cur += slen
            del branch_in_all

            for s0 in range(0, n_tot, tok_step):
                s1 = min(s0 + tok_step, n_tot)
                if has_hc:
                    h_post_a = H_all[s0:s1] + (
                        attn_out_all[s0:s1].unsqueeze(-2) * inj_w_all[s0:s1].unsqueeze(-1)
                    ).flatten(-2)
                    m_in, hyp_m, iw_m = layer.mlp_hyper_connection(h_post_a.unsqueeze(0))
                else:
                    h_post_a = H_all[s0:s1] + attn_out_all[s0:s1]
                    m_in = layer.post_attention_layernorm(h_post_a).unsqueeze(0)
                if routing_stats_out is not None:
                    num_exp = layer.mlp.experts.num_experts
                    l_key = str(l_idx)
                    if l_key not in routing_stats_out:
                        routing_stats_out[l_key] = {
                            "counts": torch.zeros(num_exp, dtype=torch.int64, device=device),
                            "weights": torch.zeros(num_exp, dtype=torch.float32, device=device),
                        }
                    st_entry = routing_stats_out[l_key]

                    def _gate_hook(_m, _inp, out, _st=st_entry, _ne=num_exp):
                        _logits, rw, sel = out
                        flat_sel = sel.reshape(-1).long()
                        _st["counts"].add_(torch.bincount(flat_sel, minlength=_ne)[:_ne])
                        _st["weights"].scatter_add_(0, flat_sel, rw.reshape(-1).float())

                    gh = layer.mlp.gate.register_forward_hook(_gate_hook)
                    try:
                        m_out = layer.mlp(m_in).squeeze(0)
                    finally:
                        gh.remove()
                else:
                    m_out = layer.mlp(m_in).squeeze(0)
                if has_hc:
                    H_all[s0:s1] = hyp_m.squeeze(0) + (
                        m_out.unsqueeze(-2) * iw_m.squeeze(0).unsqueeze(-1)
                    ).flatten(-2)
                    del hyp_m, iw_m
                else:
                    H_all[s0:s1] = h_post_a + m_out
                del h_post_a, m_in, m_out
            del attn_out_all, inj_w_all
            if routing_stats_out is not None and str(l_idx) in routing_stats_out:
                routing_stats_out[str(l_idx)]["counts"] = routing_stats_out[str(l_idx)]["counts"].cpu()
                routing_stats_out[str(l_idx)]["weights"] = routing_stats_out[str(l_idx)]["weights"].cpu()

            del layer
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if (l_idx + 1) % 8 == 0 or l_idx == 0 or (l_idx + 1) == num_layers:
                log_event(
                    "QWEN4_EXP_PROBE_LAYER_PROGRESS",
                    ckpt=ckpt_dir.name,
                    completed_layers=l_idx + 1,
                    total_layers=num_layers,
                    elapsed_s=round(time.time() - t_stream0, 2),
                )

    pre_mixer_list: list[torch.Tensor] | None = [] if return_pre_mixer else None

    if is_qwen35_moe:
        with torch.device(device), skip_weights_initialize():
            final_norm = (
                Qwen3_5MoeRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
                .to(dtype=torch.bfloat16)
                .eval()
            )
        norm_raw = _load_top_level_module_tensors(ckpt_dir, shard_map, "model.language_model.norm.")
        final_norm.weight.data.copy_(
            norm_raw["model.language_model.norm.weight"].to(device=device, dtype=torch.bfloat16)
        )
        del norm_raw
    else:
        with torch.device(device), skip_weights_initialize():
            final_norm = (
                Qwen4ExpTextGatedResidual(text_config, use_combine=False).to(dtype=torch.bfloat16).eval()
            )
        mixer_raw = _load_top_level_module_tensors(
            ckpt_dir, shard_map, "model.language_model.hyper_connection_mixer."
        )
        mixer_sd = {
            k.removeprefix("model.language_model.hyper_connection_mixer."): v.to(
                device=device, dtype=torch.bfloat16
            )
            for k, v in mixer_raw.items()
        }
        final_norm.load_state_dict(mixer_sd, strict=True)
        del mixer_raw, mixer_sd

    final_hidden_list: list[torch.Tensor] = []
    seq_cur = 0
    for slen in seq_lens:
        h_seq = H_all[seq_cur : seq_cur + slen].unsqueeze(0)
        seq_cur += slen
        if pre_mixer_list is not None:
            pre_mixer_list.append(h_seq.cpu().clone())
        final_hidden_list.append(final_norm(h_seq).cpu())
    del final_norm, H_all

    with torch.device(device), skip_weights_initialize():
        lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False, dtype=torch.bfloat16
        ).eval()
    head_raw = _load_top_level_module_tensors(ckpt_dir, shard_map, "lm_head.")
    if "lm_head.weight" in head_raw:
        lm_head.weight.data.copy_(head_raw["lm_head.weight"].to(device=device, dtype=torch.bfloat16))
    else:
        emb_raw = _load_top_level_module_tensors(ckpt_dir, shard_map, "model.language_model.embed_tokens.")
        lm_head.weight.data.copy_(
            emb_raw["model.language_model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
        )

    return final_hidden_list, pre_mixer_list, lm_head


@torch.no_grad()
def run_probe_qwen4_exp(args: argparse.Namespace) -> None:
    """Single-pass streaming fidelity probe for Qwen4Exp (text KL/PPL + MTP + Vision)."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    from warp_quant.data import (
        CHUNK,
        WEIGHTED_KL_PROTOCOL,
        aggregate_weighted_kl,
        compare_logits,
        load_batches,
        select_text_rows,
        summarize,
    )

    hub_id, info = resolve(args.model)
    base_dir = (
        Path(hub_id)
        if Path(hub_id).exists()
        else Path(resolve_snapshot(hub_id, revision=info.get("hub_revision")))
    )
    quant_dir = Path(args.quant).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log_event(
        "ROT_PROBE_START",
        model=args.model,
        base=str(base_dir),
        quant=str(quant_dir),
        eval_sets=args.eval_sets,
        eval_mtp=args.eval_mtp,
    )

    raw_base_cfg = json.loads((base_dir / "config.json").read_text())
    is_qwen35_moe = raw_base_cfg.get("text_config", {}).get("model_type") == "qwen3_5_moe_text"
    if is_qwen35_moe:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

        cfg = Qwen3_5MoeConfig.from_dict(raw_base_cfg)
    else:
        cfg = Qwen4ExpConfig.from_dict(raw_base_cfg)
    _ensure_qwen4_exp_device_kernels(cfg.text_config, device)
    processor = AutoProcessor.from_pretrained(str(base_dir))

    # ------------------------------------------------------------------
    # 1. Prepare all evaluation sequences up front so we stream base_dir ONCE and quant_dir ONCE
    # ------------------------------------------------------------------
    base_ids_all: list[torch.Tensor] = []
    base_emb_all: list[torch.Tensor | None] = []
    base_pos_all: list[torch.Tensor | None] = []

    quant_ids_all: list[torch.Tensor] = []
    quant_emb_all: list[torch.Tensor | None] = []
    quant_pos_all: list[torch.Tensor | None] = []

    text_eval_specs = []
    for eval_path_str in args.eval_sets:
        eval_path = Path(eval_path_str)
        if not eval_path.is_absolute():
            eval_path = ROOT / eval_path
        batches = load_batches(processor, str(eval_path), limit=args.limit)
        b_start = len(base_ids_all)
        q_start = len(quant_ids_all)
        for _dom, ids in batches:
            base_ids_all.append(ids)
            base_emb_all.append(None)
            base_pos_all.append(None)
            quant_ids_all.append(ids)
            quant_emb_all.append(None)
            quant_pos_all.append(None)
        text_eval_specs.append((eval_path, batches, b_start, q_start, len(batches)))

    mtp_ids_list: list[torch.Tensor] = []
    mtp_b_start = 0
    mtp_q_start = 0
    has_mtp_for_eval = bool(
        args.eval_mtp
        and _has_mtp_weights(base_dir, _load_shard_map(base_dir)[0])
        and _has_mtp_weights(quant_dir, _load_shard_map(quant_dir)[0])
    )
    if has_mtp_for_eval:
        first_eval_path = Path(args.eval_sets[0])
        if not first_eval_path.is_absolute():
            first_eval_path = ROOT / first_eval_path
        selected = select_text_rows(str(first_eval_path), args.mtp_per_domain)
        mtp_b_start = len(base_ids_all)
        mtp_q_start = len(quant_ids_all)
        for _, _domain, messages in selected:
            text = processor.apply_chat_template(messages, tokenize=False)
            ids = processor.tokenizer(
                text, add_special_tokens=False, truncation=True, max_length=512, return_tensors="pt"
            )["input_ids"]
            if ids.shape[1] >= 4:
                mtp_ids_list.append(ids)
                base_ids_all.append(ids)
                base_emb_all.append(None)
                base_pos_all.append(None)
                quant_ids_all.append(ids)
                quant_emb_all.append(None)
                quant_pos_all.append(None)

    vis_specs = []
    if args.eval_vision or args.eval_vision_guard:
        from types import MethodType, SimpleNamespace

        from warp_quant.data import device_inputs, encode_image_row, feature_metrics

        if is_qwen35_moe:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel as RopeModelCls
        else:
            from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel as RopeModelCls

        base_visual = _load_qwen4_exp_visual_from_checkpoint(base_dir, cfg.vision_config, device)
        quant_visual = _load_qwen4_exp_visual_from_checkpoint(quant_dir, cfg.vision_config, device)
        base_shard_map, _ = _load_shard_map(base_dir)
        emb_raw = _load_top_level_module_tensors(
            base_dir, base_shard_map, "model.language_model.embed_tokens."
        )
        embed_w = emb_raw["model.language_model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
        dummy_m = SimpleNamespace(config=cfg)
        dummy_m.get_vision_position_ids = MethodType(RopeModelCls.get_vision_position_ids, dummy_m)

        for vis_key, vis_arg in (
            ("vision_fidelity", args.eval_vision),
            ("vision_guard_fidelity", args.eval_vision_guard),
        ):
            if not vis_arg:
                continue
            vis_path = Path(vis_arg)
            if not vis_path.is_absolute():
                vis_path = ROOT / vis_path
            manifest_path = vis_path / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("calibration", False):
                    raise ValueError(f"Refusing to evaluate Vision on calibration fixture: {vis_path}")
                if vis_key == "vision_guard_fidelity" and not manifest.get("guard", False):
                    raise ValueError(f"Refusing to run Vision guard on non-guard fixture: {vis_path}")

            rows_path = vis_path / "rows.jsonl"
            fixture_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
            vis_ids_list: list[torch.Tensor] = []
            vis_keep_list: list[torch.Tensor] = []
            vis_labels_list: list[torch.Tensor] = []
            vis_domains_list: list[str] = []
            ref_embeds_list: list[torch.Tensor] = []
            cand_embeds_list: list[torch.Tensor] = []
            pos_ids_list: list[torch.Tensor] = []
            enc_rows = []

            for row in fixture_rows:
                encoded, keep, labels = encode_image_row(processor, vis_path, row, max_pixels=1024 * 1024)
                inputs = device_inputs(encoded, device)
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"].to(torch.bfloat16)
                image_grid_thw = inputs["image_grid_thw"]
                mm_token_type_ids = inputs.get("mm_token_type_ids")
                if mm_token_type_ids is None:
                    mm_token_type_ids = (input_ids == cfg.image_token_id).int()
                pos_3d, _ = RopeModelCls.get_rope_index(
                    dummy_m,
                    input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=inputs.get("attention_mask"),
                )

                vis_ref = base_visual(pixel_values, grid_thw=image_grid_thw).pooler_output.to(torch.bfloat16)
                vis_cand = quant_visual(pixel_values, grid_thw=image_grid_thw).pooler_output.to(
                    torch.bfloat16
                )

                f_m = feature_metrics(vis_ref, vis_cand)
                cos = F.cosine_similarity(vis_ref.float().flatten(), vis_cand.float().flatten(), dim=0).item()
                f_m["cos_sim"] = cos
                enc_rows.append(f_m)

                base_tok_emb = F.embedding(input_ids, embed_w)
                mask = (input_ids == cfg.image_token_id).unsqueeze(-1).expand_as(base_tok_emb)
                ref_emb = base_tok_emb.masked_scatter(mask, vis_ref).cpu()
                cand_emb = base_tok_emb.masked_scatter(mask, vis_cand).cpu()

                vis_ids_list.append(input_ids.cpu())
                vis_keep_list.append(keep.cpu())
                vis_labels_list.append(labels.cpu())
                vis_domains_list.append(row["domain"])
                ref_embeds_list.append(ref_emb)
                cand_embeds_list.append(cand_emb)
                pos_ids_list.append(pos_3d.cpu())

            b_ref_start = len(base_ids_all)
            for ids_v, ref_e, pos_v in zip(vis_ids_list, ref_embeds_list, pos_ids_list, strict=True):
                base_ids_all.append(ids_v)
                base_emb_all.append(ref_e)
                base_pos_all.append(pos_v)
            b_cand_start = len(base_ids_all)
            for ids_v, cand_e, pos_v in zip(vis_ids_list, cand_embeds_list, pos_ids_list, strict=True):
                base_ids_all.append(ids_v)
                base_emb_all.append(cand_e)
                base_pos_all.append(pos_v)
            q_cand_start = len(quant_ids_all)
            for ids_v, cand_e, pos_v in zip(vis_ids_list, cand_embeds_list, pos_ids_list, strict=True):
                quant_ids_all.append(ids_v)
                quant_emb_all.append(cand_e)
                quant_pos_all.append(pos_v)

            vis_specs.append(
                (
                    vis_key,
                    vis_keep_list,
                    vis_labels_list,
                    vis_domains_list,
                    enc_rows,
                    b_ref_start,
                    b_cand_start,
                    q_cand_start,
                    len(vis_ids_list),
                )
            )

        del base_visual, quant_visual, embed_w, emb_raw
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. Stream base_dir ONCE and quant_dir ONCE across all combined sequences
    # ------------------------------------------------------------------
    t0 = time.time()
    quant_routing_stats: dict[str, dict[str, torch.Tensor]] = {}
    base_final_all, base_pre_mixer_all, base_head = _stream_forward_qwen4_exp_states(
        base_dir,
        cfg.text_config,
        base_ids_all,
        device,
        inputs_embeds_override=base_emb_all,
        position_ids_override=base_pos_all,
        return_pre_mixer=bool(args.eval_mtp),
    )
    quant_final_all, quant_pre_mixer_all, quant_head = _stream_forward_qwen4_exp_states(
        quant_dir,
        cfg.text_config,
        quant_ids_all,
        device,
        inputs_embeds_override=quant_emb_all,
        position_ids_override=quant_pos_all,
        return_pre_mixer=bool(args.eval_mtp),
        routing_stats_out=quant_routing_stats,
    )
    stream_elapsed_s = round(time.time() - t0, 2)

    # ------------------------------------------------------------------
    # 3. Compute Text KL/PPL, MTP, and Vision metrics from the single-pass outputs
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}

    for eval_path, batches, b_start, q_start, n_b in text_eval_specs:
        nll = {"base": 0.0, "quant": 0.0}
        count = 0
        tok_kls = []
        row_dump = []
        dom = defaultdict(lambda: {"base": 0.0, "quant": 0.0, "count": 0, "kl": 0.0, "kl_n": 0})

        for i, (domain, ids) in enumerate(batches):
            targets = ids[:, 1:].to(device)
            lb = base_head(base_final_all[b_start + i].to(device))[:, :-1]
            lq = quant_head(quant_final_all[q_start + i].to(device))[:, :-1]
            d = dom[domain]
            row_nll = {"base": 0.0, "quant": 0.0}
            row_kl_sum, row_kl_n = 0.0, 0
            for s in range(0, lq.shape[1], CHUNK):
                sl = slice(s, min(s + CHUNK, lq.shape[1]))
                lbs = F.log_softmax(lb[:, sl].float(), dim=-1)
                lqs = F.log_softmax(lq[:, sl].float(), dim=-1)
                tgt_sl = targets[:, sl]
                for name, ls in (("base", lbs), ("quant", lqs)):
                    loss = -ls.gather(-1, tgt_sl.unsqueeze(-1)).sum().item()
                    nll[name] += loss
                    d[name] += loss
                    row_nll[name] += loss
                kl = F.kl_div(lqs, lbs, log_target=True, reduction="none").sum(-1).flatten()
                n_tok = kl.numel()
                count += n_tok
                d["count"] += n_tok
                tok_kls.append(kl.cpu())
                d["kl"] += kl.sum().item()
                d["kl_n"] += n_tok
                row_kl_sum += kl.sum().item()
                row_kl_n += n_tok
                del lbs, lqs, kl
            del lb, lq
            row_dump.append(
                {
                    "row": i,
                    "domain": domain,
                    "tokens": row_kl_n,
                    "mean_kl": row_kl_sum / max(row_kl_n, 1),
                    "base_nll": row_nll["base"],
                    "quant_nll": row_nll["quant"],
                }
            )

        kls = torch.cat(tok_kls)
        ppl_base = math.exp(nll["base"] / count)
        ppl_quant = math.exp(nll["quant"] / count)
        mean_kl = kls.mean().item()
        p999_kl = torch.quantile(kls.double(), 0.999).item()
        try:
            weighted_kl = aggregate_weighted_kl(dom)
        except ValueError:
            weighted_kl = None

        domains_summary = {
            k: {
                "tokens": v["kl_n"],
                "mean_kl": round(v["kl"] / max(v["kl_n"], 1), 6),
                "ppl_base": round(math.exp(v["base"] / max(v["count"], 1)), 4),
                "ppl_quant": round(math.exp(v["quant"] / max(v["count"], 1)), 4),
            }
            for k, v in sorted(dom.items())
        }
        summary = {
            "eval_set": eval_path.name,
            "rows": n_b,
            "tokens": count,
            "ppl_base": round(ppl_base, 4),
            "ppl_quant": round(ppl_quant, 4),
            "ppl_delta_pct": round((ppl_quant - ppl_base) / ppl_base * 100, 3),
            "mean_kl": round(mean_kl, 6),
            "weighted_kl": round(weighted_kl, 6) if weighted_kl is not None else None,
            "weighted_kl_protocol": WEIGHTED_KL_PROTOCOL if weighted_kl is not None else None,
            "p999_kl": round(p999_kl, 4),
            "elapsed_s": stream_elapsed_s,
            "domains": domains_summary,
        }
        results[eval_path.stem] = summary
        log_event("ROT_PROBE_SET", **summary)
        print(json.dumps(summary, indent=2))

        if args.dump_rows_dir:
            dump_dir = Path(args.dump_rows_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            out_rows = dump_dir / f"{eval_path.stem}_rows.jsonl"
            with out_rows.open("w") as f:
                for r in row_dump:
                    f.write(json.dumps(r) + "\n")

    if args.eval_mtp and mtp_ids_list:
        assert base_pre_mixer_all is not None and quant_pre_mixer_all is not None
        base_shard_map, _ = _load_shard_map(base_dir)
        emb_raw = _load_top_level_module_tensors(
            base_dir, base_shard_map, "model.language_model.embed_tokens."
        )
        embed_w = emb_raw["model.language_model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)

        base_mtp = _load_qwen4_exp_mtp_from_checkpoint(base_dir, cfg.text_config, device)
        quant_mtp = _load_qwen4_exp_mtp_from_checkpoint(quant_dir, cfg.text_config, device)
        num_mtp_exp = quant_mtp.layers[0].mlp.experts.num_experts
        mtp_st = {
            "counts": torch.zeros(num_mtp_exp, dtype=torch.int64, device=device),
            "weights": torch.zeros(num_mtp_exp, dtype=torch.float32, device=device),
        }

        def _mtp_gate_hook(_m, _inp, out):
            _logits, rw, sel = out
            flat_sel = sel.reshape(-1).long()
            mtp_st["counts"].add_(torch.bincount(flat_sel, minlength=num_mtp_exp)[:num_mtp_exp])
            mtp_st["weights"].scatter_add_(0, flat_sel, rw.reshape(-1).float())

        mtp_gh = quant_mtp.layers[0].mlp.gate.register_forward_hook(_mtp_gate_hook)

        isolated_rows = []
        e2e_rows = []
        try:
            for i, ids in enumerate(mtp_ids_list):
                ids_dev = ids.to(device)
                next_ids = ids_dev[:, 1:-1]
                labels = ids_dev[:, 2:]
                if is_qwen35_moe:
                    h_b = base_final_all[mtp_b_start + i][:, :-2, :].to(device)
                    h_q = quant_final_all[mtp_q_start + i][:, :-2, :].to(device)
                    pos = torch.arange(0, ids_dev.shape[1] - 2, device=device).unsqueeze(0)
                else:
                    h_b = base_pre_mixer_all[mtp_b_start + i][:, :-2, :].to(device)
                    h_q = quant_pre_mixer_all[mtp_q_start + i][:, :-2, :].to(device)
                    pos = torch.arange(1, ids_dev.shape[1] - 1, device=device).unsqueeze(0)
                emb = F.embedding(next_ids, embed_w)
                mtp_h_base = base_mtp(h_b, emb, pos)
                mtp_h_iso = quant_mtp(h_b, emb, pos)
                mtp_h_e2e = quant_mtp(h_q, emb, pos)
                ref_logits = base_head(mtp_h_base)
                iso_logits = base_head(mtp_h_iso)
                e2e_logits = base_head(mtp_h_e2e)
                isolated_rows.append(compare_logits(ref_logits, iso_logits, labels))
                e2e_rows.append(compare_logits(ref_logits, e2e_logits, labels))
        finally:
            mtp_gh.remove()

        quant_routing_stats["mtp.0"] = {
            "counts": mtp_st["counts"].cpu(),
            "weights": mtp_st["weights"].cpu(),
        }

        del base_mtp, quant_mtp, embed_w, emb_raw
        if device.type == "cuda":
            torch.cuda.empty_cache()

        mtp_summary = {"isolated": summarize(isolated_rows), "e2e": summarize(e2e_rows)}
        results["mtp_fidelity"] = mtp_summary
        log_event(
            "ROT_PROBE_MTP",
            isolated_mean_kl=mtp_summary["isolated"]["mean_kl"],
            e2e_mean_kl=mtp_summary["e2e"]["mean_kl"],
        )
        print(json.dumps({"mtp_fidelity": mtp_summary}, indent=2))

    for (
        vis_key,
        vis_keep_list,
        vis_labels_list,
        vis_domains_list,
        enc_rows,
        b_ref_start,
        b_cand_start,
        q_cand_start,
        n_v,
    ) in vis_specs:
        iso_vqa_rows = []
        e2e_vqa_rows = []
        for i in range(n_v):
            keep_cpu = vis_keep_list[i].cpu()
            labels_dev = vis_labels_list[i].to(device)
            ref_logits = base_head(base_final_all[b_ref_start + i][:, keep_cpu].to(device))
            iso_logits = base_head(base_final_all[b_cand_start + i][:, keep_cpu].to(device))
            e2e_logits = quant_head(quant_final_all[q_cand_start + i][:, keep_cpu].to(device))
            iso_vqa_rows.append(
                {"domain": vis_domains_list[i], **compare_logits(ref_logits, iso_logits, labels_dev)}
            )
            e2e_vqa_rows.append(
                {"domain": vis_domains_list[i], **compare_logits(ref_logits, e2e_logits, labels_dev)}
            )

        sq_err = sum(r["squared_error"] for r in enc_rows)
        ref_energy = sum(r["reference_energy"] for r in enc_rows)
        vis_summary = {
            "encoder": {
                "rows": len(enc_rows),
                "relative_rmse": (sq_err / ref_energy) ** 0.5,
                "max_row_relative_rmse": max(r["relative_rmse"] for r in enc_rows),
                "mean_cos_sim": sum(r["cos_sim"] for r in enc_rows) / len(enc_rows),
            },
            "isolated": summarize(iso_vqa_rows),
            "isolated_domains": {
                d: summarize([r for r in iso_vqa_rows if r["domain"] == d])
                for d in sorted({r["domain"] for r in iso_vqa_rows})
            },
            "e2e": summarize(e2e_vqa_rows),
            "e2e_domains": {
                d: summarize([r for r in e2e_vqa_rows if r["domain"] == d])
                for d in sorted({r["domain"] for r in e2e_vqa_rows})
            },
        }
        results[vis_key] = vis_summary
        log_event(
            "ROT_PROBE_VISION" if vis_key == "vision_fidelity" else "ROT_PROBE_VISION_GUARD",
            encoder_relative_rmse=vis_summary["encoder"]["relative_rmse"],
            isolated_mean_kl=vis_summary["isolated"]["mean_kl"],
            e2e_mean_kl=vis_summary["e2e"]["mean_kl"],
        )
        print(json.dumps({vis_key: vis_summary}, indent=2))

    del base_final_all, quant_final_all, base_pre_mixer_all, quant_pre_mixer_all, base_head, quant_head
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if quant_routing_stats:
        rankings = _write_hot_expert_rankings(
            quant_dir,
            quant_routing_stats,
            source_desc="probe_multimodal_and_text_eval_pass",
        )
        log_event(
            "QWEN4_EXP_HOT_CACHE_RANKINGS_WRITTEN",
            output=str(quant_dir / "runtime" / "static_hot_cache_rankings.json"),
            layers=len(rankings),
        )

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2) + "\n")

    run_done(quant=str(quant_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser(
        "quantize", help="Streaming W4A16-HGPTQ + 4-bit Custom-LUT4 PLE quantizer for Qwen4Exp"
    )
    q.add_argument("--model", default="flash-next")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--group-size", type=int, choices=[32, 64, 128], default=64)
    q.add_argument("--hadamard-block-size", type=int, default=64)
    q.add_argument("--actorder", choices=["none", "weight", "static"], default="static")
    q.add_argument("--dampening-frac", type=float, default=0.01)
    q.add_argument("--ple-group-size", type=int, choices=[0, 32, 64, 128], default=32)
    q.add_argument("--ple-cd-passes", type=int, default=1)
    q.add_argument("--quantize-mtp", action=argparse.BooleanOptionalAction, default=True)
    q.add_argument("--quantize-vision", action=argparse.BooleanOptionalAction, default=True)
    q.add_argument("--component-quant-mode", choices=["gptq", "rtn"], default="gptq")
    q.add_argument("--vision-calib-dir", default="data/vision-calib-v2")
    q.add_argument("--vision-calib-samples", type=int, default=None)
    q.add_argument("--component-calib-samples", type=int, default=64)
    q.add_argument("--calib", default=None)
    q.add_argument("--num-samples", type=int, default=None)
    q.add_argument("--max-seq-length", type=int, default=None)
    q.add_argument("--window", choices=["head", "tail", "random"], default=None)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--sequential-offload-device", choices=["cpu", "cuda"], default="cuda")
    q.add_argument(
        "--reuse-ple-vision-from",
        default=None,
        help="Optional existing checkpoint dir to reuse unchanged model-ple-*.safetensors, model-vision.safetensors, and model-embed-head.safetensors from",
    )

    p = sub.add_parser("probe", help="Streaming layer-by-layer fidelity probe for Qwen4Exp")
    p.add_argument("--model", default="flash-next")
    p.add_argument("--quant", required=True)
    p.add_argument("--eval-sets", nargs="+", default=["data/probe-v4.jsonl", "data/probe-guard.jsonl"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-mtp", action="store_true")
    p.add_argument("--mtp-per-domain", type=int, default=2)
    p.add_argument("--eval-vision", default=None)
    p.add_argument("--eval-vision-guard", default=None)
    p.add_argument("--dump-rows-dir", default=None)
    p.add_argument("--output-json", default=None)

    args = parser.parse_args()
    try:
        if args.cmd == "quantize":
            run_quantize_qwen4_exp(args)
        elif args.cmd == "probe":
            run_probe_qwen4_exp(args)
    except BaseException as exc:
        run_fail(error=str(exc))
        raise


if __name__ == "__main__":
    main()
