"""Clean W4A16 + Block-Hadamard rotation experiment (Text, Vision, and Native MTP).

Uses current llmcompressor (QuIPModifier + GPTQModifier + oneshot) and
compressed-tensors (TransformConfig + PackedQuantizationCompressor) APIs
without legacy experiment dependencies.

Subcommands:
  quantize  Run W4A16 GPTQ (+ optional block-Hadamard input rotation) on main text,
            plus optional W4A16 (+ Hadamard) on Native MTP and Vision linears.
  probe     Evaluate a quantized checkpoint (with or without transform_config) on
            probe-v4 / probe-guard text KL, plus optional MTP and Vision fidelity.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor

from warp_quant.cachepath import resolve_snapshot
from warp_quant.lablog import log_event, run_done, run_fail
from warp_quant.models import model_class, quant_recipe, resolve
from warp_quant.stamp import (
    build_generation_config,
    check_template_renders,
    ensure_license_files,
    ensure_processor_configs,
    stamp_release_template,
)

ROOT = Path(__file__).resolve().parents[2]

# Unanchored prefix + anchored suffix regexes so targets match identically in:
#   1. HuggingFace / llmcompressor (`model.language_model.layers.0...`, `model.visual...`)
#   2. vLLM VL (`language_model.model.layers.0...`, `visual...`)
#   3. vLLM CausalLM (`model.layers.0...`)
#   4. vLLM Native MTP (`mtp.layers.0...`)
# This avoids vLLM's CompressedTensorsConfig.apply_vllm_mapper omission on
# transform_config and vLLM's "Linear" class-vs-name regex bug.
TEXT_TARGET_RE = (
    r"re:.*(?<!mtp\.)layers\.\d+\.(self_attn|linear_attn|mlp(?:\.(?:experts\.\d+|shared_expert))?)\."
    r"(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"
)
TEXT_HAD_TARGET_RE = TEXT_TARGET_RE
MTP_TARGET_RE = (
    r"re:.*mtp\.layers\.\d+\.(self_attn|mlp(?:\.(?:experts\.\d+|shared_expert))?)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
MTP_HAD_TARGET_RE = MTP_TARGET_RE
VISION_GROUP_TARGET_RE = (
    r"re:.*visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc1)|merger\.linear_fc[12])$"
)
VISION_FC2_TARGET_RE = r"re:.*visual\.blocks\.\d+\.mlp\.linear_fc2$"

BASE_IGNORE = [
    "lm_head",
    r"re:.*linear_attn\.in_proj_a$",
    r"re:.*linear_attn\.in_proj_b$",
    r"re:.*mlp\.gate$",
    r"re:.*mlp\.shared_expert_gate$",
]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


_HAD_CACHE: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


def _hadamard_rotate_last_dim(
    tensor: torch.Tensor,
    block_size: int,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Right-multiply the last dimension of tensor by block-diagonal H_b^T / sqrt(b)."""
    from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix

    if not _is_power_of_two(block_size):
        raise ValueError(f"Hadamard block_size must be a power of 2, got {block_size}")
    orig_shape = tensor.shape
    k_in = orig_shape[-1]
    if k_in % block_size != 0:
        raise ValueError(f"Last dim={k_in} is not divisible by hadamard_block_size={block_size}")
    cache_key = (block_size, compute_dtype, tensor.device)
    had_t_scaled = _HAD_CACHE.get(cache_key)
    if had_t_scaled is None:
        had = deterministic_hadamard_matrix(block_size, compute_dtype, tensor.device)
        had_t_scaled = (had.T / math.sqrt(block_size)).contiguous()
        _HAD_CACHE[cache_key] = had_t_scaled
    t_cast = tensor.to(compute_dtype).reshape(-1, k_in // block_size, block_size)
    rotated = torch.matmul(t_cast, had_t_scaled)
    return rotated.reshape(orig_shape).to(tensor.dtype)


def _attach_expert_hadamard_hooks(experts: nn.Module, had_in: int, had_down: int) -> list:
    """Attach online Hadamard input transforms before gate_up_proj (had_in) and before down_proj (had_down)."""
    handles = []
    if had_in > 0:
        handles.append(
            experts.register_forward_pre_hook(
                lambda _m, args, bs=had_in: (
                    _hadamard_rotate_last_dim(args[0], bs, compute_dtype=torch.float32),
                    *args[1:],
                )
            )
        )
    if had_down > 0:
        orig_apply_gate = experts._apply_gate
        orig_forward = experts.forward

        def _apply_gate_had(proj_out: torch.Tensor, _orig=orig_apply_gate, _bs=had_down) -> torch.Tensor:
            return _hadamard_rotate_last_dim(_orig(proj_out), _bs, compute_dtype=torch.float32)

        def _eager_forward_had(
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            if getattr(getattr(experts, "config", None), "_experts_implementation", "eager") != "eager":
                return orig_forward(hidden_states, top_k_index, top_k_weights)
            final_hidden_states = torch.zeros_like(hidden_states)
            expert_mask = F.one_hot(top_k_index, num_classes=experts.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                e_idx = expert_idx[0]
                if e_idx == experts.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[e_idx])
                cur = hidden_states[token_idx]
                cur = _apply_gate_had(F.linear(cur, experts.gate_up_proj[e_idx]))
                cur = F.linear(cur, experts.down_proj[e_idx]) * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, cur.to(final_hidden_states.dtype))
            return final_hidden_states

        experts._apply_gate = _apply_gate_had
        experts.forward = _eager_forward_had
    return handles


def _hadamard_rotate_weight(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    """Right-multiply weight [N, K] by block-diagonal H_b^T / sqrt(b) in FP64."""
    return _hadamard_rotate_last_dim(weight, block_size, compute_dtype=torch.float64)


def _pack_w4_tensor(
    weight: torch.Tensor,
    strategy: str,
    group_size: int | None,
) -> dict[str, torch.Tensor]:
    """Quantize a 2D BF16 weight tensor to symmetric INT4 and pack in compressed-tensors format."""
    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    w_f32 = weight.float()
    n_out, k_in = w_f32.shape
    if strategy == "group":
        if group_size is None or k_in % group_size != 0:
            raise ValueError(f"Invalid group_size={group_size} for in_features={k_in}")
        grouped = w_f32.view(n_out, k_in // group_size, group_size)
        scale = (grouped.abs().amax(dim=-1) / 7.0).clamp(min=1e-7).to(torch.bfloat16)
    elif strategy == "channel":
        scale = (w_f32.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-7).to(torch.bfloat16)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy=strategy,
            group_size=group_size if strategy == "group" else None,
        ),
    )
    compressed = PackedQuantizationCompressor.compress(
        {"weight": weight, "weight_scale": scale},
        scheme,
    )
    return {k: v.contiguous() for k, v in compressed.items()}


def _gptq_pack_tensor(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    num_samples: torch.Tensor,
    strategy: str,
    group_size: int | None,
    actorder: str | None,
    dampening_frac: float,
    *,
    cd_passes: int = 2,
) -> dict[str, torch.Tensor]:
    """Quantize a 2D BF16 weight tensor using MSE-init Cholesky GPTQ + H-CD and pack losslessly to INT4."""
    from compressed_tensors.quantization import QuantizationArgs
    from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight

    from warp_quant.rco.options import find_mse_scale, pack_candidate, refine_hessian_cd

    w_f32 = weight.detach().float()
    n_out, k_in = w_f32.shape
    if strategy == "group":
        if group_size is None or k_in % group_size != 0:
            raise ValueError(f"Invalid group_size={group_size} for in_features={k_in}")
        eff_gs = group_size
    elif strategy == "channel":
        eff_gs = k_in
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    scale = find_mse_scale(w_f32, 4, eff_gs)
    qargs = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy=strategy,
        group_size=group_size if strategy == "group" else None,
        actorder=actorder if strategy == "group" else None,
    )
    zp = torch.zeros_like(scale, dtype=torch.int32)
    norm_h = (hessian / num_samples.clamp(min=1).to(hessian.device)).to(torch.float32).contiguous()

    q_w_batch, _losses, used_rtn = quantize_weight(
        weights=w_f32.clone().unsqueeze(0),
        hessians=norm_h.clone().unsqueeze(0),
        scale=scale.unsqueeze(0),
        zero_point=zp.unsqueeze(0),
        global_scale=None,
        quant_args=qargs,
        blocksize=128,
        percdamp=dampening_frac,
    )
    if used_rtn[0].item():
        raise RuntimeError("GPTQ fell back to RTN during component quantization")
    q_w = q_w_batch[0]
    scale_exp = scale.float().repeat_interleave(eff_gs, dim=1)
    codes = torch.clamp(torch.round(q_w.float() / scale_exp), -8, 7).to(torch.int8)
    if cd_passes > 0:
        codes, scale = refine_hessian_cd(
            w_f32,
            norm_h,
            codes,
            scale,
            4,
            eff_gs,
            passes=cd_passes,
            dampening_frac=dampening_frac,
        )
    return {k: v.contiguous().cpu() for k, v in pack_candidate(codes, scale, 4).items()}


@torch.no_grad()
def _gptq_calibrate_mtp(
    model: nn.Module,
    base_dir: Path,
    ds,
    *,
    group_size: int,
    hadamard_block_size: int,
    actorder: str | None,
    dampening_frac: float,
    max_calib_samples: int = 64,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run Cholesky GPTQ (+ optional Hadamard) on the Native MTP linear layers using text calibration samples."""
    import re

    from llmcompressor.modeling.moe.context import moe_calibration_context
    from llmcompressor.modifiers.gptq.helpers import accumulate_hessian, make_empty_hessian

    from warp_quant.data import align_mtp

    device = next(model.parameters()).device
    mtp = _load_mtp_module_from_checkpoint(base_dir, model.config, device, linearize_experts=True)
    embed = model.get_input_embeddings()

    mtp_pat = re.compile(MTP_TARGET_RE.removeprefix("re:"))
    mtp_had_pat = re.compile(MTP_HAD_TARGET_RE.removeprefix("re:"))
    target_mods: dict[str, nn.Linear] = {}
    hessians: dict[str, torch.Tensor] = {}
    num_samples: dict[str, torch.Tensor] = {}
    hooks = []

    for name, mod in mtp.named_modules():
        if not isinstance(mod, nn.Linear) or not mtp_pat.match(f"mtp.{name}"):
            continue
        use_had = hadamard_block_size > 0 and bool(mtp_had_pat.match(f"mtp.{name}"))
        if use_had:
            mod.weight.data = _hadamard_rotate_weight(mod.weight.data, hadamard_block_size)
        target_mods[name] = mod
        hessians[name] = make_empty_hessian(mod, device)
        num_samples[name] = torch.zeros(1, device=device, dtype=torch.int64)

        def _make_hook(m_name: str, m_ref: nn.Linear, m_use_had: bool):
            def _pre_hook(_module, hook_args):
                with torch.amp.autocast("cuda", enabled=False):
                    x = hook_args[0]
                    if m_use_had:
                        x = _hadamard_rotate_last_dim(x, hadamard_block_size, compute_dtype=torch.float32)
                    accumulate_hessian(x, m_ref, hessians[m_name], num_samples[m_name])
                return (x, *hook_args[1:])

            return _pre_hook

        hooks.append(mod.register_forward_pre_hook(_make_hook(name, mod, use_had)))

    n_run = min(len(ds), max_calib_samples)
    try:
        with moe_calibration_context():
            for i in range(n_run):
                row = ds[i]
                ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                ids = ids[:, :2048]
                if ids.shape[1] < 4:
                    continue
                base_out = model.model.language_model(input_ids=ids, use_cache=False)
                h_base, next_ids, pos, _ = align_mtp(ids, base_out.last_hidden_state)
                emb = embed(next_ids)
                _ = mtp(h_base, emb, pos)
    finally:
        for h in hooks:
            h.remove()

    packed_by_prefix: dict[str, dict[str, torch.Tensor]] = {}
    for name, mod in target_mods.items():
        packed_by_prefix[f"mtp.{name}"] = _gptq_pack_tensor(
            mod.weight.data,
            hessians.pop(name),
            num_samples.pop(name),
            strategy="group",
            group_size=group_size,
            actorder=actorder,
            dampening_frac=dampening_frac,
        )
    del mtp
    torch.cuda.empty_cache()
    return packed_by_prefix


@torch.no_grad()
def _gptq_calibrate_vision(
    model: nn.Module,
    processor,
    vision_calib_dir: Path,
    *,
    group_size: int,
    hadamard_block_size: int,
    actorder: str | None,
    dampening_frac: float,
    max_calib_samples: int | None = None,
    max_pixels: int = 1024 * 1024,
    token_exp: float | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run Cholesky GPTQ (+ optional Hadamard) on all Vision linear layers using disjoint calibration images."""
    from llmcompressor.modifiers.gptq.helpers import accumulate_hessian, make_empty_hessian

    from warp_quant.data import device_inputs, encode_image_row

    manifest_path = vision_calib_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"Vision calibration manifest not found at {manifest_path} "
            "(materialize with 'mise run build-vision-data' or pass --no-quantize-vision)"
        )
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("calibration", False):
        raise ValueError(
            f"Refusing to calibrate Vision on non-calibration fixture: {vision_calib_dir} (calibration={manifest.get('calibration')})"
        )
    if token_exp is None:
        token_exp = 0.25 if manifest.get("suite") == "v2" else 0.0

    device = next(model.parameters()).device
    visual = model.model.visual
    target_mods: dict[str, nn.Linear] = {}
    orig_weights: dict[str, torch.Tensor] = {}
    had_blocks: dict[str, int] = {}
    hessians: dict[str, torch.Tensor] = {}
    num_samples: dict[str, torch.Tensor] = {}
    hooks = []

    for name, mod in visual.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        target_mods[name] = mod
        orig_weights[name] = mod.weight.data.cpu().clone()
        k_in = mod.weight.shape[1]
        had_b = min(hadamard_block_size, k_in & -k_in) if hadamard_block_size > 0 else 0
        had_blocks[name] = had_b
        if had_b > 0:
            mod.weight.data = _hadamard_rotate_weight(mod.weight.data, had_b)
        hessians[name] = make_empty_hessian(mod, device)
        num_samples[name] = torch.zeros(1, device=device, dtype=torch.int64)

        def _make_hook(m_name: str, m_ref: nn.Linear, b_size: int):
            def _pre_hook(_module, hook_args):
                with torch.amp.autocast("cuda", enabled=False):
                    x = hook_args[0]
                    if b_size > 0:
                        x = _hadamard_rotate_last_dim(x, b_size, compute_dtype=torch.float32)
                    if token_exp > 0.0:
                        x_3d = x.unsqueeze(0) if x.dim() == 2 else x
                        scale = (1024.0 / max(x_3d.shape[1], 1)) ** token_exp
                        accumulate_hessian(x_3d * scale, m_ref, hessians[m_name], num_samples[m_name])
                    else:
                        accumulate_hessian(x, m_ref, hessians[m_name], num_samples[m_name])
                return (x, *hook_args[1:])

            return _pre_hook

        hooks.append(mod.register_forward_pre_hook(_make_hook(name, mod, had_b)))

    rows_path = vision_calib_dir / "rows.jsonl"
    calib_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if max_calib_samples is not None:
        calib_rows = calib_rows[:max_calib_samples]
    try:
        for row in calib_rows:
            encoded, _, _ = encode_image_row(processor, vision_calib_dir, row, max_pixels)
            inputs = device_inputs(encoded, device)
            _ = visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    finally:
        for h in hooks:
            h.remove()

    packed_by_prefix: dict[str, dict[str, torch.Tensor]] = {}
    for name, mod in target_mods.items():
        k_in = mod.weight.shape[1]
        strat = "group" if k_in % group_size == 0 else "channel"
        g_size = group_size if strat == "group" else None
        packed_by_prefix[f"model.visual.{name}"] = _gptq_pack_tensor(
            mod.weight.data,
            hessians.pop(name),
            num_samples.pop(name),
            strategy=strat,
            group_size=g_size,
            actorder=actorder if strat == "group" else None,
            dampening_frac=dampening_frac,
        )
        mod.weight.data.copy_(orig_weights.pop(name).to(device=device, dtype=mod.weight.dtype))
    torch.cuda.empty_cache()
    return packed_by_prefix


def _load_shard_map(model_dir: Path) -> tuple[dict[str, str], dict | None]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
        return dict(index_data["weight_map"]), index_data
    return {}, None


def _clean_and_augment_shards(
    save_dir: Path,
    *,
    quantize_mtp: bool,
    quantize_vision: bool,
    group_size: int,
    hadamard_block_size: int,
    precomputed_packed: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, int]:
    """Strip extraneous *.v_input.weight keys and optionally quantize MTP / Vision in-place."""
    index_path = save_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
        shard_names = sorted(set(index_data["weight_map"].values()))
    else:
        index_data = None
        shard_names = [p.name for p in sorted(save_dir.glob("*.safetensors"))]

    mtp_prefixes = tuple(
        [f"mtp.layers.0.self_attn.{p}_proj" for p in ("q", "k", "v", "o")]
        + [f"mtp.layers.0.mlp.{p}_proj" for p in ("gate", "up", "down")]
        + [f"mtp.layers.0.mlp.shared_expert.{p}_proj" for p in ("gate", "up", "down")]
    )
    mtp_weight_keys = {f"{p}.weight" for p in mtp_prefixes}

    def is_vision_linear_weight(key: str) -> bool:
        if not (key.startswith("model.visual.") and key.endswith(".weight")):
            return False
        return any(
            part in key
            for part in (
                ".attn.qkv.weight",
                ".attn.proj.weight",
                ".mlp.linear_fc1.weight",
                ".mlp.linear_fc2.weight",
                "merger.linear_fc1.weight",
                "merger.linear_fc2.weight",
            )
        )

    counts = {
        "stripped_transform_weights": 0,
        "mtp_quantized": 0,
        "vision_quantized": 0,
        "vision_fc2_channel": 0,
    }
    new_weight_map: dict[str, str] = {}
    total_bytes = 0

    for shard_name in shard_names:
        shard_path = save_dir / shard_name
        tensors = load_file(str(shard_path))
        updated: dict[str, torch.Tensor] = {}
        modified = False

        for key, tensor in tensors.items():
            # 1. Strip tied HadamardTransform parameter tensor (*.v_input.weight)
            #    so neither transformers nor vLLM AutoWeightsLoader sees an unknown param.
            if key.endswith(".v_input.weight"):
                counts["stripped_transform_weights"] += 1
                modified = True
                continue

            # 2. Optional Native MTP W4A16 (+ Hadamard on LinearBase, unrotated on RoutedExperts)
            if quantize_mtp and key == "mtp.layers.0.mlp.experts.gate_up_proj":
                num_exp, two_inter, _ = tensor.shape
                inter = two_inter // 2
                for e in range(num_exp):
                    for proj_name, w_slice in (
                        ("gate_proj", tensor[e, :inter].contiguous()),
                        ("up_proj", tensor[e, inter:].contiguous()),
                    ):
                        prefix = f"mtp.layers.0.mlp.experts.{e}.{proj_name}"
                        if precomputed_packed and prefix in precomputed_packed:
                            packed = precomputed_packed[prefix]
                        else:
                            w = w_slice
                            if hadamard_block_size > 0:
                                w = _hadamard_rotate_weight(w, hadamard_block_size)
                            packed = _pack_w4_tensor(w, strategy="group", group_size=group_size)
                        for suffix, pt in packed.items():
                            full_key = f"{prefix}.{suffix}"
                            updated[full_key] = pt
                            new_weight_map[full_key] = shard_name
                            total_bytes += pt.numel() * pt.element_size()
                        counts["mtp_quantized"] += 1
                modified = True
                continue

            if quantize_mtp and key == "mtp.layers.0.mlp.experts.down_proj":
                num_exp = tensor.shape[0]
                for e in range(num_exp):
                    prefix = f"mtp.layers.0.mlp.experts.{e}.down_proj"
                    if precomputed_packed and prefix in precomputed_packed:
                        packed = precomputed_packed[prefix]
                    else:
                        w = tensor[e].contiguous()
                        if hadamard_block_size > 0:
                            w = _hadamard_rotate_weight(w, hadamard_block_size)
                        packed = _pack_w4_tensor(w, strategy="group", group_size=group_size)
                    for suffix, pt in packed.items():
                        full_key = f"{prefix}.{suffix}"
                        updated[full_key] = pt
                        new_weight_map[full_key] = shard_name
                        total_bytes += pt.numel() * pt.element_size()
                    counts["mtp_quantized"] += 1
                modified = True
                continue

            if quantize_mtp and key in mtp_weight_keys:
                prefix = key.removesuffix(".weight")
                if precomputed_packed and prefix in precomputed_packed:
                    packed = precomputed_packed[prefix]
                else:
                    w = tensor
                    if hadamard_block_size > 0:
                        w = _hadamard_rotate_weight(w, hadamard_block_size)
                    packed = _pack_w4_tensor(w, strategy="group", group_size=group_size)
                for suffix, pt in packed.items():
                    full_key = f"{prefix}.{suffix}"
                    updated[full_key] = pt
                    new_weight_map[full_key] = shard_name
                    total_bytes += pt.numel() * pt.element_size()
                counts["mtp_quantized"] += 1
                modified = True
                continue

            # 3. Optional Vision W4A16 (+ Hadamard)
            if quantize_vision and is_vision_linear_weight(key):
                prefix = key.removesuffix(".weight")
                w = tensor
                k_in = w.shape[1]
                if k_in % group_size != 0:
                    counts["vision_fc2_channel"] += 1
                if precomputed_packed and prefix in precomputed_packed:
                    packed = precomputed_packed[prefix]
                else:
                    if k_in % group_size != 0:
                        strat = "channel"
                        g_size = None
                    else:
                        strat = "group"
                        g_size = group_size
                    had_b = min(hadamard_block_size, k_in & -k_in) if hadamard_block_size > 0 else 0
                    if had_b > 0:
                        w = _hadamard_rotate_weight(w, had_b)
                    packed = _pack_w4_tensor(w, strategy=strat, group_size=g_size)
                for suffix, pt in packed.items():
                    full_key = f"{prefix}.{suffix}"
                    updated[full_key] = pt
                    new_weight_map[full_key] = shard_name
                    total_bytes += pt.numel() * pt.element_size()
                counts["vision_quantized"] += 1
                modified = True
                continue

            updated[key] = tensor
            new_weight_map[key] = shard_name
            total_bytes += tensor.numel() * tensor.element_size()

        if modified:
            save_file(updated, str(shard_path))

    if index_data is not None:
        index_data["weight_map"] = dict(sorted(new_weight_map.items()))
        if "metadata" in index_data and isinstance(index_data["metadata"], dict):
            index_data["metadata"]["total_size"] = total_bytes
        index_path.write_text(json.dumps(index_data, indent=2, sort_keys=True) + "\n")

    return counts


def _assert_packed_alignment(save_dir: Path) -> None:
    """Verify every packed weight satisfies Marlin alignment (N%64==0 for text/MTP, N%16==0 for Vision)."""
    bad = []
    for shard in sorted(save_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            keys = set(f.keys())
            for k in keys:
                if not k.endswith("weight_shape"):
                    continue
                packed = k[: -len("weight_shape")] + "weight_packed"
                if packed not in keys:
                    continue
                logical = f.get_tensor(k)
                n, k_in = int(logical[0]), int(logical[1])
                if ".ple.ple_embedding.ngram_embedding." in k:
                    if n <= 0 or k_in % 2 != 0:
                        bad.append(f"{k[: -len('_shape')]} shape=({n}, {k_in})")
                    continue
                align = 16 if k.startswith("model.visual.") else 64
                if n % align != 0 or k_in % 16 != 0:
                    bad.append(f"{k[: -len('_shape')]} shape=({n}, {k_in})")
    if bad:
        raise AssertionError(f"packed tensors violating Marlin alignment: {bad}")
    print(f"_assert_packed_alignment: OK ({save_dir})")


def _update_config_for_components(
    save_dir: Path,
    *,
    group_size: int,
    actorder: str | None,
    hadamard_block_size: int,
    quantize_mtp: bool,
    quantize_vision: bool,
    vision_fc2_channel: bool,
) -> None:
    """Ensure config.json has clean regex targets for Text, MTP, and Vision in both quantization_config and transform_config."""
    cfg_path = save_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    qcfg = cfg["quantization_config"]

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

    qcfg["config_groups"] = config_groups
    qcfg["ignore"] = list(BASE_IGNORE)

    if hadamard_block_size > 0:
        main_had_targets = [TEXT_HAD_TARGET_RE]
        if quantize_mtp:
            main_had_targets.append(MTP_HAD_TARGET_RE)
        if quantize_vision:
            # Check if vision_fc2 needs a smaller Hadamard block size (e.g. K=4304 -> max power-of-2 divisor is 16)
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
                        "ignore": list(BASE_IGNORE),
                    },
                    {
                        "targets": main_had_targets,
                        "location": "input",
                        "inverse": False,
                        "ignore": list(BASE_IGNORE),
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
        qcfg["transform_config"] = {"config_groups": t_groups}
    else:
        qcfg["transform_config"] = {}

    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def run_quantize(args: argparse.Namespace) -> None:
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.quip import QuIPModifier

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

    cls_name = model_class(hub_id).__name__
    if cls_name.startswith("Qwen4Exp") or cls_name == "Qwen3_5MoeForConditionalGeneration":
        from warp_quant.streaming_moe import run_quantize_qwen4_exp

        return run_quantize_qwen4_exp(args)

    ensure_processor_configs(
        hub_id,
        info.get("processor_hub_fallback"),
        info.get("processor_hub_fallback_revision"),
    )
    ensure_license_files(None, hub_id, info.get("license_files"), dry_run=True)

    recipe_defaults = quant_recipe(info)
    calib = args.calib or recipe_defaults["calib"]
    num_samples = args.num_samples if args.num_samples is not None else recipe_defaults["num_samples"]
    max_seq_length = (
        args.max_seq_length if args.max_seq_length is not None else recipe_defaults["max_seq_length"]
    )
    window = args.window or recipe_defaults["window"]
    actorder = None if args.actorder == "none" else args.actorder

    log_event(
        "ROT_QUANT_START",
        model=args.model,
        snapshot=str(hub_id),
        output_dir=str(output_dir),
        group_size=args.group_size,
        hadamard_block_size=args.hadamard_block_size,
        actorder=actorder,
        quantize_mtp=args.quantize_mtp,
        quantize_vision=args.quantize_vision,
        calib=calib,
        num_samples=num_samples,
        max_seq_length=max_seq_length,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    model_cls = model_class(hub_id)
    if model_cls.__name__ == "Qwen3_5MoeForConditionalGeneration":
        from llmcompressor.modeling.moe.linearize import load_quantizable_moe

        with load_quantizable_moe(model_cls):
            model = model_cls.from_pretrained(hub_id, dtype="auto", device_map={"": 0})
    else:
        model = model_cls.from_pretrained(hub_id, dtype="auto", device_map={"": 0})
    processor = AutoProcessor.from_pretrained(hub_id)

    scheme = QuantizationScheme(
        targets=[TEXT_TARGET_RE],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy="group",
            group_size=args.group_size,
            actorder=actorder,
        ),
    )

    class _HCDGPTQModifier(GPTQModifier):
        """GPTQModifier subclass adding MSE scale initialization + 2-pass Gauss-Seidel H-CD."""

        def compress_module_list(self, module_list):
            import contextlib

            from compressed_tensors.offload import disable_offloading
            from compressed_tensors.utils import getattr_chain
            from llmcompressor.modifiers.gptq.base import (
                CompressionLogger,
                assign_batches,
                prepare_batch,
                update_batch_qparams,
            )
            from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight

            from warp_quant.rco.options import decode_candidate, find_mse_scale, refine_hessian_cd_3d

            for batch in assign_batches(module_list, self.batched_quantization, self.block_size):
                quant_args = getattr_chain(batch[0], "quantization_scheme.weights")
                batch_qparams = [module.weight_observer.get_qparams() for module in batch]
                names = [self._module_names[module] for module in batch]

                with (
                    torch.no_grad(),
                    disable_offloading(),
                    contextlib.ExitStack() as ctx_stack,
                ):
                    weights, hessians, scales, zero_points, global_scales = prepare_batch(
                        batch, batch_qparams, self._hessians, self._num_samples
                    )
                    bsz, n_out_b, k_in_b = weights.shape
                    eff_gs = int(quant_args.group_size) if quant_args.group_size else k_in_b
                    num_bits = int(quant_args.num_bits)
                    if quant_args.symmetric and global_scales is None:
                        mse_s = find_mse_scale(
                            weights.float().reshape(bsz * n_out_b, k_in_b), num_bits, eff_gs
                        ).view_as(scales)
                        scales = mse_s.to(dtype=scales.dtype)

                    orig_weights = weights.float().clone()
                    orig_hessians = hessians.float().clone()
                    comp_loggers = [ctx_stack.enter_context(CompressionLogger(m)) for m in batch]
                    quantized, losses, used_rtn_fallback = quantize_weight(
                        weights=weights,
                        hessians=hessians,
                        scale=scales,
                        zero_point=zero_points,
                        global_scale=global_scales,
                        quant_args=quant_args,
                        blocksize=self.block_size,
                        percdamp=self.dampening_frac,
                    )
                    if quant_args.symmetric and global_scales is None and not used_rtn_fallback.any().item():
                        qmin = -(1 << (num_bits - 1))
                        qmax = (1 << (num_bits - 1)) - 1
                        s_exp = scales.float().repeat_interleave(eff_gs, dim=-1)
                        codes_3d = torch.clamp(torch.round(quantized.float() / s_exp), qmin, qmax).to(
                            torch.int8
                        )
                        codes_3d, new_scales = refine_hessian_cd_3d(
                            orig_weights,
                            orig_hessians,
                            codes_3d,
                            scales,
                            num_bits,
                            eff_gs,
                            passes=2,
                            dampening_frac=self.dampening_frac,
                        )
                        scales = new_scales.to(dtype=scales.dtype)
                        for b_idx in range(bsz):
                            quantized[b_idx] = decode_candidate(
                                codes_3d[b_idx], new_scales[b_idx], eff_gs, dtype=quantized.dtype
                            )
                    del orig_weights, orig_hessians

                    for index, comp_logger in enumerate(comp_loggers):
                        comp_logger.set_results(name="GPTQ", loss=losses[index].item())
                        if used_rtn_fallback[index].item():
                            self._rtn_fallback_module_names.append(names[index])

                    ctx_stack.close()
                    update_batch_qparams(batch, quantized, scales, zero_points, global_scales, quant_args)
                    self._num_compressed_modules += len(batch)

    gptq_mod = _HCDGPTQModifier(
        config_groups={"group_0": scheme},
        actorder=actorder,
        ignore=list(BASE_IGNORE),
        dampening_frac=args.dampening_frac,
        block_size=128,
    )

    if args.hadamard_block_size > 0:
        quip_mod = QuIPModifier(
            rotations=["v"],
            transform_type="hadamard",
            transform_block_size=args.hadamard_block_size,
            targets=[TEXT_HAD_TARGET_RE],
            ignore=list(BASE_IGNORE),
        )
        recipe = [quip_mod, gptq_mod]
    else:
        recipe = [gptq_mod]

    ds = get_calibration_dataset(processor, calib, num_samples, max_seq_length, window=window, seed=args.seed)

    precomputed_packed: dict[str, dict[str, torch.Tensor]] = {}
    if args.component_quant_mode == "gptq":
        if args.quantize_mtp:
            mtp_packed = _gptq_calibrate_mtp(
                model,
                Path(hub_id),
                ds,
                group_size=args.group_size,
                hadamard_block_size=args.hadamard_block_size,
                actorder=actorder,
                dampening_frac=args.dampening_frac,
                max_calib_samples=args.component_calib_samples,
            )
            precomputed_packed.update(mtp_packed)
        if args.quantize_vision:
            if not args.vision_calib_dir:
                raise ValueError(
                    "--vision-calib-dir is required when --quantize-vision is used with --component-quant-mode gptq"
                )
            vis_calib_path = Path(args.vision_calib_dir)
            if not vis_calib_path.is_absolute():
                vis_calib_path = ROOT / vis_calib_path
            vis_packed = _gptq_calibrate_vision(
                model,
                processor,
                vis_calib_path,
                group_size=args.group_size,
                hadamard_block_size=args.hadamard_block_size,
                actorder=actorder,
                dampening_frac=args.dampening_frac,
                max_calib_samples=args.vision_calib_samples,
            )
            precomputed_packed.update(vis_packed)

    _rtn_seen: dict[int, list[str]] = {}
    _orig_finalize = GPTQModifier.on_finalize

    def _capture_finalize(self, *a, **k):
        _rtn_seen[id(self)] = list(self._rtn_fallback_module_names)
        return _orig_finalize(self, *a, **k)

    GPTQModifier.on_finalize = _capture_finalize
    t0 = time.time()
    try:
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=max_seq_length,
            num_calibration_samples=num_samples,
            sequential_offload_device=args.sequential_offload_device,
            shuffle_calibration_samples=True,
        )
    finally:
        GPTQModifier.on_finalize = _orig_finalize
    quant_Wall_s = round(time.time() - t0, 2)

    rtn = [n for names in _rtn_seen.values() for n in names]
    if rtn:
        raise RuntimeError(f"GPTQ fell back to RTN on {len(rtn)} modules: {rtn[:5]}")

    model.save_pretrained(str(output_dir), save_compressed=True)
    processor.save_pretrained(str(output_dir))

    # Clean *.v_input.weight and optionally quantize MTP / Vision in-place
    shard_counts = _clean_and_augment_shards(
        output_dir,
        quantize_mtp=args.quantize_mtp,
        quantize_vision=args.quantize_vision,
        group_size=args.group_size,
        hadamard_block_size=args.hadamard_block_size,
        precomputed_packed=precomputed_packed,
    )
    _update_config_for_components(
        output_dir,
        group_size=args.group_size,
        actorder=actorder,
        hadamard_block_size=args.hadamard_block_size,
        quantize_mtp=args.quantize_mtp,
        quantize_vision=args.quantize_vision,
        vision_fc2_channel=shard_counts["vision_fc2_channel"] > 0,
    )

    _assert_packed_alignment(output_dir)

    # Stamp serving generation_config + release chat_template
    gc = build_generation_config(processor, info, hub_id, {})
    (output_dir / "generation_config.json").write_text(json.dumps(gc, indent=2) + "\n")
    tpl_sha = stamp_release_template(str(output_dir))
    check_template_renders(str(output_dir))
    ensure_license_files(str(output_dir), hub_id, info.get("license_files"))

    receipt = {
        "model": args.model,
        "base_snapshot": Path(hub_id).name,
        "group_size": args.group_size,
        "hadamard_block_size": args.hadamard_block_size,
        "actorder": actorder,
        "dampening_frac": args.dampening_frac,
        "quantize_mtp": args.quantize_mtp,
        "quantize_vision": args.quantize_vision,
        "component_quant_mode": args.component_quant_mode,
        "shard_counts": shard_counts,
        "calib": calib,
        "num_samples": num_samples,
        "max_seq_length": max_seq_length,
        "quant_wall_s": quant_Wall_s,
        "template_sha": tpl_sha,
    }
    (output_dir / "rot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    log_event("ROT_QUANT_DONE", **receipt)
    run_done(output_dir=str(output_dir))
    print(json.dumps(receipt, indent=2))


def _load_moe_quant_model_direct(path: Path, device: str) -> nn.Module:
    """Directly construct 3D Qwen3_5MoeForConditionalGeneration on device and stream-decompress shards."""
    import re
    from contextlib import ExitStack

    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor.utils.dev import skip_weights_initialize
    from transformers import Qwen3_5MoeConfig

    raw_cfg = json.loads((path / "config.json").read_text())
    qcfg = raw_cfg["quantization_config"]
    g_size = qcfg["config_groups"]["group_0"]["weights"]["group_size"]
    clean_cfg_dict = {k: v for k, v in raw_cfg.items() if k != "quantization_config"}
    cfg = Qwen3_5MoeConfig.from_dict(clean_cfg_dict)
    cls = model_class(str(path))

    with torch.device(device), skip_weights_initialize():
        model = cls._from_config(cfg, dtype=torch.bfloat16).eval()

    group_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="group", group_size=g_size),
    )
    channel_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="channel", group_size=None),
    )

    shards = [s for s in sorted(path.glob("*.safetensors")) if s.name != "model_mtp.safetensors"]
    exp_pat = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$"
    )
    loaded_params: set[str] = set()

    with ExitStack() as stack:
        key_to_f = {}
        for sp in shards:
            f = stack.enter_context(safe_open(str(sp), framework="pt", device="cpu"))
            for k in f.keys():
                key_to_f[k] = f

        packed_prefixes = sorted(
            k.removesuffix(".weight_packed") for k in key_to_f if k.endswith(".weight_packed")
        )
        packed_prefix_set = set(packed_prefixes)

        # 1. Batch-decompress MoE routed experts per layer directly into 3D gate_up_proj / down_proj
        num_layers = cfg.text_config.num_hidden_layers
        for l_idx in range(num_layers):
            experts = model.model.language_model.layers[l_idx].mlp.experts
            num_exp = experts.num_experts
            inter = experts.intermediate_dim
            hidden = experts.hidden_dim
            first_key = f"model.language_model.layers.{l_idx}.mlp.experts.0.gate_proj"
            if first_key not in packed_prefix_set:
                raise RuntimeError(f"Missing packed expert weights for layer {l_idx}: {first_key}")
            for proj_name, out_dim, in_dim in (
                ("gate_proj", inter, hidden),
                ("up_proj", inter, hidden),
                ("down_proj", hidden, inter),
            ):
                wps = [
                    key_to_f[
                        f"model.language_model.layers.{l_idx}.mlp.experts.{e}.{proj_name}.weight_packed"
                    ].get_tensor(
                        f"model.language_model.layers.{l_idx}.mlp.experts.{e}.{proj_name}.weight_packed"
                    )
                    for e in range(num_exp)
                ]
                wss = [
                    key_to_f[
                        f"model.language_model.layers.{l_idx}.mlp.experts.{e}.{proj_name}.weight_scale"
                    ].get_tensor(
                        f"model.language_model.layers.{l_idx}.mlp.experts.{e}.{proj_name}.weight_scale"
                    )
                    for e in range(num_exp)
                ]
                deq = PackedQuantizationCompressor.decompress(
                    {
                        "weight_packed": torch.cat(wps, dim=0).to(device),
                        "weight_scale": torch.cat(wss, dim=0).to(device),
                        "weight_shape": torch.tensor([num_exp * out_dim, in_dim]),
                    },
                    group_scheme,
                )["weight"].view(num_exp, out_dim, in_dim)
                if proj_name == "gate_proj":
                    experts.gate_up_proj.data[:, :inter].copy_(deq)
                elif proj_name == "up_proj":
                    experts.gate_up_proj.data[:, inter:].copy_(deq)
                    loaded_params.add(f"model.language_model.layers.{l_idx}.mlp.experts.gate_up_proj")
                else:
                    experts.down_proj.data.copy_(deq)
                    loaded_params.add(f"model.language_model.layers.{l_idx}.mlp.experts.down_proj")
                del wps, wss, deq

        # 2. Decompress all non-expert packed 2D linears (attention, GDN, vision)
        for prefix in packed_prefixes:
            if exp_pat.match(prefix) or prefix.startswith("mtp."):
                continue
            wp = key_to_f[f"{prefix}.weight_packed"].get_tensor(f"{prefix}.weight_packed").to(device)
            ws = key_to_f[f"{prefix}.weight_scale"].get_tensor(f"{prefix}.weight_scale").to(device)
            wsh = key_to_f[f"{prefix}.weight_shape"].get_tensor(f"{prefix}.weight_shape")
            sch = channel_scheme if ws.shape[1] == 1 else group_scheme
            deq = PackedQuantizationCompressor.decompress(
                {"weight_packed": wp, "weight_scale": ws, "weight_shape": wsh},
                sch,
            )["weight"]
            model.get_submodule(prefix).weight.data.copy_(deq)
            loaded_params.add(f"{prefix}.weight")
            del wp, ws, deq

        # 3. Copy all remaining dense parameters/buffers
        param_map = dict(model.named_parameters())
        buf_map = dict(model.named_buffers())
        for k, f in key_to_f.items():
            if k.startswith("mtp.") or k.endswith((".weight_packed", ".weight_scale", ".weight_shape")):
                continue
            t = f.get_tensor(k)
            if k in param_map:
                param_map[k].data.copy_(t)
                loaded_params.add(k)
            elif k in buf_map:
                buf_map[k].data.copy_(t)
            else:
                raise RuntimeError(f"Unexpected tensor key in MoE checkpoint: {k}")

    if getattr(cfg.text_config, "tie_word_embeddings", False):
        model.tie_weights()
        if "model.language_model.embed_tokens.weight" in loaded_params:
            loaded_params.add("lm_head.weight")
    missing_params = sorted(set(param_map) - loaded_params)
    if missing_params:
        raise RuntimeError(f"Missing parameters in MoE checkpoint: {missing_params[:10]}")
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model


def load_quant_model_with_transforms(path: str | Path, device: str = "cuda:0") -> nn.Module:
    """Load a compressed-tensors checkpoint, decompress INT4 weights, and attach online transforms."""
    from compressed_tensors.transform import TransformConfig, TransformScheme, apply_transform_config

    path = Path(path)
    raw_cfg = json.loads((path / "config.json").read_text())
    cls = model_class(str(path))
    if cls.__name__ == "Qwen3_5MoeForConditionalGeneration":
        model = _load_moe_quant_model_direct(path, device)
    else:
        model = cls.from_pretrained(str(path), dtype="auto", device_map={"": device})
        model.eval()
        if hasattr(model, "hf_quantizer") and hasattr(model.hf_quantizer, "compressor"):
            model.hf_quantizer.compressor.decompress_model(model)

    _apply_rco_code_offsets(model, path)

    tcfg_dict = raw_cfg.get("quantization_config", {}).get("transform_config")
    if tcfg_dict and tcfg_dict.get("config_groups"):
        full_tcfg = TransformConfig.model_validate(tcfg_dict)
        online_groups = {}
        for name, scheme in full_tcfg.config_groups.items():
            online_apply = [a for a in scheme.apply if a.is_online()]
            if online_apply:
                online_groups[name] = TransformScheme(
                    type=scheme.type,
                    head_dim=scheme.head_dim,
                    precision=scheme.precision,
                    randomize=scheme.randomize,
                    requires_grad=False,
                    apply=online_apply,
                )
        if online_groups:
            apply_transform_config(model, TransformConfig(config_groups=online_groups))
            if cls.__name__ == "Qwen3_5MoeForConditionalGeneration" and "v" in online_groups:
                had_dim = online_groups["v"].head_dim
                v_targets = " ".join(t for a in online_groups["v"].apply for t in a.targets)
                if "experts" in v_targets:
                    for layer in model.model.language_model.layers:
                        exp = layer.mlp.experts
                        _attach_expert_hadamard_hooks(
                            exp,
                            min(had_dim, exp.hidden_dim & -exp.hidden_dim),
                            min(had_dim, exp.intermediate_dim & -exp.intermediate_dim),
                        )
    return model


def _apply_rco_code_offsets(model: nn.Module, path: Path) -> None:
    """Apply half-integer code offsets recorded in rco_manifest.json after CT weight decompression."""
    rco_manifest_path = path / "rco_manifest.json"
    if not rco_manifest_path.is_file():
        return
    rco_meta = json.loads(rco_manifest_path.read_text())
    if not isinstance(rco_meta, dict):
        raise ValueError(f"Expected dict in {rco_manifest_path}")
    code_offsets = rco_meta.get("code_offsets")
    if not code_offsets:
        return
    if not isinstance(code_offsets, dict):
        raise ValueError(f"Expected dict for code_offsets in {rco_manifest_path}")

    shard_map, _ = _load_shard_map(path)
    shard_by_name = {sp.name: sp for sp in sorted(path.glob("*.safetensors"))}

    for prefix, spec in code_offsets.items():
        if not isinstance(spec, dict) or "code_offset" not in spec or "group_size" not in spec:
            raise ValueError(f"Malformed code_offset spec for '{prefix}' in {rco_manifest_path}")
        try:
            c_off = float(spec["code_offset"])
            gs = int(spec["group_size"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric code_offset spec for '{prefix}' in {rco_manifest_path}") from exc
        if not math.isfinite(c_off) or not (-1.0 <= c_off <= 1.0):
            raise ValueError(f"Invalid code_offset={c_off} for '{prefix}' (must be in [-1.0, 1.0])")
        if gs <= 0 or (gs & (gs - 1)) != 0:
            raise ValueError(f"Invalid group_size={gs} for '{prefix}' (must be a positive power of 2)")
        if c_off == 0.0:
            continue

        try:
            mod = model.get_submodule(prefix)
        except AttributeError as exc:
            raise ValueError(f"Unknown module prefix '{prefix}' in {rco_manifest_path}") from exc
        weight = getattr(mod, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"Module '{prefix}' does not have a 2D weight tensor")
        n_out, k_in = weight.shape
        if k_in % gs != 0:
            raise ValueError(f"Weight shape {(n_out, k_in)} for '{prefix}' not divisible by group_size={gs}")

        w_scale = getattr(mod, "weight_scale", None)
        if w_scale is None:
            scale_key = f"{prefix}.weight_scale"
            mapped_name = shard_map.get(scale_key)
            if mapped_name is not None:
                if mapped_name not in shard_by_name:
                    raise ValueError(
                        f"Shard '{mapped_name}' for '{scale_key}' is not a local safetensors file in {path}"
                    )
                with safe_open(str(shard_by_name[mapped_name]), framework="pt", device="cpu") as sf:
                    if scale_key in sf.keys():
                        w_scale = sf.get_tensor(scale_key)
            else:
                for sp in shard_by_name.values():
                    with safe_open(str(sp), framework="pt", device="cpu") as sf:
                        if scale_key in sf.keys():
                            w_scale = sf.get_tensor(scale_key)
                            break
        if w_scale is None:
            raise ValueError(f"Missing '{prefix}.weight_scale' for code_offset fixup in {path}")
        expected_shape = (n_out, k_in // gs)
        if tuple(w_scale.shape) != expected_shape:
            raise ValueError(
                f"Scale shape {tuple(w_scale.shape)} != expected {expected_shape} for '{prefix}'"
            )

        s_dev = w_scale.to(device=mod.weight.device, dtype=torch.float32)
        w_adj = mod.weight.data.float() + (
            c_off * s_dev.unsqueeze(-1).expand(n_out, k_in // gs, gs).reshape(n_out, k_in)
        )
        mod.weight.data.copy_(w_adj.to(mod.weight.dtype))


def _build_mtp_module(text_config) -> nn.Module:
    from warp_quant.data import FixedInputMTP

    if getattr(text_config, "num_experts", 0) > 0:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5RMSNorm,
            Qwen3_5TextRotaryEmbedding,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

        cfg = copy.deepcopy(text_config)
        cfg.num_hidden_layers = 1
        cfg.layer_types = ["full_attention"]
        cfg._attn_implementation = "eager"
        cfg.dtype = torch.bfloat16
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        mtp = FixedInputMTP.__new__(FixedInputMTP)
        nn.Module.__init__(mtp)
        mtp.config = cfg
        mtp.fc = nn.Linear(2 * h, h, bias=False)
        mtp.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=eps)
        mtp.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=eps)
        mtp.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(cfg, layer_idx=0)])
        mtp.norm = Qwen3_5RMSNorm(h, eps=eps)
        mtp.rotary = Qwen3_5TextRotaryEmbedding(cfg)
        return mtp
    return FixedInputMTP(text_config)


def _load_mtp_module_from_checkpoint(
    ckpt_dir: Path,
    base_config,
    device: torch.device,
    *,
    linearize_experts: bool = False,
) -> nn.Module:
    """Instantiate FixedInputMTP and load BF16 or W4A16 (+ Hadamard) MTP weights from ckpt_dir."""
    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from compressed_tensors.transform import (
        TransformArgs,
        TransformConfig,
        TransformScheme,
        apply_transform_config,
    )

    raw_cfg = json.loads((ckpt_dir / "config.json").read_text())
    qcfg = raw_cfg.get("quantization_config", {})
    is_moe_mtp = getattr(base_config.text_config, "num_experts", 0) > 0
    mtp = _build_mtp_module(base_config.text_config).to(device=device, dtype=torch.bfloat16).eval()

    # Collect all mtp.* tensors from checkpoint shards
    shard_map, _ = _load_shard_map(ckpt_dir)
    if shard_map:
        mtp_shards = sorted({ckpt_dir / s for k, s in shard_map.items() if k.startswith("mtp.")})
    else:
        mtp_shards = sorted(ckpt_dir.glob("*.safetensors"))

    raw_mtp: dict[str, torch.Tensor] = {}
    for sp in mtp_shards:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    raw_mtp[k] = f.get_tensor(k)

    # Decompress any packed MTP linear weights
    state_dict: dict[str, torch.Tensor] = {}
    packed_prefixes = [k.removesuffix(".weight_packed") for k in raw_mtp if k.endswith(".weight_packed")]
    for k, v in raw_mtp.items():
        if not any(k.startswith(p + ".") for p in packed_prefixes):
            state_dict[k.removeprefix("mtp.")] = v.to(torch.bfloat16)

    if packed_prefixes:
        g_size = qcfg["config_groups"]["group_0"]["weights"]["group_size"]
        scheme = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True, strategy="group", group_size=g_size
            ),
        )
        for prefix in packed_prefixes:
            deq = PackedQuantizationCompressor.decompress(
                {
                    "weight_packed": raw_mtp[f"{prefix}.weight_packed"],
                    "weight_scale": raw_mtp[f"{prefix}.weight_scale"],
                    "weight_shape": raw_mtp[f"{prefix}.weight_shape"],
                },
                scheme,
            )["weight"]
            state_dict[f"{prefix.removeprefix('mtp.')}.weight"] = deq.to(torch.bfloat16)

    if is_moe_mtp:
        from llmcompressor.modeling.moe.linearize import linearize_moe

        if packed_prefixes:
            if linearize_experts:
                linearize_moe(mtp)
                mtp.load_state_dict(state_dict, strict=True)
            else:
                num_exp = mtp.layers[0].mlp.experts.num_experts
                gates = [state_dict.pop(f"layers.0.mlp.experts.{e}.gate_proj.weight") for e in range(num_exp)]
                ups = [state_dict.pop(f"layers.0.mlp.experts.{e}.up_proj.weight") for e in range(num_exp)]
                downs = [state_dict.pop(f"layers.0.mlp.experts.{e}.down_proj.weight") for e in range(num_exp)]
                state_dict["layers.0.mlp.experts.gate_up_proj"] = torch.cat(
                    [torch.stack(gates, dim=0), torch.stack(ups, dim=0)], dim=1
                )
                state_dict["layers.0.mlp.experts.down_proj"] = torch.stack(downs, dim=0)
                mtp.load_state_dict(state_dict, strict=True)
        else:
            mtp.load_state_dict(state_dict, strict=True)
            if linearize_experts:
                linearize_moe(mtp)
    else:
        mtp.load_state_dict(state_dict, strict=True)
    mtp = mtp.to(device=device, dtype=torch.bfloat16).eval()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # If MTP weights were rotated with Hadamard, attach online input rotation to mtp.layers.0.*
    tcfg_dict = qcfg.get("transform_config", {})
    if packed_prefixes and tcfg_dict and tcfg_dict.get("config_groups"):
        v_scheme = next(iter(tcfg_dict["config_groups"].values()))
        had_dim = v_scheme["head_dim"]
        mtp_online_tcfg = TransformConfig(
            config_groups={
                "v": TransformScheme(
                    type="hadamard",
                    head_dim=had_dim,
                    apply=[
                        TransformArgs(
                            targets=[
                                r"re:^layers\.0\.(self_attn|mlp(?:\.(?:experts\.\d+|shared_expert))?)\."
                                r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
                            ],
                            location="input",
                            inverse=False,
                            ignore=list(BASE_IGNORE),
                        )
                    ],
                )
            }
        )
        apply_transform_config(mtp, mtp_online_tcfg)
        if is_moe_mtp and not linearize_experts:
            v_targets = " ".join(t for a in v_scheme.get("apply", []) for t in a.get("targets", []))
            if "experts" in v_targets:
                exp = mtp.layers[0].mlp.experts
                _attach_expert_hadamard_hooks(
                    exp,
                    min(had_dim, exp.hidden_dim & -exp.hidden_dim),
                    min(had_dim, exp.intermediate_dim & -exp.intermediate_dim),
                )

    return mtp


@torch.no_grad()
def evaluate_mtp_fidelity(
    base_model: nn.Module,
    quant_model: nn.Module,
    processor,
    base_dir: Path,
    quant_dir: Path,
    eval_path: Path,
    per_domain: int = 2,
    max_tokens: int = 512,
) -> dict[str, dict]:
    """Evaluate Native MTP teacher-forced KL & top-1 agreement (both isolated and end-to-end)."""
    from warp_quant.data import align_mtp, compare_logits, select_text_rows, summarize

    device = next(base_model.parameters()).device
    base_mtp = _load_mtp_module_from_checkpoint(base_dir, base_model.config, device)
    quant_mtp = _load_mtp_module_from_checkpoint(quant_dir, base_model.config, device)
    embed = base_model.get_input_embeddings()
    lm_head = base_model.get_output_embeddings()

    selected = select_text_rows(str(eval_path), per_domain)
    isolated_rows = []
    e2e_rows = []
    try:
        for _, _domain, messages in selected:
            text = processor.apply_chat_template(messages, tokenize=False)
            ids = processor.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )["input_ids"].to(device)
            if ids.shape[1] < 4:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                base_out = base_model.model.language_model(input_ids=ids, use_cache=False)
                quant_out = quant_model.model.language_model(input_ids=ids, use_cache=False)
                h_base, next_ids, pos, labels = align_mtp(ids, base_out.last_hidden_state)
                h_quant, _, _, _ = align_mtp(ids, quant_out.last_hidden_state)
                emb = embed(next_ids)
                mtp_h_base = base_mtp(h_base, emb, pos)
                mtp_h_iso = quant_mtp(h_base, emb, pos)
                mtp_h_e2e = quant_mtp(h_quant, emb, pos)
                ref_logits = lm_head(mtp_h_base)
                iso_logits = lm_head(mtp_h_iso)
                e2e_logits = lm_head(mtp_h_e2e)
            isolated_rows.append(compare_logits(ref_logits, iso_logits, labels))
            e2e_rows.append(compare_logits(ref_logits, e2e_logits, labels))
    finally:
        del base_mtp, quant_mtp
        torch.cuda.empty_cache()
    return {
        "isolated": summarize(isolated_rows),
        "e2e": summarize(e2e_rows),
    }


@torch.no_grad()
def evaluate_vision_fidelity(
    base_model: nn.Module,
    quant_model: nn.Module,
    vision_data_dir: Path,
    max_pixels: int = 1024 * 1024,
) -> dict[str, dict]:
    """Evaluate ViT+Merger encoder error plus isolated and end-to-end VQA answer logit KL."""
    from warp_quant.data import compare_logits, device_inputs, encode_image_row, feature_metrics, summarize

    manifest_path = vision_data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("calibration", False):
            raise ValueError(
                f"Refusing to evaluate Vision on calibration fixture: {vision_data_dir} (calibration=True)"
            )

    device = next(base_model.parameters()).device
    processor = AutoProcessor.from_pretrained(str(base_model.config._name_or_path), local_files_only=True)
    rows_path = vision_data_dir / "rows.jsonl"
    fixture_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    encoder_rows = []
    isolated_rows = []
    e2e_rows = []
    orig_base_visual = base_model.model.visual
    try:
        for row in fixture_rows:
            encoded, keep, labels = encode_image_row(processor, vision_data_dir, row, max_pixels)
            inputs = device_inputs(encoded, device)
            keep_dev, labels_dev = keep.to(device), labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # 1. Direct ViT + Merger encoder output comparison
                vis_ref = orig_base_visual(
                    inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]
                ).pooler_output
                vis_cand = quant_model.model.visual(
                    inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]
                ).pooler_output
                f_m = feature_metrics(vis_ref, vis_cand)
                cos = F.cosine_similarity(vis_ref.float().flatten(), vis_cand.float().flatten(), dim=0).item()
                f_m["cos_sim"] = cos
                encoder_rows.append(f_m)

                # 2. BF16 reference VQA answer logits
                base_model.model.visual = orig_base_visual
                ref_logits = base_model(**inputs, logits_to_keep=keep_dev, use_cache=False).logits

                # 3. Isolated Vision-only VQA answer logits (quantized visual + BF16 language model)
                base_model.model.visual = quant_model.model.visual
                iso_logits = base_model(**inputs, logits_to_keep=keep_dev, use_cache=False).logits

                # 4. End-to-end VQA answer logits (quantized visual + quantized language model)
                e2e_logits = quant_model(**inputs, logits_to_keep=keep_dev, use_cache=False).logits

            isolated_rows.append(
                {"domain": row["domain"], **compare_logits(ref_logits, iso_logits, labels_dev)}
            )
            e2e_rows.append({"domain": row["domain"], **compare_logits(ref_logits, e2e_logits, labels_dev)})
    finally:
        base_model.model.visual = orig_base_visual

    sq_err = sum(r["squared_error"] for r in encoder_rows)
    ref_energy = sum(r["reference_energy"] for r in encoder_rows)
    encoder_summary = {
        "rows": len(encoder_rows),
        "relative_rmse": (sq_err / ref_energy) ** 0.5,
        "max_row_relative_rmse": max(r["relative_rmse"] for r in encoder_rows),
        "mean_cos_sim": sum(r["cos_sim"] for r in encoder_rows) / len(encoder_rows),
    }
    return {
        "encoder": encoder_summary,
        "isolated": summarize(isolated_rows),
        "isolated_domains": {
            d: summarize([r for r in isolated_rows if r["domain"] == d])
            for d in sorted({r["domain"] for r in isolated_rows})
        },
        "e2e": summarize(e2e_rows),
        "e2e_domains": {
            d: summarize([r for r in e2e_rows if r["domain"] == d])
            for d in sorted({r["domain"] for r in e2e_rows})
        },
    }


def _available_cuda_bytes(dev: torch.device) -> int:
    free_b = int(torch.cuda.mem_get_info(dev)[0])
    if getattr(torch.cuda.get_device_properties(dev), "is_integrated", 0):
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                        return max(free_b, avail_kb * 1024)
        except OSError:
            pass
    return free_b


@torch.no_grad()
def _run_pair_low_mem(base: nn.Module, quant: nn.Module, batches):
    """Same exact math as quants.probe.run_pair, but offloads base logits to CPU before running quant."""
    from collections import defaultdict

    from warp_quant.data import CHUNK

    nll = {"base": 0.0, "quant": 0.0}
    count = 0
    tok_kls = []
    row_dump = []
    dom = defaultdict(lambda: {"base": 0.0, "quant": 0.0, "count": 0, "kl": 0.0, "kl_n": 0})
    dev = next(base.parameters()).device

    for i, (domain, ids) in enumerate(batches):
        ids = ids.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lb_out = base(input_ids=ids, use_cache=False).logits[:, :-1]
            logit_bytes = lb_out.numel() * lb_out.element_size()
            keep_on_gpu = dev.type == "cuda" and _available_cuda_bytes(dev) > max(
                8 * (1024**3), logit_bytes * 3
            )
            lb_buf = lb_out if keep_on_gpu else lb_out.cpu()
            del lb_out
            try:
                lq = quant(input_ids=ids, use_cache=False).logits[:, :-1]
            except torch.cuda.OutOfMemoryError:
                if keep_on_gpu:
                    lb_buf = lb_buf.cpu()
                    keep_on_gpu = False
                    torch.cuda.empty_cache()
                    lq = quant(input_ids=ids, use_cache=False).logits[:, :-1]
                else:
                    raise
        targets = ids[:, 1:]
        d = dom[domain]
        row_nll = {"base": 0.0, "quant": 0.0}
        row_kl_sum, row_kl_n = 0.0, 0
        for s in range(0, lq.shape[1], CHUNK):
            sl = slice(s, min(s + CHUNK, lq.shape[1]))
            lb_sl = lb_buf[:, sl] if keep_on_gpu else lb_buf[:, sl].to(dev, non_blocking=True)
            lbs = F.log_softmax(lb_sl.float(), dim=-1)
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
        del lb_buf, lq
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

    return nll, count, torch.cat(tok_kls), dom, row_dump


def run_probe(args: argparse.Namespace) -> None:
    from warp_quant.data import (
        WEIGHTED_KL_PROTOCOL,
        aggregate_weighted_kl,
        load_batches,
    )

    hub_id, info = resolve(args.model)
    base_dir = (
        Path(hub_id)
        if Path(hub_id).exists()
        else Path(resolve_snapshot(hub_id, revision=info.get("hub_revision")))
    )
    cls_name = model_class(str(base_dir)).__name__
    if cls_name.startswith("Qwen4Exp") or cls_name == "Qwen3_5MoeForConditionalGeneration":
        from warp_quant.streaming_moe import run_probe_qwen4_exp

        return run_probe_qwen4_exp(args)

    quant_dir = Path(args.quant).resolve()

    log_event(
        "ROT_PROBE_START",
        model=args.model,
        base=str(base_dir),
        quant=str(quant_dir),
        eval_sets=args.eval_sets,
        eval_mtp=args.eval_mtp,
    )

    processor = AutoProcessor.from_pretrained(str(base_dir))
    quant_model = load_quant_model_with_transforms(quant_dir, device="cuda:0")
    torch.cuda.empty_cache()
    base_model = model_class(str(base_dir)).from_pretrained(str(base_dir), dtype="auto", device_map={"": 0})
    base_model.eval()
    torch.cuda.empty_cache()

    results: dict[str, dict] = {}
    first_batches = None

    for eval_path_str in args.eval_sets:
        eval_path = Path(eval_path_str)
        if not eval_path.is_absolute():
            eval_path = ROOT / eval_path
        batches = load_batches(processor, str(eval_path), limit=args.limit)
        if first_batches is None:
            first_batches = batches

        t0 = time.time()
        nll, count, kls, dom, row_dump = _run_pair_low_mem(base_model, quant_model, batches)
        elapsed_s = round(time.time() - t0, 2)

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
            "rows": len(batches),
            "tokens": count,
            "ppl_base": round(ppl_base, 4),
            "ppl_quant": round(ppl_quant, 4),
            "ppl_delta_pct": round((ppl_quant - ppl_base) / ppl_base * 100, 3),
            "mean_kl": round(mean_kl, 6),
            "weighted_kl": round(weighted_kl, 6) if weighted_kl is not None else None,
            "weighted_kl_protocol": WEIGHTED_KL_PROTOCOL if weighted_kl is not None else None,
            "p999_kl": round(p999_kl, 4),
            "elapsed_s": elapsed_s,
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

    if args.eval_mtp:
        first_eval_path = Path(args.eval_sets[0])
        if not first_eval_path.is_absolute():
            first_eval_path = ROOT / first_eval_path
        mtp_summary = evaluate_mtp_fidelity(
            base_model,
            quant_model,
            processor,
            base_dir,
            quant_dir,
            first_eval_path,
            per_domain=args.mtp_per_domain,
        )
        results["mtp_fidelity"] = mtp_summary
        log_event(
            "ROT_PROBE_MTP",
            isolated_mean_kl=mtp_summary["isolated"]["mean_kl"],
            e2e_mean_kl=mtp_summary["e2e"]["mean_kl"],
        )
        print(json.dumps({"mtp_fidelity": mtp_summary}, indent=2))

    if args.eval_vision:
        vis_dir = Path(args.eval_vision)
        if not vis_dir.is_absolute():
            vis_dir = ROOT / vis_dir
        vis_summary = evaluate_vision_fidelity(base_model, quant_model, vis_dir)
        results["vision_fidelity"] = vis_summary
        log_event(
            "ROT_PROBE_VISION",
            encoder_relative_rmse=vis_summary["encoder"]["relative_rmse"],
            isolated_mean_kl=vis_summary["isolated"]["mean_kl"],
            e2e_mean_kl=vis_summary["e2e"]["mean_kl"],
        )
        print(json.dumps({"vision_fidelity": vis_summary}, indent=2))

    if getattr(args, "eval_vision_guard", None):
        guard_dir = Path(args.eval_vision_guard)
        if not guard_dir.is_absolute():
            guard_dir = ROOT / guard_dir
        guard_manifest = json.loads((guard_dir / "manifest.json").read_text())
        if not guard_manifest.get("guard", False):
            raise ValueError(
                f"Refusing to run Vision guard on non-guard fixture: {guard_dir} (guard={guard_manifest.get('guard')})"
            )
        guard_summary = evaluate_vision_fidelity(base_model, quant_model, guard_dir)
        results["vision_guard_fidelity"] = guard_summary
        log_event(
            "ROT_PROBE_VISION_GUARD",
            encoder_relative_rmse=guard_summary["encoder"]["relative_rmse"],
            isolated_mean_kl=guard_summary["isolated"]["mean_kl"],
            e2e_mean_kl=guard_summary["e2e"]["mean_kl"],
        )
        print(json.dumps({"vision_guard_fidelity": guard_summary}, indent=2))

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2) + "\n")

    run_done(quant=str(quant_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quantize", help="Quantize with GPTQ + optional block-Hadamard rotation")
    q.add_argument("--model", default="2b")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--group-size", type=int, choices=[32, 64, 128], default=64)
    q.add_argument(
        "--hadamard-block-size",
        type=int,
        default=64,
        help="0 for unrotated GPTQ baseline; 32/64/128/256 for online block-Hadamard",
    )
    q.add_argument(
        "--ple-group-size",
        type=int,
        choices=[0, 32, 64, 128],
        default=32,
        help="Group size for 4-bit Custom-LUT4 Layer-1 PLE tables (0 to keep BF16)",
    )
    q.add_argument(
        "--ple-cd-passes",
        type=int,
        default=1,
        help="Coordinate-descent centroid refinement passes for Custom-LUT4 PLE",
    )
    q.add_argument("--actorder", choices=["none", "weight", "static"], default="static")
    q.add_argument("--dampening-frac", type=float, default=0.01)
    q.add_argument(
        "--quantize-mtp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize mtp.layers.0.* projections to W4A16 (+ Hadamard) (--no-quantize-mtp to skip)",
    )
    q.add_argument(
        "--quantize-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize model.visual.* linear layers to W4A16 (+ Hadamard) (--no-quantize-vision to skip)",
    )
    q.add_argument(
        "--component-quant-mode",
        choices=["gptq", "rtn"],
        default="gptq",
        help="Quantize MTP and Vision with Hessian-calibrated Cholesky GPTQ or weight-only RTN",
    )
    q.add_argument(
        "--vision-calib-dir",
        default="data/vision-calib-v2",
        help="Disjoint calibration image fixture dir (with calibration=true in manifest.json)",
    )
    q.add_argument(
        "--vision-calib-samples",
        type=int,
        default=None,
        help="Optional cap on Vision calibration images (default: all rows in --vision-calib-dir)",
    )
    q.add_argument(
        "--component-calib-samples", type=int, default=64, help="Number of calibration sequences for MTP GPTQ"
    )
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

    p = sub.add_parser("probe", help="Evaluate text KL + optional MTP fidelity on a checkpoint")
    p.add_argument("--model", default="2b")
    p.add_argument("--quant", required=True)
    p.add_argument("--eval-sets", nargs="+", default=["data/probe-v4.jsonl", "data/probe-guard.jsonl"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-mtp", action="store_true")
    p.add_argument("--mtp-per-domain", type=int, default=2)
    p.add_argument("--eval-vision", default=None, help="Path to vision probe fixture dir")
    p.add_argument(
        "--eval-vision-guard",
        default=None,
        help="Path to OOD vision guard fixture dir (guard=true in manifest.json)",
    )
    p.add_argument("--dump-rows-dir", default=None)
    p.add_argument("--output-json", default=None)

    args = parser.parse_args()
    try:
        if args.cmd == "quantize":
            run_quantize(args)
        elif args.cmd == "probe":
            run_probe(args)
    except BaseException as exc:
        run_fail(error=str(exc))
        raise


if __name__ == "__main__":
    main()
