"""Layer-by-layer H64-rotated multi-option GPTQ candidate bank builder for RCO."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from llmcompressor.modifiers.gptq.helpers import accumulate_hessian, make_empty_hessian

from warp_quant.hadamard_gptq import _hadamard_rotate_last_dim, _hadamard_rotate_weight
from warp_quant.lablog import log_event
from warp_quant.rco.options import QuantOption, decode_candidate, quantize_candidate, resolve_options


def get_text_model(model: nn.Module) -> tuple[nn.Module, str]:
    """Return (text_model, prefix) where prefix is e.g. 'model.language_model' or 'model'."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model, "model.language_model"
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model, "language_model"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "model"
    raise ValueError(f"Cannot locate language_model.layers on {type(model).__name__}")


def discover_text_groups(model: nn.Module) -> list[dict]:
    """Return ordered list of vLLM-compatible fused linear groups across all decoder layers."""
    text_model, prefix = get_text_model(model)
    groups: list[dict] = []
    for layer_idx, layer in enumerate(text_model.layers):
        l_pref = f"{prefix}.layers.{layer_idx}"
        if hasattr(layer, "self_attn"):
            groups.append(
                {
                    "group_idx": len(groups),
                    "layer_idx": layer_idx,
                    "stage": "attn",
                    "kind": "self_attn_qkv",
                    "members": [
                        f"{l_pref}.self_attn.q_proj",
                        f"{l_pref}.self_attn.k_proj",
                        f"{l_pref}.self_attn.v_proj",
                    ],
                }
            )
            groups.append(
                {
                    "group_idx": len(groups),
                    "layer_idx": layer_idx,
                    "stage": "attn",
                    "kind": "self_attn_o",
                    "members": [f"{l_pref}.self_attn.o_proj"],
                }
            )
        elif hasattr(layer, "linear_attn"):
            groups.append(
                {
                    "group_idx": len(groups),
                    "layer_idx": layer_idx,
                    "stage": "attn",
                    "kind": "linear_attn_qkvz",
                    "members": [
                        f"{l_pref}.linear_attn.in_proj_qkv",
                        f"{l_pref}.linear_attn.in_proj_z",
                    ],
                }
            )
            groups.append(
                {
                    "group_idx": len(groups),
                    "layer_idx": layer_idx,
                    "stage": "attn",
                    "kind": "linear_attn_out",
                    "members": [f"{l_pref}.linear_attn.out_proj"],
                }
            )
        else:
            raise ValueError(f"Layer {layer_idx} has neither self_attn nor linear_attn")

        groups.append(
            {
                "group_idx": len(groups),
                "layer_idx": layer_idx,
                "stage": "mlp",
                "kind": "mlp_gate_up",
                "members": [
                    f"{l_pref}.mlp.gate_proj",
                    f"{l_pref}.mlp.up_proj",
                ],
            }
        )
        groups.append(
            {
                "group_idx": len(groups),
                "layer_idx": layer_idx,
                "stage": "mlp",
                "kind": "mlp_down",
                "members": [f"{l_pref}.mlp.down_proj"],
            }
        )
    return groups


def attach_online_hadamard_hooks(
    model: nn.Module,
    groups: list[dict],
    hadamard_block_size: int,
) -> list:
    """Attach online H_b input rotation pre-hooks to all group member Linear modules."""
    if hadamard_block_size <= 0:
        return []
    handles = []
    for grp in groups:
        for member_name in grp["members"]:
            mod = model.get_submodule(member_name)
            handles.append(
                mod.register_forward_pre_hook(
                    lambda _m, args, bs=hadamard_block_size: (
                        _hadamard_rotate_last_dim(args[0], bs, compute_dtype=torch.float32),
                        *args[1:],
                    )
                )
            )
    return handles


@torch.no_grad()
def build_rco_bank(
    model: nn.Module,
    calib_rows: list[torch.Tensor],
    option_specs: list[str | QuantOption] | tuple[str | QuantOption, ...],
    bank_dir: Path | str,
    *,
    hadamard_block_size: int = 64,
    dampening_frac: float = 0.01,
    propagate_option_idx: int = 0,
    rotate_weights_in_place: bool = True,
) -> dict:
    """Build a multi-option HGPTQ candidate bank block-by-block from identical FP32 Hessians.

    Each block calibrates its attention/GDN stage first, solves all options, installs
    `options[propagate_option_idx]` for attention, then calibrates and solves the MLP stage
    before propagating the quantized block outputs to the next layer.
    """
    options = resolve_options(option_specs)
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = bank_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    text_model, _ = get_text_model(model)
    groups = discover_text_groups(model)
    num_layers = len(text_model.layers)

    # 1. Rotate eligible weights by H_b^T / sqrt(b) if requested
    if rotate_weights_in_place and hadamard_block_size > 0:
        for grp in groups:
            for member_name in grp["members"]:
                mod = model.get_submodule(member_name)
                mod.weight.data = _hadamard_rotate_weight(mod.weight.data, hadamard_block_size)

    # 2. Precompute initial token embeddings on CPU
    hidden_rows: list[torch.Tensor] = []
    total_tokens = 0
    for ids in calib_rows:
        inp = ids.to(device=device, dtype=torch.long)
        if inp.ndim == 1:
            inp = inp.unsqueeze(0)
        if inp.shape[1] < 2:
            continue
        emb = text_model.embed_tokens(inp).detach().to(torch.bfloat16).cpu()
        hidden_rows.append(emb)
        total_tokens += int(inp.shape[1])

    if not hidden_rows:
        raise ValueError("No valid calibration sequences with length >= 2")

    # Cache rotary embeddings per sequence length
    rotary_cache: dict[int, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = {}

    def _get_pos(seq_len: int, hidden_sample: torch.Tensor):
        cached = rotary_cache.get(seq_len)
        if cached is None:
            pos_3d = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, 1, -1)
            text_pos = torch.arange(seq_len, device=device).view(1, -1)
            pos_emb = text_model.rotary_emb(hidden_sample, pos_3d)
            cached = (pos_emb, text_pos)
            rotary_cache[seq_len] = cached
        return cached

    had_hooks = attach_online_hadamard_hooks(model, groups, hadamard_block_size)
    group_costs: list[tuple[int, ...]] = [() for _ in groups]
    matrix_summaries: dict[str, list[dict]] = {}

    t0 = time.time()
    try:
        for layer_idx in range(num_layers):
            lt0 = time.time()
            layer = text_model.layers[layer_idx]
            layer_groups = [g for g in groups if g["layer_idx"] == layer_idx]
            attn_groups = [g for g in layer_groups if g["stage"] == "attn"]
            mlp_groups = [g for g in layer_groups if g["stage"] == "mlp"]
            layer_bundle: dict[str, list[dict]] = {}

            for _stage_name, stage_groups in [("attn", attn_groups), ("mlp", mlp_groups)]:
                stage_members = [m for g in stage_groups for m in g["members"]]
                hessians = {m: make_empty_hessian(model.get_submodule(m), device) for m in stage_members}
                nsamples = {m: torch.zeros(1, device=device, dtype=torch.int64) for m in stage_members}
                stage_hooks = []

                for m_name in stage_members:
                    mod = model.get_submodule(m_name)

                    def _make_accum_hook(
                        key: str,
                        ref: nn.Linear,
                        h_map: dict = hessians,
                        n_map: dict = nsamples,
                    ):
                        def _hook(_module, args):
                            with torch.amp.autocast("cuda", enabled=False):
                                accumulate_hessian(args[0], ref, h_map[key], n_map[key])

                        return _hook

                    stage_hooks.append(mod.register_forward_pre_hook(_make_accum_hook(m_name, mod)))

                try:
                    for h_cpu in hidden_rows:
                        h_in = h_cpu.to(device=device, non_blocking=True)
                        pos_emb, text_pos = _get_pos(h_in.shape[1], h_in)
                        _ = layer(
                            h_in,
                            position_embeddings=pos_emb,
                            attention_mask=None,
                            position_ids=text_pos,
                            past_key_values=None,
                            use_cache=False,
                        )
                finally:
                    for h in stage_hooks:
                        h.remove()

                # Solve all QuantOptions for each member in this stage
                for grp in stage_groups:
                    g_costs = [0] * len(options)
                    for m_name in grp["members"]:
                        mod = model.get_submodule(m_name)
                        w_rot = mod.weight.data
                        h_rot = hessians.pop(m_name)
                        n_smp = nsamples.pop(m_name)
                        m_opts: list[dict] = []
                        m_summary: list[dict] = []
                        for k, opt in enumerate(options):
                            cand = quantize_candidate(
                                w_rot,
                                h_rot,
                                n_smp,
                                opt,
                                dampening_frac=dampening_frac,
                            )
                            m_opts.append(cand)
                            g_costs[k] += int(cand["bytes"])
                            m_summary.append(
                                {
                                    "option": opt.name,
                                    "num_bits": opt.num_bits,
                                    "group_size": opt.group_size,
                                    "bytes": int(cand["bytes"]),
                                    "hessian_loss": float(cand["hessian_loss"]),
                                    "gptq_loss": float(cand["gptq_loss"]),
                                    "weight_mse": float(cand["weight_mse"]),
                                    "rel_mse": float(cand["rel_mse"]),
                                }
                            )
                        layer_bundle[m_name] = m_opts
                        matrix_summaries[m_name] = m_summary
                        # Install propagated option's decoded weight so downstream stage/layers see realistic activations
                        prop_cand = m_opts[propagate_option_idx]
                        mod.weight.data.copy_(
                            decode_candidate(
                                prop_cand["codes"].to(device),
                                prop_cand["scales"].to(device),
                                int(prop_cand["group_size"]),
                                dtype=mod.weight.dtype,
                            )
                        )
                        del h_rot, n_smp
                    group_costs[grp["group_idx"]] = tuple(g_costs)

            # Propagate hidden states through the completed quantized block to feed layer_idx + 1
            for row_idx, h_cpu in enumerate(hidden_rows):
                h_in = h_cpu.to(device=device, non_blocking=True)
                pos_emb, text_pos = _get_pos(h_in.shape[1], h_in)
                h_out = layer(
                    h_in,
                    position_embeddings=pos_emb,
                    attention_mask=None,
                    position_ids=text_pos,
                    past_key_values=None,
                    use_cache=False,
                )
                if isinstance(h_out, tuple):
                    h_out = h_out[0]
                hidden_rows[row_idx] = h_out.detach().to(torch.bfloat16).cpu()

            torch.save(layer_bundle, layers_dir / f"layer_{layer_idx:03d}.pt")
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            log_event(
                "RCO_BANK_LAYER",
                layer=layer_idx,
                total_layers=num_layers,
                elapsed_s=round(time.time() - lt0, 2),
            )
    finally:
        for h in had_hooks:
            h.remove()

    manifest = {
        "schema_version": 1,
        "options": [
            {
                "name": o.name,
                "num_bits": o.num_bits,
                "group_size": o.group_size,
                "mse_scale": o.mse_scale,
                "actorder": o.actorder,
                "bpw": o.bpw,
            }
            for o in options
        ],
        "hadamard_block_size": hadamard_block_size,
        "dampening_frac": dampening_frac,
        "propagate_option_idx": propagate_option_idx,
        "calib_rows": len(hidden_rows),
        "calib_tokens": total_tokens,
        "num_layers": num_layers,
        "groups": groups,
        "costs": [list(c) for c in group_costs],
        "all_low_bytes": sum(min(c) for c in group_costs),
        "all_high_bytes": sum(max(c) for c in group_costs),
        "matrices": matrix_summaries,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (bank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_rco_bank(bank_dir: Path | str) -> tuple[dict, dict[str, list[dict]]]:
    """Load bank manifest and all per-layer candidate bundles from bank_dir."""
    bank_dir = Path(bank_dir)
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    candidates: dict[str, list[dict]] = {}
    layers_dir = bank_dir / "layers"
    for layer_idx in range(int(manifest["num_layers"])):
        layer_data = torch.load(
            layers_dir / f"layer_{layer_idx:03d}.pt", map_location="cpu", weights_only=True
        )
        candidates.update(layer_data)
    return manifest, candidates
