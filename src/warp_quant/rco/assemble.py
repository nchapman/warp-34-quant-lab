"""Assemble a multi-group compressed-tensors checkpoint from an RCO candidate bank and allocation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor

from warp_quant.hadamard_gptq import (
    BASE_IGNORE,
    LM_HEAD_TARGET_RE,
    MTP_FC_TARGET_RE,
    MTP_HAD_TARGET_RE,
    MTP_TARGET_RE,
    TEXT_HAD_TARGET_RE,
    VISION_FC2_TARGET_RE,
    VISION_GROUP_TARGET_RE,
    _assert_packed_alignment,
    _clean_and_augment_shards,
    _pack_w8_mse_tensor,
)
from warp_quant.rco.bank import load_rco_bank
from warp_quant.rco.options import pack_candidate
from warp_quant.stamp import (
    build_generation_config,
    check_template_renders,
    ensure_license_files,
    ensure_processor_configs,
    stamp_release_template,
)


def _unanchored_layer_target_re(member_name: str) -> str:
    """Convert e.g. 'model.language_model.layers.3.self_attn.q_proj' to an HF+vLLM compatible regex."""
    idx = member_name.find("layers.")
    if idx < 0:
        raise ValueError(f"Expected 'layers.' in member name, got {member_name}")
    suffix = member_name[idx:].replace(".", r"\.")
    return rf"re:.*(?<!mtp\.){suffix}$"


@torch.no_grad()
def calibrate_lm_head_w8(
    lm_head_weight: torch.Tensor,
    post_norm_states: list[torch.Tensor],
    *,
    group_size: int = 64,
    dampening_frac: float = 0.01,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Calibrate untied lm_head to unrotated W8 (MSE-init Cholesky GPTQ) from post-norm hidden states."""
    from warp_quant.hadamard_gptq import _gptq_pack_tensor

    dev = torch.device(device) if device is not None else lm_head_weight.device
    k_in = int(lm_head_weight.shape[1])
    hessian = torch.zeros((k_in, k_in), dtype=torch.float32, device=dev)
    total_tokens = 0
    for h in post_norm_states:
        if not isinstance(h, torch.Tensor):
            continue
        x = h.to(device=dev, dtype=torch.float32).reshape(-1, k_in)
        if x.shape[0] == 0:
            continue
        hessian.addmm_(x.T, x, beta=1.0, alpha=2.0)
        total_tokens += int(x.shape[0])
    if total_tokens == 0:
        raise ValueError("calibrate_lm_head_w8 received no valid post-norm hidden states")
    return _gptq_pack_tensor(
        lm_head_weight.to(device=dev),
        hessian,
        torch.tensor([total_tokens], dtype=torch.int64, device=dev),
        strategy="group",
        group_size=group_size,
        actorder=None,
        dampening_frac=dampening_frac,
        num_bits=8,
        cd_passes=0,
    )


def build_rco_quantization_config(
    base_cfg: dict,
    manifest: dict,
    choices: list[int] | tuple[int, ...],
    *,
    quantize_mtp: bool = False,
    quantize_vision: bool = False,
    vision_fc2_channel: bool = False,
    component_group_size: int = 64,
    quantize_lm_head_w8: bool = False,
    quantize_mtp_fc_w8: bool = False,
) -> dict:
    """Build a compressed-tensors quantization_config with one config_group per active option."""
    options = manifest["options"]
    groups = manifest["groups"]
    hadamard_block_size = int(manifest.get("hadamard_block_size", 64))
    if len(choices) != len(groups):
        raise ValueError(f"Expected {len(groups)} group choices, got {len(choices)}")

    targets_by_opt: dict[int, list[str]] = {i: [] for i in range(len(options))}
    for grp, choice in zip(groups, choices, strict=True):
        for member in grp["members"]:
            targets_by_opt[int(choice)].append(_unanchored_layer_target_re(member))

    config_groups: dict[str, dict] = {}
    for opt_idx, opt in enumerate(options):
        t_list = targets_by_opt[opt_idx]
        if not t_list:
            continue
        config_groups[opt["name"]] = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": t_list,
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": int(opt["group_size"]),
                "num_bits": int(opt["num_bits"]),
                "observer": "minmax",
                "observer_kwargs": {},
                "scale_dtype": None,
                "strategy": "group",
                "symmetric": True,
                "type": "int",
                "zp_dtype": None,
            },
        }

    comp_targets: list[str] = []
    if quantize_mtp:
        comp_targets.append(MTP_TARGET_RE)
    if quantize_vision:
        if vision_fc2_channel:
            comp_targets.append(VISION_GROUP_TARGET_RE)
        else:
            comp_targets.extend([VISION_GROUP_TARGET_RE, VISION_FC2_TARGET_RE])

    if comp_targets:
        comp_weights = {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": component_group_size,
            "num_bits": 4,
            "observer": "minmax",
            "observer_kwargs": {},
            "scale_dtype": None,
            "strategy": "group",
            "symmetric": True,
            "type": "int",
            "zp_dtype": None,
        }
        config_groups["group_0"] = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": comp_targets,
            "weights": comp_weights,
        }
        if quantize_vision and vision_fc2_channel:
            config_groups["vision_fc2_channel"] = {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": [VISION_FC2_TARGET_RE],
                "weights": dict(comp_weights, strategy="channel", group_size=None),
            }

    w8_head_targets: list[str] = []
    if quantize_lm_head_w8:
        w8_head_targets.append(LM_HEAD_TARGET_RE)
    if quantize_mtp_fc_w8:
        w8_head_targets.append(MTP_FC_TARGET_RE)
    if w8_head_targets:
        merged = False
        for grp_cfg in config_groups.values():
            w_cfg = grp_cfg.get("weights", {})
            if (
                w_cfg.get("num_bits") == 8
                and w_cfg.get("strategy") == "group"
                and w_cfg.get("group_size") == component_group_size
                and w_cfg.get("actorder") is None
            ):
                grp_cfg["targets"].extend(w8_head_targets)
                merged = True
                break
        if not merged:
            config_groups["w8_head"] = {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": w8_head_targets,
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": component_group_size,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                    "zp_dtype": None,
                },
            }

    top_ignore = [ig for ig in BASE_IGNORE if not (quantize_lm_head_w8 and ig == "lm_head")]
    qcfg: dict = {
        "config_groups": config_groups,
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": top_ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    if hadamard_block_size > 0:
        main_had_targets = [TEXT_HAD_TARGET_RE]
        if quantize_mtp:
            main_had_targets.append(MTP_HAD_TARGET_RE)
        if quantize_vision:
            vision_inter = base_cfg.get("vision_config", {}).get("intermediate_size", 0)
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

    return qcfg


def assemble_rco_checkpoint(
    base_dir: Path | str,
    bank_dir: Path | str,
    choices: list[int] | tuple[int, ...],
    output_dir: Path | str,
    *,
    model_key: str | None = None,
    allocation_name: str = "rco",
    mtp_packed: dict[str, dict[str, torch.Tensor]] | None = None,
    vision_packed: dict[str, dict[str, torch.Tensor]] | None = None,
    lm_head_packed: dict[str, torch.Tensor] | None = None,
    scale_overrides: dict[str, torch.Tensor] | None = None,
    bf16_overrides: dict[str, torch.Tensor] | None = None,
    quantize_mtp_rtn: bool = False,
    quantize_vision_rtn: bool = False,
    quantize_lm_head_w8: bool = False,
    component_group_size: int = 64,
) -> dict:
    """Stream base safetensors shards, replace eligible text weights with packed RCO candidates, and stamp config."""
    base_dir = Path(base_dir)
    bank_dir = Path(bank_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"output_dir already exists: {output_dir}")

    manifest, candidates = load_rco_bank(bank_dir)
    groups = manifest["groups"]
    if len(choices) != len(groups):
        raise ValueError(f"choices length {len(choices)} != groups length {len(groups)}")

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _assemble_rco_checkpoint_into(
            base_dir,
            output_dir,
            manifest,
            candidates,
            choices,
            model_key=model_key,
            allocation_name=allocation_name,
            mtp_packed=mtp_packed,
            vision_packed=vision_packed,
            lm_head_packed=lm_head_packed,
            scale_overrides=scale_overrides,
            bf16_overrides=bf16_overrides,
            quantize_mtp_rtn=quantize_mtp_rtn,
            quantize_vision_rtn=quantize_vision_rtn,
            quantize_lm_head_w8=quantize_lm_head_w8,
            component_group_size=component_group_size,
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def _assemble_rco_checkpoint_into(
    base_dir: Path,
    output_dir: Path,
    manifest: dict,
    candidates: dict[str, list[dict]],
    choices: list[int] | tuple[int, ...],
    *,
    model_key: str | None = None,
    allocation_name: str = "rco",
    mtp_packed: dict[str, dict[str, torch.Tensor]] | None = None,
    vision_packed: dict[str, dict[str, torch.Tensor]] | None = None,
    lm_head_packed: dict[str, torch.Tensor] | None = None,
    scale_overrides: dict[str, torch.Tensor] | None = None,
    bf16_overrides: dict[str, torch.Tensor] | None = None,
    quantize_mtp_rtn: bool = False,
    quantize_vision_rtn: bool = False,
    quantize_lm_head_w8: bool = False,
    component_group_size: int = 64,
) -> dict:
    groups = manifest["groups"]
    options = manifest["options"]

    member_to_choice: dict[str, int] = {}
    for grp, choice in zip(groups, choices, strict=True):
        for m in grp["members"]:
            member_to_choice[m] = int(choice)

    # Copy non-safetensors metadata files from base_dir (use copyfile so HF cache 0444 modes aren't copied)
    for item in base_dir.iterdir():
        if (
            item.is_file()
            and not item.name.endswith(".safetensors")
            and item.name != "model.safetensors.index.json"
        ):
            shutil.copyfile(item, output_dir / item.name)

    base_cfg = json.loads((base_dir / "config.json").read_text())
    vision_inter = base_cfg.get("vision_config", {}).get("intermediate_size", 0)
    vision_fc2_channel = (vision_inter % component_group_size != 0) if vision_inter else False
    is_tied = bool(
        base_cfg.get("tie_word_embeddings") or base_cfg.get("text_config", {}).get("tie_word_embeddings")
    )
    want_lm_head_w8 = (quantize_lm_head_w8 or lm_head_packed is not None) and not is_tied

    index_path = base_dir / "model.safetensors.index.json"
    has_index = index_path.is_file()
    index_data = json.loads(index_path.read_text()) if has_index else None
    new_weight_map: dict[str, str] = {}
    code_offsets: dict[str, dict[str, float | int]] = {}
    w8_head_tensors: list[str] = []
    did_quantize_lm_head_w8 = False
    has_mtp_fc_in_base = False
    total_bytes = 0
    packed_count = 0

    for shard_path in sorted(base_dir.glob("*.safetensors")):
        shard_name = shard_path.name
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key == "mtp.fc.weight":
                    has_mtp_fc_in_base = True
                if key == "lm_head.weight" and want_lm_head_w8:
                    raw_lm = f.get_tensor(key)
                    if (
                        raw_lm.ndim == 2
                        and raw_lm.shape[0] % 64 == 0
                        and raw_lm.shape[1] % component_group_size == 0
                    ):
                        packed = (
                            lm_head_packed
                            if lm_head_packed is not None
                            else _pack_w8_mse_tensor(raw_lm, group_size=component_group_size)
                        )
                        for suffix, pt in packed.items():
                            full_k = f"lm_head.{suffix}"
                            out_tensors[full_k] = pt
                            new_weight_map[full_k] = shard_name
                            total_bytes += pt.numel() * pt.element_size()
                        did_quantize_lm_head_w8 = True
                        w8_head_tensors.append("lm_head")
                        continue
                if key.endswith(".weight") and key[:-7] in member_to_choice:
                    prefix = key[:-7]
                    c_idx = member_to_choice[prefix]
                    cand = candidates[prefix][c_idx]
                    c_off = float(cand.get("code_offset", 0.0))
                    if c_off != 0.0:
                        code_offsets[prefix] = {
                            "code_offset": c_off,
                            "group_size": int(cand["group_size"]),
                        }
                    scales_to_pack = (
                        scale_overrides[prefix]
                        if (scale_overrides is not None and prefix in scale_overrides)
                        else cand["scales"]
                    )
                    packed = pack_candidate(
                        cand["codes"],
                        scales_to_pack,
                        int(cand["num_bits"]),
                    )
                    for suffix, pt in packed.items():
                        full_k = f"{prefix}.{suffix}"
                        out_tensors[full_k] = pt
                        new_weight_map[full_k] = shard_name
                        total_bytes += pt.numel() * pt.element_size()
                    packed_count += 1
                else:
                    t = (
                        bf16_overrides[key].detach().cpu().to(f.get_tensor(key).dtype)
                        if (bf16_overrides is not None and key in bf16_overrides)
                        else f.get_tensor(key)
                    )
                    out_tensors[key] = t
                    new_weight_map[key] = shard_name
                    total_bytes += t.numel() * t.element_size()
        save_file(out_tensors, str(output_dir / shard_name))

    if has_index and index_data is not None:
        index_data["weight_map"] = dict(sorted(new_weight_map.items()))
        if "metadata" in index_data and isinstance(index_data["metadata"], dict):
            index_data["metadata"]["total_size"] = total_bytes
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index_data, indent=2, sort_keys=True) + "\n"
        )

    has_mtp = mtp_packed is not None or quantize_mtp_rtn
    has_vis = vision_packed is not None or quantize_vision_rtn
    want_mtp_fc_w8 = (
        has_mtp
        and has_mtp_fc_in_base
        and (quantize_lm_head_w8 or bool(mtp_packed and "mtp.fc" in mtp_packed))
    )
    if has_mtp or has_vis:
        precomputed: dict[str, dict[str, torch.Tensor]] = {}
        if mtp_packed:
            precomputed.update(mtp_packed)
        if vision_packed:
            precomputed.update(vision_packed)
        _clean_and_augment_shards(
            output_dir,
            quantize_mtp=has_mtp,
            quantize_vision=has_vis,
            group_size=component_group_size,
            hadamard_block_size=int(manifest.get("hadamard_block_size", 64)),
            precomputed_packed=precomputed if precomputed else None,
            quantize_mtp_fc_w8=want_mtp_fc_w8,
        )
        if want_mtp_fc_w8:
            w8_head_tensors.append("mtp.fc")
        out_idx_path = output_dir / "model.safetensors.index.json"
        if out_idx_path.is_file():
            out_idx = json.loads(out_idx_path.read_text())
            total_bytes = int(out_idx.get("metadata", {}).get("total_size", total_bytes))

    base_cfg["quantization_config"] = build_rco_quantization_config(
        base_cfg,
        manifest,
        choices,
        quantize_mtp=has_mtp,
        quantize_vision=has_vis,
        vision_fc2_channel=vision_fc2_channel,
        component_group_size=component_group_size,
        quantize_lm_head_w8=did_quantize_lm_head_w8,
        quantize_mtp_fc_w8=want_mtp_fc_w8,
    )
    (output_dir / "config.json").write_text(json.dumps(base_cfg, indent=2, sort_keys=True) + "\n")

    _assert_packed_alignment(output_dir)

    if model_key is not None:
        from warp_quant.models import resolve

        _, info = resolve(model_key)
        processor = AutoProcessor.from_pretrained(str(base_dir))
        processor.save_pretrained(str(output_dir))
        ensure_processor_configs(
            str(output_dir),
            fallback=info.get("processor_hub_fallback") or str(base_dir),
            fallback_revision=info.get("processor_hub_fallback_revision"),
        )
        gc = build_generation_config(processor, info, str(base_dir), {})
        (output_dir / "generation_config.json").write_text(json.dumps(gc, indent=2) + "\n")
        stamp_release_template(str(output_dir))
        check_template_renders(str(output_dir))
        ensure_license_files(str(output_dir), str(base_dir), info.get("license_files"))

    option_counts = {opt["name"]: sum(1 for c in choices if int(c) == idx) for idx, opt in enumerate(options)}
    eligible_bytes = sum(int(manifest["costs"][g_idx][int(c)]) for g_idx, c in enumerate(choices))
    receipt = {
        "schema_version": 1,
        "allocation_name": allocation_name,
        "choices": [int(c) for c in choices],
        "option_group_counts": option_counts,
        "packed_matrices": packed_count,
        "eligible_text_bytes": eligible_bytes,
        "total_safetensors_bytes": total_bytes,
    }
    if code_offsets:
        receipt["code_offsets"] = code_offsets
    if w8_head_tensors:
        receipt["w8_head_tensors"] = w8_head_tensors
    (output_dir / "rco_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
