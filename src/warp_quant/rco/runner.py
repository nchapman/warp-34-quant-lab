"""End-to-end H-RCO runner: bank construction, Riemannian KL search, and checkpoint assembly."""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoProcessor

from warp_quant.cachepath import resolve_snapshot
from warp_quant.data import get_calibration_dataset
from warp_quant.hadamard_gptq import (
    _gptq_calibrate_mtp,
    _gptq_calibrate_vision,
)
from warp_quant.lablog import log_event
from warp_quant.models import model_class, quant_recipe, resolve
from warp_quant.rco import allocation
from warp_quant.rco.assemble import assemble_rco_checkpoint, calibrate_lm_head_w8
from warp_quant.rco.bank import (
    attach_online_hadamard_hooks,
    build_rco_bank,
    get_text_model,
    load_rco_bank,
)
from warp_quant.rco.hidden_teacher import capture_teacher_hidden_states, hidden_kl
from warp_quant.rco.scale_qat import run_e2e_scale_qat
from warp_quant.rco.search import SearchModel, build_sensitivity_logits


def _enable_eval_layer_checkpointing(text_model: nn.Module):
    """Wrap each decoder layer forward in non-reentrant checkpointing even while in eval() mode."""
    orig_forwards = []
    for layer in text_model.layers:
        orig_fn = layer.forward
        orig_forwards.append((layer, orig_fn))

        def _ckpt_forward(*args, _orig=orig_fn, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(functools.partial(_orig, **kwargs), *args, use_reentrant=False)
            return _orig(*args, **kwargs)

        layer.forward = _ckpt_forward

    def _restore():
        for layer, orig_fn in orig_forwards:
            layer.forward = orig_fn

    return _restore


def prepare_calib_and_search_rows(
    base_dir: Path,
    info: dict,
    *,
    calib_samples: int = 128,
    search_samples: int = 64,
    calib_seq_len: int = 2048,
    search_seq_len: int = 1024,
    seed: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object]:
    """Load calibration dataset and split into bank calibration rows and search rows."""
    rec = quant_recipe(info)
    processor = AutoProcessor.from_pretrained(str(base_dir))
    total_needed = max(calib_samples + search_samples, rec["num_samples"])
    ds = get_calibration_dataset(
        processor,
        rec["calib"],
        num_samples=total_needed,
        max_seq_length=max(calib_seq_len, search_seq_len),
        window=rec["window"],
        seed=seed,
    )
    calib_rows: list[torch.Tensor] = []
    search_rows: list[torch.Tensor] = []
    for i in range(min(calib_samples, len(ds))):
        ids = torch.tensor(ds[i]["input_ids"][:calib_seq_len], dtype=torch.long)
        if ids.numel() >= 4:
            calib_rows.append(ids)
    # Use tail of dataset (or stride) for search rows
    start_search = min(calib_samples, max(0, len(ds) - search_samples))
    for i in range(start_search, min(start_search + search_samples, len(ds))):
        ids = torch.tensor(ds[i]["input_ids"][:search_seq_len], dtype=torch.long)
        if ids.numel() >= 4:
            search_rows.append(ids)
    return calib_rows, search_rows, ds


def run_rco_search_on_bank(
    model: nn.Module,
    bank_dir: Path | str,
    search_rows: list[torch.Tensor],
    budgets: list[float],
    *,
    steps: int = 120,
    gumbel_samples: int = 4,
    lr: float = 0.1,
    seed: int = 42,
    teacher_states: list[torch.Tensor] | None = None,
    kl_chunk_size: int = 128,
) -> dict:
    """Run Riemannian KL search for each budget fraction in `budgets` against `bank_dir`."""
    if not search_rows:
        raise ValueError("search_rows must be non-empty")
    if any(r.numel() < 2 for r in search_rows):
        raise ValueError("every row in search_rows must have at least 2 tokens")
    if teacher_states is not None and len(teacher_states) != len(search_rows):
        raise ValueError(
            f"teacher_states length {len(teacher_states)} != search_rows length {len(search_rows)}"
        )
    bank_dir = Path(bank_dir)
    manifest, candidates = load_rco_bank(bank_dir)
    groups = manifest["groups"]
    group_members = [g["members"] for g in groups]
    costs = [tuple(int(c) for c in row) for row in manifest["costs"]]
    hadamard_block_size = int(manifest.get("hadamard_block_size", 64))

    model.eval()
    device = next(model.parameters()).device
    text_model, _ = get_text_model(model)

    if teacher_states is None:
        log_event("RCO_TEACHER_CAPTURE_START", rows=len(search_rows))
        teacher_states = capture_teacher_hidden_states(model, search_rows)
        log_event("RCO_TEACHER_CAPTURE_DONE", rows=len(teacher_states))

    sens_logits = build_sensitivity_logits(group_members, candidates, costs)

    allocations_out: dict[str, dict] = {
        "all_low": {
            "fraction": 0.0,
            "choices": [0] * len(groups),
            "total_bytes": int(manifest["all_low_bytes"]),
        },
        "all_high": {
            "fraction": 1.0,
            "choices": [len(manifest["options"]) - 1] * len(groups),
            "total_bytes": int(manifest["all_high_bytes"]),
        },
    }

    had_hooks = attach_online_hadamard_hooks(model, groups, hadamard_block_size)
    # Wrap decoder layers in activation checkpointing when VRAM is tight or model is large (>8 GiB W4 text)
    high_vram = device.type == "cuda" and torch.cuda.mem_get_info(device)[0] > 48 * (1024**3)
    use_layer_ckpt = (not high_vram) or int(manifest["all_high_bytes"]) > 8 * (1024**3)
    kl_ckpt = True
    restore_ckpt = _enable_eval_layer_checkpointing(text_model) if use_layer_ckpt else (lambda: None)
    try:
        with SearchModel(model, group_members, candidates, costs) as search:

            @torch.no_grad()
            def eval_allocation_kl(choices: tuple[int, ...] | list[int], max_eval_rows: int = 32) -> float:
                search.set_assignment(choices, soft=None)
                n_ev = min(len(search_rows), max_eval_rows)
                if n_ev == 0:
                    return 0.0
                kls = []
                ev_bsz = 8 if high_vram else 1
                for b0 in range(0, n_ev, ev_bsz):
                    b_idxs = list(range(b0, min(b0 + ev_bsz, n_ev)))
                    lens = [int(search_rows[i].numel()) for i in b_idxs]
                    s_max = max(lens)
                    inp = torch.zeros((len(b_idxs), s_max), dtype=torch.long, device=device)
                    for loc_i, r_i in enumerate(b_idxs):
                        inp[loc_i, : lens[loc_i]] = search_rows[r_i].to(device=device, dtype=torch.long)
                    h_batch = text_model(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state
                    for loc_i, r_i in enumerate(b_idxs):
                        kl_val = hidden_kl(
                            h_batch[loc_i : loc_i + 1, : lens[loc_i]],
                            teacher_states[r_i],
                            model.lm_head,
                            token_chunk_size=kl_chunk_size,
                            checkpointed=False,
                        )
                        kls.append(float(kl_val.item()))
                return sum(kls) / len(kls)

            low_kl = eval_allocation_kl(allocations_out["all_low"]["choices"])
            high_kl = eval_allocation_kl(allocations_out["all_high"]["choices"])
            allocations_out["all_low"]["search_kl"] = low_kl
            allocations_out["all_high"]["search_kl"] = high_kl
            log_event("RCO_ENDPOINTS_KL", all_low_kl=round(low_kl, 6), all_high_kl=round(high_kl, 6))

            for frac in budgets:
                if frac <= 0.0 or frac >= 1.0:
                    continue
                b_info = allocation.upgrade_budget(costs, frac)
                target_bytes = int(b_info["realized_bytes"])
                tag = f"{int(round(frac * 100))}pct"

                # 1. Sensitivity baseline at exact same byte budget
                sens_alloc = allocation.solve(costs, sens_logits.tolist(), target_bytes, exact=True)
                sens_kl = eval_allocation_kl(sens_alloc.choices)
                allocations_out[f"{tag}_sensitivity"] = {
                    "fraction": frac,
                    "choices": list(sens_alloc.choices),
                    "total_bytes": sens_alloc.total_bytes,
                    "search_kl": sens_kl,
                }

                # 2. RCO Riemannian optimization
                t0 = time.time()

                def _forward_loss(_m: nn.Module, step_idx: int, _sample_idx: int) -> torch.Tensor:
                    row_i = (
                        32 + (step_idx % (len(search_rows) - 32))
                        if len(search_rows) > 32
                        else (step_idx % len(search_rows))
                    )
                    inp = search_rows[row_i].to(device=device, dtype=torch.long).unsqueeze(0)
                    # Dummy input grad on embedding output so checkpointed blocks propagate grad to weight gates
                    emb = text_model.embed_tokens(inp).detach().requires_grad_(True)
                    student_h = text_model(
                        inputs_embeds=emb, use_cache=False, return_dict=True
                    ).last_hidden_state
                    return hidden_kl(
                        student_h,
                        teacher_states[row_i],
                        model.lm_head,
                        token_chunk_size=kl_chunk_size,
                        checkpointed=kl_ckpt,
                    )

                def _on_step(rec: dict, _tag: str = tag):
                    if rec["step"] == 0 or (rec["step"] + 1) % 20 == 0 or rec["step"] + 1 == steps:
                        log_event(
                            "RCO_SEARCH_STEP",
                            budget=_tag,
                            step=rec["step"] + 1,
                            total_steps=steps,
                            loss=round(rec["loss"], 5),
                            tau=round(rec["tau"], 4),
                            grad_norm=round(rec["projected_grad_norm"], 5),
                        )

                rco_alloc, history = search.optimize(
                    _forward_loss,
                    steps=steps,
                    gumbel_samples=gumbel_samples,
                    target_bytes=target_bytes,
                    seed=seed,
                    lr=lr,
                    sensitivity_logits=sens_logits,
                    on_step=_on_step,
                )
                rco_kl = eval_allocation_kl(rco_alloc.choices)
                swaps_vs_sens = sum(
                    1 for a, b in zip(rco_alloc.choices, sens_alloc.choices, strict=True) if a != b
                )
                allocations_out[f"{tag}_rco"] = {
                    "fraction": frac,
                    "choices": list(rco_alloc.choices),
                    "total_bytes": rco_alloc.total_bytes,
                    "search_kl": rco_kl,
                    "swaps_vs_sensitivity": swaps_vs_sens,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                log_event(
                    "RCO_BUDGET_DONE",
                    budget=tag,
                    target_bytes=target_bytes,
                    sens_kl=round(sens_kl, 6),
                    rco_kl=round(rco_kl, 6),
                    swaps_vs_sens=swaps_vs_sens,
                    elapsed_s=round(time.time() - t0, 2),
                )
                (bank_dir / f"history_{tag}.json").write_text(json.dumps(history, indent=2) + "\n")
    finally:
        restore_ckpt()
        for h in had_hooks:
            h.remove()

    out_path = bank_dir / "allocations.json"
    out_path.write_text(json.dumps(allocations_out, indent=2) + "\n")
    return allocations_out


def run_rco_cli(args: argparse.Namespace) -> None:
    """CLI entrypoint for `warp-quant rco` (builds bank, runs RCO search, and optionally assembles checkpoints)."""
    hub_id, info = resolve(args.model)
    base_dir = (
        Path(hub_id)
        if Path(hub_id).exists()
        else Path(resolve_snapshot(hub_id, revision=info.get("hub_revision")))
    )
    work_dir = Path(args.work_dir).resolve()
    bank_dir = work_dir / "bank"

    calib_rows, search_rows, ds = prepare_calib_and_search_rows(
        base_dir,
        info,
        calib_samples=args.calib_samples,
        search_samples=args.search_samples,
        calib_seq_len=args.calib_seq_len,
        search_seq_len=args.search_seq_len,
        seed=args.seed,
    )

    qat_train_rows: list[torch.Tensor] = []
    if getattr(args, "scale_qat", False):
        max_qat = int(getattr(args, "scale_qat_samples", 512))
        if max_qat <= 0:
            raise ValueError(f"--scale-qat-samples must be > 0, got {max_qat}")
        qat_indices = list(range(0, args.calib_samples)) + list(
            range(args.calib_samples + args.search_samples, len(ds))
        )
        for idx in qat_indices:
            if len(qat_train_rows) >= max_qat:
                break
            ids = torch.tensor(ds[idx]["input_ids"][: args.search_seq_len], dtype=torch.long)
            if ids.numel() >= 64:
                qat_train_rows.append(ids)
        if not qat_train_rows:
            raise ValueError(
                f"No non-search calibration rows of length >= 64 available for --scale-qat (dataset size {len(ds)})"
            )

    cls = model_class(str(base_dir))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Capture teacher hidden states + optional MTP/Vision GPTQ components BEFORE mutating model weights for bank
    model = cls.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map={"": device})
    model.eval()

    teacher_dir = work_dir / "teacher_cache"
    done_file = teacher_dir / "done.json"
    expected_meta = {
        "rows": len(search_rows),
        "seed": args.seed,
        "search_seq_len": args.search_seq_len,
        "row_lengths": [int(r.numel()) for r in search_rows],
    }
    cache_valid = False
    if done_file.is_file():
        try:
            cached_meta = json.loads(done_file.read_text())
            files_exist = all((teacher_dir / f"row_{i:04d}.pt").is_file() for i in range(len(search_rows)))
            cache_valid = (
                files_exist
                and int(cached_meta.get("rows", -1)) == len(search_rows)
                and cached_meta.get("row_lengths", expected_meta["row_lengths"])
                == expected_meta["row_lengths"]
            )
        except Exception:
            cache_valid = False
    if cache_valid:
        teacher_states = [
            torch.load(teacher_dir / f"row_{i:04d}.pt", map_location="cpu", weights_only=True)
            for i in range(len(search_rows))
        ]
        if any(t.shape[1] != r.numel() for t, r in zip(teacher_states, search_rows, strict=True)):
            cache_valid = False
    if not cache_valid:
        log_event("RCO_TEACHER_CAPTURE", rows=len(search_rows))
        teacher_paths = capture_teacher_hidden_states(model, search_rows, cache_dir=teacher_dir)
        done_file.write_text(json.dumps(expected_meta, indent=2) + "\n")
        teacher_states = [torch.load(p, map_location="cpu", weights_only=True) for p in teacher_paths]

    comp_dir = work_dir / "components"
    comp_dir.mkdir(parents=True, exist_ok=True)
    want_lm_head_w8 = bool(getattr(args, "quantize_lm_head_w8", True))
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    is_tied = bool(
        getattr(model.config, "tie_word_embeddings", False) or getattr(text_cfg, "tie_word_embeddings", False)
    )
    mtp_packed = None
    vision_packed = None
    lm_head_packed = None
    if want_lm_head_w8 and not is_tied:
        lm_file = comp_dir / "lm_head_w8.pt"
        if lm_file.is_file():
            lm_head_packed = torch.load(lm_file, map_location="cpu", weights_only=True)
        else:
            log_event("RCO_LM_HEAD_W8_CALIBRATE", rows=len(teacher_states))
            lm_head_packed = calibrate_lm_head_w8(
                model.lm_head.weight.detach(),
                teacher_states,
                group_size=64,
                dampening_frac=args.dampening_frac,
                device=device,
            )
            torch.save(lm_head_packed, lm_file)

    if args.quantize_mtp:
        mtp_file = comp_dir / "mtp_w4.pt"
        if mtp_file.is_file():
            mtp_packed = torch.load(mtp_file, map_location="cpu", weights_only=True)
        else:
            log_event("RCO_MTP_CALIBRATE", quantize_fc_w8=want_lm_head_w8)
            mtp_packed = _gptq_calibrate_mtp(
                model,
                base_dir,
                ds,
                group_size=64,
                hadamard_block_size=args.hadamard_block_size,
                actorder="static",
                dampening_frac=args.dampening_frac,
                max_calib_samples=64,
                quantize_fc_w8=want_lm_head_w8,
            )
            torch.save(mtp_packed, mtp_file)

    if args.quantize_vision:
        vis_file = comp_dir / "vision_w4.pt"
        if vis_file.is_file():
            vision_packed = torch.load(vis_file, map_location="cpu", weights_only=True)
        else:
            log_event("RCO_VISION_CALIBRATE")
            processor = AutoProcessor.from_pretrained(str(base_dir))
            vision_packed = _gptq_calibrate_vision(
                model,
                processor,
                Path(args.vision_calib_dir),
                group_size=64,
                hadamard_block_size=args.hadamard_block_size,
                actorder="static",
                dampening_frac=args.dampening_frac,
            )
            torch.save(vision_packed, vis_file)

    if not (bank_dir / "manifest.json").is_file():
        log_event("RCO_BANK_BUILD_START", model=args.model, options=args.options)
        build_rco_bank(
            model,
            calib_rows,
            args.options,
            bank_dir,
            hadamard_block_size=args.hadamard_block_size,
            dampening_frac=args.dampening_frac,
            propagate_option_idx=-1,
            rotate_weights_in_place=True,
        )
        # Reload fresh base weights for subsequent stages
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = cls.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map={"": device})
        model.eval()

    qat_teacher_states: list[torch.Tensor] | None = None
    if getattr(args, "scale_qat", False) and qat_train_rows:
        qat_cache_file = work_dir / "qat_teacher_states.pt"
        if qat_cache_file.is_file():
            log_event("RCO_QAT_TEACHER_CACHE_HIT", path=str(qat_cache_file))
            qat_teacher_states = torch.load(qat_cache_file, map_location="cpu", weights_only=True)
        else:
            log_event("RCO_QAT_TEACHER_CAPTURE", rows=len(qat_train_rows))
            qat_teacher_states = [
                t for t in capture_teacher_hidden_states(model, qat_train_rows) if isinstance(t, torch.Tensor)
            ]
            torch.save(qat_teacher_states, qat_cache_file)

    clean_bf16_backup: dict[str, torch.Tensor] = {}
    if getattr(args, "scale_qat", False):
        text_m_ref, text_pref = get_text_model(model)
        for sub_name, sub_mod in text_m_ref.named_modules():
            w = getattr(sub_mod, "weight", None)
            if not isinstance(w, nn.Parameter):
                continue
            leaf = sub_name.rsplit(".", 1)[-1]
            if "norm" in leaf.lower() or leaf in ("in_proj_a", "in_proj_b"):
                clean_bf16_backup[f"{text_pref}.{sub_name}.weight"] = (
                    w.detach().to(torch.bfloat16).cpu().clone()
                )

    only_all_high = (
        bool(args.assemble)
        and list(args.assemble) == ["all_high"]
        and all(float(b) <= 0.0 or float(b) >= 1.0 for b in args.budgets)
    ) or len(args.options) == 1

    active_bank_dir = bank_dir
    if getattr(args, "scale_qat", False) and not only_all_high:
        import shutil

        manifest, candidates = load_rco_bank(bank_dir)
        low_state_file = work_dir / "all_low_scale_qat_state.pt"
        if low_state_file.is_file():
            low_qat = torch.load(low_state_file, map_location="cpu", weights_only=True)
        else:
            low_qat = run_e2e_scale_qat(
                model,
                None,
                manifest,
                candidates,
                [0] * len(manifest["groups"]),
                train_rows=qat_train_rows,
                val_rows=search_rows[:32],
                val_teacher_states=teacher_states[:32],
                train_teacher_states=qat_teacher_states,
                num_epochs=1,
                batch_size=4,
                lr=1.2e-4,
                norm_lr=1.5e-4,
                anchor_wd=0.03,
                train_scales_max_bits=2,
                train_norms=True,
                train_gdn_gates=True,
                seed=args.seed,
            )
            torch.save(low_qat, low_state_file)

        active_bank_dir = work_dir / "bank_opt"
        if not (active_bank_dir / "manifest.json").is_file():
            if active_bank_dir.exists():
                shutil.rmtree(active_bank_dir)
            shutil.copytree(bank_dir, active_bank_dir)
            for stale in [
                active_bank_dir / "allocations.json",
                *active_bank_dir.glob("history_*.json"),
            ]:
                if stale.is_file():
                    stale.unlink()
            for layer_pt in sorted((active_bank_dir / "layers").glob("layer_*.pt")):
                l_data = torch.load(layer_pt, map_location="cpu", weights_only=True)
                for m_name, cand_list in l_data.items():
                    if m_name in low_qat["scale_overrides"]:
                        cand_list[0]["scales"] = low_qat["scale_overrides"][m_name].to(torch.bfloat16).cpu()
                torch.save(l_data, layer_pt)

        with torch.no_grad():
            for full_k, new_w in low_qat["bf16_overrides"].items():
                if full_k.endswith(".weight"):
                    mod = model.get_submodule(full_k[:-7])
                    mod.weight.copy_(new_w.to(device=mod.weight.device, dtype=mod.weight.dtype))

    high_kl_chunk = (
        1024 if (torch.cuda.is_available() and torch.cuda.mem_get_info(device)[0] > 40 * (1024**3)) else 128
    )
    if (active_bank_dir / "allocations.json").is_file():
        allocations = json.loads((active_bank_dir / "allocations.json").read_text())
    elif only_all_high:
        manifest_tmp = json.loads((active_bank_dir / "manifest.json").read_text())
        n_g = len(manifest_tmp["groups"])
        n_o = len(manifest_tmp["options"])
        allocations = {
            "all_low": {
                "fraction": 0.0,
                "choices": [0] * n_g,
                "total_bytes": int(manifest_tmp["all_low_bytes"]),
                "search_kl": 0.0,
            },
            "all_high": {
                "fraction": 1.0,
                "choices": [n_o - 1] * n_g,
                "total_bytes": int(manifest_tmp["all_high_bytes"]),
                "search_kl": 0.0,
            },
        }
        (active_bank_dir / "allocations.json").write_text(json.dumps(allocations, indent=2) + "\n")
    else:
        allocations = run_rco_search_on_bank(
            model,
            active_bank_dir,
            search_rows,
            args.budgets,
            steps=args.steps,
            gumbel_samples=args.gumbel_samples,
            lr=args.lr,
            seed=args.seed,
            teacher_states=teacher_states,
            kl_chunk_size=high_kl_chunk,
        )

    if args.assemble:
        to_assemble = args.assemble if "all" not in args.assemble else list(allocations.keys())
        manifest, opt_candidates = load_rco_bank(active_bank_dir)
        low_st_cached = (
            torch.load(work_dir / "all_low_scale_qat_state.pt", map_location="cpu", weights_only=True)
            if (work_dir / "all_low_scale_qat_state.pt").is_file()
            else None
        )

        for name in to_assemble:
            if name not in allocations:
                raise ValueError(f"Unknown allocation '{name}'; available: {sorted(allocations)}")
            out_ckpt = work_dir / "checkpoints" / name
            if out_ckpt.exists():
                print(f"[skip] {out_ckpt} already exists")
                continue

            scale_overrides = None
            bf16_overrides = None
            if getattr(args, "scale_qat", False) and not name.endswith("_sensitivity"):
                state_file = work_dir / f"{name}_scale_qat_state.pt"
                if name == "all_low" and low_st_cached is not None:
                    st = low_st_cached
                elif state_file.is_file():
                    st = torch.load(state_file, map_location="cpu", weights_only=True)
                else:
                    is_all_high = name == "all_high"
                    with torch.no_grad():
                        for full_k, orig_w in clean_bf16_backup.items():
                            mod = model.get_submodule(full_k[:-7])
                            mod.weight.copy_(orig_w.to(device=mod.weight.device, dtype=mod.weight.dtype))
                        if not is_all_high and low_st_cached is not None:
                            for full_k, new_w in low_st_cached["bf16_overrides"].items():
                                if full_k.endswith(".weight"):
                                    mod = model.get_submodule(full_k[:-7])
                                    mod.weight.copy_(
                                        new_w.to(device=mod.weight.device, dtype=mod.weight.dtype)
                                    )
                    n_qat = 512 if is_all_high else 384
                    st = run_e2e_scale_qat(
                        model,
                        None,
                        manifest,
                        opt_candidates,
                        allocations[name]["choices"],
                        train_rows=qat_train_rows[:n_qat],
                        val_rows=search_rows[:32],
                        val_teacher_states=teacher_states[:32],
                        train_teacher_states=(
                            qat_teacher_states[:n_qat] if qat_teacher_states is not None else None
                        ),
                        num_epochs=1,
                        batch_size=4,
                        lr=0.0 if is_all_high else 8e-5,
                        norm_lr=1.2e-4,
                        anchor_wd=0.04,
                        train_scales_max_bits=2,
                        train_norms=True,
                        train_gdn_gates=not is_all_high,
                        seed=args.seed,
                    )
                    torch.save(st, state_file)
                scale_overrides = st["scale_overrides"]
                bf16_overrides = st["bf16_overrides"]

            log_event("RCO_ASSEMBLE", allocation=name, out=str(out_ckpt))
            assemble_rco_checkpoint(
                base_dir=base_dir,
                bank_dir=active_bank_dir,
                choices=allocations[name]["choices"],
                output_dir=out_ckpt,
                model_key=args.model,
                allocation_name=name,
                mtp_packed=mtp_packed,
                vision_packed=vision_packed,
                lm_head_packed=lm_head_packed,
                scale_overrides=scale_overrides,
                bf16_overrides=bf16_overrides,
                quantize_lm_head_w8=want_lm_head_w8,
                component_group_size=64,
            )
            print(f"[ok] assembled {out_ckpt}")

    del model, qat_teacher_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
