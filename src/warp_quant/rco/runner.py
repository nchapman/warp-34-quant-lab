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
from warp_quant.rco.assemble import assemble_rco_checkpoint
from warp_quant.rco.bank import (
    attach_online_hadamard_hooks,
    build_rco_bank,
    get_text_model,
    load_rco_bank,
)
from warp_quant.rco.hidden_teacher import capture_teacher_hidden_states, hidden_kl
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
    # Only wrap decoder layers in activation checkpointing on large models (>14B) where activations need it
    use_layer_ckpt = int(manifest["all_high_bytes"]) > 6 * (1024**3)
    restore_ckpt = _enable_eval_layer_checkpointing(text_model) if use_layer_ckpt else (lambda: None)
    try:
        with SearchModel(model, group_members, candidates, costs) as search:

            @torch.no_grad()
            def eval_allocation_kl(choices: tuple[int, ...] | list[int], max_eval_rows: int = 32) -> float:
                search.set_assignment(choices, soft=None)
                kls = []
                for idx in range(min(len(search_rows), max_eval_rows)):
                    inp = search_rows[idx].to(device=device, dtype=torch.long).unsqueeze(0)
                    student_h = text_model(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state
                    kl_val = hidden_kl(student_h, teacher_states[idx], model.lm_head, checkpointed=False)
                    kls.append(float(kl_val.item()))
                return sum(kls) / max(len(kls), 1)

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
                    return hidden_kl(student_h, teacher_states[row_i], model.lm_head, checkpointed=True)

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
    mtp_packed = None
    vision_packed = None
    if args.quantize_mtp:
        mtp_file = comp_dir / "mtp_w4.pt"
        if mtp_file.is_file():
            mtp_packed = torch.load(mtp_file, map_location="cpu", weights_only=True)
        else:
            log_event("RCO_MTP_CALIBRATE")
            mtp_packed = _gptq_calibrate_mtp(
                model,
                base_dir,
                ds,
                group_size=64,
                hadamard_block_size=args.hadamard_block_size,
                actorder="static",
                dampening_frac=args.dampening_frac,
                max_calib_samples=64,
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
        # Reload fresh base weights for search if needed (SearchModel replaces weights via parametrization anyway)

    allocations = run_rco_search_on_bank(
        model,
        bank_dir,
        search_rows,
        args.budgets,
        steps=args.steps,
        gumbel_samples=args.gumbel_samples,
        lr=args.lr,
        seed=args.seed,
        teacher_states=teacher_states,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.assemble:
        to_assemble = args.assemble if "all" not in args.assemble else list(allocations.keys())
        for name in to_assemble:
            if name not in allocations:
                raise ValueError(f"Unknown allocation '{name}'; available: {sorted(allocations)}")
            out_ckpt = work_dir / "checkpoints" / name
            if out_ckpt.exists():
                print(f"[skip] {out_ckpt} already exists")
                continue
            log_event("RCO_ASSEMBLE", allocation=name, out=str(out_ckpt))
            assemble_rco_checkpoint(
                base_dir=base_dir,
                bank_dir=bank_dir,
                choices=allocations[name]["choices"],
                output_dir=out_ckpt,
                model_key=args.model,
                allocation_name=name,
                mtp_packed=mtp_packed,
                vision_packed=vision_packed,
                component_group_size=64,
            )
            print(f"[ok] assembled {out_ckpt}")
