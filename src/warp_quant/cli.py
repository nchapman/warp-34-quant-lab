"""Unified warp-quant CLI for W4A16-HGPTQ quantization, fidelity probing, and fixture setup."""

from __future__ import annotations

import argparse
from pathlib import Path

from warp_quant.data import prepare_v2_vision_split
from warp_quant.hadamard_gptq import run_probe, run_quantize
from warp_quant.lablog import log_event, run_done, run_fail
from warp_quant.models import MODELS, quant_recipe

ROOT = Path(__file__).resolve().parents[2]


def _run_list_models() -> None:
    print(f"{'KEY':<18} {'ARCH':<14} {'UPSTREAM HUB':<42} {'TARGET QUANT REPO'}")
    print("-" * 120)
    for key, info in MODELS.items():
        print(f"{key:<18} {info.get('arch', 'qwen3_5'):<14} {info['hub']:<42} {info.get('quant_repo', '-')}")
        rec = quant_recipe(info)
        print(
            f"  recipe: g{rec['group_size']}-h{rec['hadamard_block_size']} actorder={rec['actorder']} "
            f"calib={rec['calib']} ({rec['num_samples']}x{rec['max_seq_length']}) "
            f"mtp={rec['quantize_mtp']} vision={rec['quantize_vision']}"
        )


def _run_build_vision_data(args: argparse.Namespace) -> None:
    base_dir = Path(args.data_dir)
    if not base_dir.is_absolute():
        base_dir = ROOT / base_dir
    splits = (
        [("vision-calib-v2", "calibration"), ("vision-probe-v2", "probe"), ("vision-guard-v2", "guard")]
        if args.split == "all"
        else [(f"vision-{args.split}-v2", "calibration" if args.split == "calib" else args.split)]
    )
    for folder, split_name in splits:
        target = base_dir / folder
        if target.exists():
            print(f"[skip] {target} already exists")
            continue
        log_event("VISION_DATA_BUILD", target=str(target), split=split_name)
        prepare_v2_vision_split(target, split=split_name)
        print(f"[ok] built {target} ({split_name})")
    run_done(data_dir=str(base_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warp-quant",
        description="W4A16-HGPTQ (Hadamard-rotated Cholesky GPTQ + 4-bit Custom-LUT4 PLE) quantization & probing.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-models", help="List supported Qwen3.5 / 3.6 / 3.8 models and default recipes")

    b = sub.add_parser(
        "build-vision-data",
        help="Materialize disjoint vision-calib-v2 (120), vision-probe-v2 (96), and vision-guard-v2 (48) fixtures",
    )
    b.add_argument("--data-dir", default="data", help="Parent directory for vision-*-v2 folders")
    b.add_argument(
        "--split",
        choices=["all", "calib", "probe", "guard"],
        default="all",
        help="Which v2 vision fixture split(s) to materialize",
    )

    q = sub.add_parser(
        "quantize", help="Quantize a model with W4A16-HGPTQ (Text + Shared/Routed MoE + MTP + Vision + PLE)"
    )
    q.add_argument("--model", default="35b", help="Model key from `warp-quant list-models` or HF repo/path")
    q.add_argument("--output-dir", required=True, help="Destination checkpoint directory")
    q.add_argument("--group-size", type=int, choices=[32, 64, 128], default=64)
    q.add_argument(
        "--hadamard-block-size",
        type=int,
        default=64,
        help="0 for unrotated GPTQ; 32/64/128/256 for online Walsh-Hadamard input rotation",
    )
    q.add_argument(
        "--ple-group-size",
        type=int,
        choices=[0, 32, 64, 128],
        default=32,
        help="Group size for 4-bit Custom-LUT4 W_kv-HGPTQ Layer-1 PLE tables (0 to keep BF16)",
    )
    q.add_argument(
        "--ple-cd-passes",
        type=int,
        default=1,
        help="Coordinate-descent centroid refinement passes for 4-bit Custom-LUT4 PLE",
    )
    q.add_argument("--actorder", choices=["none", "weight", "static"], default="static")
    q.add_argument("--dampening-frac", type=float, default=0.01)
    q.add_argument(
        "--quantize-mtp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize mtp.layers.0.* projections to W4A16-HGPTQ (default: enabled)",
    )
    q.add_argument(
        "--quantize-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize model.visual.* linear layers to W4A16-HGPTQ (default: enabled)",
    )
    q.add_argument(
        "--quantize-lm-head-w8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize untied lm_head (and mtp.fc when MTP is quantized) to unrotated W8_g64 GPTQ+MSE (default: enabled)",
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
        help="Optional existing checkpoint dir to reuse unchanged PLE/Vision/embed shards from",
    )
    q.add_argument(
        "--save-rco-bank-dir",
        default=None,
        help="Optional directory to save per-layer W2 routed-expert candidates + teacher QAT states during MoE streaming quantize",
    )

    p = sub.add_parser(
        "probe", help="Evaluate text KL + Native MTP + Vision fidelity on one or more quantized checkpoints"
    )
    p.add_argument("--model", default="35b")
    p.add_argument("--quant", nargs="+", required=True, help="One or more quantized checkpoint directories")
    p.add_argument("--eval-sets", nargs="+", default=["data/probe-v4.jsonl", "data/probe-guard.jsonl"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-mtp", action="store_true")
    p.add_argument("--mtp-per-domain", type=int, default=2)
    p.add_argument(
        "--eval-vision", default=None, help="Path to vision probe fixture dir (e.g. data/vision-probe-v2)"
    )
    p.add_argument(
        "--eval-vision-guard",
        default=None,
        help="Path to OOD vision guard fixture dir (e.g. data/vision-guard-v2)",
    )
    p.add_argument("--dump-rows-dir", default=None)
    p.add_argument("--output-json", default=None)
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write probe_<ckpt_name>.json when probing multiple --quant checkpoints",
    )

    r = sub.add_parser(
        "rco",
        help="Build a multi-option HGPTQ candidate bank, run Riemannian Constrained Optimization (H-RCO), and assemble checkpoints",
    )
    r.add_argument("--model", default="2b", help="Model key from `warp-quant list-models` or HF repo/path")
    r.add_argument(
        "--work-dir",
        required=True,
        help="Working directory for bank, teacher cache, and assembled checkpoints",
    )
    r.add_argument(
        "--options",
        nargs="+",
        default=["w2_g64", "w4_g64"],
        help="Ordered QuantOption names (e.g. w2_g64 w4_g64 or w2_g128 w4_g64 w8_g64)",
    )
    r.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[0.15, 0.25, 0.30],
        help="Upgrade budget fractions in (0, 1) between all_low and all_high",
    )
    r.add_argument("--hadamard-block-size", type=int, default=64)
    r.add_argument("--dampening-frac", type=float, default=0.01)
    r.add_argument("--calib-samples", type=int, default=128)
    r.add_argument("--search-samples", type=int, default=64)
    r.add_argument("--calib-seq-len", type=int, default=2048)
    r.add_argument("--search-seq-len", type=int, default=1024)
    r.add_argument("--steps", type=int, default=120)
    r.add_argument("--gumbel-samples", type=int, default=4)
    r.add_argument("--lr", type=float, default=0.1)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument(
        "--quantize-mtp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calibrate and include W4A16-HGPTQ Native MTP in assembled checkpoints",
    )
    r.add_argument(
        "--quantize-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calibrate and include W4A16-HGPTQ Vision in assembled checkpoints",
    )
    r.add_argument("--vision-calib-dir", default="data/vision-calib-v2")
    r.add_argument(
        "--quantize-lm-head-w8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize untied lm_head (and mtp.fc when MTP is quantized) to unrotated W8_g64 GPTQ+MSE (default: enabled)",
    )
    r.add_argument(
        "--scale-qat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run end-to-end continuous weight_scale and FP32-master RMSNorm KL tuning before/after RCO search",
    )
    r.add_argument(
        "--scale-qat-samples",
        type=int,
        default=512,
        help="Number of non-overlapping calibration sequences for E2E Scale+Norm KL tuning",
    )
    r.add_argument(
        "--assemble",
        nargs="*",
        default=["all"],
        help="Allocation names to assemble into checkpoints (or 'all' for all solved allocations)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "list-models":
            _run_list_models()
        elif args.cmd == "build-vision-data":
            _run_build_vision_data(args)
        elif args.cmd == "quantize":
            run_quantize(args)
        elif args.cmd == "probe":
            run_probe(args)
        elif args.cmd == "rco":
            from warp_quant.rco.runner import run_rco_cli

            run_rco_cli(args)
    except BaseException as exc:
        run_fail(error=str(exc))
        raise


if __name__ == "__main__":
    main()
