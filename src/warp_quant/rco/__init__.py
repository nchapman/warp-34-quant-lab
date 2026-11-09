"""Reusable H-RCO (Hadamard + Cholesky GPTQ + Riemannian Constrained Optimization) mixed-precision package."""

from warp_quant.rco.allocation import (
    Allocation,
    largest_reachable_budget,
    random_control,
    solve,
    upgrade_budget,
)
from warp_quant.rco.assemble import assemble_rco_checkpoint, build_rco_quantization_config
from warp_quant.rco.bank import build_rco_bank, discover_text_groups, load_rco_bank
from warp_quant.rco.hidden_teacher import capture_teacher_hidden_states, hidden_kl
from warp_quant.rco.options import (
    OPTIONS,
    QuantOption,
    decode_candidate,
    find_mse_scale,
    pack_candidate,
    packed_byte_cost,
    quantize_candidate,
    refine_hessian_cd,
    refine_hessian_cd_3d,
    resolve_option,
    resolve_options,
)
from warp_quant.rco.runner import run_rco_cli, run_rco_search_on_bank
from warp_quant.rco.search import SearchModel, build_sensitivity_logits

__all__ = [
    "OPTIONS",
    "Allocation",
    "QuantOption",
    "SearchModel",
    "assemble_rco_checkpoint",
    "build_rco_bank",
    "build_rco_quantization_config",
    "build_sensitivity_logits",
    "capture_teacher_hidden_states",
    "decode_candidate",
    "discover_text_groups",
    "find_mse_scale",
    "hidden_kl",
    "largest_reachable_budget",
    "load_rco_bank",
    "pack_candidate",
    "packed_byte_cost",
    "quantize_candidate",
    "random_control",
    "refine_hessian_cd",
    "refine_hessian_cd_3d",
    "resolve_option",
    "resolve_options",
    "run_rco_cli",
    "run_rco_search_on_bank",
    "solve",
    "upgrade_budget",
]
