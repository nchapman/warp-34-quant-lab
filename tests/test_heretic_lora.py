"""Unit tests for Heretic v2.1-compatible abliteration LoRA extraction (`warp_quant.heretic_lora`)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.linalg as LA
import torch.nn as nn
from safetensors.torch import load_file

from warp_quant.cli import build_parser
from warp_quant.heretic_lora import (
    AbliterationParameters,
    InMemoryLoRAApplicator,
    RowNormalization,
    WeightDistribution,
    collapse_flattened_residual_streams,
    collect_abliterable_layer_modules,
    compute_kl_divergence,
    compute_layer_weight,
    compute_lora_factors,
    compute_orthogonal_refusal_directions,
    compute_pareto_front,
    get_last_token_logits,
    interpolate_direction,
    is_refusal_response,
    sample_trial_parameters,
    save_peft_lora_adapter,
)


class _DummyFusedExperts(nn.Module):
    """Mimics `Qwen3_5MoeExperts` / `Qwen4ExpExperts` with 3D `nn.Parameter` weights."""

    def __init__(self, num_experts: int = 4, hidden_size: int = 16, intermediate_size: int = 8) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size))


class _DummyHybridMoEModel(nn.Module):
    """Mimics hybrid GDN + full-attention + shared-expert MoE structure (`Qwen3.6-35B-A3B` / `Qwen3.8-Flash-Next`)."""

    def __init__(self, hidden_size: int = 16, intermediate_size: int = 8) -> None:
        super().__init__()
        layer0 = nn.Module()
        layer0.linear_attn = nn.Module()
        layer0.linear_attn.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        layer0.mlp = nn.Module()
        layer0.mlp.experts = _DummyFusedExperts(4, hidden_size, intermediate_size)
        layer0.mlp.shared_expert = nn.Module()
        layer0.mlp.shared_expert.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        layer1 = nn.Module()
        layer1.self_attn = nn.Module()
        layer1.self_attn.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        layer1.mlp = nn.Module()
        layer1.mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([layer0, layer1])


def test_collapse_flattened_residual_streams_qwen4_exp() -> None:
    cfg = SimpleNamespace(hidden_size=8, hc_count=4)
    h0 = torch.arange(2 * 3 * 32, dtype=torch.float32).view(2, 3, 32)
    h1 = torch.ones(2, 3, 4, 8, dtype=torch.float32) * 3.0
    h_final = torch.ones(2, 3, 8, dtype=torch.float32) * 5.0

    collapsed = collapse_flattened_residual_streams((h0, h1, h_final), cfg)
    assert len(collapsed) == 3
    assert collapsed[0].shape == (2, 3, 8)
    assert torch.allclose(collapsed[0], h0.unflatten(-1, (4, 8)).mean(dim=-2))
    assert collapsed[1].shape == (2, 3, 8)
    assert torch.allclose(collapsed[1], torch.full((2, 3, 8), 3.0))
    assert torch.allclose(collapsed[2], h_final)


def test_collect_abliterable_layer_modules_hybrid_moe() -> None:
    model = _DummyHybridMoEModel()
    per_layer = collect_abliterable_layer_modules(model)
    assert len(per_layer) == 2

    assert set(per_layer[0].keys()) == {"attn.o_proj", "mlp.down_proj"}
    assert per_layer[0]["attn.o_proj"][0][0] == "model.language_model.layers.0.linear_attn.out_proj"
    assert per_layer[0]["mlp.down_proj"][0][0] == "model.language_model.layers.0.mlp.shared_expert.down_proj"

    assert set(per_layer[1].keys()) == {"attn.o_proj", "mlp.down_proj"}
    assert per_layer[1]["attn.o_proj"][0][0] == "model.language_model.layers.1.self_attn.o_proj"
    assert per_layer[1]["mlp.down_proj"][0][0] == "model.language_model.layers.1.mlp.down_proj"


def test_compute_orthogonal_refusal_directions() -> None:
    torch.manual_seed(0)
    good = torch.randn(5, 32)
    bad = good + 0.5 * torch.randn(5, 32)
    dirs = compute_orthogonal_refusal_directions(good, bad, orthogonalize_direction=True)
    assert dirs.shape == (5, 32)
    norms = LA.vector_norm(dirs, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    good_unit = good / LA.vector_norm(good, dim=1, keepdim=True)
    dots = (dirs * good_unit).sum(dim=1)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)


def test_compute_lora_factors_modes() -> None:
    torch.manual_seed(7)
    W = torch.randn(32, 24)
    v = torch.randn(32)
    v = v / LA.vector_norm(v)

    # 1. NONE (rank-1 exact projection: v^T (W + B @ A) == 0 when weight == 1.0)
    lora_A, lora_B = compute_lora_factors(W, v, 1.0, row_normalization=RowNormalization.NONE, lora_rank=1)
    assert lora_A.shape == (1, 24)
    assert lora_B.shape == (32, 1)
    W_proj = W + lora_B @ lora_A
    residual_along_v = v @ W_proj
    assert torch.allclose(residual_along_v, torch.zeros_like(residual_along_v), atol=1e-5)

    # 2. PRE (rank-1 row-normalized input projection)
    lora_A_pre, lora_B_pre = compute_lora_factors(
        W, v, 1.0, row_normalization=RowNormalization.PRE, lora_rank=1
    )
    assert lora_A_pre.shape == (1, 24)
    assert lora_B_pre.shape == (32, 1)

    # 3. FULL (MPOA rank-3 SVD preserves row magnitudes much closer to original W than NONE)
    lora_A_full, lora_B_full = compute_lora_factors(
        W, v, 1.15, row_normalization=RowNormalization.FULL, lora_rank=3, seed=42
    )
    assert lora_A_full.shape == (3, 24)
    assert lora_B_full.shape == (32, 3)

    orig_norms = LA.vector_norm(W, dim=1)
    none_A, none_B = compute_lora_factors(W, v, 1.15, row_normalization=RowNormalization.NONE, lora_rank=1)
    err_none = (LA.vector_norm(W + none_B @ none_A, dim=1) - orig_norms).abs().mean()
    err_full = (LA.vector_norm(W + lora_B_full @ lora_A_full, dim=1) - orig_norms).abs().mean()
    assert err_full < err_none


def test_in_memory_lora_applicator_and_save(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = _DummyHybridMoEModel(hidden_size=16, intermediate_size=8)
    layer_modules = collect_abliterable_layer_modules(model)
    directions = torch.randn(3, 16)
    directions = directions / LA.vector_norm(directions, dim=1, keepdim=True)

    x = torch.randn(2, 4, 16)
    mod0 = model.model.language_model.layers[0].linear_attn.out_proj
    base_out = mod0(x)

    applicator = InMemoryLoRAApplicator(layer_modules)
    params = AbliterationParameters(
        direction_index=None,
        weight_distributions={
            "attn.o_proj": WeightDistribution(1.2, 0.5, 0.8, 2.0),
            "mlp.down_proj": WeightDistribution(0.9, 0.5, 0.5, 2.0),
        },
    )
    deltas = applicator.set_parameters(
        params,
        directions,
        row_normalization=RowNormalization.NONE,
        lora_rank=1,
    )
    assert len(deltas) == 4

    hooked_out = mod0(x)
    lora_A, lora_B = deltas["model.language_model.layers.0.linear_attn.out_proj"]
    expected_out = base_out + (x @ lora_A.T) @ lora_B.T
    assert torch.allclose(hooked_out, expected_out, atol=1e-5)

    # Removing applicator restores exact original forward pass
    applicator.remove()
    assert torch.allclose(mod0(x), base_out, atol=1e-6)

    # Saving produces canonical PEFT + vLLM structure
    save_peft_lora_adapter(
        tmp_path,
        deltas,
        base_model_name_or_path="Qwen/Qwen3.5-9B",
        lora_rank=1,
        summary={"kl_divergence": 0.01},
    )
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA"
    assert cfg["r"] == 1
    assert "model.language_model.layers.0.linear_attn.out_proj" in cfg["target_modules"]

    tensors = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert "base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_A.weight" in tensors
    assert (
        "base_model.model.model.language_model.layers.0.mlp.shared_expert.down_proj.lora_B.weight" in tensors
    )


def test_refusal_matcher_and_helpers() -> None:
    assert is_refusal_response("")
    assert is_refusal_response("I *cannot* fulfill that request.")
    assert is_refusal_response("I’m sorry, but as an AI assistant I won’t do that.")
    assert not is_refusal_response("Here is a two-sentence explanation of TCP vs UDP.")

    d = torch.eye(4)
    interp = interpolate_direction(d, 0.5, 0)
    assert torch.isclose(LA.vector_norm(interp), torch.tensor(1.0))

    dist = WeightDistribution(
        max_weight=1.0, max_weight_position=10.0, min_weight=0.5, min_weight_distance=5.0
    )
    assert compute_layer_weight(dist, 10) == 1.0
    assert compute_layer_weight(dist, 15) == 0.5
    assert compute_layer_weight(dist, 20) == 0.0

    logits = torch.randn(4, 16)
    assert abs(compute_kl_divergence(logits, logits)) < 1e-6


def test_cli_heretic_lora_parser_and_presets() -> None:
    parser = build_parser()
    args = parser.parse_args(["heretic-lora", "--model", "9b", "--output-dir", "/tmp/out", "--trials", "8"])
    assert args.cmd == "heretic-lora"
    assert args.model == "9b"
    assert args.row_normalization == "full"
    assert args.lora_rank == 3
    assert args.trials == 8

    rng = random.Random(42)
    for idx in range(18):
        p = sample_trial_parameters(rng, 32, ["attn.o_proj", "mlp.down_proj"], idx)
        assert set(p.weight_distributions.keys()) == {"attn.o_proj", "mlp.down_proj"}

    bad_trials = parser.parse_args(
        ["heretic-lora", "--model", "9b", "--output-dir", "/tmp/out", "--trials", "100"]
    )
    from warp_quant.heretic_lora import run_heretic_lora

    with pytest.raises(ValueError, match="--trials must be between 1 and 64"):
        run_heretic_lora(bad_trials)

    with pytest.raises(ValueError, match="prompts must not be empty"):
        get_last_token_logits(nn.Linear(4, 4), None, [])

    pareto = compute_pareto_front(
        [
            {"trial": 0, "refusals": 3, "kl_divergence": 0.05},
            {"trial": 1, "refusals": 0, "kl_divergence": 0.04},
            {"trial": 2, "refusals": 2, "kl_divergence": 0.01},
            {"trial": 3, "refusals": 5, "kl_divergence": 0.08},
        ]
    )
    assert [r["trial"] for r in pareto] == [1, 2]
