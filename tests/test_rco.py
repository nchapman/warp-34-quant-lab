"""Unit tests for the reusable H-RCO mixed-precision package (src/warp_quant/rco/)."""

from __future__ import annotations

import torch
import torch.nn as nn
from compressed_tensors.compressors import PackedQuantizationCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from warp_quant.rco import (
    SearchModel,
    build_rco_quantization_config,
    build_sensitivity_logits,
    decode_candidate,
    find_mse_scale,
    hidden_kl,
    pack_candidate,
    packed_byte_cost,
    quantize_candidate,
    resolve_options,
    solve,
    upgrade_budget,
)


def test_resolve_options_and_packed_byte_cost():
    opts = resolve_options(["w2_g64", "w4_g64", "w8_g64"])
    assert [o.num_bits for o in opts] == [2, 4, 8]
    assert [o.bpw for o in opts] == [2.25, 4.25, 8.25]
    # For (128, 256):
    # w2_g64: packed = 128 * (256 // 16) * 4 = 8192; scales = 128 * (256 // 64) * 2 = 1024; shape = 16 -> 9232
    assert packed_byte_cost((128, 256), opts[0]) == 8192 + 1024 + 16
    assert packed_byte_cost((128, 256), opts[1]) == 16384 + 1024 + 16


def test_find_mse_scale_improves_w2_mse_over_minmax():
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float32) * 0.03
    # Inject a heavy tail per group
    w[:, ::64] *= 3.5
    s_mse = find_mse_scale(w, num_bits=2, group_size=64)
    s_mm = (w.view(64, 4, 64).abs().amax(dim=-1) / 1.0).clamp(min=1e-7).to(torch.bfloat16)

    def _err(scale):
        sc = scale.float().unsqueeze(-1)
        q = torch.clamp(torch.round(w.view(64, 4, 64) / sc), -2, 1) * sc
        return (w.view(64, 4, 64) - q).square().mean().item()

    assert _err(s_mse) < _err(s_mm) * 0.65


def test_pack_and_decode_candidate_roundtrip_matches_compressed_tensors():
    torch.manual_seed(1)
    for bits, gsize in [(2, 64), (2, 128), (4, 64), (8, 64)]:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        codes = torch.randint(qmin, qmax + 1, (64, 128), dtype=torch.int8)
        scales = torch.rand(64, 128 // gsize, dtype=torch.bfloat16) * 0.05 + 0.002
        decoded = decode_candidate(codes, scales, gsize, dtype=torch.bfloat16)
        packed = pack_candidate(codes, scales, bits)
        scheme = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=bits,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=gsize,
            ),
        )
        ct_deq = PackedQuantizationCompressor.decompress(packed, scheme)["weight"]
        assert torch.equal(decoded, ct_deq)


def test_multi_option_knapsack_allocation_exact_bytes():
    # 4 groups, 3 options per group (e.g. W2, W4, W8)
    costs = [
        (1000, 2000, 4000),
        (1500, 3000, 6000),
        (2000, 4000, 8000),
        (1000, 2000, 4000),
    ]
    b_info = upgrade_budget(costs, 0.25)
    target = int(b_info["realized_bytes"])
    scores = [
        (0.0, 1.0, 1.5),
        (0.0, 5.0, 6.0),
        (0.0, 2.0, 2.5),
        (0.0, 0.5, 0.8),
    ]
    alloc = solve(costs, scores, target, exact=True)
    assert alloc.total_bytes == target
    assert sum(costs[g][c] for g, c in enumerate(alloc.choices)) == target


def test_compact_search_model_ste_and_riemannian_optimization():
    torch.manual_seed(42)

    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            self.l2 = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            self.l3 = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            self.head = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            return self.l3(torch.relu(self.l2(self.l1(x))))

    model = TinyBlock().eval()
    groups = [["l1"], ["l2"], ["l3"]]
    opts = resolve_options(["w2_g64", "w4_g64"])
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        teacher_h = model(x)

    h_eye = torch.eye(64, dtype=torch.float32)
    candidates = {}
    costs = []
    for grp in groups:
        m = grp[0]
        w = model.get_submodule(m).weight.data
        c_low = quantize_candidate(w, h_eye, 1, opts[0])
        c_high = quantize_candidate(w, h_eye, 1, opts[1])
        candidates[m] = [c_low, c_high]
        costs.append((int(c_low["bytes"]), int(c_high["bytes"])))

    sens = build_sensitivity_logits(groups, candidates, costs)
    # Budget allowing 1 of the 3 groups to upgrade to w4_g64
    target_bytes = costs[0][1] + costs[1][0] + costs[2][0]

    with SearchModel(model, groups, candidates, costs) as search:
        final_alloc, history = search.optimize(
            lambda m, _step, _samp: hidden_kl(m(x), teacher_h, m.head, checkpointed=True),
            steps=6,
            gumbel_samples=2,
            target_bytes=target_bytes,
            seed=0,
            lr=0.1,
            sensitivity_logits=sens,
        )
    assert final_alloc.total_bytes == target_bytes
    assert len(history) == 6
    assert all(abs(rec["soft_budget_error_bytes"]) < 1e-3 for rec in history)


def test_build_rco_quantization_config_unanchored_regexes():
    manifest = {
        "options": [
            {"name": "w2_g64", "num_bits": 2, "group_size": 64},
            {"name": "w4_g64", "num_bits": 4, "group_size": 64},
        ],
        "hadamard_block_size": 64,
        "groups": [
            {
                "group_idx": 0,
                "layer_idx": 0,
                "members": [
                    "model.language_model.layers.0.self_attn.q_proj",
                    "model.language_model.layers.0.self_attn.k_proj",
                ],
            },
            {
                "group_idx": 1,
                "layer_idx": 0,
                "members": ["model.language_model.layers.0.mlp.down_proj"],
            },
        ],
    }
    qcfg = build_rco_quantization_config(
        {"vision_config": {"intermediate_size": 4304}},
        manifest,
        choices=[0, 1],
        quantize_mtp=True,
        quantize_vision=True,
        vision_fc2_channel=True,
    )
    assert "w2_g64" in qcfg["config_groups"]
    assert "w4_g64" in qcfg["config_groups"]
    assert "group_0" in qcfg["config_groups"]
    assert "vision_fc2_channel" in qcfg["config_groups"]
    assert qcfg["config_groups"]["w2_g64"]["weights"]["num_bits"] == 2
    assert qcfg["config_groups"]["w4_g64"]["weights"]["num_bits"] == 4
    assert qcfg["config_groups"]["w2_g64"]["targets"] == [
        r"re:.*(?<!mtp\.)layers\.0\.self_attn\.q_proj$",
        r"re:.*(?<!mtp\.)layers\.0\.self_attn\.k_proj$",
    ]


def test_refine_hessian_cd_monotonically_decreases_objective() -> None:
    from warp_quant.rco.options import (
        QuantOption,
        decode_candidate,
        quantize_candidate,
    )

    torch.manual_seed(7)
    w = torch.randn(64, 64, dtype=torch.bfloat16)
    x = torch.randn(128, 64, dtype=torch.float32)
    h = (x.T @ x) / 128.0
    opt = QuantOption("w2_g64", num_bits=2, group_size=64)

    res_no_cd = quantize_candidate(w, h, 128, opt, cd_passes=0)
    res_cd = quantize_candidate(w, h, 128, opt, cd_passes=2)
    assert float(res_cd["hessian_loss"]) <= float(res_no_cd["hessian_loss"]) + 1e-6
    q_cd = decode_candidate(res_cd["codes"], res_cd["scales"], 64)
    assert q_cd.shape == (64, 64)


def test_w2_sym_half_integer_option_beats_asymmetric_int2_on_gaussian() -> None:
    from warp_quant.rco.options import decode_candidate, quantize_candidate, resolve_option

    torch.manual_seed(19)
    w = torch.randn(64, 128, dtype=torch.bfloat16)
    x = torch.randn(256, 128, dtype=torch.float32)
    h = x.T @ x

    opt_int2 = resolve_option("w2_g64")
    opt_sym2 = resolve_option("w2_g64_sym")
    assert opt_sym2.code_offset == 0.5

    res_int2 = quantize_candidate(w, h, 256, opt_int2)
    res_sym2 = quantize_candidate(w, h, 256, opt_sym2)
    assert float(res_sym2["hessian_loss"]) < float(res_int2["hessian_loss"])

    dec = decode_candidate(res_sym2["codes"], res_sym2["scales"], 64, code_offset=0.5)
    assert dec.shape == (64, 128)


def test_apply_rco_code_offsets_from_shard_and_validation(tmp_path) -> None:
    import json

    import pytest
    import torch.nn as nn
    from safetensors.torch import save_file

    from warp_quant.hadamard_gptq import _apply_rco_code_offsets

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)

    m = _TinyModel()
    torch.nn.init.zeros_(m.lin.weight)
    scales = torch.full((16, 1), 0.25, dtype=torch.bfloat16)
    save_file({"lin.weight_scale": scales}, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lin.weight_scale": "model-00001-of-00001.safetensors"}})
    )
    (tmp_path / "rco_manifest.json").write_text(
        json.dumps({"code_offsets": {"lin": {"code_offset": 0.5, "group_size": 64}}})
    )

    _apply_rco_code_offsets(m, tmp_path)
    assert torch.allclose(m.lin.weight, torch.full_like(m.lin.weight, 0.125))

    # Reject out-of-range / NaN code_offset or invalid group_size
    (tmp_path / "rco_manifest.json").write_text(
        json.dumps({"code_offsets": {"lin": {"code_offset": 99.0, "group_size": 64}}})
    )
    with pytest.raises(ValueError, match="Invalid code_offset"):
        _apply_rco_code_offsets(m, tmp_path)


def test_refine_hessian_cd_3d_matches_2d_and_pack_tensor_channel() -> None:
    from warp_quant.hadamard_gptq import _gptq_pack_tensor
    from warp_quant.rco.options import refine_hessian_cd, refine_hessian_cd_3d
    from warp_quant.streaming_moe import _gptq_quantize_and_pack_3d_experts

    torch.manual_seed(23)
    w_3d = torch.randn(3, 32, 64, dtype=torch.float32)
    x_3d = torch.randn(3, 96, 64, dtype=torch.float32)
    h_3d = torch.bmm(x_3d.transpose(1, 2), x_3d) / 96.0
    codes_3d = torch.randint(-8, 8, (3, 32, 64), dtype=torch.int8)
    scales_3d = (w_3d.abs().amax(dim=-1, keepdim=True) / 7.0).to(torch.bfloat16)

    c_batched, s_batched = refine_hessian_cd_3d(w_3d, h_3d, codes_3d, scales_3d, 4, 64, passes=2)
    for e in range(3):
        c_e, s_e = refine_hessian_cd(w_3d[e], h_3d[e], codes_3d[e], scales_3d[e], 4, 64, passes=2)
        assert torch.equal(c_batched[e], c_e)
        assert torch.equal(s_batched[e], s_e)

    # 3D MoE expert packer (shared H and per-expert H)
    q_shared, p_shared, s_shared, shp = _gptq_quantize_and_pack_3d_experts(
        w_3d.to(torch.bfloat16), h_3d[:1], group_size=64, actorder="static", dampening_frac=0.01
    )
    assert q_shared.shape == (3, 32, 64)
    assert p_shared.shape == (3, 32, 8)
    assert s_shared.shape == (3, 32, 1)
    assert shp.tolist() == [32, 64]

    q_per_exp, p_per_exp, _, _ = _gptq_quantize_and_pack_3d_experts(
        w_3d.to(torch.bfloat16), h_3d, group_size=64, actorder="static", dampening_frac=0.01
    )
    assert q_per_exp.shape == (3, 32, 64)
    assert p_per_exp.shape == (3, 32, 8)

    # Per-channel Vision fc2 (non-power-of-2 K=80 = 16 * 5)
    w_fc2 = torch.randn(32, 80, dtype=torch.bfloat16)
    x_fc2 = torch.randn(128, 80, dtype=torch.float32)
    packed_fc2 = _gptq_pack_tensor(
        w_fc2,
        x_fc2.T @ x_fc2,
        torch.tensor(128),
        strategy="channel",
        group_size=None,
        actorder=None,
        dampening_frac=0.01,
    )
    assert packed_fc2["weight_packed"].shape == (32, 10)
    assert packed_fc2["weight_scale"].shape == (32, 1)


def test_scale_qat_with_cached_teacher_states_and_multi_quant_cli(tmp_path) -> None:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    from warp_quant.rco.bank import build_rco_bank, load_rco_bank
    from warp_quant.rco.hidden_teacher import capture_teacher_hidden_states
    from warp_quant.rco.scale_qat import run_e2e_scale_qat

    torch.manual_seed(11)
    cfg = Qwen3_5Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 1, 1],
            },
        }
    )
    teacher = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    student = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    student.load_state_dict(teacher.state_dict())

    rows = [
        torch.randint(2, 120, (12,), dtype=torch.long),
        torch.randint(2, 120, (20,), dtype=torch.long),
    ]
    cached_t = [t for t in capture_teacher_hidden_states(teacher, rows) if isinstance(t, torch.Tensor)]
    assert len(cached_t) == 2 and cached_t[0].shape == (1, 12, 64) and cached_t[1].shape == (1, 20, 64)

    bank_dir = tmp_path / "bank"
    build_rco_bank(
        teacher,
        rows,
        ["w2_g64", "w4_g64"],
        bank_dir,
        hadamard_block_size=64,
        rotate_weights_in_place=True,
    )
    manifest, cands = load_rco_bank(bank_dir)
    assert manifest["calib_rows"] == 2
    assert manifest["calib_tokens"] == 32
    assert len(manifest["groups"]) == 4

    res = run_e2e_scale_qat(
        student,
        None,
        manifest,
        cands,
        [0] * len(manifest["groups"]),
        train_rows=rows,
        val_rows=rows,
        val_teacher_states=cached_t,
        train_teacher_states=cached_t,
        num_epochs=1,
        batch_size=2,
        lr=1e-3,
    )
    assert res["best_val_kl"] <= res["init_val_kl"] + 1e-5
