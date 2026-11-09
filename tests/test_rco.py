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
