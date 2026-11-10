"""Unit tests for quants/rot_experiment.py (CPU-only)."""

import json
import re
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from warp_quant.hadamard_gptq import (
    MTP_HAD_TARGET_RE,
    MTP_TARGET_RE,
    TEXT_HAD_TARGET_RE,
    TEXT_TARGET_RE,
    VISION_FC2_TARGET_RE,
    VISION_GROUP_TARGET_RE,
    _assert_packed_alignment,
    _clean_and_augment_shards,
    _gptq_pack_tensor,
    _hadamard_rotate_weight,
    _pack_w4_tensor,
    _update_config_for_components,
)


def test_target_regexes_match_hf_and_vllm_names():
    text_pat = re.compile(TEXT_TARGET_RE.removeprefix("re:"))
    text_had_pat = re.compile(TEXT_HAD_TARGET_RE.removeprefix("re:"))
    mtp_pat = re.compile(MTP_TARGET_RE.removeprefix("re:"))
    mtp_had_pat = re.compile(MTP_HAD_TARGET_RE.removeprefix("re:"))
    vis_grp_pat = re.compile(VISION_GROUP_TARGET_RE.removeprefix("re:"))
    vis_fc2_pat = re.compile(VISION_FC2_TARGET_RE.removeprefix("re:"))

    # HF + vLLM text target names must match TEXT_TARGET_RE / TEXT_HAD_TARGET_RE and not MTP/Vision
    for name in (
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.3.linear_attn.in_proj_qkv",
        "model.language_model.layers.3.linear_attn.in_proj_z",
        "model.language_model.layers.3.linear_attn.out_proj",
        "model.language_model.layers.3.mlp.down_proj",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj",
        "model.language_model.layers.0.mlp.shared_expert.up_proj",
        "model.language_model.layers.0.mlp.shared_expert.down_proj",
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
        "model.language_model.layers.39.mlp.experts.255.down_proj",
        "language_model.model.layers.0.self_attn.k_proj",
        "model.layers.0.mlp.gate_proj",
    ):
        assert text_pat.match(name), f"Expected TEXT_TARGET_RE to match {name}"
        assert text_had_pat.match(name), f"Expected TEXT_HAD_TARGET_RE to match {name}"
        assert not mtp_pat.match(name)
        assert not vis_grp_pat.match(name)

    # Excluded layers must not match TEXT_TARGET_RE
    for excluded in (
        "lm_head",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.mlp.gate",
        "model.language_model.layers.0.mlp.shared_expert_gate",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.mlp.experts.0.gate_proj",
        "mtp.fc",
        "model.visual.blocks.0.attn.qkv",
    ):
        assert not text_pat.match(excluded), f"Expected TEXT_TARGET_RE to reject {excluded}"

    # MTP target names
    for name in (
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.mlp.down_proj",
        "mtp.layers.0.mlp.shared_expert.gate_proj",
        "mtp.layers.0.mlp.shared_expert.down_proj",
        "mtp.layers.0.mlp.experts.0.gate_proj",
        "mtp.layers.0.mlp.experts.255.down_proj",
        "model.mtp.layers.0.self_attn.o_proj",
    ):
        assert mtp_pat.match(name), f"Expected MTP_TARGET_RE to match {name}"
        assert mtp_had_pat.match(name), f"Expected MTP_HAD_TARGET_RE to match {name}"
    for excluded_mtp in (
        "mtp.fc",
        "mtp.layers.0.mlp.gate",
        "mtp.layers.0.mlp.shared_expert_gate",
    ):
        assert not mtp_pat.match(excluded_mtp), f"Expected MTP_TARGET_RE to reject {excluded_mtp}"

    # Vision target names
    for name in (
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
        "visual.blocks.26.attn.qkv",
        "visual.merger.linear_fc2",
    ):
        assert vis_grp_pat.match(name), f"Expected VISION_GROUP_TARGET_RE to match {name}"
    assert vis_fc2_pat.match("model.visual.blocks.0.mlp.linear_fc2")
    assert vis_fc2_pat.match("visual.blocks.0.mlp.linear_fc2")
    assert not vis_grp_pat.match("model.visual.patch_embed.proj")


def test_hadamard_weight_rotation_and_online_input_cancel():
    from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix

    torch.manual_seed(42)
    x = torch.randn(4, 128, dtype=torch.float32)
    w = torch.randn(64, 128, dtype=torch.float32)
    w_rot = _hadamard_rotate_weight(w, block_size=64)

    had = deterministic_hadamard_matrix(64, torch.float32, x.device) / 8.0
    x_rot = (x.view(4, 2, 64) @ had).view(4, 128)

    y_ref = F.linear(x, w)
    y_rot = F.linear(x_rot, w_rot)
    assert torch.allclose(y_ref, y_rot, rtol=1e-5, atol=1e-5)


def test_clean_and_augment_shards_strips_v_input_and_packs_mtp_vision():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        tensors = {
            "model.language_model.layers.0.self_attn.q_proj.v_input.weight": torch.ones(
                64, 64, dtype=torch.bfloat16
            ),
            "mtp.layers.0.self_attn.q_proj.weight": torch.randn(64, 128, dtype=torch.bfloat16),
            "mtp.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 128, 128, dtype=torch.bfloat16),
            "mtp.layers.0.mlp.experts.down_proj": torch.randn(2, 128, 64, dtype=torch.bfloat16),
            "mtp.fc.weight": torch.randn(64, 128, dtype=torch.bfloat16),
            "model.visual.blocks.0.attn.qkv.weight": torch.randn(192, 64, dtype=torch.bfloat16),
            "model.visual.blocks.0.mlp.linear_fc1.weight": torch.randn(4304, 1152, dtype=torch.bfloat16),
            "model.visual.blocks.0.mlp.linear_fc2.weight": torch.randn(64, 4304, dtype=torch.bfloat16),
        }
        save_file(tensors, str(d / "model.safetensors"))
        cfg = {
            "vision_config": {"intermediate_size": 4304},
            "quantization_config": {"config_groups": {}, "ignore": [], "format": "pack-quantized"},
        }
        (d / "config.json").write_text(json.dumps(cfg))

        counts = _clean_and_augment_shards(
            d,
            quantize_mtp=True,
            quantize_vision=True,
            group_size=32,
            hadamard_block_size=64,
        )
        assert counts["stripped_transform_weights"] == 1
        assert counts["mtp_quantized"] == 7
        assert counts["vision_quantized"] == 3
        assert counts["vision_fc2_channel"] == 1
        _assert_packed_alignment(d)
        with tempfile.TemporaryDirectory() as bad_tmp:
            bad_d = Path(bad_tmp)
            save_file(
                {
                    "model.language_model.layers.0.mlp.up_proj.weight_packed": torch.zeros(
                        (4304, 144), dtype=torch.int32
                    ),
                    "model.language_model.layers.0.mlp.up_proj.weight_shape": torch.tensor(
                        [4304, 1152], dtype=torch.int32
                    ),
                },
                str(bad_d / "model.safetensors"),
            )
            try:
                _assert_packed_alignment(bad_d)
            except AssertionError as exc:
                assert "violating Marlin alignment" in str(exc)
            else:
                raise AssertionError("Expected _assert_packed_alignment to reject non-visual N=4304")

        out_tensors = load_file(str(d / "model.safetensors"))
        assert not any(k.endswith(".v_input.weight") for k in out_tensors)
        assert "mtp.layers.0.self_attn.q_proj.weight_packed" in out_tensors
        assert "mtp.layers.0.mlp.experts.0.gate_proj.weight_packed" in out_tensors
        assert "mtp.layers.0.mlp.experts.1.down_proj.weight_packed" in out_tensors
        assert "mtp.fc.weight" in out_tensors
        assert "model.visual.blocks.0.mlp.linear_fc2.weight_packed" in out_tensors
        assert tuple(out_tensors["model.visual.blocks.0.mlp.linear_fc2.weight_scale"].shape) == (64, 1)

        _update_config_for_components(
            d,
            group_size=32,
            actorder="weight",
            hadamard_block_size=64,
            quantize_mtp=True,
            quantize_vision=True,
            vision_fc2_channel=True,
        )
        updated_cfg = json.loads((d / "config.json").read_text())
        qcfg = updated_cfg["quantization_config"]
        assert "vision_fc2_channel" in qcfg["config_groups"]
        assert "v_vision_fc2" in qcfg["transform_config"]["config_groups"]
        assert qcfg["transform_config"]["config_groups"]["v_vision_fc2"]["head_dim"] == 16


def test_gptq_modifier_actorder_explicit_none_is_preserved():
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor.modifiers.gptq import GPTQModifier

    for actorder in (None, "static"):
        scheme = QuantizationScheme(
            targets=[TEXT_TARGET_RE],
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=64,
                actorder=actorder,
            ),
        )
        mod = GPTQModifier(config_groups={"group_0": scheme}, actorder=actorder)
        resolved = mod.resolve_quantization_config()
        resolved_act = resolved.config_groups["group_0"].weights.actorder
        if actorder is None:
            assert resolved_act is None
        else:
            assert str(resolved_act.value if hasattr(resolved_act, "value") else resolved_act) == actorder


def test_gptq_pack_tensor_reduces_output_error():
    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from llmcompressor.modifiers.gptq.helpers import accumulate_hessian, make_empty_hessian

    torch.manual_seed(7)
    lin = torch.nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)
    # Create an outlier channel in weight and correlated activations
    lin.weight.data[:, 0] *= 6.0
    x = torch.randn(32, 16, 128, dtype=torch.bfloat16)
    h = make_empty_hessian(lin, torch.device("cpu"))
    n_samp = torch.zeros(1, dtype=torch.int64)
    accumulate_hessian(x, lin, h, n_samp)

    packed_rtn = _pack_w4_tensor(lin.weight.data, strategy="group", group_size=64)
    packed_gptq = _gptq_pack_tensor(
        lin.weight.data,
        h,
        n_samp,
        strategy="group",
        group_size=64,
        actorder="static",
        dampening_frac=0.01,
    )
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="group", group_size=64),
    )
    w_rtn = PackedQuantizationCompressor.decompress(packed_rtn, scheme)["weight"]
    w_gptq = PackedQuantizationCompressor.decompress(packed_gptq, scheme)["weight"]

    y_ref = F.linear(x.float(), lin.weight.data.float())
    err_rtn = (F.linear(x.float(), w_rtn.float()) - y_ref).pow(2).mean().item()
    err_gptq = (F.linear(x.float(), w_gptq.float()) - y_ref).pow(2).mean().item()
    assert err_gptq < err_rtn, f"Expected GPTQ output MSE ({err_gptq}) < RTN ({err_rtn})"


def test_moe_mtp_and_quant_load_with_transforms():
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

    from warp_quant.data import align_mtp
    from warp_quant.hadamard_gptq import (
        _build_mtp_module,
        _gptq_calibrate_mtp,
        _hadamard_rotate_weight,
        _load_mtp_module_from_checkpoint,
        _pack_w4_tensor,
        load_quant_model_with_transforms,
    )

    torch.manual_seed(11)
    tcfg = {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 256,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 64,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "num_experts": 2,
        "num_experts_per_tok": 2,
        "layer_types": ["full_attention"],
    }
    vcfg = {
        "model_type": "qwen3_5_moe",
        "depth": 1,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_heads": 4,
        "out_hidden_size": 64,
    }
    cfg = Qwen3_5MoeConfig(text_config=tcfg, vision_config=vcfg)
    torch.manual_seed(0)
    model = Qwen3_5MoeForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    mtp_ref = _build_mtp_module(cfg.text_config).to(torch.bfloat16).eval()
    with torch.no_grad():
        for m in (model, mtp_ref):
            for p in m.parameters():
                if p.dim() >= 2:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    torch.nn.init.ones_(p)

    with tempfile.TemporaryDirectory() as tmp:
        base_d = Path(tmp) / "base"
        quant_d = Path(tmp) / "quant"
        base_d.mkdir()
        quant_d.mkdir()
        (base_d / "config.json").write_text(json.dumps(cfg.to_dict()))
        mtp_sd = {f"mtp.{k}": v.contiguous() for k, v in mtp_ref.state_dict().items()}
        save_file(mtp_sd, str(base_d / "model_mtp.safetensors"))

        ds = [{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]} for _ in range(4)]
        packed_mtp = _gptq_calibrate_mtp(
            model,
            base_d,
            ds,
            group_size=64,
            hadamard_block_size=64,
            actorder="static",
            dampening_frac=0.01,
            max_calib_samples=4,
            quantize_fc_w8=True,
        )
        # 4 self_attn + 3 shared_expert + 2 experts * 3 + 1 W8 mtp.fc = 14 MTP linears
        assert len(packed_mtp) == 14
        assert packed_mtp["mtp.fc"]["weight_packed"].shape == (64, 32)

        save_file(mtp_sd, str(quant_d / "model_mtp.safetensors"))
        (quant_d / "config.json").write_text(
            json.dumps(
                {
                    **cfg.to_dict(),
                    "quantization_config": {
                        "config_groups": {
                            "group_0": {
                                "targets": ["Linear"],
                                "weights": {
                                    "num_bits": 4,
                                    "type": "int",
                                    "symmetric": True,
                                    "strategy": "group",
                                    "group_size": 64,
                                    "actorder": "static",
                                },
                            }
                        },
                        "ignore": [],
                        "format": "pack-quantized",
                    },
                }
            )
        )
        counts = _clean_and_augment_shards(
            quant_d,
            quantize_mtp=True,
            quantize_vision=False,
            group_size=64,
            hadamard_block_size=64,
            precomputed_packed=packed_mtp,
            quantize_mtp_fc_w8=True,
        )
        assert counts["mtp_quantized"] == 14
        assert counts["mtp_fc_w8"] == 1
        _update_config_for_components(
            quant_d,
            group_size=64,
            actorder="static",
            hadamard_block_size=64,
            quantize_mtp=True,
            quantize_vision=False,
            vision_fc2_channel=False,
            quantize_lm_head_w8=True,
            quantize_mtp_fc_w8=True,
        )
        mtp_q = _load_mtp_module_from_checkpoint(quant_d, cfg, torch.device("cpu"))
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        base_out = model.model.language_model(input_ids=ids, use_cache=False)
        h_base, next_ids, pos, _ = align_mtp(ids, base_out.last_hidden_state)
        emb = model.get_input_embeddings()(next_ids)
        out_ref = mtp_ref(h_base, emb, pos)
        out_q = mtp_q(h_base, emb, pos)
        cos = F.cosine_similarity(out_ref.float().flatten(), out_q.float().flatten(), dim=0).item()
        assert cos > 0.95, f"Expected high cosine similarity for MoE MTP, got {cos}"

        # Also test _load_moe_quant_model_direct via load_quant_model_with_transforms
        main_sd = {}
        for k, v in model.state_dict().items():
            if k == "model.language_model.layers.0.mlp.experts.gate_up_proj":
                inter = v.shape[1] // 2
                for e in range(v.shape[0]):
                    for pname, sl in (("gate_proj", slice(0, inter)), ("up_proj", slice(inter, None))):
                        w_rot = _hadamard_rotate_weight(v[e, sl], 64)
                        packed = _pack_w4_tensor(w_rot, strategy="group", group_size=64)
                        for pk, pv in packed.items():
                            main_sd[f"model.language_model.layers.0.mlp.experts.{e}.{pname}.{pk}"] = (
                                pv.contiguous()
                            )
            elif k == "model.language_model.layers.0.mlp.experts.down_proj":
                for e in range(v.shape[0]):
                    w_rot = _hadamard_rotate_weight(v[e], 64)
                    packed = _pack_w4_tensor(w_rot, strategy="group", group_size=64)
                    for pk, pv in packed.items():
                        main_sd[f"model.language_model.layers.0.mlp.experts.{e}.down_proj.{pk}"] = (
                            pv.contiguous()
                        )
            elif (
                (
                    k.startswith("model.language_model.layers.0.self_attn.")
                    or k.startswith("model.language_model.layers.0.mlp.shared_expert.")
                )
                and k.endswith(".weight")
                and v.dim() == 2
            ):
                prefix = k.removesuffix(".weight")
                w_rot = _hadamard_rotate_weight(v, 64)
                packed = _pack_w4_tensor(w_rot, strategy="group", group_size=64)
                for pk, pv in packed.items():
                    main_sd[f"{prefix}.{pk}"] = pv.contiguous()
            elif k == "lm_head.weight":
                from warp_quant.hadamard_gptq import _pack_w8_mse_tensor

                for pk, pv in _pack_w8_mse_tensor(v, group_size=64).items():
                    main_sd[f"lm_head.{pk}"] = pv.contiguous()
            else:
                main_sd[k] = v.contiguous()
        save_file(main_sd, str(quant_d / "model.safetensors"))
        quant_model = load_quant_model_with_transforms(quant_d, device="cpu")
        logits_ref = model(input_ids=ids, use_cache=False).logits
        logits_q = quant_model(input_ids=ids, use_cache=False).logits
        cos_main = F.cosine_similarity(logits_ref.float().flatten(), logits_q.float().flatten(), dim=0).item()
        assert cos_main > 0.95, f"Expected high cosine similarity for MoE main model, got {cos_main}"


def test_vision_v2_split_disjointness_and_calibration_guard():
    import io

    from PIL import Image

    from warp_quant.data import _select_v2_split
    from warp_quant.hadamard_gptq import evaluate_vision_fidelity

    def _make_png(color, size=(160, 160)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        return buf.getvalue()

    cands = [
        ("docvqa", "ocrbench:docVQA", i, _make_png((i, i * 2, i * 3)), f"q{i}", [f"a{i}"])
        for i in range(1, 7)
    ]
    # Include one undersized image (min_dim < 128) that must be skipped
    cands.append(("docvqa", "ocrbench:docVQA", 99, _make_png((9, 9, 9), size=(64, 160)), "q99", ["a99"]))

    seen = set()
    probe = _select_v2_split(cands, "vision-probe-v2-0", {"docvqa": 2}, seen, min_dim=128)
    calib = _select_v2_split(cands, "vision-calib-v2-1", {"docvqa": 3}, seen, min_dim=128)
    guard = _select_v2_split(cands, "vision-guard-v2-0", {"docvqa": 1}, seen, min_dim=128)
    p_dig = {r[3] for r in probe}
    c_dig = {r[3] for r in calib}
    g_dig = {r[3] for r in guard}
    assert len(p_dig) == 2 and len(c_dig) == 3 and len(g_dig) == 1
    assert not (p_dig & c_dig) and not (p_dig & g_dig) and not (c_dig & g_dig)

    with tempfile.TemporaryDirectory() as tmp:
        calib_dir = Path(tmp)
        (calib_dir / "manifest.json").write_text(json.dumps({"calibration": True}))
        try:
            evaluate_vision_fidelity(torch.nn.Module(), torch.nn.Module(), calib_dir)
            raise AssertionError("Expected ValueError when evaluating Vision on a calibration fixture")
        except ValueError as exc:
            assert "Refusing to evaluate Vision on calibration fixture" in str(exc)


class RotExperimentTests(unittest.TestCase):
    def test_target_regexes(self):
        test_target_regexes_match_hf_and_vllm_names()

    def test_hadamard_weight_rotation(self):
        test_hadamard_weight_rotation_and_online_input_cancel()

    def test_clean_and_augment_shards(self):
        test_clean_and_augment_shards_strips_v_input_and_packs_mtp_vision()

    def test_gptq_modifier_actorder(self):
        test_gptq_modifier_actorder_explicit_none_is_preserved()

    def test_gptq_pack_tensor(self):
        test_gptq_pack_tensor_reduces_output_error()

    def test_moe_mtp_and_quant_load(self):
        test_moe_mtp_and_quant_load_with_transforms()

    def test_vision_v2_split_and_guard(self):
        test_vision_v2_split_disjointness_and_calibration_guard()


if __name__ == "__main__":
    unittest.main()
