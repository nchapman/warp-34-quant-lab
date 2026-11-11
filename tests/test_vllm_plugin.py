from __future__ import annotations

import os
import sys
import types

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
from compressed_tensors.transform import TransformLocation

from warp_quant.hadamard_gptq import _hadamard_rotate_weight
from warp_quant.streaming_moe import dequantize_ple_rows_lut4
from warp_quant.vllm_plugin import (
    CustomLut4PLEMethod,
    _patch_ct_linear_transform,
    _patch_qwen4_exp_ple_lut4,
    fast_block_hadamard,
    register,
    repack_uint2b2_to_uint4b8,
    unpack_dequant_uint2b2,
)


def test_repack_uint2b2_to_uint4b8_2d_and_3d() -> None:
    torch.manual_seed(42)
    for shape in ((64, 128), (4, 32, 64)):
        q_int = torch.randint(-2, 2, shape, dtype=torch.int8)
        w2_packed = pack_to_int32(q_int, num_bits=2)
        w4_expected = pack_to_int32(q_int, num_bits=4)
        w4_actual = repack_uint2b2_to_uint4b8(w2_packed)
        assert w4_actual.shape == w4_expected.shape
        assert torch.equal(w4_actual, w4_expected)
        unpacked = unpack_from_int32(w4_actual, 4, torch.Size(shape))
        assert torch.equal(unpacked.to(torch.int8), q_int)


def test_unpack_dequant_uint2b2_matches_reference() -> None:
    torch.manual_seed(7)
    n, k, g = 32, 128, 64
    q_int = torch.randint(-2, 2, (n, k), dtype=torch.int8)
    scale = torch.randn(n, k // g, dtype=torch.bfloat16).abs() + 0.01
    w2_packed = pack_to_int32(q_int, num_bits=2)

    actual = unpack_dequant_uint2b2(w2_packed, scale, group_size=g, out_dtype=torch.bfloat16)
    expected = q_int.to(torch.bfloat16) * scale.repeat_interleave(g, dim=-1)
    assert torch.equal(actual, expected)


def test_fast_block_hadamard_validation_and_hadacore_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(11)
    x = torch.randn(5, 192, dtype=torch.bfloat16)
    assert torch.equal(fast_block_hadamard(x, block_size=1), x)

    y = fast_block_hadamard(x, block_size=64)
    expected = _hadamard_rotate_weight(x, block_size=64)
    assert torch.equal(y, expected)
    x_roundtrip = fast_block_hadamard(y, block_size=64)
    assert torch.allclose(x_roundtrip.float(), x.float(), atol=2e-2, rtol=2e-2)

    with pytest.raises(ValueError, match="power of 2"):
        fast_block_hadamard(x, block_size=48)
    with pytest.raises(ValueError, match="divisible by block_size"):
        fast_block_hadamard(torch.randn(2, 100), block_size=64)
    with pytest.raises(ValueError, match=">= 1"):
        fast_block_hadamard(x, block_size=0)

    calls: list[tuple[tuple[int, ...], bool]] = []
    fake_ops = types.ModuleType("vllm._custom_ops")

    def _fake_hadacore(t: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        calls.append((tuple(t.shape), inplace))
        return t.mul_(2.0)

    fake_ops.hadacore_transform = _fake_hadacore  # type: ignore[attr-defined]
    fake_vllm = types.ModuleType("vllm")
    fake_vllm._custom_ops = fake_ops  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", fake_ops)

    x_orig = x.clone()
    out = fast_block_hadamard(x, block_size=64)
    assert calls == [((5, 3, 64), True)]
    assert torch.equal(x, x_orig)
    assert torch.equal(out, x_orig * 2.0)


def test_custom_lut4_ple_method_storage_dequant_and_tp_sharding() -> None:
    torch.manual_seed(19)
    num_rows, dim, group_size = 64, 160, 32
    num_groups = dim // group_size
    codes = torch.randint(0, 16, (num_rows, dim), dtype=torch.uint8)
    w_packed = ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).contiguous()
    w_scale = (torch.randn(num_rows, num_groups, dtype=torch.float16).abs() + 0.01).contiguous()
    w_lut = torch.linspace(-1.0, 1.0, 16, dtype=torch.bfloat16)

    method = CustomLut4PLEMethod(group_size=group_size)
    shape, dtype = method.get_storage_Spec(num_rows, dim, torch.bfloat16)
    assert shape == (num_rows, 90)
    assert dtype == torch.uint8

    with pytest.raises(ValueError, match="missing required"):
        method.pack_shard_weights(0, {"weight_packed": w_packed}, dim)

    combined_rows = method.pack_shard_weights(
        shard_idx=0,
        parts={"weight_packed": w_packed, "weight_scale": w_scale, "weight_lut": w_lut},
        embedding_dim_per_shard=dim,
    )
    assert combined_rows.shape == (num_rows, 90)

    token_ids = torch.tensor([[0, 5, 17, 63], [2, 19, 33, 40]], dtype=torch.long)
    expected = dequantize_ple_rows_lut4(
        w_packed[token_ids],
        w_scale[token_ids],
        w_lut,
        group_size=group_size,
        out_dtype=torch.bfloat16,
    )
    gathered = combined_rows.index_select(0, token_ids.reshape(-1)).reshape(*token_ids.shape, 90)
    actual = method.dequantize(gathered, dim, torch.bfloat16, shard_idx=0)
    assert torch.equal(actual, expected)

    with pytest.raises(RuntimeError, match="Missing Custom-LUT4 codebook"):
        method.dequantize(gathered, dim, torch.bfloat16, shard_idx=3)

    # Verify TP=2 sharding where dim=160 splits into 80 columns per rank (g_eff = gcd(32, 80) = 16)
    rank_pieces = []
    for tp_rank in (0, 1):
        m_tp = CustomLut4PLEMethod(group_size=group_size)
        tp_shape, _ = m_tp.get_storage_Spec(num_rows, dim // 2, torch.bfloat16)
        assert tp_shape == (num_rows, 40 + (80 // 16) * 2)
        packed_rank = m_tp.pack_shard_weights(
            shard_idx=0,
            parts={"weight_packed": w_packed, "weight_scale": w_scale, "weight_lut": w_lut},
            embedding_dim_per_shard=dim // 2,
            tp_size=2,
            tp_rank=tp_rank,
        )
        g_rank = packed_rank.index_select(0, token_ids.reshape(-1)).reshape(*token_ids.shape, -1)
        rank_pieces.append(m_tp.dequantize(g_rank, dim // 2, torch.bfloat16, shard_idx=0))
    assert torch.equal(torch.cat(rank_pieces, dim=-1), expected)


def test_patch_ct_linear_transform_and_ple_with_stub_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeQuantMethod:
        def apply(
            self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            del layer
            return x + bias if bias is not None else x

    class FakeLinearTransformMethod:
        def __init__(self) -> None:
            self.input_tfms = {0: "scheme_a"}
            self.output_tfms: dict[int, str] = {}
            self.input_transform = lambda t: t * 2.0
            self.output_transform = None
            self.quant_method = FakeQuantMethod()

        def _validate_tfm_schemes(self, num_partitions: int) -> None:
            for i in range(num_partitions):
                _ = self.input_tfms[i]

        def apply(
            self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            assert bias is None
            return x

    class FakeHadamardTransform:
        def __init__(
            self,
            transforms: dict[int, object],
            layer: torch.nn.Module,
            weight_loader: object,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
        ) -> None:
            del transforms, layer, weight_loader, input_size_per_partition, output_partition_sizes
            if fake_mod.get_tensor_model_parallel_world_size() > 1:
                raise NotImplementedError("Online transforms with tensor parallelism is not supported")

    fake_dist_par = types.ModuleType("vllm.distributed.parallel_state")
    fake_dist_par.get_tensor_model_parallel_world_size = lambda: 2  # type: ignore[attr-defined]
    fake_lin = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.linear")
    fake_lin.CompressedTensorsLinearTransformMethod = FakeLinearTransformMethod  # type: ignore[attr-defined]
    fake_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.module")
    fake_mod.HadamardTransform = FakeHadamardTransform  # type: ignore[attr-defined]
    fake_mod.get_tensor_model_parallel_world_size = fake_dist_par.get_tensor_model_parallel_world_size  # type: ignore[attr-defined]

    for name, mod in {
        "vllm": types.ModuleType("vllm"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.distributed.parallel_state": fake_dist_par,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.quantization": types.ModuleType(
            "vllm.model_executor.layers.quantization"
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors": types.ModuleType(
            "vllm.model_executor.layers.quantization.compressed_tensors"
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors.transform": types.ModuleType(
            "vllm.model_executor.layers.quantization.compressed_tensors.transform"
        ),
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.linear": fake_lin,
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.module": fake_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    _patch_ct_linear_transform()

    ltm = FakeLinearTransformMethod()
    # Unfused QKV (num_partitions=3, input_tfms only has key 0) should not raise KeyError
    ltm._validate_tfm_schemes(3)
    x = torch.ones(2, 4)
    bias = torch.full((4,), 0.5)
    out = ltm.apply(torch.nn.Identity(), x, bias=bias)
    assert torch.equal(out, x * 2.0 + bias)

    # Block-Hadamard TP=2 should succeed when head_dim divides input_size_per_partition
    fake_tfm = types.SimpleNamespace(
        scheme=types.SimpleNamespace(head_dim=64),
        args=types.SimpleNamespace(location=TransformLocation.INPUT),
    )
    FakeHadamardTransform({0: fake_tfm}, torch.nn.Identity(), lambda *a: None, 128, [128])


def test_patch_qwen4_exp_ple_lut4_end_to_end_with_stub_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePLEMethod:
        @classmethod
        def from_quant_config(cls, quant_config: object, hf_text_config: object = None) -> FakePLEMethod:
            del quant_config, hf_text_config
            return cls()

    class FakeDeviceEmbedding:
        def __init__(self, quant_method: FakePLEMethod, dim_per_shard: int) -> None:
            self.quant_method = quant_method
            self.embedding_dim_per_shard = dim_per_shard
            self.params_dtype = torch.bfloat16
            self.weights: dict[int, torch.Tensor] = {}

        def load_shard_weight(self, shard_idx: int, weight: torch.Tensor) -> None:
            self.weights[shard_idx] = weight

        def forward_SingleShard(
            self, shard_idx: int, flat_ids: torch.Tensor, valid_mask: torch.Tensor
        ) -> torch.Tensor:
            del shard_idx, flat_ids, valid_mask
            raise AssertionError("unpatched forward_SingleShard should not be called for LUT4")

    class FakePinnedHostEmbedding(FakeDeviceEmbedding):
        def __init__(self, quant_method: FakePLEMethod, dim_per_shard: int) -> None:
            super().__init__(quant_method, dim_per_shard)
            self._prefetched_out = None

        def _gather_rows_uva(self, shard_idx: int, flat_ids: torch.Tensor) -> torch.Tensor:
            return self.weights[shard_idx].index_select(0, flat_ids)

    class FakeNGramEmbedding:
        def __init__(self, backend: FakeDeviceEmbedding, num_shards: int, dim_per_shard: int) -> None:
            self.backend = backend
            self.num_shards = num_shards
            self.embedding_dim_per_shard = dim_per_shard
            self.tp_size = 1
            self.tp_rank = 0

        def load_weights(self, weights: object) -> set[str]:
            del weights
            return set()

    common_mod = types.ModuleType("vllm.models.qwen4_exp.common.ngram_embedding")
    common_mod.Qwen4ExpPLEEmbeddingMethod = FakePLEMethod  # type: ignore[attr-defined]
    common_mod.Qwen4ExpPLEDeviceEmbedding = FakeDeviceEmbedding  # type: ignore[attr-defined]
    common_mod.Qwen4ExpPLEPinnedHostEmbedding = FakePinnedHostEmbedding  # type: ignore[attr-defined]
    nvidia_mod = types.ModuleType("vllm.models.qwen4_exp.nvidia.ngram_embedding")
    nvidia_mod.Qwen4ExpNGramEmbedding = FakeNGramEmbedding  # type: ignore[attr-defined]

    for name, mod in {
        "vllm": types.ModuleType("vllm"),
        "vllm.models": types.ModuleType("vllm.models"),
        "vllm.models.qwen4_exp": types.ModuleType("vllm.models.qwen4_exp"),
        "vllm.models.qwen4_exp.common": types.ModuleType("vllm.models.qwen4_exp.common"),
        "vllm.models.qwen4_exp.common.ngram_embedding": common_mod,
        "vllm.models.qwen4_exp.nvidia": types.ModuleType("vllm.models.qwen4_exp.nvidia"),
        "vllm.models.qwen4_exp.nvidia.ngram_embedding": nvidia_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    _patch_qwen4_exp_ple_lut4()

    hf_cfg = types.SimpleNamespace(ple_quantization={"format": "custom_lut4_g32_fp16scale", "group_size": 32})
    qmethod = FakePLEMethod.from_quant_config(None, hf_cfg)
    assert isinstance(qmethod, CustomLut4PLEMethod)

    backend = FakePinnedHostEmbedding(qmethod, 160)
    ngram = FakeNGramEmbedding(backend, num_shards=1, dim_per_shard=160)

    w_packed = torch.randint(0, 256, (16, 80), dtype=torch.uint8)
    w_scale = (torch.randn(16, 5, dtype=torch.float16).abs() + 0.05).contiguous()
    w_lut = torch.linspace(-0.5, 0.5, 16, dtype=torch.bfloat16)

    loaded = ngram.load_weights(
        [
            ("shard_0.weight_packed", w_packed),
            ("shard_0.weight_scale", w_scale),
            ("shard_0.weight_lut", w_lut),
        ]
    )
    assert loaded == {"shard_0.weight_packed", "shard_0.weight_scale", "shard_0.weight_lut"}
    flat_ids = torch.tensor([1, 4, 7], dtype=torch.long)
    valid_mask = torch.tensor([True, False, True])
    out = backend.forward_SingleShard(0, flat_ids, valid_mask)
    assert out.shape == (3, 160)
    assert torch.all(out[1] == 0)


def test_register_is_idempotent_without_vllm() -> None:
    register()
    register()


def test_hf_auto_map_warp_quant_config_from_checkpoint_dir(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import shutil
    from pathlib import Path

    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    import warp_quant.vllm_plugin as vp

    ckpt_dir = Path(str(tmp_path))
    shutil.copyfile(Path(vp.__file__).resolve(), ckpt_dir / "vllm_warp_quant.py")
    (ckpt_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "auto_map": {"AutoConfig": "vllm_warp_quant.WarpQuantConfig"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("WARP_QUANT_SUBPROCESS_AUTOLOAD", raising=False)

    cfg_trusted = AutoConfig.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    assert isinstance(cfg_trusted, Qwen3_5MoeConfig)
    assert os.environ.get("WARP_QUANT_SUBPROCESS_AUTOLOAD") == "1"

    cfg_untrusted = AutoConfig.from_pretrained(str(ckpt_dir), trust_remote_code=False)
    assert isinstance(cfg_untrusted, Qwen3_5MoeConfig)


def test_hadamard_transform_forward_clones_input_and_humming_named_params_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    par_mod = types.ModuleType("vllm.distributed.parallel_state")
    par_mod.get_tensor_model_parallel_world_size = lambda: 1
    ct_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors")
    lin_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.linear")
    ht_mod = types.ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.transform.module")

    class FakeLinearTransformMethod:
        pass

    class FakeHadamardTransform(torch.nn.Module):
        def __init__(
            self, transforms, layer, weight_loader, input_size_per_partition, output_partition_sizes
        ):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(4), requires_grad=False)

        def forward(self, value: torch.Tensor, part_id: int = 0) -> torch.Tensor:
            # Simulate vLLM's in-place ops.hadacore_transform mutation
            value.mul_(2.0)
            return value

    lin_mod.CompressedTensorsLinearTransformMethod = FakeLinearTransformMethod
    ht_mod.HadamardTransform = FakeHadamardTransform

    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", par_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        ct_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.linear",
        lin_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.compressed_tensors.transform.module",
        ht_mod,
    )

    _patch_ct_linear_transform()

    ht = FakeHadamardTransform({}, torch.nn.Linear(4, 4), None, 4, [4])
    x = torch.ones(2, 4)
    out = ht(x)
    # Caller's x must NOT be mutated in-place
    assert torch.all(x == 1.0)
    assert torch.all(out == 2.0)
