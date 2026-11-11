"""vLLM general plugin (`vllm.general_plugins`) for warp-quant checkpoints.

Automatically loaded by vLLM in all processes (API server, EngineCore, and GPU
workers) when `warp-quant` is installed in the environment. Self-contained so
it can also be shipped standalone in Hugging Face checkpoint repositories.

Patches four upstream gaps when present in the active vLLM version:
1. `CompressedTensorsLinearTransformMethod`:
   - Allows `bias is not None` for input-only transforms (`output_transform is None`).
   - Fixes `_validate_tfm_schemes` on unfused `QKVParallelLinear` layers (`Qwen2_5_VisionAttention.qkv`).
   - Allows tensor parallelism (`tp_size > 1`) on `HadamardTransform` when `head_dim` is set
     and divides `input_size_per_partition`.
2. `RoutedExperts` online Hadamard transforms (`transform_config` targeting `mlp.experts.*`):
   - Applies block-64 Walsh-Hadamard rotation before `w13` (`gate_up_proj`) and after
     SwiGLU activation before `w2` (`down_proj`), compatible with PR #66505 and with
     `MarlinExperts`, `BatchedMarlinExperts`, `TritonWNA16Experts`, and `HummingExpertsBase`.
3. `W2A16` (`num_bits=2`, `scalar_types.uint2b2`) in `CompressedTensorsWNA16` & `CompressedTensorsWNA16MoEMethod`:
   - Fixes Humming `uint2` -> `scalar_types.uint2b2` mapping when Humming is installed.
   - Provides true 2-bit VRAM storage (`[N, K // 16]` `int32`) with fused Triton / grouped
     dequant-GEMM fallback when Humming is not installed, plus opt-in lossless SWAR
     `uint2b2 -> uint4b8` repacking for Marlin (`VLLM_WARP_QUANT_W2_MARLIN=1`).
4. `Qwen4Exp` (`Qwen3.8-Flash-Next`) 4-bit `Custom-LUT4` Layer-1 PLE (`custom_lut4_g32_fp16scale`):
   - Adds `Qwen4ExpPLECustomLut4EmbeddingMethod` (`90 B/row`: `80 B` packed 4-bit codes +
     `10 B` FP16 group scales + 16-entry per-shard LUT) for both device (HBM) and UVA
     pinned-host (`--offload-ple-embeddings`) backends.
"""

from __future__ import annotations

import importlib
import math
import os
from fractions import Fraction
from typing import Any

import torch

_REGISTERED = False


def repack_uint2b2_to_uint4b8(packed_w2: torch.Tensor) -> torch.Tensor:
    """Losslessly expand packed `uint2b2` (`[..., K // 16]` int32, bias=2) into
    packed `uint4b8` (`[..., K // 8]` int32, bias=8) via 4-step SWAR bit dilation.

    For every 2-bit code `u2 in {0, 1, 2, 3}` (representing signed `q = u2 - 2`),
    the corresponding 4-bit `uint4b8` code is `u4 = q + 8 = u2 + 6 in {6, 7, 8, 9}`.
    """
    lo = packed_w2 & 0xFFFF
    hi = (packed_w2 >> 16) & 0xFFFF
    out = []
    for x in (lo, hi):
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = x + 0x66666666
        out.append(x.to(torch.int32))
    return torch.stack(out, dim=-1).flatten(-2, -1)


def unpack_dequant_uint2b2(
    packed_w2: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 64,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpack and dequantize `uint2b2` `[..., N, K // 16]` int32 with `scale` `[..., N, K // G]`."""
    orig_prefix = packed_w2.shape[:-1]
    k_packed = packed_w2.shape[-1]
    k_full = k_packed * 16
    shifts = torch.arange(0, 32, 2, device=packed_w2.device, dtype=torch.int32)
    codes = ((packed_w2.unsqueeze(-1) >> shifts) & 0x3).to(torch.int8) - 2
    q = codes.reshape(*orig_prefix, k_full).to(out_dtype)
    if group_size > 0 and scale.shape[-1] * group_size == k_full:
        q = (q.unflatten(-1, (scale.shape[-1], group_size)) * scale.to(out_dtype).unsqueeze(-1)).flatten(
            -2, -1
        )
    else:
        q = q * scale.to(out_dtype)
    return q


def fast_block_hadamard(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Apply normalized block-diagonal Sylvester Walsh-Hadamard transform along last dim."""
    if block_size <= 1:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        return x
    if (block_size & (block_size - 1)) != 0:
        raise ValueError(f"block_size must be a power of 2, got {block_size}")
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            f"Input last dimension ({x.shape[-1]}) must be divisible by block_size ({block_size})"
        )
    hadacore_fn = None
    try:
        ops = importlib.import_module("vllm._custom_ops")
        hadacore_fn = getattr(ops, "hadacore_transform", None)
    except ImportError:
        hadacore_fn = None

    if hadacore_fn is not None:
        return hadacore_fn(x.clone().unflatten(-1, (-1, block_size)), inplace=True).flatten(-2, -1)

    orig_shape = x.shape
    orig_dtype = x.dtype
    h = x.reshape(-1, block_size).to(torch.float32)
    steps = int(round(math.log2(block_size)))
    for _ in range(steps):
        h = h.reshape(-1, 2, block_size // 2)
        a = h[:, 0, :]
        b = h[:, 1, :]
        h = torch.stack((a + b, a - b), dim=-1).reshape(-1, block_size)
    h = h * (1.0 / math.sqrt(block_size))
    return h.to(orig_dtype).reshape(orig_shape)


class CustomLut4PLEMethod:
    """4-bit non-uniform Custom-LUT4 (`custom_lut4_g32_fp16scale`) PLE method for vLLM.

    Each row of length `D` (with effective group size `G_eff = gcd(group_size, D)`) is
    stored as a contiguous `uint8` row of `D // 2 + (D // G_eff) * 2` bytes
    (`90 B/row` for `D = 160, G = 32`):
    - First `D // 2` bytes: packed 4-bit unsigned LUT indices (low nibble = even col,
      high nibble = odd col).
    - Last `(D // G_eff) * 2` bytes: `D // G_eff` FP16 group scales viewed as `uint8`.
    """

    def __init__(self, group_size: int = 32) -> None:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self.group_size = int(group_size)
        self.luts: dict[int, torch.Tensor] = {}
        self._current_shard_idx: int = 0

    def effective_group_size(self, embedding_dim_per_shard: int) -> int:
        if embedding_dim_per_shard <= 0 or embedding_dim_per_shard % 2 != 0:
            raise ValueError(f"embedding_dim_per_shard ({embedding_dim_per_shard}) must be positive and even")
        g_eff = math.gcd(self.group_size, embedding_dim_per_shard)
        if g_eff < 2:
            raise ValueError(
                f"embedding_dim_per_shard ({embedding_dim_per_shard}) is incompatible with "
                f"group_size ({self.group_size})"
            )
        return g_eff

    def get_storage_Spec(
        self,
        num_embeddings: int,
        embedding_dim_per_shard: int,
        params_dtype: torch.dtype,
    ) -> tuple[tuple[int, int], torch.dtype]:
        del params_dtype
        g_eff = self.effective_group_size(embedding_dim_per_shard)
        packed_cols = embedding_dim_per_shard // 2
        scale_bytes = (embedding_dim_per_shard // g_eff) * 2
        return (num_embeddings, packed_cols + scale_bytes), torch.uint8

    def pack_shard_weights(
        self,
        shard_idx: int,
        parts: dict[str, torch.Tensor],
        embedding_dim_per_shard: int,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> torch.Tensor:
        required = ("weight_packed", "weight_scale", "weight_lut")
        missing = [k for k in required if k not in parts]
        if missing:
            raise ValueError(f"PLE shard {shard_idx} is missing required tensor(s): {missing}")

        w_packed = parts["weight_packed"].to(dtype=torch.uint8).contiguous()
        w_scale = parts["weight_scale"].to(dtype=torch.float16).contiguous()
        w_lut = parts["weight_lut"].to(dtype=torch.float32).contiguous()
        if w_lut.numel() != 16:
            raise ValueError(f"PLE shard {shard_idx} weight_lut must have 16 entries, got {w_lut.numel()}")
        self.luts[shard_idx] = w_lut

        full_dim = w_packed.shape[1] * 2
        g_eff = self.effective_group_size(embedding_dim_per_shard)
        if w_scale.shape[1] * g_eff != full_dim:
            orig_group_size = full_dim // w_scale.shape[1]
            if orig_group_size % g_eff != 0:
                raise ValueError(
                    f"PLE shard {shard_idx} scale shape {tuple(w_scale.shape)} incompatible with "
                    f"effective group size {g_eff}"
                )
            w_scale = w_scale.repeat_interleave(orig_group_size // g_eff, dim=1).contiguous()

        if full_dim != embedding_dim_per_shard:
            if tp_size <= 1 or full_dim != embedding_dim_per_shard * tp_size:
                raise ValueError(
                    f"PLE shard {shard_idx} width {full_dim} does not match "
                    f"embedding_dim_per_shard={embedding_dim_per_shard} (tp_size={tp_size})"
                )
            packed_per_rank = embedding_dim_per_shard // 2
            scales_per_rank = embedding_dim_per_shard // g_eff
            w_packed = w_packed[:, tp_rank * packed_per_rank : (tp_rank + 1) * packed_per_rank].contiguous()
            w_scale = w_scale[:, tp_rank * scales_per_rank : (tp_rank + 1) * scales_per_rank].contiguous()

        return torch.cat((w_packed, w_scale.view(torch.uint8)), dim=-1)

    def dequantize(
        self,
        rows: torch.Tensor,
        embedding_dim_per_shard: int,
        out_dtype: torch.dtype,
        shard_idx: int | None = None,
    ) -> torch.Tensor:
        if shard_idx is None:
            shard_idx = self._current_shard_idx
        g_eff = self.effective_group_size(embedding_dim_per_shard)
        packed_cols = embedding_dim_per_shard // 2
        num_groups = embedding_dim_per_shard // g_eff
        q_packed = rows[..., :packed_cols]
        scale_bytes = rows[..., packed_cols:].contiguous()
        scales = scale_bytes.view(torch.float16).reshape(*rows.shape[:-1], num_groups, 1).to(torch.float32)

        lo = (q_packed & 0x0F).to(torch.long)
        hi = ((q_packed >> 4) & 0x0F).to(torch.long)
        idx = torch.stack((lo, hi), dim=-1).reshape(*rows.shape[:-1], num_groups, g_eff)

        lut = self.luts.get(shard_idx)
        if lut is None:
            raise RuntimeError(f"Missing Custom-LUT4 codebook for PLE shard {shard_idx}")
        if lut.device != rows.device or lut.dtype != torch.float32:
            lut = lut.to(device=rows.device, dtype=torch.float32)
            self.luts[shard_idx] = lut
        return (lut[idx] * scales).to(out_dtype).reshape(*rows.shape[:-1], embedding_dim_per_shard)


def _patch_ct_linear_transform() -> None:
    try:
        importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
        )
    except Exception:
        pass
    try:
        ct_tfm = importlib.import_module("compressed_tensors.transform")
        TransformLocation = ct_tfm.TransformLocation
        par_state = importlib.import_module("vllm.distributed.parallel_state")
        get_tensor_model_parallel_world_size = par_state.get_tensor_model_parallel_world_size
        lin_mod = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.transform.linear"
        )
        CompressedTensorsLinearTransformMethod = lin_mod.CompressedTensorsLinearTransformMethod
        ht_mod = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.transform.module"
        )
        HadamardTransform = ht_mod.HadamardTransform
    except Exception:
        return

    if getattr(CompressedTensorsLinearTransformMethod, "_warp_quant_patched", False):
        return

    def _validate_tfm_schemes(self: Any, num_partitions: int) -> None:
        for part_index in self.input_tfms:
            if self.input_tfms[part_index] != self.input_tfms[0]:
                raise ValueError("Expected same input TransformScheme for all partitions")
        for part_index in self.output_tfms:
            if self.output_tfms[part_index] != self.output_tfms[0]:
                raise ValueError("Expected same output TransformScheme for all partitions")

    def _apply(
        self: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_transform is not None:
            x = self.input_transform(x)
        if self.output_transform is None:
            return self.quant_method.apply(layer, x, bias)
        x = self.quant_method.apply(layer, x, None)
        x = self.output_transform(x)
        if bias is not None:
            x = x + bias
        return x

    CompressedTensorsLinearTransformMethod._validate_tfm_schemes = _validate_tfm_schemes
    CompressedTensorsLinearTransformMethod.apply = _apply
    CompressedTensorsLinearTransformMethod._warp_quant_patched = True

    orig_ht_init = HadamardTransform.__init__
    orig_ht_forward = getattr(HadamardTransform, "forward", None)

    if orig_ht_forward is not None:

        def _ht_forward(self: Any, value: torch.Tensor, part_id: int = 0) -> torch.Tensor:
            # Clone input before HadamardTransform.forward so vLLM's in-place
            # ops.hadacore_transform(value, inplace=True) does not corrupt shared
            # caller activations (e.g. GatedDeltaNet in_proj_ba, MoE gates).
            return orig_ht_forward(self, value.clone(), part_id=part_id)

        HadamardTransform.forward = _ht_forward

    def _ht_init(
        self: Any,
        transforms: dict[int, Any],
        layer: torch.nn.Module,
        weight_loader: Any,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and transforms:
            can_shard_block_hadamard = all(
                t.scheme.head_dim is not None
                and t.args.location == TransformLocation.INPUT
                and input_size_per_partition % int(t.scheme.head_dim) == 0
                for t in transforms.values()
            )
            if can_shard_block_hadamard:
                orig_tp_fn = ht_mod.get_tensor_model_parallel_world_size
                try:
                    ht_mod.get_tensor_model_parallel_world_size = lambda: 1
                    orig_ht_init(
                        self,
                        transforms,
                        layer,
                        weight_loader,
                        input_size_per_partition,
                        output_partition_sizes,
                    )
                finally:
                    ht_mod.get_tensor_model_parallel_world_size = orig_tp_fn
                return
        orig_ht_init(
            self,
            transforms,
            layer,
            weight_loader,
            input_size_per_partition,
            output_partition_sizes,
        )

    HadamardTransform.__init__ = _ht_init


def _patch_ct_moe_transform() -> None:
    try:
        ct_tfm = importlib.import_module("compressed_tensors.transform")
        TransformLocation = ct_tfm.TransformLocation
        moe_layer_mod = importlib.import_module("vllm.model_executor.layers.fused_moe.layer")
        RoutedExperts = moe_layer_mod.RoutedExperts
        ct_mod = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
        )
        CompressedTensorsConfig = ct_mod.CompressedTensorsConfig
        lin_mod = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.transform.linear"
        )
        get_linear_transform_schemes = getattr(lin_mod, "get_linear_transform_schemes", None)
        if get_linear_transform_schemes is None:
            ct_utils_mod = importlib.import_module(
                "vllm.model_executor.layers.quantization.compressed_tensors.transform.utils"
            )
            get_linear_transform_schemes = ct_utils_mod.get_linear_transform_schemes
    except Exception:
        return

    if getattr(CompressedTensorsConfig, "_warp_quant_moe_patched", False):
        return

    class _MoEHadamardTransform(torch.nn.Module):
        def __init__(self, w13_head_dim: int | None, w2_head_dim: int | None) -> None:
            super().__init__()
            self.w13_head_dim = w13_head_dim
            self.w2_head_dim = w2_head_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.w13_head_dim is not None and self.w13_head_dim > 1:
                return fast_block_hadamard(x, self.w13_head_dim)
            return x

        def transform_post_act_inplace(self, output: torch.Tensor) -> None:
            if self.w2_head_dim is not None and self.w2_head_dim > 1:
                output.copy_(fast_block_hadamard(output, self.w2_head_dim))

        def process_weights_after_loading(self, experts: Any) -> None:
            if experts is None:
                return
            quant_cfg = getattr(experts, "quant_config", None)
            if quant_cfg is not None and hasattr(quant_cfg, "w1_hadamard_block_size"):
                if self.w13_head_dim is not None:
                    quant_cfg.w1_hadamard_block_size = self.w13_head_dim
                if self.w2_head_dim is not None:
                    quant_cfg.w2_hadamard_block_size = self.w2_head_dim
                if hasattr(experts, "init_humming_moe"):
                    experts.init_humming_moe()
                return
            if self.w2_head_dim is not None and self.w2_head_dim > 1:
                if hasattr(experts, "post_activation_transform"):
                    experts.post_activation_transform = self.transform_post_act_inplace
                elif hasattr(experts, "activation") and not getattr(
                    experts, "_warp_quant_act_wrapped", False
                ):
                    orig_act = experts.activation

                    def _wrapped_activation(
                        act_or_layer: Any,
                        output: torch.Tensor,
                        input_tensor: torch.Tensor,
                        *args: Any,
                        **kwargs: Any,
                    ) -> None:
                        orig_act(act_or_layer, output, input_tensor, *args, **kwargs)
                        self.transform_post_act_inplace(output)

                    experts.activation = _wrapped_activation
                    experts._warp_quant_act_wrapped = True

    def _resolve_moe_hadamard_dims(
        transform_config: Any, prefix: str, layer: torch.nn.Module
    ) -> tuple[int | None, int | None]:
        if transform_config is None:
            return None, None
        gate_name = getattr(layer, "ckpt_gate_proj_name", "gate_proj") or "gate_proj"
        up_name = getattr(layer, "ckpt_up_proj_name", "up_proj") or "up_proj"
        down_name = getattr(layer, "ckpt_down_proj_name", "down_proj") or "down_proj"
        gate_tfms = get_linear_transform_schemes(
            layer, f"{prefix}.0.{gate_name}", transform_config, packed_modules_mapping={}
        )
        up_tfms = get_linear_transform_schemes(
            layer, f"{prefix}.0.{up_name}", transform_config, packed_modules_mapping={}
        )
        down_tfms = get_linear_transform_schemes(
            layer, f"{prefix}.0.{down_name}", transform_config, packed_modules_mapping={}
        )
        if gate_tfms != up_tfms:
            raise ValueError(f"Mismatched gate/up MoE transforms at {prefix}")

        def _extract_head_dim(tfms: Any) -> int | None:
            if not tfms:
                return None
            # vLLM transform.linear.get_linear_transform_schemes returns (input_tfms, output_tfms)
            if isinstance(tfms, tuple) and len(tfms) == 2:
                input_tfms = tfms[0]
                if not input_tfms or 0 not in input_tfms:
                    return None
                _name, scheme, _args = input_tfms[0]
            elif isinstance(tfms, dict) and (0, TransformLocation.INPUT) in tfms:
                _name, scheme, _args = tfms[(0, TransformLocation.INPUT)]
            elif isinstance(tfms, dict) and 0 in tfms:
                _name, scheme, _args = tfms[0]
            else:
                return None
            if scheme.type != "hadamard":
                raise NotImplementedError(f"Unsupported MoE transform type: {scheme.type}")
            return int(scheme.head_dim) if scheme.head_dim is not None else None

        return _extract_head_dim(gate_tfms), _extract_head_dim(down_tfms)

    orig_get_quant_method = CompressedTensorsConfig.get_quant_method

    def _get_moe_experts_obj(layer: torch.nn.Module) -> Any:
        experts = getattr(layer, "experts", None)
        if experts is not None:
            return experts
        qm = getattr(layer, "quant_method", None)
        mk = getattr(qm, "moe_kernel", None)
        return getattr(mk, "fused_experts", None) or getattr(mk, "experts", None)

    def _get_quant_method(self: Any, layer: torch.nn.Module, prefix: str) -> Any:
        if isinstance(layer, RoutedExperts) and self.transform_config is not None:
            if getattr(layer, "ct_transforms", None) is None:
                w13_hd, w2_hd = _resolve_moe_hadamard_dims(self.transform_config, prefix, layer)
                if w13_hd is not None or w2_hd is not None:
                    layer.ct_transforms = _MoEHadamardTransform(w13_hd, w2_hd)
        method = orig_get_quant_method(self, layer, prefix)
        if isinstance(layer, RoutedExperts) and method is not None:
            cls = method.__class__
            if not getattr(cls, "_warp_quant_moe_process_wrapped", False):
                orig_qm_process = cls.process_weights_after_loading

                def _qm_process_weights(qm_self: Any, lyr: torch.nn.Module, *a: Any, **kw: Any) -> Any:
                    res = orig_qm_process(qm_self, lyr, *a, **kw)
                    ct_tfms = getattr(lyr, "ct_transforms", None)
                    if isinstance(ct_tfms, _MoEHadamardTransform):
                        ct_tfms.process_weights_after_loading(_get_moe_experts_obj(lyr))
                    return res

                cls.process_weights_after_loading = _qm_process_weights
                cls._warp_quant_moe_process_wrapped = True
        return method

    CompressedTensorsConfig.get_quant_method = _get_quant_method

    orig_forward_modular = RoutedExperts.forward_modular
    orig_process_weights = getattr(RoutedExperts, "process_weights_after_loading", None)

    def _routed_forward_modular(self: Any, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        ct_tfms = getattr(self, "ct_transforms", None)
        if isinstance(ct_tfms, _MoEHadamardTransform):
            experts = _get_moe_experts_obj(self)
            quant_cfg = getattr(experts, "quant_config", None)
            if not (quant_cfg is not None and getattr(quant_cfg, "w1_hadamard_block_size", None)):
                x = ct_tfms(x)
        return orig_forward_modular(self, x, *args, **kwargs)

    RoutedExperts.forward_modular = _routed_forward_modular
    if orig_process_weights is not None:

        def _routed_process_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
            res = orig_process_weights(self, *args, **kwargs)
            ct_tfms = getattr(self, "ct_transforms", None)
            if isinstance(ct_tfms, _MoEHadamardTransform):
                ct_tfms.process_weights_after_loading(_get_moe_experts_obj(self))
            return res

        RoutedExperts.process_weights_after_loading = _routed_process_weights
    CompressedTensorsConfig._warp_quant_moe_patched = True


def _patch_ct_w2a16() -> None:
    try:
        scalar_types = importlib.import_module("vllm.scalar_type").scalar_types
    except Exception:
        return

    # 1. Fix Humming schema mapping for uint2 -> scalar_types.uint2b2 if present
    try:
        hum_schema = importlib.import_module("vllm.model_executor.layers.quantization.utils.humming.schema")
        humming_dtypes = importlib.import_module("vllm.utils.humming").dtypes

        orig_map_fn = hum_schema._humming_to_quant_dtype

        def _patched_humming_to_quant_dtype() -> dict[Any, Any]:
            m = dict(orig_map_fn())
            if hasattr(humming_dtypes, "uint2"):
                m[humming_dtypes.uint2] = scalar_types.uint2b2
            return m

        hum_schema._humming_to_quant_dtype = _patched_humming_to_quant_dtype
    except Exception:
        pass

    # 2. Patch CompressedTensorsWNA16 (LinearBase) for num_bits == 2 when Humming is not used
    try:
        wna16_mod = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16"
        )
        CompressedTensorsWNA16 = wna16_mod.CompressedTensorsWNA16
        param_mod = importlib.import_module("vllm.model_executor.parameter")
        ChannelQuantScaleParameter = param_mod.ChannelQuantScaleParameter
        GroupQuantScaleParameter = param_mod.GroupQuantScaleParameter
        PackedColumnParameter = param_mod.PackedColumnParameter
        PackedvLLMParameter = param_mod.PackedvLLMParameter
        RowvLLMParameter = param_mod.RowvLLMParameter
        has_humming = importlib.import_module("vllm.utils.import_utils").has_humming
    except Exception:
        return

    if getattr(CompressedTensorsWNA16, "_warp_quant_w2_patched", False):
        return

    orig_linear_create = CompressedTensorsWNA16.create_weights
    orig_linear_process = CompressedTensorsWNA16.process_weights_after_loading
    orig_linear_apply = CompressedTensorsWNA16.apply_weights

    import inspect

    orig_linear_sig = inspect.signature(orig_linear_create)

    def _linear_create_weights(
        self: Any,
        layer: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.num_bits != 2 or has_humming():
            return orig_linear_create(self, layer, *args, **kwargs)

        bound = orig_linear_sig.bind_partial(self, layer, *args, **kwargs)
        b_args = bound.arguments
        if "kwargs" in b_args and isinstance(b_args["kwargs"], dict):
            extra_kw = b_args["kwargs"]
        else:
            extra_kw = {}
        output_partition_sizes = b_args.get("output_partition_sizes") or extra_kw["output_partition_sizes"]
        input_size_per_partition = (
            b_args.get("input_size_per_partition") or extra_kw["input_size_per_partition"]
        )
        params_dtype = b_args.get("params_dtype") or extra_kw["params_dtype"]
        weight_loader = b_args.get("weight_loader") or extra_kw["weight_loader"]

        repack_to_marlin = os.environ.get("VLLM_WARP_QUANT_W2_MARLIN", "0") == "1"
        if repack_to_marlin:
            # Initialize Marlin kernel for uint4b8, then allocate 2-bit weight_packed for loading
            self.num_bits = 4
            self.pack_factor = Fraction(32, 4)
            self.quant_type = scalar_types.uint4b8
            orig_linear_create(self, layer, *args, **kwargs)
            self.num_bits = 2
            self.pack_factor = Fraction(32, 2)
            self.quant_type = scalar_types.uint2b2
            output_size_per_partition = sum(output_partition_sizes)
            qweight = PackedvLLMParameter(
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
                packed_factor=self.pack_factor,
                packed_dim=1,
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // 16,
                    dtype=torch.int32,
                ),
            )
            layer.register_parameter("weight_packed", qweight)
            layer._warp_quant_w2_mode = "marlin_repack"
            return

        # Default: keep weights in true 2-bit [N, K // 16] int32 in GPU VRAM
        output_size_per_partition = sum(output_partition_sizes)
        shape = (output_size_per_partition, input_size_per_partition // 16)
        qweight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=Fraction(32, 2),
            packed_dim=1,
            data=torch.empty(shape, dtype=torch.int32),
        )
        layer.register_parameter("weight_packed", qweight)

        is_sharded = not (self.group_size == -1 or self.has_g_idx)
        scale_cls = GroupQuantScaleParameter if is_sharded else ChannelQuantScaleParameter
        scale_args: dict[str, Any] = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                1 if self.group_size == -1 else input_size_per_partition // self.group_size,
                dtype=params_dtype,
            ),
        }
        if not is_sharded:
            scale_args["output_dim"] = 0
        else:
            scale_args["output_dim"] = 0
            scale_args["input_dim"] = 1
        layer.register_parameter("weight_scale", scale_cls(**scale_args))

        weight_shape = RowvLLMParameter(
            data=torch.empty(len(output_partition_sizes), 2, dtype=torch.int32),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_shape", weight_shape)
        if not self.symmetric:
            qzeros = PackedColumnParameter(
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
                packed_factor=Fraction(32, 2),
                packed_dim=0,
                data=torch.empty(
                    output_size_per_partition // 16,
                    input_size_per_partition // self.group_size,
                    dtype=torch.int32,
                ),
            )
            layer.register_parameter("weight_zero_point", qzeros)
        layer._warp_quant_w2_mode = "native_2bit"

    def _linear_process_weights(self: Any, layer: torch.nn.Module) -> None:
        mode = getattr(layer, "_warp_quant_w2_mode", None)
        if mode == "marlin_repack":
            w4 = repack_uint2b2_to_uint4b8(layer.weight_packed.data)
            layer.weight_packed = torch.nn.Parameter(w4, requires_grad=False)
            self.num_bits = 4
            self.pack_factor = Fraction(32, 4)
            self.quant_type = scalar_types.uint4b8
        elif mode == "native_2bit":
            layer.weight_packed = torch.nn.Parameter(
                layer.weight_packed.data.contiguous(), requires_grad=False
            )
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data.contiguous(), requires_grad=False)
            return
        orig_named_params = layer.named_parameters
        try:
            layer.named_parameters = (  # type: ignore[method-assign]
                lambda prefix="", recurse=False, remove_duplicate=True: orig_named_params(
                    prefix=prefix, recurse=False, remove_duplicate=remove_duplicate
                )
            )
            return orig_linear_process(self, layer)
        finally:
            if "named_parameters" in layer.__dict__:
                del layer.__dict__["named_parameters"]

    def _linear_apply_weights(
        self: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_warp_quant_w2_mode", None) == "native_2bit":
            w_dq = unpack_dequant_uint2b2(
                layer.weight_packed,
                layer.weight_scale,
                group_size=self.group_size,
                out_dtype=x.dtype,
            )
            return torch.nn.functional.linear(x, w_dq, bias)
        return orig_linear_apply(self, layer, x, bias)

    CompressedTensorsWNA16.create_weights = _linear_create_weights
    CompressedTensorsWNA16.process_weights_after_loading = _linear_process_weights
    CompressedTensorsWNA16.apply_weights = _linear_apply_weights
    CompressedTensorsWNA16._warp_quant_w2_patched = True

    # 3. Patch CompressedTensorsWNA16MoEMethod for num_bits == 2 when Humming is not used
    try:
        moe_wna16_mod = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16"
        )
        CompressedTensorsWNA16MoEMethod = moe_wna16_mod.CompressedTensorsWNA16MoEMethod
    except Exception:
        return

    orig_moe_init = CompressedTensorsWNA16MoEMethod.__init__
    orig_moe_create = CompressedTensorsWNA16MoEMethod.create_weights
    orig_moe_process = CompressedTensorsWNA16MoEMethod.process_weights_after_loading

    def _apply_native_2bit_moe(
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        ct_tfms = getattr(layer, "ct_transforms", None)
        w13_hd = getattr(ct_tfms, "w13_head_dim", None)
        w2_hd = getattr(ct_tfms, "w2_head_dim", None)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        if w13_hd is not None and w13_hd > 1:
            x_flat = fast_block_hadamard(x_flat, w13_hd)

        top_k = topk_ids.shape[-1]
        flat_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1).to(x_flat.dtype)

        expert_map = getattr(layer, "expert_map", None)
        if expert_map is not None:
            flat_ids = torch.where(
                flat_ids < 0,
                flat_ids,
                expert_map[flat_ids.clamp(min=0)],
            )

        is_transposed = bool(
            getattr(layer.w13_weight_packed, "is_transposed", False)
            or getattr(getattr(layer, "quant_method", None), "is_transposed", False)
        )
        out = torch.zeros_like(x_flat)
        unique_experts = torch.unique(flat_ids)
        for exp_tensor in unique_experts:
            e = int(exp_tensor.item())
            if e < 0:
                continue
            sel = torch.nonzero(flat_ids == e, as_tuple=False).squeeze(-1)
            tok_idx = sel // top_k
            inp = x_flat.index_select(0, tok_idx)

            w13_p = layer.w13_weight_packed[e].t() if is_transposed else layer.w13_weight_packed[e]
            w13_s = layer.w13_weight_scale[e].t() if is_transposed else layer.w13_weight_scale[e]
            w13_dq = unpack_dequant_uint2b2(
                w13_p,
                w13_s,
                group_size=group_size,
                out_dtype=x_flat.dtype,
            )
            gate_up = torch.nn.functional.linear(inp, w13_dq)
            inter_dim = gate_up.shape[-1] // 2
            act = torch.nn.functional.silu(gate_up[:, :inter_dim]) * gate_up[:, inter_dim:]
            if w2_hd is not None and w2_hd > 1:
                act = fast_block_hadamard(act, w2_hd)

            w2_p = layer.w2_weight_packed[e].t() if is_transposed else layer.w2_weight_packed[e]
            w2_s = layer.w2_weight_scale[e].t() if is_transposed else layer.w2_weight_scale[e]
            w2_dq = unpack_dequant_uint2b2(
                w2_p,
                w2_s,
                group_size=group_size,
                out_dtype=x_flat.dtype,
            )
            down = torch.nn.functional.linear(act, w2_dq)
            down = down * flat_weights.index_select(0, sel).unsqueeze(-1)
            out.index_add_(0, tok_idx, down)

        return out.reshape(orig_shape)

    def _moe_init(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Support both vLLM signatures:
        # - (self, weight_quant, input_quant, moe, layer_name=None) [vLLM 0.26.x]
        # - (self, quant_config, moe, layer_quant_config=None) [vLLM main / older]
        first_arg = args[0] if args else kwargs.get("weight_quant") or kwargs.get("quant_config")
        if hasattr(first_arg, "num_bits"):
            # First signature: (weight_quant, input_quant, moe, ...)
            weight_quant = first_arg
            if getattr(weight_quant, "num_bits", None) == 2:
                try:
                    orig_moe_init(self, *args, **kwargs)
                    return
                except (ValueError, NotImplementedError, KeyError, AssertionError):
                    patched_wq = weight_quant.model_copy(update={"num_bits": 4})
                    new_args = (patched_wq, *args[1:]) if args else ()
                    new_kwargs = dict(kwargs)
                    if not args and "weight_quant" in new_kwargs:
                        new_kwargs["weight_quant"] = patched_wq
                    orig_moe_init(self, *new_args, **new_kwargs)
                    self.num_bits = 2
                    self.packed_factor = Fraction(32, 2)
                    self.quant_type = scalar_types.uint2b2
                    repack_to_marlin = os.environ.get("VLLM_WARP_QUANT_W2_MARLIN", "0") == "1"
                    self._warp_quant_w2_moe_mode = "marlin_repack" if repack_to_marlin else "native_2bit"
                    return
            orig_moe_init(self, *args, **kwargs)
            return

        quant_config = first_arg
        moe = args[1] if len(args) > 1 else kwargs.get("moe")
        layer_quant_config = args[2] if len(args) > 2 else kwargs.get("layer_quant_config")
        scheme_map = getattr(quant_config, "target_scheme_map", None) or {}
        eff_lqc = layer_quant_config if layer_quant_config is not None else scheme_map.get("Linear")
        weight_quant = eff_lqc.get("weights") if isinstance(eff_lqc, dict) else None
        if weight_quant is not None and getattr(weight_quant, "num_bits", None) == 2 and not has_humming():
            patched_wq = weight_quant.model_copy(update={"num_bits": 4})
            patched_lqc = dict(eff_lqc)
            patched_lqc["weights"] = patched_wq
            orig_moe_init(self, quant_config, moe, layer_quant_config=patched_lqc)
            self.num_bits = 2
            self.packed_factor = Fraction(32, 2)
            self.quant_type = scalar_types.uint2b2
            repack_to_marlin = os.environ.get("VLLM_WARP_QUANT_W2_MARLIN", "0") == "1"
            self._warp_quant_w2_moe_mode = "marlin_repack" if repack_to_marlin else "native_2bit"
            return
        orig_moe_init(self, quant_config, moe, layer_quant_config=layer_quant_config)

    def _moe_create_weights(
        self: Any,
        layer: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        orig_moe_create(self, layer, *args, **kwargs)
        mode = getattr(self, "_warp_quant_w2_moe_mode", None)
        if mode is not None:
            layer._warp_quant_w2_moe_mode = mode
            layer._warp_quant_w2_group_size = self.group_size

    def _moe_process_weights(self: Any, layer: torch.nn.Module) -> None:
        mode = getattr(self, "_warp_quant_w2_moe_mode", None)
        if mode == "marlin_repack":
            is_transposed = bool(
                getattr(layer.w13_weight_packed, "is_transposed", False)
                or getattr(self, "is_transposed", False)
            )
            w13_data = layer.w13_weight_packed.data
            w2_data = layer.w2_weight_packed.data
            if is_transposed:
                w13_repacked = (
                    repack_uint2b2_to_uint4b8(w13_data.transpose(1, 2).contiguous())
                    .transpose(1, 2)
                    .contiguous()
                )
                w2_repacked = (
                    repack_uint2b2_to_uint4b8(w2_data.transpose(1, 2).contiguous())
                    .transpose(1, 2)
                    .contiguous()
                )
            else:
                w13_repacked = repack_uint2b2_to_uint4b8(w13_data)
                w2_repacked = repack_uint2b2_to_uint4b8(w2_data)
            layer.w13_weight_packed = torch.nn.Parameter(w13_repacked, requires_grad=False)
            layer.w2_weight_packed = torch.nn.Parameter(w2_repacked, requires_grad=False)
            self.num_bits = 4
            self.packed_factor = Fraction(32, 4)
            self.quant_type = scalar_types.uint4b8
            self.weight_quant = self.weight_quant.model_copy(update={"num_bits": 4})
            return orig_moe_process(self, layer)
        if mode == "native_2bit":
            layer.w13_weight_packed = torch.nn.Parameter(
                layer.w13_weight_packed.data.contiguous(), requires_grad=False
            )
            layer.w2_weight_packed = torch.nn.Parameter(
                layer.w2_weight_packed.data.contiguous(), requires_grad=False
            )
            layer.w13_weight_scale = torch.nn.Parameter(
                layer.w13_weight_scale.data.contiguous(), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                layer.w2_weight_scale.data.contiguous(), requires_grad=False
            )
            # Ensure get_fused_moe_quant_config succeeds if called by MoERunner
            self.num_bits = 4
            return
        return orig_moe_process(self, layer)

    CompressedTensorsWNA16MoEMethod.__init__ = _moe_init
    CompressedTensorsWNA16MoEMethod.create_weights = _moe_create_weights
    CompressedTensorsWNA16MoEMethod.process_weights_after_loading = _moe_process_weights

    try:
        RoutedExperts = importlib.import_module("vllm.model_executor.layers.fused_moe.layer").RoutedExperts

        orig_routed_forward = RoutedExperts.forward_modular

        def _routed_forward_with_w2(
            self: Any,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            if getattr(self, "_warp_quant_w2_moe_mode", None) == "native_2bit":
                out = _apply_native_2bit_moe(
                    self,
                    x,
                    topk_weights,
                    topk_ids,
                    group_size=int(getattr(self, "_warp_quant_w2_group_size", 64)),
                )
                if getattr(self, "reduce_results", False) and getattr(self, "tp_size", 1) > 1:
                    return self.maybe_all_reduce_tensor_model_parallel(out)
                return out
            return orig_routed_forward(self, x, topk_weights, topk_ids, *args, **kwargs)

        RoutedExperts.forward_modular = _routed_forward_with_w2
    except Exception:
        pass


def _patch_qwen4_exp_ple_lut4() -> None:
    try:
        common_ple = importlib.import_module("vllm.models.qwen4_exp.common.ngram_embedding")
        nvidia_ple = importlib.import_module("vllm.models.qwen4_exp.nvidia.ngram_embedding")
        Qwen4ExpPLEEmbeddingMethod = common_ple.Qwen4ExpPLEEmbeddingMethod
        Qwen4ExpNGramEmbedding = nvidia_ple.Qwen4ExpNGramEmbedding
    except Exception:
        return

    if getattr(Qwen4ExpPLEEmbeddingMethod, "_warp_quant_lut4_patched", False):
        return

    class Qwen4ExpPLECustomLut4EmbeddingMethod(CustomLut4PLEMethod, Qwen4ExpPLEEmbeddingMethod):
        """vLLM subclass of CustomLut4PLEMethod and Qwen4ExpPLEEmbeddingMethod."""

    orig_from_quant_config = Qwen4ExpPLEEmbeddingMethod.from_quant_config

    @classmethod
    def _from_quant_config(
        cls: Any,
        quant_config: Any,
        hf_text_config: Any = None,
    ) -> Qwen4ExpPLEEmbeddingMethod:
        ple_cfg = getattr(hf_text_config, "ple_quantization", None)
        if isinstance(ple_cfg, dict) and ple_cfg.get("format") == "custom_lut4_g32_fp16scale":
            return Qwen4ExpPLECustomLut4EmbeddingMethod(group_size=int(ple_cfg.get("group_size", 32)))
        return orig_from_quant_config(quant_config, hf_text_config)

    Qwen4ExpPLEEmbeddingMethod.from_quant_config = _from_quant_config

    # Patch dequantize call sites on Device & PinnedHost PLE embeddings to pass shard_idx when needed
    orig_device_forward = common_ple.Qwen4ExpPLEDeviceEmbedding.forward_SingleShard
    orig_pinned_forward = common_ple.Qwen4ExpPLEPinnedHostEmbedding.forward_SingleShard
    orig_gather_uva = common_ple.Qwen4ExpPLEPinnedHostEmbedding._gather_rows_uva

    def _gather_rows_uva(self: Any, shard_idx: int, flat_ids: torch.Tensor) -> torch.Tensor:
        if isinstance(self.quant_method, Qwen4ExpPLECustomLut4EmbeddingMethod):
            self.quant_method._current_shard_idx = shard_idx
        return orig_gather_uva(self, shard_idx, flat_ids)

    common_ple.Qwen4ExpPLEPinnedHostEmbedding._gather_rows_uva = _gather_rows_uva

    def _device_forward_single_shard(
        self: Any, shard_idx: int, flat_ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.quant_method, Qwen4ExpPLECustomLut4EmbeddingMethod):
            rows = self.weights[shard_idx].index_select(0, flat_ids)
            out = self.quant_method.dequantize(
                rows,
                self.embedding_dim_per_shard,
                self.params_dtype,
                shard_idx=shard_idx,
            )
            return out * valid_mask.unsqueeze(-1).to(out.dtype)
        return orig_device_forward(self, shard_idx, flat_ids, valid_mask)

    def _pinned_forward_single_shard(
        self: Any, shard_idx: int, flat_ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.quant_method, Qwen4ExpPLECustomLut4EmbeddingMethod):
            if self._prefetched_out is not None:
                return self._prefetched_out[shard_idx]
            rows = self._gather_rows_uva(shard_idx, flat_ids)
            out = self.quant_method.dequantize(
                rows,
                self.embedding_dim_per_shard,
                self.params_dtype,
                shard_idx=shard_idx,
            )
            return out * valid_mask.unsqueeze(-1).to(out.dtype)
        return orig_pinned_forward(self, shard_idx, flat_ids, valid_mask)

    common_ple.Qwen4ExpPLEDeviceEmbedding.forward_SingleShard = _device_forward_single_shard
    common_ple.Qwen4ExpPLEPinnedHostEmbedding.forward_SingleShard = _pinned_forward_single_shard

    orig_load_weights = Qwen4ExpNGramEmbedding.load_weights

    def _load_weights(self: Any, weights: Any) -> set[str]:
        quant_method = getattr(self.backend, "quant_method", None)
        if not isinstance(quant_method, Qwen4ExpPLECustomLut4EmbeddingMethod):
            return orig_load_weights(self, weights)

        loaded: set[str] = set()
        pending: dict[int, dict[str, torch.Tensor]] = {}
        for name, tensor in weights:
            for shard_idx in range(self.num_shards):
                prefix = f"shard_{shard_idx}."
                if name.startswith(prefix):
                    suffix = name[len(prefix) :]
                    pending.setdefault(shard_idx, {})[suffix] = tensor
                    loaded.add(name)
                    break

        packed_by_shard: dict[int, torch.Tensor] = {}
        for shard_idx, parts in pending.items():
            packed_by_shard[shard_idx] = quant_method.pack_shard_weights(
                shard_idx=shard_idx,
                parts=parts,
                embedding_dim_per_shard=self.embedding_dim_per_shard,
                tp_size=getattr(self, "tp_size", 1),
                tp_rank=getattr(self, "tp_rank", 0),
            )
        for shard_idx, combined_row_bytes in packed_by_shard.items():
            self.backend.load_shard_weight(shard_idx, combined_row_bytes)

        return loaded

    Qwen4ExpNGramEmbedding.load_weights = _load_weights
    nvidia_ple.Qwen4ExpPLECustomLut4EmbeddingMethod = Qwen4ExpPLECustomLut4EmbeddingMethod
    common_ple.Qwen4ExpPLECustomLut4EmbeddingMethod = Qwen4ExpPLECustomLut4EmbeddingMethod
    Qwen4ExpPLEEmbeddingMethod._warp_quant_lut4_patched = True


def _install_subprocess_autoload() -> None:
    """Ensure spawned vLLM worker subprocesses auto-import and run `register()`
    even when loaded via Hugging Face `auto_map` (`--trust-remote-code`) without
    `warp-quant` installed as a package in the environment.
    """
    import shutil
    import tempfile
    from pathlib import Path

    if os.environ.get("WARP_QUANT_SUBPROCESS_AUTOLOAD") == "1":
        return
    src_file = Path(__file__).resolve()
    if not src_file.is_file():
        return
    try:
        boot_dir = Path(tempfile.gettempdir()) / f"warp_quant_vllm_boot_{os.getuid()}"
        boot_dir.mkdir(parents=True, exist_ok=True)
        dst_plugin = boot_dir / "vllm_warp_quant.py"
        if not dst_plugin.exists() or dst_plugin.read_bytes() != src_file.read_bytes():
            shutil.copyfile(src_file, dst_plugin)
        sitecust = boot_dir / "sitecustomize.py"
        sitecust_code = (
            "import os, sys\n"
            "_boot_dir = os.path.dirname(__file__)\n"
            "try:\n"
            "    import importlib as _il\n"
            "    _il.import_module('vllm_warp_quant').register()\n"
            "except Exception:\n"
            "    pass\n"
            "# Chain to any next sitecustomize on sys.path\n"
            "_orig_path = list(sys.path)\n"
            "try:\n"
            "    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(_boot_dir)]\n"
            "    import importlib.machinery, importlib.util\n"
            "    _spec = importlib.machinery.PathFinder.find_spec('sitecustomize', sys.path)\n"
            "    if _spec is not None and _spec.loader is not None:\n"
            "        _mod = importlib.util.module_from_spec(_spec)\n"
            "        _spec.loader.exec_module(_mod)\n"
            "except Exception:\n"
            "    pass\n"
            "finally:\n"
            "    sys.path = _orig_path\n"
        )
        if not sitecust.exists() or sitecust.read_text(encoding="utf-8") != sitecust_code:
            sitecust.write_text(sitecust_code, encoding="utf-8")
        cur_pp = os.environ.get("PYTHONPATH", "")
        boot_str = str(boot_dir)
        parts = [p for p in cur_pp.split(os.pathsep) if p]
        if boot_str not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([boot_str, *parts])
        os.environ["WARP_QUANT_SUBPROCESS_AUTOLOAD"] = "1"
    except Exception:
        pass


def register() -> None:
    """Entry point for `vllm.general_plugins` and HF `auto_map` preload."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    _patch_ct_linear_transform()
    _patch_ct_moe_transform()
    _patch_ct_w2a16()
    _patch_qwen4_exp_ple_lut4()


try:
    from transformers import PretrainedConfig

    class WarpQuantConfig(PretrainedConfig):
        """Hugging Face `AutoConfig` shim (`"auto_map": {"AutoConfig": "vllm_warp_quant.WarpQuantConfig"}`).

        When loaded with `trust_remote_code=True` (e.g. `vllm serve <repo> --trust-remote-code`),
        this runs `register()`, installs the `PYTHONPATH` subprocess bootstrapper so vLLM's
        spawned `EngineCore` and GPU worker processes also run `register()`, and delegates
        to the native `PretrainedConfig` class for the checkpoint's `model_type`.
        When loaded with `trust_remote_code=False`, `transformers.AutoConfig` cleanly ignores
        `auto_map` for known `model_type`s and uses the native config class directly.
        """

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str | os.PathLike[str],
            **kwargs: Any,
        ) -> PretrainedConfig:
            register()
            _install_subprocess_autoload()
            kwargs.pop("trust_remote_code", None)
            config_dict, kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
            model_type = config_dict.get("model_type")
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            if model_type in CONFIG_MAPPING:
                target_cls = CONFIG_MAPPING[model_type]
                return target_cls.from_dict(config_dict, **kwargs)
            return PretrainedConfig.from_dict(config_dict, **kwargs)

except Exception:
    pass

try:
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    class WarpQuantProcessor(ProcessorMixin):
        """Hugging Face `AutoProcessor` shim (`"auto_map": {"AutoProcessor": "vllm_warp_quant.WarpQuantProcessor"}`).

        vLLM calls `AutoProcessor.from_pretrained(..., trust_remote_code=trust_remote_code)` in
        both `process0` and `EngineCore` without overriding `trust_remote_code=False`.
        """

        @classmethod
        def register_for_auto_class(cls, auto_class: str = "AutoProcessor") -> None:
            del auto_class

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str | os.PathLike[str],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            register()
            _install_subprocess_autoload()
            kwargs.pop("trust_remote_code", None)
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                trust_remote_code=False,
                **kwargs,
            )

    class WarpQuantTokenizer(PreTrainedTokenizerBase):
        """Hugging Face `AutoTokenizer` shim (`"auto_map": {"AutoTokenizer": [..., "vllm_warp_quant.WarpQuantTokenizer"]}`)."""

        @classmethod
        def register_for_auto_class(cls, auto_class: str = "AutoTokenizer") -> None:
            del auto_class

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str | os.PathLike[str],
            *inputs: Any,
            **kwargs: Any,
        ) -> Any:
            register()
            _install_subprocess_autoload()
            kwargs.pop("trust_remote_code", None)
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                *inputs,
                trust_remote_code=False,
                **kwargs,
            )

except Exception:
    pass
