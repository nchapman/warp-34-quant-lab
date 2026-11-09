"""Declarative quantization options, MSE scale search, Cholesky GPTQ candidate solver, and lossless CT packing."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantOption:
    name: str
    num_bits: int
    group_size: int
    mse_scale: bool = True
    actorder: str | None = "weight"

    def validate(self) -> None:
        if self.num_bits not in (2, 3, 4, 8):
            raise ValueError(f"Unsupported num_bits={self.num_bits} in {self.name}")
        if self.group_size <= 0 or (self.group_size & (self.group_size - 1)) != 0:
            raise ValueError(f"group_size must be a positive power of 2, got {self.group_size}")
        if self.actorder not in (None, "none", "weight", "static"):
            raise ValueError(f"Unsupported actorder={self.actorder}")

    @property
    def bpw(self) -> float:
        return float(self.num_bits) + 16.0 / float(self.group_size)


OPTIONS: dict[str, QuantOption] = {
    "w2_g64": QuantOption("w2_g64", num_bits=2, group_size=64),
    "w2_g128": QuantOption("w2_g128", num_bits=2, group_size=128),
    "w4_g32": QuantOption("w4_g32", num_bits=4, group_size=32),
    "w4_g64": QuantOption("w4_g64", num_bits=4, group_size=64),
    "w4_g128": QuantOption("w4_g128", num_bits=4, group_size=128),
    "w8_g64": QuantOption("w8_g64", num_bits=8, group_size=64),
    "w8_g128": QuantOption("w8_g128", num_bits=8, group_size=128),
}

_SPEC_RE = re.compile(r"^w([248])_g(\d+)$")


def resolve_option(spec: str | QuantOption) -> QuantOption:
    if isinstance(spec, QuantOption):
        spec.validate()
        return spec
    if spec in OPTIONS:
        return OPTIONS[spec]
    m = _SPEC_RE.match(spec)
    if not m:
        raise ValueError(f"Unknown QuantOption spec '{spec}' (expected e.g. 'w2_g64', 'w4_g64', 'w8_g64')")
    opt = QuantOption(name=spec, num_bits=int(m.group(1)), group_size=int(m.group(2)))
    opt.validate()
    return opt


def resolve_options(
    specs: list[str | QuantOption] | tuple[str | QuantOption, ...],
) -> tuple[QuantOption, ...]:
    if len(specs) < 2:
        raise ValueError("RCO requires at least two quantization options")
    opts = tuple(resolve_option(s) for s in specs)
    names = [o.name for o in opts]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate quantization options: {names}")
    return opts


@torch.no_grad()
def find_mse_scale(
    w_f32: torch.Tensor, num_bits: int, group_size: int, grid_points: int = 81
) -> torch.Tensor:
    """Vectorized MSE clipping search for symmetric signed integer grid [-2^(b-1), 2^(b-1)-1]."""
    n_out, k_in = w_f32.shape
    if k_in % group_size != 0:
        raise ValueError(f"in_features={k_in} not divisible by group_size={group_size}")
    grouped = w_f32.view(n_out, k_in // group_size, group_size)
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    amax = grouped.abs().amax(dim=-1)
    # Use max(|qmin|, qmax) as denominator when num_bits==2 so negative tail (-2) is also covered at r=1.0
    denom = float(qmax) if num_bits > 2 else 1.5
    base_scale = (amax / denom).clamp(min=1e-7)
    best_scale = (amax / float(max(qmax, 1))).clamp(min=1e-7).to(torch.bfloat16).float()
    best_err = torch.full_like(base_scale, float("inf"))
    for r in torch.linspace(0.20, 1.05, grid_points, device=w_f32.device, dtype=torch.float32):
        cand = (base_scale * r).clamp(min=1e-7).to(torch.bfloat16).float()
        q = torch.clamp(torch.round(grouped / cand.unsqueeze(-1)), qmin, qmax) * cand.unsqueeze(-1)
        err = (grouped - q).square().sum(dim=-1)
        mask = err < best_err
        best_err = torch.where(mask, err, best_err)
        best_scale = torch.where(mask, cand, best_scale)
    return best_scale.to(torch.bfloat16)


def packed_byte_cost(shape: tuple[int, int], option: QuantOption) -> int:
    """Exact byte cost of compressed-tensors (weight_packed [int32], weight_scale [bf16], weight_shape [int64])."""
    n_out, k_in = shape
    if k_in % option.group_size != 0:
        raise ValueError(f"k_in={k_in} not divisible by group_size={option.group_size}")
    lanes = 32 // option.num_bits
    if k_in % lanes != 0:
        raise ValueError(f"k_in={k_in} not divisible by packing lanes={lanes}")
    packed_bytes = n_out * (k_in // lanes) * 4
    scale_bytes = n_out * (k_in // option.group_size) * 2
    shape_bytes = 2 * 8
    return packed_bytes + scale_bytes + shape_bytes


@torch.no_grad()
def refine_hessian_cd(
    weight_f32: torch.Tensor,
    norm_hessian: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    num_bits: int,
    group_size: int,
    *,
    passes: int = 4,
    dampening_frac: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Seidel coordinate descent over group scales and integer codes on Tr((W - Q) H (W - Q)^T)."""
    if passes <= 0:
        return codes, scales
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    w_f = weight_f32.detach().float()
    h_f = norm_hessian.detach().float().clone()
    diag_mean = h_f.diagonal().mean().clamp(min=1e-12)
    h_f.diagonal().add_(dampening_frac * diag_mean)
    h_diag = h_f.diagonal().clamp(min=1e-12)

    c_mat = codes.to(device=w_f.device, dtype=torch.float32).clone()
    s_mat = scales.to(device=w_f.device, dtype=torch.bfloat16).float().clone()
    n_out, k_in = w_f.shape
    num_groups = k_in // group_size
    idx = torch.arange(k_in, device=w_f.device).view(num_groups, group_size)
    h_blocks = h_f[idx.unsqueeze(2), idx.unsqueeze(1)].contiguous()

    q_mat = (c_mat.view(n_out, num_groups, group_size) * s_mat.unsqueeze(-1)).view(n_out, k_in)
    rh = (w_f - q_mat) @ h_f

    for _ in range(passes):
        for g in range(num_groups):
            g0, g1 = g * group_size, (g + 1) * group_size
            s_g = s_mat[:, g]
            c_g = c_mat[:, g0:g1]
            q_g_old = q_mat[:, g0:g1].clone()
            q_g = q_g_old.clone()
            rh_g = rh[:, g0:g1].clone()
            h_gg = h_blocks[g]
            hd_g = h_diag[g0:g1]

            # 1. Exact 1-group Newton scale step before column code sweep
            ch_g = c_g @ h_gg
            num = (c_g * rh_g).sum(dim=-1)
            den = (c_g * ch_g).sum(dim=-1).clamp(min=1e-12)
            s_new = (s_g + num / den).clamp(min=1e-7).to(torch.bfloat16).float()
            ds = s_new - s_g
            s_g = s_new
            q_g = c_g * s_g.unsqueeze(-1)
            rh_g = rh_g - ch_g * ds.unsqueeze(-1)

            # 2. Exact coordinate-descent code sweep within group g
            for j in range(group_size):
                q_j = q_g[:, j]
                q_star = q_j + rh_g[:, j] / hd_g[j]
                c_new = torch.clamp(torch.round(q_star / s_g), qmin, qmax)
                q_new = c_new * s_g
                dq = q_new - q_j
                c_g[:, j] = c_new
                q_g[:, j] = q_new
                rh_g.addmm_(dq.unsqueeze(1), h_gg[j : j + 1, :], alpha=-1.0)

            # 3. Exact 1-group Newton scale polish after column code sweep
            ch_g = c_g @ h_gg
            num = (c_g * rh_g).sum(dim=-1)
            den = (c_g * ch_g).sum(dim=-1).clamp(min=1e-12)
            s_new = (s_g + num / den).clamp(min=1e-7).to(torch.bfloat16).float()
            s_g = s_new
            q_g = c_g * s_g.unsqueeze(-1)

            s_mat[:, g] = s_g
            q_mat[:, g0:g1] = q_g
            dq_g = q_g - q_g_old
            rh.addmm_(dq_g, h_f[g0:g1, :], alpha=-1.0)

    return c_mat.to(torch.int8), s_mat.to(torch.bfloat16)


@torch.no_grad()
def quantize_candidate(
    weight_rot: torch.Tensor,
    hessian_rot: torch.Tensor,
    num_samples: torch.Tensor | int,
    option: QuantOption,
    *,
    dampening_frac: float = 0.01,
    blocksize: int = 128,
    cd_passes: int | None = None,
) -> dict[str, torch.Tensor | int | float | str]:
    """Run Cholesky GPTQ + Hessian coordinate descent for one QuantOption."""
    from compressed_tensors.quantization import QuantizationArgs
    from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight

    option.validate()
    if weight_rot.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape={tuple(weight_rot.shape)}")
    n_out, k_in = weight_rot.shape
    w_f32 = weight_rot.detach().float().clone()
    if option.mse_scale:
        scale = find_mse_scale(w_f32, option.num_bits, option.group_size)
    else:
        qmax = float(max((1 << (option.num_bits - 1)) - 1, 1))
        grouped = w_f32.view(n_out, k_in // option.group_size, option.group_size)
        scale = (grouped.abs().amax(dim=-1) / qmax).clamp(min=1e-7).to(torch.bfloat16)

    zp = torch.zeros_like(scale, dtype=torch.int32)
    actorder = None if option.actorder in (None, "none") else option.actorder
    qargs = QuantizationArgs(
        num_bits=option.num_bits,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=option.group_size,
        actorder=actorder,
    )
    if isinstance(num_samples, torch.Tensor):
        denom = num_samples.clamp(min=1).to(hessian_rot.device, dtype=torch.float32)
    else:
        denom = float(max(int(num_samples), 1))
    norm_h = (hessian_rot.detach().to(torch.float32) / denom).contiguous()

    # Pass clones because quantize_weight overwrites weights and hessians in-place under actorder!
    q_w_batch, losses, used_rtn = quantize_weight(
        weights=w_f32.clone().unsqueeze(0),
        hessians=norm_h.clone().unsqueeze(0),
        scale=scale.unsqueeze(0),
        zero_point=zp.unsqueeze(0),
        global_scale=None,
        quant_args=qargs,
        blocksize=blocksize,
        percdamp=dampening_frac,
    )
    if used_rtn[0].item():
        raise RuntimeError(f"GPTQ fell back to RTN for option {option.name}")
    q_w = q_w_batch[0]

    # Recover exact integer codes in [-2^(b-1), 2^(b-1)-1]
    qmin = -(1 << (option.num_bits - 1))
    qmax = (1 << (option.num_bits - 1)) - 1
    scale_exp = scale.float().repeat_interleave(option.group_size, dim=1)
    codes = torch.clamp(torch.round(q_w.float() / scale_exp), qmin, qmax).to(torch.int8)

    if cd_passes is None:
        cd_passes = 4 if option.num_bits <= 3 else (2 if option.num_bits == 4 else 0)
    if cd_passes > 0:
        codes, scale = refine_hessian_cd(
            w_f32,
            norm_h,
            codes,
            scale,
            option.num_bits,
            option.group_size,
            passes=cd_passes,
            dampening_frac=dampening_frac,
        )
        scale_exp = scale.float().repeat_interleave(option.group_size, dim=1)

    # Exact Hessian-weighted quadratic reconstruction error Tr((W - Q) H (W - Q)^T)
    decoded_f32 = codes.float() * scale_exp
    diff = w_f32 - decoded_f32
    hessian_loss = float((diff @ norm_h * diff).sum().item())
    weight_mse = float(diff.square().mean().item())
    weight_norm_sq = float(w_f32.square().sum().item())

    return {
        "option": option.name,
        "num_bits": option.num_bits,
        "group_size": option.group_size,
        "shape": (n_out, k_in),
        "codes": codes.contiguous().cpu(),
        "scales": scale.contiguous().cpu(),
        "bytes": packed_byte_cost((n_out, k_in), option),
        "hessian_loss": hessian_loss,
        "gptq_loss": float(losses[0].item()),
        "weight_mse": weight_mse,
        "rel_mse": float(diff.square().sum().item() / max(weight_norm_sq, 1e-12)),
    }


def decode_candidate(
    codes: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode signed int8 codes and BF16 group scales to a 2D weight tensor."""
    n_out, k_in = codes.shape
    return (
        (
            codes.to(torch.float32).view(n_out, k_in // group_size, group_size)
            * scales.to(torch.float32).unsqueeze(-1)
        )
        .view(n_out, k_in)
        .to(dtype)
    )


def pack_candidate(
    codes: torch.Tensor,
    scales: torch.Tensor,
    num_bits: int,
) -> dict[str, torch.Tensor]:
    """Lossless little-endian offset-binary int32 packing matching compressed-tensors."""
    if num_bits not in (2, 4, 8):
        raise ValueError(f"pack-quantized int32 packing requires num_bits in (2, 4, 8), got {num_bits}")
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    if (codes < qmin).any() or (codes > qmax).any():
        raise ValueError(f"codes out of signed {num_bits}-bit range [{qmin}, {qmax}]")
    lanes = 32 // num_bits
    n_out, k_in = codes.shape
    if k_in % lanes != 0:
        raise ValueError(f"k_in={k_in} not divisible by lanes={lanes}")
    unsigned = (codes.to(torch.int64) + (1 << (num_bits - 1))).reshape(n_out, k_in // lanes, lanes)
    shifts = torch.arange(lanes, device=codes.device, dtype=torch.int64) * num_bits
    packed = (unsigned << shifts).sum(dim=-1).to(torch.int32)
    return {
        "weight_packed": packed.contiguous(),
        "weight_scale": scales.to(torch.bfloat16).contiguous(),
        "weight_shape": torch.tensor([n_out, k_in], dtype=torch.int64, device=codes.device),
    }
