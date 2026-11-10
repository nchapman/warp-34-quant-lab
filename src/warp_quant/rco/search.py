"""Multi-option Riemannian Constrained Optimization (RCO) search over a frozen candidate bank."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils import parametrize

from warp_quant.rco import allocation
from warp_quant.rco.options import decode_candidate


class _Weight(nn.Module):
    """Multi-option Gumbel-STE parametrization supporting either compact (codes, scales) or dense BF16 candidates."""

    def __init__(
        self,
        options_list: list,
        state: dict,
        group_idx: int,
        device: torch.device,
        *,
        cache_decoded: bool = False,
    ) -> None:
        super().__init__()
        self.state = state
        self.group_idx = group_idx
        self.num_options = len(options_list)
        is_compact_input = isinstance(options_list[0], dict) and "codes" in options_list[0]
        self._is_compact = is_compact_input and not cache_decoded
        self._group_sizes: list[int] = []
        self._code_offsets: list[float] = []
        if is_compact_input and cache_decoded:
            for k, item in enumerate(options_list):
                dec = decode_candidate(
                    item["codes"].detach().to(device=device),
                    item["scales"].detach().to(device=device),
                    int(item["group_size"]),
                    dtype=torch.bfloat16,
                    code_offset=float(item.get("code_offset", 0.0)),
                )
                self.register_buffer(f"dense_{k}", dec)
        elif self._is_compact:
            for k, item in enumerate(options_list):
                self.register_buffer(f"codes_{k}", item["codes"].detach().to(device=device, dtype=torch.int8))
                self.register_buffer(
                    f"scales_{k}", item["scales"].detach().to(device=device, dtype=torch.bfloat16)
                )
                self._group_sizes.append(int(item["group_size"]))
                self._code_offsets.append(float(item.get("code_offset", 0.0)))
        else:
            for k, tensor in enumerate(options_list):
                self.register_buffer(f"dense_{k}", tensor.detach().to(device=device, dtype=torch.bfloat16))

    def _decode(self, k: int) -> torch.Tensor:
        if self._is_compact:
            codes = getattr(self, f"codes_{k}")
            scales = getattr(self, f"scales_{k}")
            return decode_candidate(
                codes,
                scales,
                self._group_sizes[k],
                dtype=torch.bfloat16,
                code_offset=self._code_offsets[k],
            )
        return getattr(self, f"dense_{k}")

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        choice = self.state["choices"][self.group_idx]
        hard = self._decode(choice)
        soft = self.state.get("soft")
        if soft is None:
            return hard
        gates = soft[self.group_idx].to(device=hard.device, dtype=torch.float32)
        base_f32 = self._decode(0).float() if choice != 0 else hard.float()
        out_f32 = hard.float()
        for k in range(1, self.num_options):
            g_k = gates[k]
            w_k = hard.float() if choice == k else self._decode(k).float()
            out_f32 = out_f32 + (g_k - g_k.detach()) * (w_k - base_f32)
        return out_f32.to(hard.dtype)


def _normal(alpha: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    p = alpha.softmax(-1)
    return p * (costs - (p * costs).sum(-1, keepdim=True))


def _project(vector: torch.Tensor, alpha: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    n = _normal(alpha, costs)
    norm = n.square().sum()
    return vector - n * ((vector * n).sum() / norm) if norm > 0 else vector


@torch.no_grad()
def _retract(alpha: torch.Tensor, costs: torch.Tensor, target: float) -> None:
    def expected(shift: float) -> float:
        return ((alpha + shift * costs).softmax(-1) * costs).sum().item()

    lo, hi = -1.0, 1.0
    for _ in range(80):
        if expected(lo) <= target <= expected(hi):
            break
        lo *= 2.0
        hi *= 2.0
    else:
        raise ValueError("soft budget has no finite retraction")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if expected(mid) < target:
            lo = mid
        else:
            hi = mid
    alpha.add_(((lo + hi) / 2.0) * costs)
    alpha.sub_(alpha.mean(-1, keepdim=True))


def build_sensitivity_logits(
    groups: list[list[str]],
    candidates: dict[str, list],
    costs: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    """Initialize group option preference logits from log-distortion (`-(distortion + 1e-8).log()`)."""
    num_groups = len(groups)
    num_opts = len(costs[0])
    distortion = torch.zeros((num_groups, num_opts), dtype=torch.float64)
    for g_idx, group in enumerate(groups):
        high_err = sum(float(candidates[m][num_opts - 1].get("weight_mse", 0.0)) for m in group)
        for k in range(num_opts - 1):
            opt_err = sum(float(candidates[m][k].get("weight_mse", 0.0)) for m in group)
            distortion[g_idx, k] = max(opt_err - high_err, 0.0)
    logits = -(distortion + 1e-8).log()
    return logits - logits.mean(-1, keepdim=True)


class SearchModel:
    """Context-managed multi-option RCO parametrization over a frozen candidate bank."""

    def __init__(
        self,
        model: nn.Module,
        groups: list[list[str]],
        candidates: dict[str, list],
        costs: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
        *,
        control_device: str = "cpu",
    ) -> None:
        self.model = model
        self.groups = [list(g) for g in groups]
        self.costs = tuple(tuple(int(c) for c in row) for row in costs)
        names = [n for g in self.groups for n in g]
        if not self.groups or any(not g for g in self.groups) or len(set(names)) != len(names):
            raise ValueError("groups must be non-empty and disjoint")
        if set(names) != set(candidates) or len(self.costs) != len(self.groups):
            raise ValueError("groups, candidates, and costs must align")
        num_opts = len(self.costs[0])
        if num_opts < 2 or any(len(row) != num_opts for row in self.costs):
            raise ValueError("all groups must have the same number of options (>= 2)")
        allocation.solve(
            self.costs, [[0.0] * num_opts for _ in self.groups], sum(min(r) for r in self.costs), exact=True
        )
        for name in names:
            m = model.get_submodule(name)
            if parametrize.is_parametrized(m, "weight"):
                raise ValueError(f"existing weight parametrization unsupported on {name}")
            opts = candidates[name]
            if len(opts) != num_opts:
                raise ValueError(f"candidate option count mismatch on {name}")
        self.candidates = candidates
        self.num_options = num_opts
        self.logits = nn.Parameter(
            torch.zeros(len(self.groups), num_opts, dtype=torch.float64, device=control_device)
        )
        self.state: dict = {"choices": (0,) * len(self.groups), "soft": None}
        self._installed: list[nn.Module] = []
        self._offloaded: dict[nn.Module, tuple[torch.Tensor, torch.device]] = {}
        self._flags: list[tuple[nn.Parameter, bool]] = []
        self._modes: list[tuple[nn.Module, bool]] = []
        self._active = False

    def __enter__(self) -> SearchModel:
        if self._active:
            raise RuntimeError("SearchModel already active")
        self._flags = [(p, p.requires_grad) for p in self.model.parameters()]
        self._modes = [(m, m.training) for m in self.model.modules()]
        self._active = True
        self.model.requires_grad_(False)
        self.model.eval()
        total_decoded_bytes = sum(
            self.model.get_submodule(name).weight.numel() * 2 * self.num_options
            for g in self.groups
            for name in g
        )
        cache_decoded = total_decoded_bytes < 40 * (1024**3)
        try:
            for i, group in enumerate(self.groups):
                for name in group:
                    m = self.model.get_submodule(name)
                    dev = m.weight.device
                    if not cache_decoded and dev.type == "cuda":
                        self._offloaded[m] = (m.weight.data.cpu(), dev)
                        m.weight.data = torch.empty(0, dtype=m.weight.dtype, device=dev)
                    parametrize.register_parametrization(
                        m,
                        "weight",
                        _Weight(self.candidates[name], self.state, i, dev, cache_decoded=cache_decoded),
                        unsafe=True,
                    )
                    self._installed.append(m)
            if self._offloaded and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.state["soft"] = None
        for m in reversed(self._installed):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=False)
            if m in self._offloaded:
                w_cpu, dev = self._offloaded.pop(m)
                m.weight.data = w_cpu.to(device=dev)
        self._installed.clear()
        self._offloaded.clear()
        for p, flag in self._flags:
            p.requires_grad_(flag)
        for m, mode in self._modes:
            m.training = mode
        self._active = False

    def set_assignment(
        self, choices: tuple[int, ...] | list[int], *, soft: torch.Tensor | None = None
    ) -> None:
        if len(choices) != len(self.groups) or any(c not in range(self.num_options) for c in choices):
            raise ValueError("invalid option choices")
        self.state.update(choices=tuple(choices), soft=soft)

    def optimize(
        self,
        forward_loss,
        *,
        steps: int,
        gumbel_samples: int,
        target_bytes: int,
        seed: int,
        lr: float = 0.1,
        sensitivity_logits: torch.Tensor | None = None,
        initialization: str | None = None,
        on_step=None,
    ) -> tuple[allocation.Allocation, list[dict]]:
        """Optimize group allocation logits on the exact-budget Riemannian manifold."""
        if not self._active:
            raise RuntimeError("use SearchModel as a context manager")
        if steps < 1 or gumbel_samples < 1 or lr <= 0:
            raise ValueError("positive steps, gumbel_samples, and lr required")
        allocation.solve(
            self.costs, [[0.0] * self.num_options for _ in self.groups], target_bytes, exact=True
        )
        low = sum(min(r) for r in self.costs)
        high = sum(max(r) for r in self.costs)
        if not low < target_bytes < high:
            raise ValueError("search requires an interior soft budget; use set_assignment at endpoints")
        if sensitivity_logits is None and initialization != "uniform_smoke":
            raise ValueError("supply sensitivity_logits or explicitly select uniform_smoke")
        alpha = self.logits
        init = (
            torch.zeros_like(alpha)
            if sensitivity_logits is None
            else torch.as_tensor(sensitivity_logits, device=alpha.device, dtype=alpha.dtype)
        )
        if init.shape != alpha.shape or not torch.isfinite(init).all():
            raise ValueError("invalid sensitivity logits")
        scale = float(high - low)
        c = (
            torch.tensor(
                [[v - min(row) for v in row] for row in self.costs],
                device=alpha.device,
                dtype=alpha.dtype,
            )
            / scale
        )
        target = (target_bytes - low) / scale
        with torch.no_grad():
            alpha.copy_(init)
        _retract(alpha, c, target)
        rng = torch.Generator(device=alpha.device).manual_seed(seed)
        m = torch.zeros_like(alpha)
        v = torch.zeros_like(alpha)
        history: list[dict] = []
        for step in range(steps):
            tau = 0.01 + 0.99 * (1.0 + math.cos(math.pi * step / max(1, steps - 1))) / 2.0
            alpha.grad = None
            losses, budgets = [], []
            for sample_idx in range(gumbel_samples):
                u = torch.rand(alpha.shape, generator=rng, device=alpha.device, dtype=alpha.dtype).clamp(
                    1e-12, 1.0 - 1e-12
                )
                scores = alpha - (-u.log()).log()
                chosen = allocation.solve(
                    self.costs, scores.detach().cpu().tolist(), target_bytes, exact=True
                )
                self.set_assignment(chosen.choices, soft=(scores / tau).softmax(-1))
                loss = forward_loss(self.model, step, sample_idx)
                if loss.ndim or not torch.isfinite(loss):
                    raise ValueError("callback must return finite scalar loss")
                (loss / gumbel_samples).backward()
                losses.append(float(loss.item()))
                budgets.append(chosen.total_bytes)
            if alpha.grad is None or not torch.isfinite(alpha.grad).all():
                raise ValueError("missing or nonfinite allocation gradient")
            with torch.no_grad():
                g = _project(alpha.grad, alpha, c)
                m.mul_(0.9).add_(g, alpha=0.1)
                v.mul_(0.999).addcmul_(g, g, value=0.001)
                direction = (m / (1.0 - 0.9 ** (step + 1))) / (
                    (v / (1.0 - 0.999 ** (step + 1))).sqrt() + 1e-8
                )
                alpha.add_(_project(direction, alpha, c), alpha=-lr)
                _retract(alpha, c, target)
                m.copy_(_project(m, alpha, c))
                expected = low + scale * (alpha.softmax(-1) * c).sum().item()
            rec = {
                "step": step,
                "loss": sum(losses) / len(losses),
                "tau": tau,
                "grad_norm": float(alpha.grad.norm().item()),
                "projected_grad_norm": float(g.norm().item()),
                "hard_bytes_min": min(budgets),
                "hard_bytes_max": max(budgets),
                "expected_bytes": expected,
                "soft_budget_error_bytes": expected - target_bytes,
            }
            history.append(rec)
            if on_step is not None:
                on_step(rec)
        final = allocation.solve(
            self.costs, alpha.detach().log_softmax(-1).cpu().tolist(), target_bytes, exact=True
        )
        self.set_assignment(final.choices)
        return final, history
