"""Post-final-norm BF16 teacher hidden-state cache and memory-bounded chunked full-vocabulary KL."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint


def _frozen_head(head: nn.Module) -> None:
    if parametrize.is_parametrized(head, "weight") or any(p.requires_grad for p in head.parameters()):
        raise ValueError("teacher head must be frozen and unparametrized")


def hidden_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    head: nn.Module,
    *,
    token_chunk_size: int = 128,
    checkpointed: bool = True,
) -> torch.Tensor:
    """Mean KL(reference || candidate) over every nonpadding next-token position.

    Unpadded [batch, sequence, hidden] inputs; teacher can reside on CPU. Each
    checkpoint includes BOTH head projections and FP32 normalization/reduction,
    so no vocabulary-sized activation survives a chunk until backward.
    """
    _frozen_head(head)
    if student.ndim != 3 or student.shape != teacher.shape or student.shape[1] < 2 or student.shape[0] < 1:
        raise ValueError("matching nonempty [batch, sequence>=2, hidden] required")
    if type(token_chunk_size) is not int or token_chunk_size < 1:
        raise ValueError("positive integer chunk size required")
    if (
        student.dtype != teacher.dtype
        or student.dtype != head.weight.dtype
        or student.shape[-1] != head.weight.shape[1]
        or student.device != head.weight.device
    ):
        raise ValueError("hidden/head dtype, width and student device must match")
    x = student[:, :-1].reshape(-1, student.shape[-1])
    ref = teacher.detach()[:, :-1].reshape(-1, teacher.shape[-1])

    def chunk_loss(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            lp = head(reference.to(candidate.device)).float().log_softmax(-1)
        cq = head(candidate).float().log_softmax(-1)
        return (lp.exp() * (lp - cq)).sum()

    total = student.new_zeros((), dtype=torch.float32)
    for start in range(0, len(x), token_chunk_size):
        args = (x[start : start + token_chunk_size], ref[start : start + token_chunk_size])
        loss = (
            checkpoint(chunk_loss, *args, use_reentrant=False)
            if checkpointed and torch.is_grad_enabled() and student.requires_grad
            else chunk_loss(*args)
        )
        total = total + loss
    return total / len(x)


@torch.no_grad()
def capture_teacher_hidden_states(
    model: nn.Module,
    search_rows: list[torch.Tensor],
    cache_dir: Path | None = None,
) -> list[torch.Tensor | Path]:
    """Capture post-final-norm BF16 hidden states [1, S, hidden] for search_rows."""
    if model.training:
        raise ValueError("teacher capture requires eval mode")
    device = next(model.parameters()).device
    lm = (
        model.model.language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model")
        else model.model
    )
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    saved: list[torch.Tensor | Path] = []
    for idx, ids in enumerate(search_rows):
        inp = ids.to(device=device, dtype=torch.long)
        if inp.ndim == 1:
            inp = inp.unsqueeze(0)
        hidden = (
            lm(input_ids=inp, use_cache=False, return_dict=True)
            .last_hidden_state.detach()
            .to(torch.bfloat16)
            .cpu()
        )
        if not torch.isfinite(hidden).all():
            raise ValueError(f"Non-finite teacher hidden state on search row {idx}")
        if cache_dir is not None:
            p = cache_dir / f"row_{idx:04d}.pt"
            torch.save(hidden, p)
            saved.append(p)
        else:
            saved.append(hidden)
    return saved
