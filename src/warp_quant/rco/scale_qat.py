"""End-to-end continuous weight_scale and FP32-master RMSNorm/GDN-gate KL fine-tuning over frozen low-bit codes."""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from warp_quant.lablog import log_event
from warp_quant.rco.bank import attach_online_hadamard_hooks, get_text_model
from warp_quant.rco.hidden_teacher import hidden_kl


class _TrainableScaleLinear(nn.Module):
    """Parametrizes a linear weight with frozen integer codes and trainable/frozen BF16 group scales."""

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        device: torch.device,
        *,
        trainable: bool = True,
        code_offset: float = 0.0,
    ) -> None:
        super().__init__()
        n_out, k_in = codes.shape
        self.n_out = n_out
        self.k_in = k_in
        self.group_size = group_size
        self.trainable = trainable
        eff_codes = codes.detach().to(device=device, dtype=torch.float32)
        if code_offset != 0.0:
            eff_codes = eff_codes + code_offset
        self.register_buffer("codes_bf16", eff_codes.to(torch.bfloat16))
        self.register_buffer("init_scales", scales.detach().to(device=device, dtype=torch.float32))
        if trainable:
            self.scales = nn.Parameter(scales.detach().to(device=device, dtype=torch.float32).clone())
        else:
            self.register_buffer("scales", scales.detach().to(device=device, dtype=torch.float32).clone())

    def forward(self, _orig: torch.Tensor) -> torch.Tensor:
        if self.trainable:
            s_bf16 = self.scales.to(torch.bfloat16) + (self.scales - self.scales.detach()).to(torch.bfloat16)
        else:
            s_bf16 = self.scales.to(torch.bfloat16)
        return (self.codes_bf16.view(self.n_out, -1, self.group_size) * s_bf16.unsqueeze(-1)).view(
            self.n_out, self.k_in
        )


class _FP32MasterBF16Weight(nn.Module):
    """Parametrizes a BF16 weight (e.g. RMSNorm or GDN gate) with an FP32 master parameter and BF16 STE."""

    def __init__(self, init_weight: torch.Tensor) -> None:
        super().__init__()
        w_f32 = init_weight.detach().to(dtype=torch.float32)
        self.register_buffer("init_master", w_f32.clone())
        self.master = nn.Parameter(w_f32.clone())

    def forward(self, _orig: torch.Tensor) -> torch.Tensor:
        return self.master.to(torch.bfloat16) + (self.master - self.master.detach()).to(torch.bfloat16)


def run_e2e_scale_qat(
    student_model: nn.Module,
    teacher_model: nn.Module | None,
    manifest: dict,
    candidates: dict[str, list[dict]],
    choices: list[int] | tuple[int, ...],
    train_rows: list[torch.Tensor],
    val_rows: list[torch.Tensor],
    val_teacher_states: list[torch.Tensor],
    *,
    train_teacher_states: list[torch.Tensor] | None = None,
    num_epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1.5e-4,
    norm_lr: float | None = None,
    anchor_wd: float = 0.02,
    train_scales_max_bits: int = 2,
    train_norms: bool = True,
    train_gdn_gates: bool = True,
    seed: int = 42,
) -> dict:
    """Tune continuous BF16 weight_scale (for <= train_scales_max_bits) and FP32-master RMSNorm/GDN-gate tensors."""
    if not train_rows:
        raise ValueError("train_rows must be non-empty")
    if teacher_model is None and train_teacher_states is None:
        raise ValueError("Either teacher_model or train_teacher_states must be provided")
    if train_teacher_states is not None and len(train_teacher_states) < len(train_rows):
        raise ValueError(
            f"train_teacher_states length {len(train_teacher_states)} < train_rows length {len(train_rows)}"
        )
    device = next(student_model.parameters()).device
    student_model.eval()
    student_model.requires_grad_(False)
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    s_text, _ = get_text_model(student_model)
    t_text = get_text_model(teacher_model)[0] if teacher_model is not None else None
    high_vram = device.type == "cuda" and torch.cuda.mem_get_info(device)[0] > 40 * (1024**3)
    kl_chunk_size = 512 if high_vram else 128
    # Always checkpoint hidden_kl chunks during backward: vocab=248,320 would otherwise retain ~24 GiB per 2k seq
    kl_checkpointed = True
    groups = manifest["groups"]
    had_block_size = int(manifest.get("hadamard_block_size", 64))
    had_hooks = attach_online_hadamard_hooks(student_model, groups, had_block_size)

    scale_modules: dict[str, tuple[nn.Module, _TrainableScaleLinear]] = {}
    scale_params: list[nn.Parameter] = []
    scale_inits: list[torch.Tensor] = []

    for grp, choice in zip(groups, choices, strict=True):
        c_idx = int(choice)
        for m_name in grp["members"]:
            mod = student_model.get_submodule(m_name)
            cand = candidates[m_name][c_idx]
            is_trainable = int(cand["num_bits"]) <= int(train_scales_max_bits) and lr > 0.0
            ts = _TrainableScaleLinear(
                cand["codes"],
                cand["scales"],
                int(cand["group_size"]),
                mod.weight.device,
                trainable=is_trainable,
                code_offset=float(cand.get("code_offset", 0.0)),
            )
            parametrize.register_parametrization(mod, "weight", ts)
            scale_modules[m_name] = (mod, ts)
            if is_trainable:
                scale_params.append(ts.scales)
                scale_inits.append(ts.init_scales)

    bf16_modules: dict[str, tuple[nn.Module, _FP32MasterBF16Weight]] = {}
    bf16_params: list[nn.Parameter] = []
    bf16_inits: list[torch.Tensor] = []

    for mod_name, mod in list(student_model.named_modules()):
        if not mod_name.startswith("model.language_model.") and not mod_name.startswith("model.layers."):
            continue
        if (
            mod_name in scale_modules
            or not hasattr(mod, "weight")
            or not isinstance(mod.weight, torch.Tensor)
        ):
            continue
        full_name = f"{mod_name}.weight"
        is_norm = train_norms and ("norm" in mod_name.lower()) and mod.weight.ndim == 1
        is_gdn_gate = train_gdn_gates and (
            mod_name.endswith(".linear_attn.in_proj_a") or mod_name.endswith(".linear_attn.in_proj_b")
        )
        if (is_norm or is_gdn_gate) and mod.weight.dtype == torch.bfloat16:
            fp32_wrap = _FP32MasterBF16Weight(mod.weight)
            parametrize.register_parametrization(mod, "weight", fp32_wrap)
            bf16_modules[full_name] = (mod, fp32_wrap)
            bf16_params.append(fp32_wrap.master)
            bf16_inits.append(fp32_wrap.init_master)

    eff_norm_lr = norm_lr if norm_lr is not None else (lr if lr > 0 else 1.5e-4)

    @torch.no_grad()
    def eval_val_kl() -> float:
        n_val = min(len(val_rows), len(val_teacher_states))
        if n_val == 0:
            return 0.0
        kls = []
        val_bsz = 8 if high_vram else 1
        for b0 in range(0, n_val, val_bsz):
            b_idxs = list(range(b0, min(b0 + val_bsz, n_val)))
            if len(b_idxs) == 1:
                i0 = b_idxs[0]
                inp = val_rows[i0].to(device=device, dtype=torch.long).unsqueeze(0)
                h_s = s_text(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state
                kl = hidden_kl(
                    h_s,
                    val_teacher_states[i0],
                    student_model.lm_head,
                    token_chunk_size=kl_chunk_size,
                    checkpointed=False,
                )
                kls.append(float(kl.item()))
            else:
                lens = [int(val_rows[i].numel()) for i in b_idxs]
                s_max = max(lens)
                inp = torch.zeros((len(b_idxs), s_max), dtype=torch.long, device=device)
                for loc_i, r_i in enumerate(b_idxs):
                    inp[loc_i, : lens[loc_i]] = val_rows[r_i].to(device=device, dtype=torch.long)
                h_batch = s_text(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state
                for loc_i, r_i in enumerate(b_idxs):
                    kl = hidden_kl(
                        h_batch[loc_i : loc_i + 1, : lens[loc_i]],
                        val_teacher_states[r_i],
                        student_model.lm_head,
                        token_chunk_size=kl_chunk_size,
                        checkpointed=False,
                    )
                    kls.append(float(kl.item()))
        return sum(kls) / len(kls)

    try:
        init_val_kl = eval_val_kl()
        log_event(
            "RCO_SCALE_QAT_START",
            train_rows=len(train_rows),
            val_rows=min(len(val_rows), len(val_teacher_states)),
            scale_scalars=sum(p.numel() for p in scale_params),
            bf16_scalars=sum(p.numel() for p in bf16_params),
            init_val_kl=round(init_val_kl, 6),
        )

        param_groups = []
        if scale_params:
            param_groups.append({"params": scale_params, "lr": lr, "base_lr": lr, "weight_decay": 0.0})
        if bf16_params:
            param_groups.append(
                {"params": bf16_params, "lr": eff_norm_lr, "base_lr": eff_norm_lr, "weight_decay": 0.0}
            )
        if not param_groups:
            raise ValueError(
                "No trainable parameters selected in run_e2e_scale_qat "
                "(check lr, train_scales_max_bits, train_norms, train_gdn_gates)"
            )
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

        rng = torch.Generator().manual_seed(seed)
        steps_per_epoch = math.ceil(len(train_rows) / batch_size)
        total_steps = num_epochs * steps_per_epoch
        step = 0
        t0 = time.time()

        best_val_kl = init_val_kl
        best_scales: dict[str, torch.Tensor] = {
            m_name: ts.scales.detach().to(torch.bfloat16).cpu().clone()
            for m_name, (_, ts) in scale_modules.items()
        }
        best_bf16: dict[str, torch.Tensor] = {
            k: wrap.master.detach().to(torch.bfloat16).cpu().clone() for k, (_, wrap) in bf16_modules.items()
        }
        history: list[dict] = []

        for epoch in range(1, num_epochs + 1):
            perm = torch.randperm(len(train_rows), generator=rng).tolist()
            ep_losses: list[float] = []
            for b0 in range(0, len(train_rows), batch_size):
                batch_indices = perm[b0 : b0 + batch_size]
                if step < 0.08 * total_steps:
                    lr_mult = (step + 1) / max(1.0, 0.08 * total_steps)
                else:
                    prog = (step - 0.08 * total_steps) / max(1.0, 0.92 * total_steps)
                    lr_mult = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog))
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["base_lr"] * lr_mult

                optimizer.zero_grad()
                if high_vram and len(batch_indices) > 1:
                    b_loss = 0.0
                    micro_bsz = 2
                    for m0 in range(0, len(batch_indices), micro_bsz):
                        m_indices = batch_indices[m0 : m0 + micro_bsz]
                        lens = [int(train_rows[idx].numel()) for idx in m_indices]
                        s_max = max(lens)
                        inp_batch = torch.zeros((len(m_indices), s_max), dtype=torch.long, device=device)
                        for loc_i, idx in enumerate(m_indices):
                            inp_batch[loc_i, : lens[loc_i]] = train_rows[idx].to(
                                device=device, dtype=torch.long
                            )
                        if train_teacher_states is None:
                            assert t_text is not None
                            with torch.no_grad():
                                h_t_batch = t_text(
                                    input_ids=inp_batch, use_cache=False, return_dict=True
                                ).last_hidden_state.detach()
                        else:
                            h_t_batch = None
                        h_s_batch = s_text(
                            input_ids=inp_batch, use_cache=False, return_dict=True
                        ).last_hidden_state
                        micro_losses = []
                        for loc_i, idx in enumerate(m_indices):
                            rl = lens[loc_i]
                            h_t_i = (
                                train_teacher_states[idx]
                                if train_teacher_states is not None
                                else h_t_batch[loc_i : loc_i + 1, :rl]
                            )
                            micro_losses.append(
                                hidden_kl(
                                    h_s_batch[loc_i : loc_i + 1, :rl],
                                    h_t_i,
                                    student_model.lm_head,
                                    token_chunk_size=kl_chunk_size,
                                    checkpointed=kl_checkpointed,
                                )
                            )
                        micro_loss = torch.stack(micro_losses).sum() / len(batch_indices)
                        micro_loss.backward()
                        b_loss += float(micro_loss.item())
                else:
                    b_loss = 0.0
                    for idx in batch_indices:
                        inp = train_rows[idx].to(device=device, dtype=torch.long).unsqueeze(0)
                        if train_teacher_states is not None:
                            h_t = train_teacher_states[idx]
                        else:
                            assert t_text is not None
                            with torch.no_grad():
                                h_t = t_text(
                                    input_ids=inp, use_cache=False, return_dict=True
                                ).last_hidden_state.detach()
                        h_s = s_text(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state
                        loss = hidden_kl(
                            h_s,
                            h_t,
                            student_model.lm_head,
                            token_chunk_size=kl_chunk_size,
                            checkpointed=kl_checkpointed,
                        ) / len(batch_indices)
                        loss.backward()
                        b_loss += float(loss.item())

                torch.nn.utils.clip_grad_norm_(scale_params + bf16_params, 1.0)
                optimizer.step()

                if anchor_wd > 0:
                    with torch.no_grad():
                        for p, p0 in zip(scale_params, scale_inits, strict=True):
                            p.sub_(p - p0, alpha=(lr * lr_mult) * anchor_wd)
                        for p, p0 in zip(bf16_params, bf16_inits, strict=True):
                            p.sub_(p - p0, alpha=(eff_norm_lr * lr_mult) * anchor_wd)

                ep_losses.append(b_loss)
                step += 1

                if step == 1 or step % 40 == 0 or step == total_steps:
                    recent_kl = sum(ep_losses[-min(len(ep_losses), 20) :]) / min(len(ep_losses), 20)
                    log_event(
                        "RCO_SCALE_QAT_STEP",
                        epoch=epoch,
                        step=step,
                        total_steps=total_steps,
                        lr=round(eff_norm_lr * lr_mult, 7),
                        train_kl=round(recent_kl, 5),
                        elapsed_s=round(time.time() - t0, 1),
                    )

            val_kl = eval_val_kl()
            rec = {
                "epoch": epoch,
                "train_kl": sum(ep_losses) / max(len(ep_losses), 1),
                "val_kl": val_kl,
                "elapsed_s": round(time.time() - t0, 2),
            }
            history.append(rec)
            log_event(
                "RCO_SCALE_QAT_EPOCH",
                epoch=epoch,
                train_kl=round(rec["train_kl"], 6),
                val_kl=round(val_kl, 6),
                elapsed_s=rec["elapsed_s"],
            )
            if val_kl <= best_val_kl:
                best_val_kl = val_kl
                best_scales = {
                    m_name: ts.scales.detach().to(torch.bfloat16).cpu().clone()
                    for m_name, (_, ts) in scale_modules.items()
                }
                best_bf16 = {
                    k: wrap.master.detach().to(torch.bfloat16).cpu().clone()
                    for k, (_, wrap) in bf16_modules.items()
                }

        return {
            "init_val_kl": init_val_kl,
            "best_val_kl": best_val_kl,
            "history": history,
            "scale_overrides": best_scales,
            "bf16_overrides": best_bf16,
        }
    finally:
        for h in had_hooks:
            h.remove()
        for _m_name, (mod, _) in scale_modules.items():
            if parametrize.is_parametrized(mod, "weight"):
                parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
        for _full_name, (mod, _) in bf16_modules.items():
            if parametrize.is_parametrized(mod, "weight"):
                parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
