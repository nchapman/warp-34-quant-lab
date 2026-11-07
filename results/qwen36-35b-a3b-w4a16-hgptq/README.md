---
base_model: Qwen/Qwen3.6-35B-A3B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w4a16
- moe
- compressed-tensors
- vllm
---

# Qwen3.6-35B-A3B-W4A16-HGPTQ

All-W4A16 Cholesky GPTQ (`group_size=64`, `actorder="static"`) with online block-Hadamard input rotation (`head_dim=64`, `QuIPModifier` / $H_{64}$ + `HadaCore`) of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) across **Text (`30,970` linears: `130` rotated `self_attn` / `linear_attn` + `120` rotated `mlp.shared_expert` + `30,720` rotated `mlp.experts`), Native MTP (`775` linears: `4` rotated `self_attn` + `3` rotated `mlp.shared_expert` + `768` rotated `mlp.experts`), and Vision (`110` rotated linears)** (`19.243 GiB` total safetensors payload).

- **Base model:** [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (`995ad96eacd98c81ed38be0c5b274b04031597b0`)
- **Text & MTP calibration:** 512 samples × 2048 tokens (`1,048,576` tokens) from `calib-cascade-qwenfinals.jsonl` pinned in [warp-3090-quant-lab-frozen-data](https://huggingface.co/datasets/nchapman/warp-3090-quant-lab-frozen-data); Native MTP (`775` linears) calibrated on captured post-norm target hidden states and shifted token embeddings
- **Vision calibration:** 120 12-domain calibration images (`vision-calib-v2`, strictly disjoint from `vision-probe-v2` and `vision-guard-v2`); `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ (`vision_fc2_channel`) + `head_dim=16` online Hadamard (`v_vision_fc2`)
- **Static hot-expert cache rankings:** `runtime/static_hot_cache_rankings.json` (and root `static_hot_cache_rankings.json`) + `runtime/expert_routing_profile.json` pre-profiled across all 40 MoE layers + MTP (`top_64 = 42.58%` hit rate / `45.17%` routing mass; `top_128 = 69.12%` hit rate / `71.54%` routing mass)
- **Chat template:** [froggeric Qwen-Fixed-Chat-Templates (v22.5)](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) ships as `chat_template.jinja` (`sha256: e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`) — the artifact's only template
- **Default sampling** (`generation_config.json`): `temperature 0.6, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`)
- **Recipe and tooling:** [warp-3090-model-lab](https://github.com/nchapman/warp-3090-model-lab)

`linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `embed_tokens` / `lm_head`, `mtp.fc`, `visual.patch_embed.proj`, norms, and biases stay BF16.

## Measured Fidelity (`Qwen3.6-35B-A3B-W4A16-HGPTQ` v2 vs. v1 & `g128-h0`)

| Metric | `Qwen3.6-35B-A3B-W4A16-GPTQ` (`g128-h0`) | v1 (`experts` unrotated, `shared_expert` BF16) | **v2 (this repo: $H_{64}$ on `experts` + `shared_expert`)** |
|---|---:|---:|---:|
| Safetensors payload | 20.678 GiB (22,203,234,720 B) | 19.419 GiB (20,851,089,304 B) | **19.243 GiB (20,661,694,432 B)** |
| `probe-v4` weightedKL (`cascade90-ml10`) ↓ | — *(legacy: 0.111)* | 0.088640 | **0.090070** (`chat: 0.032678`, `math: 0.029619`, `science: 0.029975`, `if: 0.034597`) |
| `probe-guard` token mean KL / weightedKL (76 rows, 19,534 tok) ↓ | — | 0.051031 / 0.061633 | **0.048598 / 0.057899 (−6.1% weightedKL, `no_robots: 0.041832`)** |
| Native MTP isolated KL / Top-1 agreement (11,583 pos) ↑ | *(BF16 in old repo)* | 0.011587 / 95.93% | **0.011066 (−4.5%) / 95.93%** |
| Native MTP end-to-end KL / Top-1 agreement ↑ | — | 0.122539 / 89.17% | **0.126571 / 88.85%** |
| Vision `vision-probe-v2` (96 images, 302 tokens) encoder cos / iso KL (Top-1) / e2e KL (Top-1) | — | — | **0.9798 / 0.007872 (98.34%) / 0.030865 (96.69%)** |
| Vision OOD Guard `vision-guard-v2` (48 images, 135 tokens) encoder cos / iso KL (Top-1) / e2e KL (Top-1) | — | — | **0.9845 / 0.023903 (99.26%) / 0.043429 (98.52%)** |

## Inference Server & Runtime Notes

1. **Full $H_{64}$ Input Hadamard Rotation Across Attention, Shared Experts, and Routed MoE Experts**:
   - Every quantized linear in Text (`30,970` linears) and Native MTP (`775` linears) — including `self_attn`, `linear_attn`, `mlp.shared_expert.{gate_proj,up_proj,down_proj}`, and `mlp.experts.0..255.{gate_proj,up_proj,down_proj}` — has its weights pre-rotated offline along the input dimension ($W' = W H_{64} / 8$) prior to Cholesky GPTQ quantization (`group_size=64`, `actorder="static"`).
   - At runtime, apply block-64 Walsh-Hadamard transform ($x \mapsto x H_{64} / 8$) to the input activations before `gate_proj`/`up_proj` (`K=2048`) and to the post-SwiGLU intermediate activations before `down_proj` (`K=512`), matching `quantization_config.transform_config.config_groups.v`.
   - Only `mlp.gate` and `mlp.shared_expert_gate` remain in BF16 (listed in `quantization_config.ignore`).
