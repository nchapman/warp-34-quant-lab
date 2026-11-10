---
base_model: Qwen/Qwen3.5-2B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w4a16
- compressed-tensors
- vllm
---

# Qwen3.5-2B-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ + Hessian Coordinate Descent + FP32-Master Norm QAT (`HGPTQ`)** quantization of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) across **Text (`150` linears in `W4`), Native MTP (`7` transformer linears in `W4` + `mtp.fc` in `W8_g64`), and Vision (`98` linears in `W4`)**, reducing the checkpoint payload from `4.24 GiB` in BF16 to **`1.83 GiB` (`2.31×` smaller)**.

> **Summary:** Shrinks the model `2.31×` (`4.24 GiB → 1.83 GiB`) to fit comfortably in 2–4 GB memory budgets with minimal quality loss (`+0.72%` holdout perplexity delta, `0.0591` weighted KL divergence). Combining MSE-initialized Cholesky GPTQ + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`) + end-to-end FP32-master `RMSNorm` KL tuning cuts `probe-v4` weighted KL by `−18.8%` (`0.0728 → 0.0591`), holdout perplexity delta from `+1.64%` to **`+0.72%`**, Vision isolated Top-1 to **`97.35%`** (`95.56%` E2E on `vision-guard-v2`), and speculative drafter Top-1 to **`92.78%`** (`81.64%` E2E).

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-2B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-2B`) | **`Qwen3.5-2B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `4.24 GiB` (`4,548,221,488 B`) | **`1.83 GiB` (`1,968,201,664 B`)** | **`2.31×` compression** (`−2.41 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `3.8302` | **`3.7784` (`−1.35%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0591` / `0.2038`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `12.7820` | **`12.8746` (`+0.72%`)** | Token mean KL `0.0500` (weighted KL `0.0630`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.5969`) | **`92.78%` Isolated Top-1 (`NLL 2.6173`)** | Isolated MTP KL `0.0201` (end-to-end Top-1 `81.64%`, KL `0.3566`) |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`97.35%` Isolated Top-1 · `93.71%` E2E Top-1** | Isolated ViT KL `0.0130`, E2E KL `0.0737` (`vision-guard-v2` E2E Top-1 `95.56%`, KL `0.0562`) |

## What Is Quantized

- **Quantized (`256` linear layers total in `model.safetensors-00001-of-00001.safetensors`)**:
  - **Language (`150` linears in `W4A16-HGPTQ`)**: All 24 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + MSE-initialized Cholesky GPTQ (`group_size=64`) + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`) + end-to-end FP32-master `RMSNorm` KL tuning.
  - **Native MTP (`7` transformer linears in `W4A16-HGPTQ` + `mtp.fc` in `W8_g64`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`), plus `mtp.fc` (`[2048, 4096]`) in unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`).
  - **Vision (`98` linears in `W4A16-HGPTQ`)**: All 24 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - Tied `embed_tokens` / `lm_head` (`tie_word_embeddings: true`), `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`) & **release chat template** (`chat_template.jinja`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized linear layers have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$, where $H_b$ is the deterministic $b \times b$ Sylvester Walsh–Hadamard matrix; $b=64$ for `v` and $b=16$ for `v_vision_fc2`) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_b / \sqrt{b}$) is applied to input activations before the W4A16 GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
  - Because $H_b$ is deterministic (`randomize: false`), no transform weight matrices are stored in `.safetensors` (`0` storage overhead).
