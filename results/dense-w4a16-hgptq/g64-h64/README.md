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

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) across **Text (`150` linears), Native MTP (`7` linears), and Vision (`98` linears)**, reducing the checkpoint payload from `4.24 GiB` in BF16 to **`1.84 GiB` (`2.30×` smaller)**.

> **Summary:** Shrinks the model `2.3×` (`4.24 GiB → 1.84 GiB`) to fit comfortably in 2–4 GB memory budgets with very little quality loss (`+1.7%` perplexity delta, `0.073` KL divergence). Outperforms standard unrotated 4-bit (`g32`) and mixed 4/8-bit quantization while using less memory, keeping `97.5%` vision top-1 agreement and `91.9%` speculative drafter agreement.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-2B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-2B`) | **`Qwen3.5-2B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `4.24 GiB` (`4,548,221,488 B`) | **`1.84 GiB` (`1,976,323,128 B`)** | **`2.30×` compression** (`−2.40 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `3.8302` | **`3.8942` (`+1.67%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0728` / `0.2543`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `12.7820` | **`12.9914` (`+1.64%`)** | Token mean KL `0.0572` (weighted KL `0.0725`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.5969`) | **`91.87%` Top-1 (`NLL 2.6162`)** | Isolated MTP KL `0.0255` (end-to-end Top-1 `80.49%`, KL `0.4430`) |
| **Vision (VQA & Charts)** (48 images, 157 answer tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9544` cos sim · `97.45%` Top-1** | Isolated ViT KL `0.0152` (end-to-end Top-1 `94.90%`, KL `0.0654`) |

## What Is Quantized

- **Quantized (`255` linear layers total: `248` in `model.safetensors` + `7` in `model_mtp.safetensors`)**:
  - **Language (`150` linears)**: All 24 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - **Native MTP (`7` linears)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`).
  - **Vision (`98` linears)**: All 24 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`) (`head_dim=64`, `group_size=64`).
- **Kept in BF16**:
  - `embed_tokens` / `lm_head`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mtp.fc`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized linear layers have weights pre-rotated offline along the input dimension ($W' = W H_{64} / 8$, where $H_{64}$ is the deterministic $64 \times 64$ Sylvester Walsh–Hadamard matrix) prior to Cholesky GPTQ quantization.
  - At runtime, an online block-64 Walsh–Hadamard transform ($x' = x H_{64} / 8$) is applied to input activations before the W4A16 GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
  - Because $H_{64}$ is deterministic (`randomize: false`), no transform weight matrices are stored in `.safetensors` (`0` storage overhead).
