---
base_model: Qwen/Qwen3.8-27B
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

# Qwen3.8-27B-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) across **Text (`400` linears), Native MTP (`7` linears), and Vision (`110` linears)**, reducing the checkpoint payload from `51.75 GiB` in BF16 to **`17.33 GiB` (`2.99×` smaller)**.

> **Summary:** Compresses a `51.8 GiB` dense model by `3.0×` (`17.33 GiB`, saving `34.4 GiB` of VRAM) so it fits on a **single 24 GB GPU (RTX 3090 / 4090)** with ~6.5 GiB left for KV cache. At 27B scale, 4-bit Hadamard GPTQ is nearly lossless: text perplexity matches BF16 (`8.54 → 8.52`), speculative drafter top-1 agreement is `96.2%`, and vision top-1 agreement is `97.5%`.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.8-27B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.8-27B`) | **`Qwen3.8-27B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `51.75 GiB` (`55,563,006,776 B`) | **`17.33 GiB` (`18,612,467,976 B`)** | **`2.99×` compression** (`−34.41 GiB`) |
| **Text Perplexity** (378 prompts, 852k tokens) ↓ | `8.5438` | **`8.5189` (`−0.29%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.1160` / `0.3016`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 23k tokens) ↓ | `60.3169` | **`56.1987` (`−6.83%`)** | Token mean KL `0.0639` (weighted KL `0.0811`) |
| **Native MTP (Speculative Drafter)** (12,339 positions) | `100.0%` Top-1 (`NLL 2.9579`) | **`96.22%` Top-1 (`NLL 2.9660`)** | Isolated MTP KL `0.0109` (end-to-end Top-1 `87.14%`, KL `0.2131`) |
| **Vision (VQA & Charts)** (48 images, 157 answer tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9940` cos sim · `97.45%` Top-1** | Isolated ViT KL `0.0157` (end-to-end Top-1 `89.81%`, KL `0.2671`) |

## What Is Quantized

- **Quantized (`517` linear layers total: `510` in `model.safetensors` + `7` in `model_mtp.safetensors`)**:
  - **Language (`400` linears)**: All 64 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - **Native MTP (`7` linears)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens` / `lm_head`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mtp.fc`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized linear layers have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$, where $H_b$ is the deterministic $b \times b$ Sylvester Walsh–Hadamard matrix; $b=64$ for `v` and $b=16$ for `v_vision_fc2`) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_b / \sqrt{b}$) is applied to input activations before the W4A16 GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
  - Because $H_b$ is deterministic (`randomize: false`), no transform weight matrices are stored in `.safetensors` (`0` storage overhead).
