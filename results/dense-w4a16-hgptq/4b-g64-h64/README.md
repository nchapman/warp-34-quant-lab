---
base_model: Qwen/Qwen3.5-4B
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

# Qwen3.5-4B-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) across **Text (`200` linears), Native MTP (`7` linears), and Vision (`98` linears)**, reducing the checkpoint payload from `8.68 GiB` in BF16 to **`3.21 GiB` (`2.71×` smaller)**.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-4B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-4B`) | **`Qwen3.5-4B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `8.68 GiB` (`9,319,828,096 B`) | **`3.21 GiB` (`3,442,125,688 B`)** | **`2.71×` compression** (`−5.47 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `4.0299` | **`4.1446` (`+2.85%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0949` / `0.2540`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `12.2886` | **`12.5282` (`+1.95%`)** | Token mean KL `0.0633` (weighted KL `0.0863`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.2029`) | **`93.47%` Top-1 (`NLL 2.2230`)** | Isolated MTP KL `0.0185` (end-to-end Top-1 `84.74%`, KL `0.2068`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9798` cos sim · `99.67%` Top-1** | Isolated ViT KL `0.0047` (end-to-end Top-1 `92.72%`, KL `0.0825`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9840` cos sim · `98.52%` Top-1** | Isolated ViT KL `0.0020` (end-to-end Top-1 `97.04%`, KL `0.0315`) |

## What Is Quantized

- **Quantized (`305` linear layers total: `298` in `model.safetensors` + `7` in `model_mtp.safetensors`)**:
  - **Language (`200` linears)**: All 32 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - **Native MTP (`7` linears)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`).
  - **Vision (`98` linears)**: All 24 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`, `intermediate_size=4096`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`) quantized with `group_size=64` Cholesky GPTQ + `head_dim=64` online Hadamard rotation (`v`).
- **Kept in BF16**:
  - `embed_tokens` / `lm_head`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mtp.fc`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized linear layers have weights pre-rotated offline along the input dimension ($W' = W H_{64} / \sqrt{64}$, where $H_{64}$ is the deterministic $64 \times 64$ Sylvester Walsh–Hadamard matrix) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_{64} / \sqrt{64}$) is applied to input activations before the W4A16 GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
  - Because $H_{64}$ is deterministic (`randomize: false`), no transform weight matrices are stored in `.safetensors` (`0` storage overhead).
