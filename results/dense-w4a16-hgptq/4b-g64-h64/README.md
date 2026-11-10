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

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`, `MSE + 2-Pass H-CD + FP32 Norm/GDN QAT + W8_g64 mtp.fc`)** quantization of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) across **Text (`200` linears), Native MTP (`7` W4A16 linears + `1` `W8_g64` `mtp.fc`), and Vision (`98` linears)**, reducing the checkpoint payload from `8.68 GiB` in BF16 to **`3.19 GiB` (`2.72×` smaller)**.

> **Summary:** Compresses the model `2.72×` (`8.68 GiB → 3.19 GiB`, saving `5.49 GiB` of VRAM) so it runs easily on 6–8 GB GPUs. Tracks the BF16 base model closely on text (`-1.89%` perplexity delta, `0.0811` weighted KL divergence, `0.0533` holdout mean KL) and preserves near-lossless vision accuracy (`99.34%` top-1 match on VQA/OCR and `99.26%` on UI/math) alongside `94.26%` speculative drafter agreement.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-4B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-4B`) | **`Qwen3.5-4B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `8.68 GiB` (`9,319,828,096 B`) | **`3.19 GiB` (`3,429,434,952 B`)** | **`2.72×` compression** (`−5.49 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `4.0299` | **`3.9537` (`-1.89%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0811` / `0.2133`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `12.2886` | **`12.3692` (`+0.66%`)** | Token mean KL `0.0533` (weighted KL `0.0741`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.2029`) | **`94.26%` Top-1 (`NLL 2.2149`)** | Isolated MTP KL `0.0150` (end-to-end Top-1 `85.63%`, KL `0.1793`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`99.34%` Top-1 (`NLL 0.4829`)** | Isolated ViT KL `0.0050` (end-to-end Top-1 `92.38%`, KL `0.0889`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`99.26%` Top-1 (`NLL 0.5449`)** | Isolated ViT KL `0.0032` (end-to-end Top-1 `95.56%`, KL `0.0173`) |

## What Is Quantized

- **Quantized (`306` linear layers total: `298` in `model.safetensors` + `8` in `model_mtp.safetensors`)**:
  - **Language (`200` linears)**: All 32 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, MSE scale search, 2-pass H-CD) + End-to-End FP32-master `RMSNorm` + GDN-gate KL QAT.
  - **Native MTP (`7` W4A16 linears + `1` `W8_g64` linear)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`), plus `mtp.fc` in unrotated `W8_g64`.
  - **Vision (`98` linears)**: All 24 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`, `intermediate_size=4096`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`) quantized with `group_size=64` Cholesky GPTQ + `head_dim=64` online Hadamard rotation (`v`).
- **Kept in BF16**:
  - Tied `embed_tokens` / `lm_head`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized linear layers have weights pre-rotated offline along the input dimension ($W' = W H_{64} / \sqrt{64}$, where $H_{64}$ is the deterministic $64 \times 64$ Sylvester Walsh–Hadamard matrix) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_{64} / \sqrt{64}$) is applied to input activations before the W4A16 GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
  - Because $H_{64}$ is deterministic (`randomize: false`), no transform weight matrices are stored in `.safetensors` (`0` storage overhead).
