---
base_model: Qwen/Qwen3.5-0.8B
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

# Qwen3.5-0.8B-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) across **Text (`150` linears), Native MTP (`7` linears), and Vision (`50` linears)**, reducing the checkpoint payload from `1.63 GiB` in BF16 to **`0.79 GiB` (`2.06×` smaller)**.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-0.8B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-0.8B`) | **`Qwen3.5-0.8B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `1.63 GiB` (`1,746,942,600 B`) | **`0.79 GiB` (`846,805,696 B`)** | **`2.06×` compression** (`−0.84 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `4.1695` | **`4.3593` (`+4.55%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0901` / `0.3149`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `15.4384` | **`15.7561` (`+2.06%`)** | Token mean KL `0.1041` (weighted KL `0.1446`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.8764`) | **`89.71%` Top-1 (`NLL 2.9123`)** | Isolated MTP KL `0.0364` (end-to-end Top-1 `76.24%`, KL `0.5369`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9696` cos sim · `92.38%` Top-1** | Isolated ViT KL `0.0313` (end-to-end Top-1 `89.07%`, KL `0.1358`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9692` cos sim · `98.52%` Top-1** | Isolated ViT KL `0.0115` (end-to-end Top-1 `92.59%`, KL `0.0867`) |

## What Is Quantized

- **Quantized (`207` linear layers total: `200` in `model.safetensors` + `7` in `model_mtp.safetensors`)**:
  - **Language (`150` linears)**: All 24 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - **Native MTP (`7` linears)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`).
  - **Vision (`50` linears)**: All 12 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`, `intermediate_size=3072`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`) quantized with `group_size=64` Cholesky GPTQ + `head_dim=64` online Hadamard rotation (`v`).
- **Kept in BF16**:
  - `embed_tokens` / `lm_head`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mtp.fc`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized linear layers have weights pre-rotated offline along the input dimension ($W' = W H_{64} / \sqrt{64}$, where $H_{64}$ is the deterministic $64 \times 64$ Sylvester Walsh–Hadamard matrix) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_{64} / \sqrt{64}$) is applied to input activations before the W4A16 GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
  - Because $H_{64}$ is deterministic (`randomize: false`), no transform weight matrices are stored in `.safetensors` (`0` storage overhead).
