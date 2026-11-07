---
base_model: Qwen/Qwen3.5-9B
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

# Qwen3.5-9B-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) across **Text (`200` linears), Native MTP (`7` linears), and Vision (`110` linears)**, reducing the checkpoint payload from `17.98 GiB` in BF16 to **`7.62 GiB` (`2.36×` smaller)**.

> **Summary:** Shrinks an `18 GiB` model down to `7.62 GiB` (`2.4×` smaller), taking it from needing a 24 GB card down to fitting on 10–12 GB GPUs with room for context. One of the cleanest quants in the series: text divergence is very low (`0.073` KL, `+1.9%` perplexity), vision top-1 agreement is `98.1%`, and speculative drafter agreement reaches `94.8%`.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-9B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-9B`) | **`Qwen3.5-9B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `17.98 GiB` (`19,306,310,880 B`) | **`7.62 GiB` (`8,179,266,384 B`)** | **`2.36×` compression** (`−10.36 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `3.5754` | **`3.6428` (`+1.89%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0726` / `0.2199`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `10.5643` | **`10.7110` (`+1.39%`)** | Token mean KL `0.0395` (weighted KL `0.0566`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.0275`) | **`94.80%` Top-1 (`NLL 2.0406`)** | Isolated MTP KL `0.0128` (end-to-end Top-1 `86.81%`, KL `0.2049`) |
| **Vision (VQA & Charts)** (48 images, 157 answer tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9813` cos sim · `98.09%` Top-1** | Isolated ViT KL `0.0076` (end-to-end Top-1 `94.27%`, KL `0.0390`) |

## What Is Quantized

- **Quantized (`317` linear layers total: `310` in `model.safetensors` + `7` in `model_mtp.safetensors`)**:
  - **Language (`200` linears)**: All 32 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
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
