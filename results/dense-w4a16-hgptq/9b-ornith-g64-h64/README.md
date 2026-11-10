---
base_model: ornith-ai/Ornith-1.5-9B
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

# Ornith-1.5-9B-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`, `MSE + 2-Pass H-CD + FP32 Norm/GDN QAT + W8_g64 Head`)** quantization of [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) across **Text (`200` W4A16 linears + `1` `W8_g64` `lm_head`), Native MTP (`7` W4A16 linears + `1` `W8_g64` `mtp.fc`), and Vision (`110` linears)**, reducing the checkpoint payload from `17.98 GiB` in BF16 to **`6.67 GiB` (`2.70×` smaller)**.

> **Summary:** Shrinks `Ornith-1.5-9B` by `2.70×` (`17.98 GiB → 6.67 GiB`) to fit on 8–10 GB GPUs. Preserves `97.02%–97.04%` vision top-1 agreement and `94.68%` speculative drafter agreement (`-15.34%` text perplexity delta, `0.1440` weighted KL divergence, `0.0965` holdout mean KL).

## Quality vs. BF16 Base Model (`ornith-ai/Ornith-1.5-9B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`ornith-ai/Ornith-1.5-9B`) | **`Ornith-1.5-9B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `17.98 GiB` (`19,306,303,864 B`) | **`6.67 GiB` (`7,157,574,568 B`)** | **`2.70×` compression** (`−11.31 GiB`) |
| **Text Perplexity** (378 prompts, 843k tokens) ↓ | `6.6829` | **`5.6579` (`-15.34%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.1440` / `0.5751`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `15.0115` | **`14.8568` (`-1.03%`)** | Token mean KL `0.0965` (weighted KL `0.1336`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.8335`) | **`94.68%` Top-1 (`NLL 2.8539`)** | Isolated MTP KL `0.0127` (end-to-end Top-1 `79.17%`, KL `0.4402`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`97.02%` Top-1 (`NLL 0.5313`)** | Isolated ViT KL `0.0205` (end-to-end Top-1 `94.70%`, KL `0.1109`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`97.04%` Top-1 (`NLL 0.9373`)** | Isolated ViT KL `0.0107` (end-to-end Top-1 `94.81%`, KL `0.0741`) |

## What Is Quantized

- **Quantized (`319` linear layers total)**:
  - **Language (`200` W4A16 linears + `1` `W8_g64` linear)**: All 32 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, MSE scale search, 2-pass H-CD) + End-to-End FP32-master `RMSNorm` + GDN-gate KL QAT, plus `lm_head` in unrotated `W8_g64`.
  - **Native MTP (`7` W4A16 linears + `1` `W8_g64` linear)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`), plus `mtp.fc` in unrotated `W8_g64`.
  - **Vision (`110` linears)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized linear layers have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$, where $H_b$ is the deterministic $b \times b$ Sylvester Walsh–Hadamard matrix; $b=64$ for `v` and $b=16$ for `v_vision_fc2`) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_b / \sqrt{b}$) is applied to input activations before the W4A16 GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
  - Because $H_b$ is deterministic (`randomize: false`), no transform weight matrices are stored in `.safetensors` (`0` storage overhead).
