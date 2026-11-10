---
base_model: ukisai/Swift-1.5-Qwen3.8-27b
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

# Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ + Hessian Coordinate Descent + FP32-Master Norm QAT (`HGPTQ`)** quantization of [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) across **Text (`400` backbone linears in `W4` + `lm_head` in `W8_g64`), Native MTP (`7` transformer linears in `W4` + `mtp.fc` in `W8_g64`), and Vision (`110` linears in `W4`)**, reducing the checkpoint payload from `51.75 GiB` in BF16 to **`16.14 GiB` (`3.21×` smaller, saving `35.61 GiB` of VRAM)** to fit comfortably on a **single 24 GB GPU**. Tracks the BF16 fine-tune closely across text (`0.1074` weighted KL divergence, `0.0555` holdout mean KL), speculative drafting (`96.43%` top-1 agreement), and vision (`98.01%–99.26%` top-1 agreement).

## Quality vs. BF16 Base Model (`ukisai/Swift-1.5-Qwen3.8-27b`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`ukisai/Swift-1.5-Qwen3.8-27b`) | **`Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `51.75 GiB` (`55,563,006,776 B`) | **`16.14 GiB` (`17,328,747,912 B`)** | **`3.21×` compression** (`−35.61 GiB`) |
| **Text Perplexity** (378 prompts, 852k tokens) ↓ | `9.6265` | **`9.0881` (`−5.59%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.1074` / `0.2831`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 23k tokens) ↓ | `79.6973` | **`72.1364` (`−9.49%`)** | Token mean KL `0.0555` (weighted KL `0.0720`) |
| **Native MTP (Speculative Drafter)** (12,339 positions) | `100.0%` Top-1 (`NLL 3.1061`) | **`96.43%` Top-1 (`NLL 3.1229`)** | Isolated MTP KL `0.0090` (end-to-end Top-1 `89.03%`, KL `0.2086`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`98.01%` Top-1 (`NLL 0.5043`)** | Isolated ViT KL `0.0071` (end-to-end Top-1 `91.39%`, KL `0.1444`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`99.26%` Top-1 (`NLL 0.8749`)** | Isolated ViT KL `0.0024` (end-to-end Top-1 `92.59%`, KL `0.1440`) |

## What Is Quantized

- **Quantized (`519` linear layers total)**:
  - **Language (`400` W4A16 linears + `1` `W8_g64` linear)**: All 64 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, MSE scale search, 2-pass H-CD) + End-to-End FP32-master `RMSNorm` + GDN-gate KL QAT, plus `lm_head` in unrotated `W8_g64`.
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
