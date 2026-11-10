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

4-bit Hadamard + GPTQ (`W4A16`, group size 64) quantization of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) across Text, Vision, and the Native MTP speculative drafter. It shrinks the 18.0 GiB BF16 model down to **6.67 GiB** (2.70× smaller), fitting cleanly on **10–12 GB GPUs** with 3–5 GiB of KV-cache headroom (or on an 8 GB GPU for shorter contexts; for long-context 8 GB serving, see [`Qwen3.5-9B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO) at 5.54 GiB).

- **VRAM & GPU fit:** 6.67 GiB total (saving 11.31 GiB vs. BF16) across 200 text backbone linears, 110 vision linears, and 8 Native MTP linears, with `lm_head` and `mtp.fc` in 8-bit (`W8_g64`).
- **Quantization recipe:** 64×64 Walsh–Hadamard input rotation + MSE-initialized Cholesky GPTQ + 2-pass Hessian Coordinate Descent (`H-CD`) + end-to-end FP32-master `RMSNorm` KL tuning.
- **Fidelity vs. BF16:** 0.0638 weighted KL (0.0488 mean KL, −0.77% PPL) on `probe-v4`, 0.0419 mean KL (+0.78% PPL) on the `probe-guard` holdout, 94.86% / 87.79% isolated/E2E MTP top-1 agreement, and 99.34% / 97.35% isolated/E2E vision top-1 agreement.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-9B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-9B`) | **`Qwen3.5-9B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `17.98 GiB` (`19,306,310,880 B`) | **`6.67 GiB` (`7,161,432,368 B`)** | **`2.70×` compression** (`−11.31 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `3.5754` | **`3.4999` (`−2.11%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0638` / `0.1900`** | Full-vocabulary FP32 KL (`math` `0.0144`, `chat` `0.0204`, `if` `0.0218`, `science` `0.0311`) |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `10.5643` | **`10.6466` (`+0.78%`)** | Token mean KL `0.0395` (weighted KL `0.0586`, 9-lang Belebele macro KL `0.0525`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.0275`) | **`94.86%` Isolated Top-1 (`NLL 2.0408`)** | Isolated MTP KL `0.0105` (end-to-end Top-1 `87.79%`, KL `0.1725`) |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`99.34%` Isolated Top-1 · `97.35%` E2E Top-1** | Isolated ViT KL `0.0072`, E2E KL `0.0246` (`vision-guard-v2` E2E Top-1 `94.07%`, KL `0.0305`) |

## What Is Quantized

- **Quantized (`319` linear layers total across `model.safetensors-00001-of-00004.safetensors` .. `model.safetensors-00004-of-00004.safetensors`)**:
  - **Language (`200` linears in `W4A16-HGPTQ`)**: All 32 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + MSE-initialized Cholesky GPTQ (`group_size=64`) + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`) + end-to-end FP32-master `RMSNorm` KL tuning.
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 4096]`) and `mtp.fc` (`[4096, 8192]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`), saving `0.95 GiB` (`970.7 MiB`) with `+0.00001` weighted KL impact and zero online Hadamard overhead.
  - **Native MTP (`7` transformer linears in `W4A16-HGPTQ`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears in `W4A16-HGPTQ`)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`) & **release chat template** (`chat_template.jinja`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized linear layers have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$, where $H_b$ is the deterministic $b \times b$ Sylvester Walsh–Hadamard matrix; $b=64$ for `v` and $b=16$ for `v_vision_fc2`) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_b / \sqrt{b}$) is applied to input activations before the W4A16 GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
  - Because $H_b$ is deterministic (`randomize: false`), no transform weight matrices are stored in `.safetensors` (`0` storage overhead).
