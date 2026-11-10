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

4-bit Hadamard + GPTQ (`W4A16`, group size 64) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) across Text, Vision, and the Native MTP speculative drafter. It shrinks the 51.8 GiB BF16 model down to **16.14 GiB** (3.21× smaller) so the full multimodal model and speculative drafter fit comfortably on a single **24 GB GPU** (RTX 3090 / 4090 / 5090).

- **VRAM & GPU fit:** 16.14 GiB total (15.67 GiB language + MTP, 0.47 GiB vision), leaving ~7.8 GiB on a 24 GB card for KV cache, vision tokens, MTP drafting, and CUDA graphs.
- **Quantization recipe:** 64×64 Walsh–Hadamard input rotation + MSE-initialized Cholesky GPTQ + 2-pass Hessian Coordinate Descent (`H-CD`) + end-to-end FP32-master `RMSNorm` KL tuning, with `lm_head` and `mtp.fc` in 8-bit (`W8_g64`).
- **Fidelity vs. BF16:** 0.1002 weighted KL (0.2583 mean KL, −4.09% PPL) on `probe-v4`, 0.0530 mean KL (0.0739 weighted KL, −7.23% PPL) on the `probe-guard` holdout, 96.64% / 89.16% isolated/E2E MTP top-1 agreement, and 97.68% / 92.05% isolated/E2E vision top-1 agreement (99.26% / 94.81% on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`Qwen/Qwen3.8-27B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.8-27B`) | **`Qwen3.8-27B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `51.75 GiB` (`55,563,006,776 B`) | **`16.14 GiB` (`17,330,005,464 B`)** | **`3.21×` compression** (`−35.61 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `8.5332` | **`8.1841` (`−4.09%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.1002` / `0.2583`** | Full-vocabulary FP32 KL (`math` `0.0718`, `chat` `0.0223`, `if` `0.0354`, `science` `0.0607`, `swe` `0.7075`) |
| **General & Multilingual Holdout Perplexity** (76 prompts, 23k tokens) ↓ | `61.0805` | **`56.6664` (`−7.23%`)** | Token mean KL `0.0530` (weighted KL `0.0739`, 9-lang Belebele macro KL `0.0628`) |
| **Native MTP (Speculative Drafter)** (12,339 positions) | `100.0%` Top-1 (`NLL 2.9579`) | **`96.64%` Isolated Top-1 (`NLL 2.9813`)** | Isolated MTP KL `0.0090` (end-to-end Top-1 `89.16%`, KL `0.1782`) |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`97.68%` Isolated Top-1 · `92.05%` E2E Top-1** | Isolated ViT KL `0.0056`, E2E KL `0.1808` (`vision-guard-v2` Iso Top-1 `99.26%`, E2E Top-1 `94.81%`, KL `0.0856`) |

## What Is Quantized

- **Quantized (`519` linear layers total across `model-00001-of-00018.safetensors` .. `model-00018-of-00018.safetensors`)**:
  - **Language (`400` linears in `W4A16-HGPTQ`)**: All 64 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + MSE-initialized Cholesky GPTQ (`group_size=64`) + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`) + end-to-end FP32-master `RMSNorm` KL tuning.
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 5120]`) and `mtp.fc` (`[5120, 10240]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`), saving `1.20 GiB` with `+0.00001` weighted KL impact and zero online Hadamard overhead.
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
