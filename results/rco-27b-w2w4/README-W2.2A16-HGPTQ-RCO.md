---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w2a16
- w4a16
- compressed-tensors
- vllm
---

# Qwen3.8-27B-W2.2A16-HGPTQ-RCO

2.20-bit (`W2`/`W4` mixed backbone + `W8` head, group size 64) **Hadamard + Cholesky GPTQ with Riemannian Co-design (`HGPTQ-RCO`)** quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) across **Text (`400` backbone linears + `lm_head`), Native MTP (`8` linears), and Vision (`110` linears)**, reducing the checkpoint payload from `51.75 GiB` in BF16 to **`11.04 GiB` (`4.69×` smaller; `10.58 GiB` text-only)**.

> **Summary:** Compresses a `51.8 GiB` dense multimodal model by `4.69×` (`11.04 GiB` full VLM+MTP / `10.58 GiB` text-only, saving `40.7 GiB` of VRAM). Upgrading just the top `10%` (`+580 MiB`) most sensitive fused `vLLM` linear groups (`13/32` Full-Attention, `20/96` GDN Linear-Attention, `14/128` MLP) via Riemannian Co-design + end-to-end Scale/Norm QAT captures **45% of the entire `0%→30%` `probe-v4` KL reduction** (`0.3477 → 0.3137`). Use this checkpoint for **ultra-long-context `16 GB` serving** or **headless `12 GB` text-only serving** (`--language-model-only`); for general `16 GB` multimodal + MTP serving, prefer [`Qwen3.8-27B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO) (`12.18 GiB`).

| Variant | Avg Backbone Bits | Full VLM+MTP Size | Text-Only Size | Target GPU Tier |
|---|---:|---:|---:|---|
| **`Qwen3.8-27B-W2.2A16-HGPTQ-RCO` (this repo)** | **`2.20` (`10%` `W4`)** | **`11.04 GiB`** | **`10.58 GiB`** | **Headless `12 GB` (text-only) / ultra-long-context `16 GB`** |
| [`Qwen3.8-27B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO) | `2.60` (`30%` `W4`) | `12.18 GiB` | `11.71 GiB` | Recommended `16 GB` GPU (full VLM + MTP + CUDA graphs) |
| [`Qwen3.8-27B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ) | `4.00` (`100%` `W4`) | `17.33 GiB` | `16.86 GiB` | `24 GB` GPU (RTX 3090 / 4090 / 5090) |

## Quality vs. BF16 Base Model (`Qwen/Qwen3.8-27B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.8-27B`) | **`Qwen3.8-27B-W2.2A16-HGPTQ-RCO`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `51.75 GiB` (`55,563,006,776 B`) | **`11.04 GiB` (`11,856,436,968 B`)** | **`4.69×` compression** (`−40.70 GiB`; `10.58 GiB` text-only) |
| **Text Divergence vs. BF16 (`probe-v4`)** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.3137` / `0.4597`** | 378 prompts, 840k tokens (`math` `0.2051`, `chat` `0.2213`, `if` `0.2112`, `science` `0.2150`) |
| **General & Multilingual Holdout (`probe-guard`)** (token mean KL / weighted KL) ↓ | `0.0000` | **`0.3835` / `0.4784`** | 76 prompts, 23k tokens (held-out FineWeb-Edu + 9-lang Belebele macro KL `0.4647`) |
| **Native MTP (Speculative Drafter)** (12,339 positions) | `100.0%` Top-1 (`NLL 2.9579`) | **`96.64%` Isolated Top-1 (`NLL 2.9813`)** | Isolated MTP KL `0.0090` (end-to-end Top-1 `72.10%`, KL `0.7131`) |
| **Vision (`vision-probe-v2`)** (48 images, 151 answer tokens) | `100.0%` Top-1 | **`97.68%` Isolated Top-1 · `86.09%` E2E Top-1** | Isolated ViT KL `0.0056`, E2E KL `0.5599` (`vision-guard-v2` E2E Top-1 `85.19%`, KL `0.5783`) |

## What Is Quantized

- **Quantized (`519` linear layers total across `model-00001-of-00018.safetensors` .. `model-00018-of-00018.safetensors`)**:
  - **Language Backbone (`400` linears across `256` fused `vLLM` groups, `2.20` bits/weight)**:
    - **`51` linears in `W4A16-HGPTQ` (`w4_g64`)**: `13 / 32` (`40.6%`) Full-Attention groups, `20 / 96` (`20.8%`) GatedDeltaNet Linear-Attention groups, and `14 / 128` (`10.9%`) MLP groups selected via non-myopic Gumbel-Softmax Riemannian Co-design (`54` group swaps vs. greedy Hessian sensitivity), calibrated with MSE-initialized Cholesky GPTQ + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`).
    - **`349` linears in `W2A16-HGPTQ` (`w2_g64`)**: Remaining backbone groups quantized with symmetric 2-bit (`{-2, -1, 0, 1}`) Hadamard + Cholesky GPTQ + 3-pass `H-CD` and polished end-to-end via continuous `log_scale` + FP32-master `RMSNorm`/GDN-gate KL distillation on `data/calib-cascade-v1.jsonl`.
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 5120]`) and `mtp.fc` (`[5120, 10240]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`), saving `1.20 GiB` with `+0.00001` weighted KL impact and zero online Hadamard overhead.
  - **Native MTP (`7` transformer linears in `W4A16-HGPTQ`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears in `W4A16-HGPTQ`)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`) & **release chat template** (`chat_template.jinja`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Zero-patch stock `vLLM` + Hugging Face `compressed-tensors` compatibility**:
  - Uses standard `pack-quantized` int32 shards with `w2_g64` (`num_bits: 2`), `w4_g64` / `group_0` (`num_bits: 4`), `vision_fc2_channel` (`num_bits: 4`), and `w8_head` (`num_bits: 8`) config groups.
  - All `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj` sibling projections share identical bit-widths within every layer so `vLLM` fused `QKVParallelLinear` and `MergedColumnParallelLinear` load directly without patches.
- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized backbone, MTP, and Vision linear layers have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$, where $H_b$ is the deterministic $b \times b$ Sylvester Walsh–Hadamard matrix; $b=64$ for `v` and $b=16$ for `v_vision_fc2`) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_b / \sqrt{b}$) is applied to input activations before the GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
