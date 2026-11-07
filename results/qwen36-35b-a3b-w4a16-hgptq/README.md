---
base_model: Qwen/Qwen3.6-35B-A3B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w4a16
- moe
- compressed-tensors
- vllm
---

# Qwen3.6-35B-A3B-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) across **Text (`30,970` linears), Native MTP (`775` linears), and Vision (`110` linears)**, reducing the checkpoint payload from `66.97 GiB` in BF16 to **`19.24 GiB` (`3.48×` smaller)**.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.6-35B-A3B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.6-35B-A3B`) | **`Qwen3.6-35B-A3B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `66.97 GiB` (`71,903,776,776 B`) | **`19.24 GiB` (`20,661,694,432 B`)** | **`3.48×` compression** (`−47.72 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `4.0208` | **`4.0078` (`−0.32%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0868` / `0.2277`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `25.7581` | **`26.5040` (`+2.90%`)** | Token mean KL `0.0496` (weighted KL `0.0615`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.0268`) | **`95.71%` Top-1 (`NLL 2.0395`)** | Isolated MTP KL `0.0113` (end-to-end Top-1 `89.92%`, KL `0.1174`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9797` cos sim · `98.34%` Top-1** | Isolated ViT KL `0.0077` (end-to-end Top-1 `95.70%`, KL `0.0225`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9845` cos sim · `99.26%` Top-1** | Isolated ViT KL `0.0241` (end-to-end Top-1 `98.52%`, KL `0.0240`) |

## What Is Quantized

- **Quantized (`31,855` linear layers total)**:
  - **Language (`30,970` linears)**: `130` attention projections (`self_attn`, `linear_attn`) + `120` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) + `30,720` routed MoE expert projections (`40` layers × `256` experts × `3` projections) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - **Native MTP (`775` linears)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `768` routed MoE expert projections (`256` experts × `3` projections) (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens` / `lm_head`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `mtp.fc`, `visual.patch_embed.proj`, norms, and biases.
- **Static Hot-Expert Cache Rankings**:
  - `static_hot_cache_rankings.json` (and `runtime/static_hot_cache_rankings.json` + `runtime/expert_routing_profile.json`) provides pre-profiled expert rankings (`0..255` ordered hottest-first per layer `"0".."39"`, plus `"mtp.0"` in `expert_routing_profile.json`; `top_64 = 42.58%` hit rate, `top_128 = 69.12%` hit rate).
- **Default sampling** (`generation_config.json`): `temperature 0.6, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Rotation Across Attention, Shared Experts, and Routed MoE Experts**:
  - Every quantized linear in Text (`30,970` linears) and Native MTP (`775` linears) — including `self_attn`, `linear_attn`, `mlp.shared_expert.{gate_proj,up_proj,down_proj}`, and `mlp.experts.0..255.{gate_proj,up_proj,down_proj}` — has its weights pre-rotated offline along the input dimension ($W' = W H_{64} / 8$) prior to Cholesky GPTQ quantization (`group_size=64`, `actorder="static"`).
  - At runtime, apply block-64 Walsh–Hadamard transform ($x \mapsto x H_{64} / 8$) to the input activations before `gate_proj`/`up_proj` (`K=2048`) and to the post-SwiGLU intermediate activations before `down_proj` (`K=512`), matching `quantization_config.transform_config.config_groups.v`.
  - Only `mlp.gate` and `mlp.shared_expert_gate` remain in BF16 (listed in `quantization_config.ignore`).
