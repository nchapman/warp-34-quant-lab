---
base_model: ornith-ai/Ornith-1.5-35B-A3B
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

# Ornith-1.5-35B-A3B-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) across **Text (`30,970` linears), Native MTP (`775` linears), and Vision (`110` linears)**, reducing the checkpoint payload from `66.97 GiB` in BF16 to **`19.24 GiB` (`3.48×` smaller)**.

> **Summary:** Compresses `Ornith-1.5-35B-A3B` by `3.5×` (`67.0 GiB → 19.24 GiB`) to fit within a **24 GB GPU** budget. Main-suite text tracking (`0.124` KL) and vision (`97.0%` top-1 agreement) are solid, with two caveats inherited from the sharp fine-tune: out-of-distribution holdout prompts show higher divergence (`0.256` KL), and speculative drafter agreement is lower (`90.9%` vs. `95.7%` on base `Qwen3.6-35B-A3B`) because the upstream fine-tune shifted the backbone away from its un-updated MTP head.

## Quality vs. BF16 Base Model (`ornith-ai/Ornith-1.5-35B-A3B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`ornith-ai/Ornith-1.5-35B-A3B`) | **`Ornith-1.5-35B-A3B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `66.97 GiB` (`71,903,871,064 B`) | **`19.24 GiB` (`20,661,694,432 B`)** | **`3.48×` compression** (`−47.72 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `10.8033` | **`10.2988` (`−4.67%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.1245` / `0.2462`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `249.8651` | **`241.2982` (`−3.43%`)** | Token mean KL `0.1841` (weighted KL `0.2558`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 4.4308`) | **`90.90%` Top-1 (`NLL 4.4442`)** | Isolated MTP KL `0.0237` (end-to-end Top-1 `73.64%`, KL `0.2789`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9805` cos sim · `97.02%` Top-1** | Isolated ViT KL `0.0268` (end-to-end Top-1 `94.04%`, KL `0.1085`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9850` cos sim · `97.04%` Top-1** | Isolated ViT KL `0.0150` (end-to-end Top-1 `100.00%`, KL `0.0310`) |

## What Is Quantized

- **Quantized (`31,855` linear layers total)**:
  - **Language (`30,970` linears)**: `130` attention projections (`self_attn`, `linear_attn`) + `120` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) + `30,720` routed MoE expert projections (`40` layers × `256` experts × `3` projections) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - **Native MTP (`775` linears)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `768` routed MoE expert projections (`256` experts × `3` projections) (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens` / `lm_head`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `mtp.fc`, `visual.patch_embed.proj`, norms, and biases.
- **Static Hot-Expert Cache Rankings**:
  - `static_hot_cache_rankings.json` (and `runtime/static_hot_cache_rankings.json` + `runtime/expert_routing_profile.json`) provides pre-profiled expert rankings (`0..255` ordered hottest-first per layer `"0".."39"`, plus `"mtp.0"` in `expert_routing_profile.json`; `top_64 = 40.91%` hit rate, `top_128 = 67.12%` hit rate).
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Rotation Across Attention, Shared Experts, and Routed MoE Experts**:
  - Every quantized linear in Text (`30,970` linears) and Native MTP (`775` linears) — including `self_attn`, `linear_attn`, `mlp.shared_expert.{gate_proj,up_proj,down_proj}`, and `mlp.experts.0..255.{gate_proj,up_proj,down_proj}` — has its weights pre-rotated offline along the input dimension ($W' = W H_{64} / 8$) prior to Cholesky GPTQ quantization (`group_size=64`, `actorder="static"`).
  - At runtime, apply block-64 Walsh–Hadamard transform ($x \mapsto x H_{64} / 8$) to the input activations before `gate_proj`/`up_proj` (`K=2048`) and to the post-SwiGLU intermediate activations before `down_proj` (`K=512`), matching `quantization_config.transform_config.config_groups.v`.
  - Only `mlp.gate` and `mlp.shared_expert_gate` remain in BF16 (listed in `quantization_config.ignore`).
