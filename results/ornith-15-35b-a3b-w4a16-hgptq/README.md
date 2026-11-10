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

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`, `MSE + 2-Pass H-CD + FP32 Norm/GDN QAT + W8_g64 Head`)** quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) across **Text (`30,970` W4A16 linears + `1` `W8_g64` `lm_head`), Native MTP (`775` W4A16 linears + `1` `W8_g64` `mtp.fc`), and Vision (`110` linears)**, reducing the checkpoint payload from `66.97 GiB` in BF16 to **`18.78 GiB` (`3.57×` smaller)**.

> **Summary:** Compresses `Ornith-1.5-35B-A3B` by `3.57×` (`67.0 GiB → 18.78 GiB`, `−48.19 GiB` of VRAM) to fit within a **24 GB GPU** budget. Main-suite text tracking (`0.1139` weighted KL, `0.1647` holdout mean KL) and vision (`96.36%–97.04%` top-1 agreement) are solid, with two caveats inherited from the sharp fine-tune: out-of-distribution holdout prompts show higher divergence (`0.2289` weighted KL), and speculative drafter agreement is lower (`91.44%` vs. `95.80%` on base `Qwen3.6-35B-A3B`) because the upstream fine-tune shifted the backbone away from its un-updated MTP head.

## Quality vs. BF16 Base Model (`ornith-ai/Ornith-1.5-35B-A3B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`ornith-ai/Ornith-1.5-35B-A3B`) | **`Ornith-1.5-35B-A3B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `66.97 GiB` (`71,903,871,064 B`) | **`18.78 GiB` (`20,160,901,544 B`)** | **`3.57×` compression** (`−48.19 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `10.8033` | **`10.2627` (`−5.00%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.1139` / `0.2303`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `249.8651` | **`257.9844` (`+3.25%`)** | Token mean KL `0.1647` (weighted KL `0.2289`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 4.4308`) | **`91.44%` Top-1 (`NLL 4.4458`)** | Isolated MTP KL `0.0215` (end-to-end Top-1 `73.34%`, KL `0.2596`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9856` cos sim · `96.36%` Top-1** | Isolated ViT KL `0.0305` (end-to-end Top-1 `94.37%`, KL `0.0875`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9900` cos sim · `97.04%` Top-1** | Isolated ViT KL `0.0162` (end-to-end Top-1 `96.30%`, KL `0.0411`) |

## What Is Quantized

- **Quantized (`31,857` linear layers total)**:
  - **Language (`30,970` W4A16 linears + `1` `W8_g64` linear)**: `130` attention projections (`self_attn`, `linear_attn`) + `120` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) + `30,720` routed MoE expert projections (`40` layers × `256` experts × `3` projections) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, MSE scale search, 2-pass H-CD) + End-to-End FP32-master Norm/GDN KL QAT, plus `lm_head` in unrotated `W8_g64`.
  - **Native MTP (`775` W4A16 linears + `1` `W8_g64` linear)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `768` routed MoE expert projections (`256` experts × `3` projections) (`head_dim=64`, `group_size=64`), plus `mtp.fc` in unrotated `W8_g64`.
  - **Vision (`110` linears)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `visual.patch_embed.proj`, norms, and biases.
- **Static Hot-Expert Cache Rankings**:
  - `static_hot_cache_rankings.json` (and `runtime/static_hot_cache_rankings.json` + `runtime/expert_routing_profile.json`) provides pre-profiled expert rankings (`0..255` ordered hottest-first per layer `"0".."39"`, plus `"mtp.0"` in `expert_routing_profile.json`; `top_64 = 41.00%` hit rate, `top_128 = 67.20%` hit rate).
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Rotation Across Attention, Shared Experts, and Routed MoE Experts**:
  - Every quantized linear in Text (`30,970` linears) and Native MTP (`775` linears) — including `self_attn`, `linear_attn`, `mlp.shared_expert.{gate_proj,up_proj,down_proj}`, and `mlp.experts.0..255.{gate_proj,up_proj,down_proj}` — has its weights pre-rotated offline along the input dimension ($W' = W H_{64} / 8$) prior to Cholesky GPTQ quantization (`group_size=64`, `actorder="static"`).
  - At runtime, apply block-64 Walsh–Hadamard transform ($x \mapsto x H_{64} / 8$) to the input activations before `gate_proj`/`up_proj` (`K=2048`) and to the post-SwiGLU intermediate activations before `down_proj` (`K=512`), matching `quantization_config.transform_config.config_groups.v`.
  - Only `mlp.gate` and `mlp.shared_expert_gate` remain in BF16 (listed in `quantization_config.ignore`).
