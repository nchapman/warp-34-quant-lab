---
base_model: Qwen/Qwen3.6-35B-A3B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w2a16
- w4a16
- mixed-precision
- moe
- rco
- compressed-tensors
- vllm
---

# Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO

Sub-3-bit (`W2.6A16`, group size 64) **Hadamard + Cholesky GPTQ + Riemannian Constrained Optimization (`HGPTQ-RCO`)** mixed-precision (`W2_g64` / `W4_g64` / `W8_g64`) quantization of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) across **Text (`30,970` W2/W4 linears + `lm_head` in `W8_g64`), Native MTP (`775` W4A16 linears + `mtp.fc` in `W8_g64`), and Vision (`110` W4A16 linears)**, reducing the checkpoint payload from `66.97 GiB` in BF16 (`18.78 GiB` in pure `W4A16`) to **`13.15 GiB` (`5.09×` smaller than BF16, saving `5.63 GiB` vs. `W4A16`)**—fitting the **entire 35B-A3B MoE model + Vision tower + Native MTP drafter on a single 16 GB GPU** with `~2.85 GiB` of KV-cache and activation headroom. Keeps `100%` of Attention/GDN (`130` linears), `100%` of Shared Experts (`120` linears), `100%` of Native MTP (`775` linears), and `100%` of Vision (`110` linears) in `W4_g64` (plus `lm_head` and `mtp.fc` in `W8_g64`), while allocating `25%` (`10/40` layers = `2,560` experts) of routed MoE experts to `W4_g64` and `75%` (`30/40` layers = `7,680` experts) to `W2_g64` (`2.50 bpw` across routed experts, `2.62 bpw` overall across the language backbone).

## Quality vs. BF16 Base Model (`Qwen/Qwen3.6-35B-A3B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model and the `18.78 GiB` pure 4-bit [`Qwen3.6-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ) checkpoint:

| Evaluation | BF16 Base (`Qwen/Qwen3.6-35B-A3B`) | `W4A16-HGPTQ` (`4.00 bpw`) | **`W2.6A16-HGPTQ-RCO` (`2.62 bpw`)** | Notes |
|---|---:|---:|---:|---|
| **Safetensors Size** | `66.97 GiB` (`71,903,776,776 B`) | `18.78 GiB` (`20,160,901,544 B`) | **`13.15 GiB` (`14,121,080,536 B`)** | **`5.09×` compression** (`−53.82 GiB` vs. BF16, `−5.63 GiB` vs. W4A16) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `4.0208` | `3.7522` (`−6.68%`) | **`3.7230` (`−7.41%`)** | Chat, math (`0.0775` KL), science (`0.0871` KL), coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | `0.0847` / `0.2103` | **`0.2121` / `0.3885`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `25.7581` | `24.5254` (`−4.79%`) | **`21.2581` (`−17.47%`)** | Token mean KL `0.1943` (weighted KL `0.2080`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.0268`) | `95.80%` Top-1 (`KL 0.0105`) | **`95.80%` Top-1 (`NLL 2.0363`)** | Isolated MTP KL `0.0105` (end-to-end Top-1 `83.47%`, KL `0.2470`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `0.9860` cos sim · `99.34%` Top-1 | **`0.9860` cos sim · `99.34%` Top-1** | Isolated ViT KL `0.0072` (end-to-end Top-1 `91.72%`, KL `0.1385`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `0.9899` cos sim · `98.52%` Top-1 | **`0.9899` cos sim · `98.52%` Top-1** | Isolated ViT KL `0.0048` (end-to-end Top-1 `93.33%`, KL `0.1300`) |

## What Is Quantized (`HGPTQ-RCO` 16 GB Allocation)

- **Strictly Standard Bit-Widths (`{2, 4, 8}` bits, `group_size=64`)**:
  - **Always-Active Backbone (`100%` in `W4_g64`, `250` linears)**: All `130` hybrid GatedDeltaNet + full-attention projections (`self_attn`, `linear_attn`) and all `120` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) across all 40 layers stay in `W4_g64` (`head_dim=64`, MSE scale search + 2-pass H-CD).
  - **Routed MoE Experts (`2.50 bpw` average across `30,720` linears)**:
    - **`W4_g64` (`10` layers = `2,560` experts = `7,680` linears)**: Layers `30..39` (`model.language_model.layers.30..39.mlp.experts.0..255.{gate_proj,up_proj,down_proj}`).
    - **`W2_g64` (`30` layers = `7,680` experts = `23,040` linears)**: Layers `0..29` (`model.language_model.layers.0..29.mlp.experts.0..255.{gate_proj,up_proj,down_proj}`).
  - **Untied Output & Drafter Fusion Heads (`W8_g64`, `2` linears)**: `lm_head` (`248320 × 2048`) and `mtp.fc` (`2048 × 4096`) quantized with unrotated 8-bit Cholesky GPTQ + MSE scale search (`group_size=64`).
  - **Native MTP (`775` linears in `W4_g64`)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `768` routed MoE expert projections (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears in `W4_g64`)**: All 27 ViT blocks + 2 spatial merger projections (`visual.blocks.*.mlp.linear_fc2` uses per-channel GPTQ + `head_dim=16` Hadamard rotation `v_vision_fc2`; all other Vision linears use `group_size=64` + `head_dim=64`).
  - **End-to-End FP32-Master Norm & GDN-Gate KL QAT**: Post-allocation KL-divergence QAT over all layer `RMSNorm`s, `q_norm`/`k_norm`, GatedDeltaNet scalar projections (`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`), and `mlp.shared_expert_gate` against exact BF16 teacher final-norm states.
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `visual.patch_embed.proj`, norms, and biases.
- **Static Hot-Expert Cache Rankings**:
  - `static_hot_cache_rankings.json` (and `runtime/static_hot_cache_rankings.json` + `runtime/expert_routing_profile.json`) provides pre-profiled expert rankings (`0..255` ordered hottest-first per layer `"0".."39"`, plus `"mtp.0"` in `expert_routing_profile.json`; `top_64 = 43.15%` hit rate, `top_128 = 69.58%` hit rate).
- **Default sampling** (`generation_config.json`): `temperature 0.6, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).
