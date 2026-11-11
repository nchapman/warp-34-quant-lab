---
base_model: vllm-sr/d3-lite
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w4a16
- compressed-tensors
- decision-model
- code-readout-v1
---

# d3-lite-W4A16-HGPTQ

4-bit Walsh–Hadamard + Cholesky GPTQ (`W4A16-HGPTQ`, group size 64) quantization of [vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite) (`vllm-sr/d3` `code-readout-v1` single-pass decision model). It compresses the `1.59 GiB` BF16 checkpoint (`readout.safetensors` + backbone) to **`0.78 GiB`** (**`2.05×` smaller**) while preserving `91.67%` top-1 decision agreement and `0.998205` readout logit cosine similarity against the BF16 reference.

- **Footprint:** `0.78 GiB` total across `150` quantized text linears, plus `50` vision tower & merger linears, plus the exact BF16 `readout.safetensors` (`255 × 1024` code-readout head).
- **Quantization recipe:** 64×64 Sylvester Walsh–Hadamard input rotation (`H_64`) + MSE-initialized Cholesky GPTQ (`group_size=64`, `actorder="static"`, `dampening_frac=0.01`), calibrated layer-by-layer under `noncausal_full_attention` attention.
- **Architecture & recurrence:** `24` layers (`hidden_size=1024`). Single-pass hybrid GatedDeltaNet + full-attention forward pass.
- **Canonical weight layout:** All safetensors keys use canonical `model.language_model.*` (and `model.visual.*`) prefixes with `"quantization_status": "compressed"`, loading cleanly with `missing: 0, unexpected: 0` in `Qwen3_5Model.from_pretrained`, `D3.from_pretrained`, and `warp-34`.

## Quality vs. BF16 Base Model (`vllm-sr/d3-lite`)

Single-pass `code-readout-v1` evaluation across multi-question `choice`, `yes_no`, `multi_select`, and `scale` decision tasks (with forward + reverse option-permutation averaging):

| Evaluation | BF16 Base (`vllm-sr/d3-lite`) | **`d3-lite-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** (incl. `readout.safetensors`) | `1.59 GiB` (`1,707,072,248 B`) | **`0.78 GiB` (`833,893,048 B`)** | **`2.05×` compression** (`−0.81 GiB`) |
| **Decision Top-1 Agreement** (`12` questions) ↑ | `100.00%` | **`91.67%`** | Forward + reverse permutation-averaged readout |
| **Option Probability KL Divergence** (mean / max nats) ↓ | `0.000000` | **`0.005819` / `0.058474`** | Per-question normalized option distribution KL |
| **255-Code Readout Logit Cosine Similarity** ↑ | `1.000000` | **`0.998205`** | Full 255-code softmax KL `0.007149` nats |
| **Vision Tower (`vision-probe-v2`)** (96 images, 50,596 visual tokens) | `1.000000` cos sim | **`0.974470` mean cos (`0.953063` min)** | `vision-guard-v2` mean cos `0.974600` (min `0.953356`) |

## What Is Quantized

- **Quantized (`200` linear layers in `W4A16-HGPTQ`)**:
  - **Language (`150` linears)**: All `24` hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - **Vision (`50` linears)**: All 12 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus `visual.merger.{linear_fc1,linear_fc2}` in `g64-h64`.
- **Kept in BF16**:
  - `readout.safetensors` (`weight` `[255, 1024]`), `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, norms, biases, and `visual.patch_embed.proj`.

## Usage

```python
from d3_runtime import D3

model = D3.from_pretrained("nchapman/d3-lite-W4A16-HGPTQ", device="cuda:0")
result = model.system_one(
    state="Customer ordered item #4412, delivered on time, requested an invoice copy.",
    questions={
        "intent": {
            "type": "choice",
            "instructions": "What does the customer need?",
            "criteria": {"invoice": "Copy of invoice", "refund": "Return and refund"},
        }
    },
)
print(result)
```
