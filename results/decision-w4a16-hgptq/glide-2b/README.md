---
base_model: fastino/GLiDE-2B
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

# GLiDE-2B-W4A16-HGPTQ

4-bit Walsh–Hadamard + Cholesky GPTQ (`W4A16-HGPTQ`, group size 64) quantization of [fastino/GLiDE-2B](https://huggingface.co/fastino/GLiDE-2B) (`GLiDE` `code-readout-v1` single-pass decision model). It compresses the `3.51 GiB` BF16 checkpoint (`readout.safetensors` + backbone) to **`1.63 GiB`** (**`2.15×` smaller**) while preserving `100.00%` top-1 decision agreement and `0.999073` readout logit cosine similarity against the BF16 reference.

- **Footprint:** `1.63 GiB` total across `150` quantized text linears (text-only decision checkpoint), plus the exact BF16 `readout.safetensors` (`255 × 2048` code-readout head).
- **Quantization recipe:** 64×64 Sylvester Walsh–Hadamard input rotation (`H_64`) + MSE-initialized Cholesky GPTQ (`group_size=64`, `actorder="static"`, `dampening_frac=0.01`), calibrated layer-by-layer under `noncausal_full_attention` attention.
- **Architecture & recurrence:** `24` layers (`hidden_size=2048`). Single-pass hybrid GatedDeltaNet + full-attention forward pass.
- **Canonical weight layout:** All safetensors keys use canonical `model.language_model.*` (and `model.visual.*`) prefixes with `"quantization_status": "compressed"`, loading cleanly with `missing: 0, unexpected: 0` in `Qwen3_5Model.from_pretrained`, `Glide.from_pretrained`, and `warp-34`.

## Quality vs. BF16 Base Model (`fastino/GLiDE-2B`)

Single-pass `code-readout-v1` evaluation across multi-question `choice`, `yes_no`, `multi_select`, and `scale` decision tasks (with forward + reverse option-permutation averaging):

| Evaluation | BF16 Base (`fastino/GLiDE-2B`) | **`GLiDE-2B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** (incl. `readout.safetensors`) | `3.51 GiB` (`3,764,736,616 B`) | **`1.63 GiB` (`1,750,331,856 B`)** | **`2.15×` compression** (`−1.88 GiB`) |
| **Decision Top-1 Agreement** (`12` questions) ↑ | `100.00%` | **`100.00%`** | Forward + reverse permutation-averaged readout |
| **Option Probability KL Divergence** (mean / max nats) ↓ | `0.000000` | **`0.000158` / `0.000933`** | Per-question normalized option distribution KL |
| **255-Code Readout Logit Cosine Similarity** ↑ | `1.000000` | **`0.999073`** | Full 255-code softmax KL `0.002010` nats |

## What Is Quantized

- **Quantized (`150` linear layers in `W4A16-HGPTQ`)**:
  - **Language (`150` linears)**: All `24` hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - **Vision (`0` linears)**: None (text-only checkpoint).
- **Kept in BF16**:
  - `readout.safetensors` (`weight` `[255, 2048]`), `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, norms, biases.

## Usage

```python
from glide import Glide

model = Glide.from_pretrained("nchapman/GLiDE-2B-W4A16-HGPTQ", device="cuda:0")
result = model.decide(
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
