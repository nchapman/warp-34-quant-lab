---
base_model: vllm-sr/d3-flash
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

# d3-flash-W4A16-HGPTQ

4-bit Walsh–Hadamard + Cholesky GPTQ (`W4A16-HGPTQ`, group size 64) quantization of [vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash) (`vllm-sr/d3` `code-readout-v1` single-pass decision model). It compresses the `15.64 GiB` BF16 checkpoint (`readout.safetensors` + backbone) to **`5.56 GiB`** (**`2.81×` smaller**) while preserving `100.00%` top-1 decision agreement and `0.999728` readout logit cosine similarity against the BF16 reference.

- **Footprint:** `5.56 GiB` total across `200` quantized text linears, plus `137` vision tower & merger linears, plus the exact BF16 `readout.safetensors` (`255 × 4096` code-readout head).
- **Quantization recipe:** 64×64 Sylvester Walsh–Hadamard input rotation (`H_64`) + MSE-initialized Cholesky GPTQ (`group_size=64`, `actorder="static"`, `dampening_frac=0.01`), calibrated layer-by-layer under `noncausal_full_attention` attention.
- **Architecture & recurrence:** `32` layers (`hidden_size=4096`). Single-pass hybrid GatedDeltaNet + full-attention forward pass.
- **Canonical weight layout:** All safetensors keys use canonical `model.language_model.*` (and `model.visual.*`) prefixes with `"quantization_status": "compressed"`, loading cleanly with `missing: 0, unexpected: 0` in `Qwen3_5Model.from_pretrained`, `D3.from_pretrained`, and `warp-34`.

## Quality vs. BF16 Base Model (`vllm-sr/d3-flash`)

Single-pass `code-readout-v1` evaluation across multi-question `choice`, `yes_no`, `multi_select`, and `scale` decision tasks (with forward + reverse option-permutation averaging):

| Evaluation | BF16 Base (`vllm-sr/d3-flash`) | **`d3-flash-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** (incl. `readout.safetensors`) | `15.64 GiB` (`16,789,656,976 B`) | **`5.56 GiB` (`5,970,640,640 B`)** | **`2.81×` compression** (`−10.08 GiB`) |
| **Decision Top-1 Agreement** (`12` questions) ↑ | `100.00%` | **`100.00%`** | Forward + reverse permutation-averaged readout |
| **Option Probability KL Divergence** (mean / max nats) ↓ | `0.000000` | **`0.000086` / `0.000709`** | Per-question normalized option distribution KL |
| **255-Code Readout Logit Cosine Similarity** ↑ | `1.000000` | **`0.999728`** | Full 255-code softmax KL `0.000168` nats |
| **Vision Tower (`vision-probe-v2`)** (96 images, 50,596 visual tokens) | `1.000000` cos sim | **`0.984221` mean cos (`0.915869` min)** | `vision-guard-v2` mean cos `0.987428` (min `0.935132`) |

## What Is Quantized

- **Quantized (`337` linear layers in `W4A16-HGPTQ`)**:
  - **Language (`200` linears)**: All `32` hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - **Vision (`137` linears)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1}` in `g64-h64`, `visual.blocks.*.mlp.linear_fc2` in per-channel GPTQ + `head_dim=16` Hadamard) plus `visual.merger.{linear_fc1,linear_fc2}`.
- **Kept in BF16**:
  - `readout.safetensors` (`weight` `[255, 4096]`), `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, norms, biases, and `visual.patch_embed.proj`.

## Usage

```python
from d3_runtime import D3

model = D3.from_pretrained("nchapman/d3-flash-W4A16-HGPTQ", device="cuda:0")
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
