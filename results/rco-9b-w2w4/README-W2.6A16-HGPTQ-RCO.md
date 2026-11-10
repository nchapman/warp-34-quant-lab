---
base_model: Qwen/Qwen3.5-9B
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

# Qwen3.5-9B-W2.6A16-HGPTQ-RCO

Mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.6A16`, group size 64) quantization of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) built with Riemannian Co-design (`HGPTQ-RCO`). It compresses the 18.0 GiB BF16 model down to **5.54 GiB** (3.24× smaller; 5.11 GiB text-only) so the full multimodal model and Native MTP drafter fit on a single **8 GB GPU** (RTX 3070 / 4060 / 4060 Ti 8GB / 5060).

- **VRAM & GPU fit:** 5.54 GiB full VLM+MTP (5.11 GiB text-only), leaving ~2.45 GiB on an 8 GB card for KV cache, vision tokens, speculative decoding, and CUDA graphs.
- **Quantization recipe:** Upgrades the top 30% most sensitive fused `vLLM` backbone groups (9/16 Full-Attention, 18/48 GDN Linear-Attention, 22/64 MLP) to `W4_g64`, keeps Vision and Native MTP transformer blocks in `W4_g64` (`lm_head` and `mtp.fc` in `W8_g64`), and polishes the remaining `W2_g64` scales + FP32-master `RMSNorm`/GDN gates end-to-end.
- **Fidelity vs. BF16:** 0.2661 weighted KL (0.3435 mean KL) on `probe-v4`, 0.3639 mean KL (+0.47% PPL) on the `probe-guard` holdout, 94.86% / 77.46% isolated/E2E MTP top-1 agreement, and 99.34% / 90.73% isolated/E2E vision top-1 agreement.

| Variant | Avg Backbone Bits | Full VLM+MTP Size | Text-Only Size | Target GPU Tier |
|---|---:|---:|---:|---|
| **`Qwen3.5-9B-W2.6A16-HGPTQ-RCO` (this repo)** | **`2.60` (`30%` `W4`)** | **`5.54 GiB`** | **`5.11 GiB`** | **`8 GB` GPU (full VLM + MTP + CUDA graphs)** |
| [`Qwen3.5-9B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-9B-W4A16-HGPTQ) | `4.00` (`100%` `W4`) | `6.67 GiB` | `6.24 GiB` | `10–12 GB` GPU (or `8 GB` shorter context) |

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-9B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-9B`) | **`Qwen3.5-9B-W2.6A16-HGPTQ-RCO`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `17.98 GiB` (`19,306,310,880 B`) | **`5.54 GiB` (`5,951,375,352 B`)** | **`3.24×` compression** (`−12.44 GiB`) |
| **Text Perplexity (`probe-v4`)** (378 prompts, 840k tokens) ↓ | `3.5754` | **`2.8545` (`−20.16%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16 (`probe-v4`)** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.2661` / `0.3435`** | 378 prompts, 840k tokens (`math` `0.1048`, `science` `0.1707`, `if` `0.1979`, `chat` `0.2261`) |
| **General & Multilingual Holdout (`probe-guard`)** (76 prompts, 20k tokens) ↓ | `10.5643` | **`10.6139` (`+0.47%`)** | Token mean KL `0.3639` (weighted KL `0.4571`, 9-lang Belebele KL `0.4559`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.0275`) | **`94.86%` Isolated Top-1 · `77.46%` E2E Top-1** | Isolated MTP KL `0.0105`, E2E KL `0.4125` |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`99.34%` Isolated Top-1 · `90.73%` E2E Top-1** | Isolated ViT KL `0.0073`, E2E KL `0.1966` (`vision-guard-v2` Iso Top-1 `98.52%`, E2E Top-1 `85.93%`) |

## What Is Quantized

- **Quantized (`319` linear layers total across `model.safetensors-00001-of-00004.safetensors` .. `model.safetensors-00004-of-00004.safetensors`)**:
  - **Language Backbone (`200` linears across `128` fused `vLLM` groups, `2.60` bits/weight)**:
    - **`65` linears in `W4A16-HGPTQ` (`w4_g64`)**: `9 / 16` (`56.3%`) Full-Attention groups, `18 / 48` (`37.5%`) GatedDeltaNet Linear-Attention groups, and `22 / 64` (`34.4%`) MLP groups selected via non-myopic Gumbel-Softmax Riemannian Co-design (`60` group swaps vs. greedy Hessian sensitivity), calibrated with MSE-initialized Cholesky GPTQ + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`).
    - **`135` linears in `W2A16-HGPTQ` (`w2_g64`)**: Remaining backbone groups quantized with symmetric 2-bit (`{-2, -1, 0, 1}`) Hadamard + Cholesky GPTQ + 3-pass `H-CD` and polished end-to-end via continuous `log_scale` + FP32-master `RMSNorm`/GDN-gate KL distillation on `data/calib-cascade-v1.jsonl`.
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 4096]`) and `mtp.fc` (`[4096, 8192]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`), saving `0.95 GiB` (`970.7 MiB`) with `+0.0001` weighted KL impact and zero online Hadamard overhead.
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
