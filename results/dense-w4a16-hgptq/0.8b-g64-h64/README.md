---
base_model: Qwen/Qwen3.5-0.8B
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

# Qwen3.5-0.8B-W4A16-HGPTQ

4-bit Hadamard + GPTQ (`W4A16`, group size 64) quantization of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) across Text, Vision, and the Native MTP speculative drafter. It cuts the 1.63 GiB BF16 checkpoint in half to **0.79 GiB** (2.07× smaller), fitting the entire multimodal model and speculative drafter in under 850 MiB of memory.

- **VRAM & footprint:** 0.79 GiB total across all 150 text linears, 50 vision linears, and 8 Native MTP linears (`mtp.fc` in `W8_g64`, tied `lm_head` in BF16).
- **Quantization recipe:** 64×64 Walsh–Hadamard input rotation + MSE-initialized Cholesky GPTQ + 2-pass Hessian Coordinate Descent (`H-CD`) + end-to-end FP32-master `RMSNorm` KL tuning.
- **Fidelity vs. BF16:** 0.0735 weighted KL (0.1816 mean KL) on `probe-v4`, 0.0887 weighted KL (+3.17% PPL) on the `probe-guard` holdout, 90.24% / 78.36% isolated/E2E MTP top-1 agreement, and 94.04% / 88.74% isolated/E2E vision top-1 agreement (94.81% E2E on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-0.8B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-0.8B`) | **`Qwen3.5-0.8B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `1.63 GiB` (`1,746,942,600 B`) | **`0.79 GiB` (`844,779,168 B`)** | **`2.07×` compression** (`−0.84 GiB`) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `4.1695` | **`4.3015` (`+3.17%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0735` / `0.2833`** | Full-vocabulary FP32 KL (`math` `0.0271`, `science` `0.0370`, `if` `0.0392`, `chat` `0.0471`) |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `15.4384` | **`16.3150` (`+5.68%`)** | Token mean KL `0.0706` (weighted KL `0.0887`, `no_robots` KL `0.0554`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.8764`) | **`90.24%` Isolated Top-1 (`NLL 2.8997`)** | Isolated MTP KL `0.0288` (end-to-end Top-1 `78.36%`, KL `0.4515`) |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`94.04%` Isolated Top-1 · `88.08%` E2E Top-1** | Isolated ViT KL `0.0266`, E2E KL `0.1388` (`vision-guard-v2` E2E Top-1 `94.81%`, KL `0.0884`) |

## What Is Quantized

- **Quantized (`208` linear layers total in `model.safetensors-00001-of-00001.safetensors`)**:
  - **Language (`150` linears in `W4A16-HGPTQ`)**: All 24 hybrid GatedDeltaNet + full-attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) with online block-Hadamard rotation (`head_dim=64`) + MSE-initialized Cholesky GPTQ (`group_size=64`) + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`) + end-to-end FP32-master `RMSNorm` KL tuning.
  - **Native MTP (`7` transformer linears in `W4A16-HGPTQ` + `mtp.fc` in `W8_g64`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`), plus `mtp.fc` (`[1024, 2048]`) in unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`).
  - **Vision (`50` linears in `W4A16-HGPTQ`)**: All 12 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`, `intermediate_size=3072`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`) quantized with `group_size=64` Cholesky GPTQ + `head_dim=64` online Hadamard rotation (`v`).
- **Kept in BF16**:
  - Tied `embed_tokens` / `lm_head` (`tie_word_embeddings: true`), `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`) & **release chat template** (`chat_template.jinja`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized linear layers have weights pre-rotated offline along the input dimension ($W' = W H_{64} / \sqrt{64}$, where $H_{64}$ is the deterministic $64 \times 64$ Sylvester Walsh–Hadamard matrix) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_{64} / \sqrt{64}$) is applied to input activations before the W4A16 GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
  - Because $H_{64}$ is deterministic (`randomize: false`), no transform weight matrices are stored in `.safetensors` (`0` storage overhead).
