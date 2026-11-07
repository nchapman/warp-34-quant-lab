---
base_model: Qwen/Qwen3.8-Flash-Next
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w4a16
- moe
- ple
- compressed-tensors
- vllm
---

# Qwen3.8-Flash-Next-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`) + 4-bit `Custom-LUT4` Layer-1 PLE** quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) across **Text (`74,028` linears), Layer-1 Per-Layer Embedding (`51.2B` params), Native MTP (`1,543` linears), and Vision (`110` linears)**, reducing the checkpoint payload from `335.28 GiB` (`360 GB`) in BF16 to **`93.42 GiB` (`3.59×` smaller)** across `59` safetensors shards.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.8-Flash-Next`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.8-Flash-Next`) | **`Qwen3.8-Flash-Next-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `335.28 GiB` (`360,000,192,888 B`) | **`93.42 GiB` (`100,306,498,784 B`)** | **`3.59×` compression** (`−241.86 GiB` across `59` shards) |
| **Text Perplexity** (378 prompts, 852k tokens) ↓ | `3.6085` | **`3.5963` (`−0.34%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0504` / `0.0920`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 23k tokens) ↓ | `24.2483` | **`26.8842` (`+10.87%`)** | Token mean KL `0.1007` (weighted KL `0.1308`, English chat PPL `22.31 → 22.93`) |
| **Native MTP (Speculative Drafter)** (12,339 positions) | `100.0%` Top-1 (`NLL 2.6008`) | **`95.01%` Top-1 (`NLL 2.6179`)** | Isolated MTP KL `0.0217` (end-to-end Top-1 `87.55%`, KL `0.1689`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9871` cos sim · `96.69%` Top-1** | Isolated ViT KL `0.0171` (end-to-end Top-1 `94.04%`, KL `0.0746`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9903` cos sim · `99.26%` Top-1** | Isolated ViT KL `0.0098` (end-to-end Top-1 `97.04%`, KL `0.0556`) |

## What Is Quantized

- **Text Trunk (`48` hybrid GatedDeltaNet + QSA-Indexed-Attention + 512-expert MoE layers, `62.651 GiB`):**
  - `156` attention projections (`linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj`, `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`) quantized with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - `144` shared MoE expert projections (`48` layers × `3` projections: `mlp.shared_expert.{gate_proj,up_proj,down_proj}`) quantized with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - `73,728` routed MoE expert projections (`48` layers × `512` experts × `3` projections: `mlp.experts.0..511.{gate_proj,up_proj,down_proj}`) quantized with online block-Hadamard input rotation (`head_dim=64` on both $K_{\text{in}}=2560$ and $K_{\text{down}}=640$) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
- **Layer-1 N-Gram Per-Layer Embedding (`51.2B` parameters / `102.4 GB` BF16 → `26.822 GiB` across `8` shards `model-ple-000..007.safetensors`):**
  - All `128` n-gram embedding shards (`320,000,456` rows × `160` dim) quantized to 4-bit non-uniform `Custom-LUT4` (`group_size=32`, `90 B/row`: `80 B` packed 4-bit nibbles + `10 B` FP16 scales) using exact downstream `W_kv`-HGPTQ error feedback ($H = W_k^\top W_k + W_v^\top W_v$) + 1-pass Coordinate Descent so quantization error is orthogonal to the downstream `layers.1.ple.{key_proj,value_proj}` subspaces.
  - Each shard stores linear tables (`weight_packed`, `weight_scale`, `weight_lut`, `weight_shape`), and `45` rows (`90 B/row`) fit into a single `4096 B` OS page (`98.9%` density) for single-page-fault `mmap` / `io_uring` lookup.
- **Native MTP (`1,543` linears, `1.340 GiB` in `model-mtp.safetensors`):**
  - `4` rotated `self_attn` projections (`head_dim=64`) + `3` rotated `mlp.shared_expert` projections (`head_dim=64`) + `1,536` rotated `group_size=64` routed MoE expert projections (`512` experts, `head_dim=64`).
- **Vision Tower (`110` rotated linears, `0.225 GiB` in `model-vision.safetensors`):**
  - `27` transformer blocks (`qkv`, `proj`, `linear_fc1`, `linear_fc2`) + `merger` (`linear_fc1`, `linear_fc2`) quantized with online block-Hadamard rotation + Cholesky GPTQ (`group_size=64`; `visual.blocks.*.mlp.linear_fc2` uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard `v_vision_fc2`).
- **Kept in BF16/FP32**:
  - `embed_tokens`, `lm_head`, `hyper_connection_mixer`, `*.attn_hyper_connection.*`, `*.mlp_hyper_connection.*`, `*.linear_attn.{in_proj_a,in_proj_b,conv1d,A_log,dt_bias,norm}`, `*.self_attn.indexer.*`, `layers.1.ple.{key_proj,value_proj,norm_key,norm_query,norm_conv,conv1d}`, `*.mlp.{gate,shared_expert_gate}`, `mtp.{fc,pre_fc_norm_embedding,pre_fc_norm_hidden,norm,hyper_connection_mixer}`, `visual.patch_embed.proj`, norms, and biases.

## Shard Layout & Memory Tiering (`93.418 GiB` total)

| Component | Shards | Size | Format | Recommended Placement (`1x / 2x RTX 3090`) |
|---|---:|---:|---|---|
| Embed + LM Head + Mixer | `model-embed-head.safetensors` | `2.380 GiB` | BF16 | GPU VRAM resident |
| 48 Target Layers (Attn + Shared MLP + HC + Norms + PLE Proj) | `model-layer-00..47.safetensors` | `~3.85 GiB` | W4A16-HGPTQ (`g64-h64`) + BF16 | GPU VRAM resident |
| 48 Target Layers Routed Experts (`48 × 512` experts) | `model-layer-00..47.safetensors` | `~58.80 GiB` | W4A16-HGPTQ (`g64-h64`) | Hot cache in GPU VRAM + Cold experts in pinned DDR (`HostExpertStore` / UVA) |
| Layer-1 PLE N-Gram Table (`128` shards, `51.2B` params) | `model-ple-000..007.safetensors` | `26.822 GiB` | 4-bit `Custom-LUT4` (`g32`, `90 B/row`) | Linux page cache (`mmap`) / CPU offload |
| Native MTP Layer (`512` experts + Shared MLP + QSA Attn) | `model-mtp.safetensors` | `1.340 GiB` | W4A16-HGPTQ (`g64-h64`) | Hot cache in GPU VRAM + Cold experts in pinned DDR |
| Vision Tower (`27` blocks + merger) | `model-vision.safetensors` | `0.225 GiB` | W4A16-HGPTQ (`g64-h64` / `channel-h16`) | GPU VRAM resident (when multimodal enabled) |

## Static Expert Hot-Cache Rankings (`runtime/static_hot_cache_rankings.json`)

Both `runtime/static_hot_cache_rankings.json` and root `static_hot_cache_rankings.json` map layer indices `"0"` through `"47"` to global expert IDs (`0..511`) ordered from **most-frequently routed (hottest) to least-frequently routed**. Full per-layer activation counts, routing weight mass, and `mtp.0` rankings are provided in `runtime/expert_routing_profile.json` (`9,804,460` expert assignments per target layer):

| Layer | Active Experts | Top-5 Hottest Expert IDs | `top_32` Hit Rate | `top_64` Hit Rate | `top_88` Hit Rate | `top_128` Hit Rate | `top_256` Hit Rate |
|---|---:|---|---:|---:|---:|---:|---:|
| Layer `0` | `507 / 512` | `[306, 470, 279, 280, 266]` | `13.92%` | `23.95%` | `30.89%` | `41.41%` | `69.73%` |
| Layer `1` | `511 / 512` | `[130, 308, 320, 58, 383]` | `14.72%` | `25.46%` | `32.62%` | `43.51%` | `72.17%` |
| Layer `12` | `512 / 512` | `[34, 272, 139, 63, 166]` | `15.72%` | `26.99%` | `34.46%` | `45.40%` | `73.58%` |
| Layer `23` | `512 / 512` | `[499, 361, 242, 183, 84]` | `16.75%` | `28.12%` | `35.07%` | `45.19%` | `70.84%` |
| Layer `35` | `512 / 512` | `[189, 479, 18, 372, 341]` | `21.92%` | `34.77%` | `42.34%` | `52.95%` | `77.96%` |
| Layer `47` | `511 / 512` | `[242, 398, 157, 110, 55]` | `31.52%` | `45.60%` | `54.00%` | `65.32%` | `87.93%` |
| `mtp.0` | `512 / 512` | `[439, 433, 410, 257, 451]` | `26.80%` | `39.96%` | `47.72%` | `58.51%` | `81.41%` |
