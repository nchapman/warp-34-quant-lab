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

4-bit Hadamard + GPTQ (`W4A16`, group size 64) and 4-bit `Custom-LUT4` Layer-1 PLE quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) across Text, Layer-1 Per-Layer Embeddings, Vision, and the Native MTP speculative drafter. It compresses the 335.3 GiB (360 GB) BF16 checkpoint down to **93.42 GiB** (3.59× smaller) across 59 safetensors shards.

- **VRAM & shard breakdown:** 93.42 GiB total (62.7 GiB text MoE + 26.8 GiB 4-bit LUT Layer-1 PLE + 1.3 GiB Native MTP + 0.2 GiB vision) across 74,028 text linears, 1,543 MTP linears, and 110 vision linears.
- **Quantization recipe:** 64×64 Walsh–Hadamard input rotation + Cholesky GPTQ (`group_size=64`) on all 512 routed experts, shared experts, Attention/GDN, MTP, and Vision, plus KV-projected HGPTQ + 4-bit Lloyd–Max `Custom-LUT4` on the 51.2B-parameter Layer-1 PLE table.
- **Fidelity vs. BF16:** 0.0499 weighted KL (0.0563 mean KL, −0.34% PPL) on `probe-v4`, 0.1008 mean KL on the `probe-guard` holdout, 95.01% isolated MTP top-1 agreement, and 96.69%–99.26% vision top-1 agreement.

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

## Why 4-Bit `Custom-LUT4` + `W_kv`-HGPTQ for the `51.2B` Layer-1 N-Gram Table

In `Qwen/Qwen3.8-Flash-Next`, Layer 1 (`layers.1.ple`) contains a `51.2B`-parameter (`102.4 GB` / `95.37 GiB` BF16) N-gram hash-lookup table (`320,000,456` rows × `160` columns across `16` hash heads: `8` bigram + `8` trigram). Every token looks up `16` rows (`2560` total features) that feed **only** `layers.1.ple.key_proj` ($W_k$) and `layers.1.ple.value_proj` ($W_v$).

Before choosing `Custom-LUT4 + W_kv-HGPTQ` (`group_size=32`), we evaluated low-rank SVD (`rank 144/160` loses `5.66%` energy / `23.8%` RMSE because the `160`-dim spectrum is flat), orthogonal rotation (row crest factor is already `2.89`, below Gaussian `3.18`), 8-bit `FP8`/`INT8`, and existing 4-bit formats (`NVFP4`, uniform `INT4`, and `IQ4_NL`):

1. **Crossing the 96–128 GB Memory Cliff (`26.82 GiB` vs. `47.68 GiB` FP8):**
   - Leaving the PLE table in 8-bit `FP8` (`51.20 GB` / `47.68 GiB`) brings total model weights to **`114.3 GiB`** (`66.6 GiB` quantized trunk/MTP/Vision + `47.7 GiB` PLE), which exceeds a `96 GB` GPU (`RTX PRO 6000` / `4× RTX 3090`) and overflows the Linux page cache when `~45 GiB` of cold MoE experts are pinned in host RAM on a 96–128 GB system.
   - Compressing PLE to **`26.82 GiB` (`4.50 bits/weight`)** brings the entire `180B` model down to **`93.42 GiB`** (fitting inside `96 GB` VRAM or staying 100% resident in OS page cache) and cuts random memory-sector / CPU cache-line reads per lookup by `40%` (`90 B/row` vs. `160 B/row`).
2. **Non-Uniform Empirical Centroids (`Custom-LUT4`):**
   - Normalized 32-element PLE groups are super-Gaussian ($\kappa \approx 4.6$). Uniform `INT4` wastes outer quantization bins on sparse tails, while `NVFP4` (`E2M1`) has coarse steps near the core. A 16-entry non-uniform lookup table fitted to the empirical PLE distribution concentrates levels across the high-density core and fits inside a single CUDA warp register (`__shfl_sync`).
3. **Analytical Downstream `W_kv`-HGPTQ + 1-Pass Coordinate Descent:**
   - Because any finite calibration set touches `<0.1%` of the `320M` hash rows, activation-based GPTQ cannot calibrate the table. However, SVD of each head's downstream `key_proj` slice $W_{k,h} \in \mathbb{R}^{10240 \times 160}$ reveals a dominant rank-1 singular direction ($\sigma_1 = 5.66\text{–}6.88$ vs. $\sigma_2 \approx 2.2$, a `25×–47×` energy amplification) aligned at `0.9501` cosine similarity with the embedding table's top singular vector.
   - Because each head $h \in \{0,\dots,15\}$ feeds only $(W_{k,h}, W_{v,h})$, the exact downstream quadratic error has a closed-form, calibration-free $160 \times 160$ Hessian:
     $$H_h = \frac{1}{2}\left(\frac{W_{k,h}^\top W_{k,h}}{\|W_{k,h}\|_F^2 / 160} + \frac{W_{v,h}^\top W_{v,h}}{\|W_{v,h}\|_F^2 / 160}\right) + \lambda D \in \mathbb{R}^{160 \times 160}$$
   - Running Cholesky GPTQ (`group_size=32`, static actorder) + 1 pass of coordinate-descent scale refinement against $H_h$ steers quantization noise out of $W_{k,h}$'s amplified singular direction across all `320M` rows, cutting `key_proj` reconstruction error by **`29.6%` vs. `NVFP4 g16`** and **`22.3%` vs. `32 GB` `INT4 g16`**:

| 4-Bit PLE Scheme (`320M × 160`) | Effective `bpw` | Total PLE Size | `E` Rel-RMSE | **`key_proj(E)` Rel-RMSE** ↓ | **`value_proj(E)` Rel-RMSE** ↓ |
|---|---:|---:|---:|---:|---:|
| `NVFP4 E2M1` (`group_size=16`, RTN) | `4.50` | `28.80 GB` (`26.82 GiB`) | `9.53%` | `9.45%` | `9.53%` |
| Uniform `INT4` symmetric (`group_size=32`, RTN) | `4.50` | `28.80 GB` (`26.82 GiB`) | `9.42%` | `9.37%` | `9.42%` |
| Uniform `INT4` symmetric (`group_size=16`, RTN) | `5.00` | `32.00 GB` (`29.80 GiB`) | `8.67%` | `8.57%` | `8.68%` |
| Non-uniform `IQ4_NL` (`group_size=32`, RTN) | `4.50` | `28.80 GB` (`26.82 GiB`) | `8.29%` | `7.43%` | `8.29%` |
| **`Custom-LUT4 + W_kv-HGPTQ + CD` (`group_size=32`, This Checkpoint)** | **`4.50`** | **`28.80 GB` (`26.82 GiB`)** | **`7.99%`** | **`6.66%`** | **`7.93%`** |

4. **Zero Page-Straddle `4 KiB` Slotting (`45 rows/page`):**
   - Each token performs `16` random row lookups; with unaligned contiguous rows, `39.8%` of tokens at 8-bit (`160 B/row`) and `29.9%` at 4-bit (`90 B/row`) have at least one head straddling two `4,096 B` OS/NVMe pages. At `90 B/row` (`80 B` packed nibbles + `10 B` FP16 scales), packing `45` complete rows into each `4,096 B` page uses `4,050 / 4,096 B` (**`98.88%` page density**) with **`0.000%` page-boundary straddles**.

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
