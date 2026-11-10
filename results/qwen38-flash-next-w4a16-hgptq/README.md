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

4-bit Hadamard + GPTQ (`W4A16`, group size 64) and 4-bit `Custom-LUT4` Layer-1 PLE quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) across Text, Layer-1 Per-Layer Embeddings, Vision, and the Native MTP speculative drafter. It compresses the 335.3 GiB (360 GB) BF16 checkpoint down to **92.84 GiB** (3.61× smaller) across 59 safetensors shards.

- **VRAM & shard breakdown:** 92.84 GiB total (64.5 GiB text MoE + 26.8 GiB 4-bit LUT Layer-1 PLE + 1.3 GiB Native MTP + 0.2 GiB vision) across 74,028 text linears, 1 `W8_g64` LM head, 1,543 MTP linears, and 110 vision linears (for 48 GB GPUs or 80 GB all-in HBM, see [`Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO) at 44.93 GiB trunk / 71.75 GiB total).
- **Quantization recipe:** 64×64 Walsh–Hadamard input rotation + MSE-initialized Cholesky GPTQ + 2-pass Hessian Coordinate Descent (`group_size=64`) on all 512 routed experts, shared experts, Attention/GDN, MTP, and Vision (`lm_head` in `W8_g64`), plus KV-projected HGPTQ + 2-pass H-CD + closed-form $W_{kv}$ MMSE compensation on the 51.2B-parameter Layer-1 PLE table, polished with end-to-end FP32-master Norm/GDN/HC/PLE KL QAT.
- **Fidelity vs. BF16:** 0.0421 weighted KL (0.0798 mean KL, −1.11% PPL) on `probe-v4`, 0.0673 mean KL (0.0884 weighted KL, −5.56% PPL) on the `probe-guard` holdout, 95.41% / 88.39% isolated/E2E MTP top-1 agreement, and 97.35% / 93.71% isolated/E2E vision top-1 agreement (99.26% / 96.30% on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`Qwen/Qwen3.8-Flash-Next`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.8-Flash-Next`) | **`Qwen3.8-Flash-Next-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `335.28 GiB` (`360,000,192,888 B`) | **`92.84 GiB` (`99,690,665,368 B`)** | **`3.61×` compression** (`−242.43 GiB` across `59` shards) |
| **Text Perplexity** (378 prompts, 852k tokens) ↓ | `3.6085` | **`3.5683` (`−1.11%`)** | Chat (`0.0157` KL), math (`0.0274` KL), science (`0.0274` KL), coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0421` / `0.0798`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 23k tokens) ↓ | `24.2483` | **`22.8992` (`−5.56%`)** | Token mean KL `0.0673` (weighted KL `0.0884`, `no_robots` KL `0.0517`, PPL `22.31 → 20.61`) |
| **Native MTP (Speculative Drafter)** (12,339 positions) | `100.0%` Top-1 (`NLL 2.6008`) | **`95.41%` Top-1 (`NLL 2.6186`)** | Isolated MTP KL `0.0184` (end-to-end Top-1 `88.39%`, KL `0.1473`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9906` cos sim · `97.35%` Top-1** | Isolated ViT KL `0.0171` (end-to-end Top-1 `93.71%`, KL `0.0602`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9930` cos sim · `99.26%` Top-1** | Isolated ViT KL `0.0087` (end-to-end Top-1 `96.30%`, KL `0.0593`) |

## What Is Quantized

- **Text Trunk (`48` hybrid GatedDeltaNet + QSA-Indexed-Attention + 512-expert MoE layers, `62.651 GiB` + `1.807 GiB` embed/head):**
  - `156` attention projections (`linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj`, `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`) quantized with online block-Hadamard rotation (`head_dim=64`) + MSE-initialized Cholesky GPTQ + 2-pass H-CD (`group_size=64`, `actorder="static"`).
  - `144` shared MoE expert projections (`48` layers × `3` projections: `mlp.shared_expert.{gate_proj,up_proj,down_proj}`) quantized with online block-Hadamard rotation (`head_dim=64`) + MSE-initialized Cholesky GPTQ + 2-pass H-CD (`group_size=64`, `actorder="static"`).
  - `73,728` routed MoE expert projections (`48` layers × `512` experts × `3` projections: `mlp.experts.0..511.{gate_proj,up_proj,down_proj}`) quantized with online block-Hadamard input rotation (`head_dim=64` on both $K_{\text{in}}=2560$ and $K_{\text{down}}=512$) + MSE-initialized Cholesky GPTQ + 2-pass H-CD (`group_size=64`, `actorder="static"`).
  - `1` untied output head (`lm_head`, `248320 × 2560`) quantized with unrotated 8-bit Cholesky GPTQ + MSE scale search (`W8_g64`).
- **Layer-1 N-Gram Per-Layer Embedding (`51.2B` parameters / `102.4 GB` BF16 → `26.822 GiB` across `8` shards `model-ple-000..007.safetensors`):**
  - All `128` n-gram embedding shards (`320,000,456` rows × `160` dim) quantized to 4-bit non-uniform `Custom-LUT4` (`group_size=32`, `90 B/row`: `80 B` packed 4-bit nibbles + `10 B` FP16 scales) using $\operatorname{diag}(H_m)$-weighted MSE scale search + exact downstream `W_kv`-HGPTQ error feedback ($H = W_k^\top W_k + W_v^\top W_v$) + 2-pass Gauss–Seidel H-CD + closed-form FP16 group scale refinement + population $W_{kv}$ MMSE bias/cross-covariance compensation folded into `layers.1.ple.{key_proj,value_proj}`.
  - Each shard stores linear tables (`weight_packed`, `weight_scale`, `weight_lut`, `weight_shape`), and `45` rows (`90 B/row`) fit into a single `4096 B` OS page (`98.9%` density) for single-page-fault `mmap` / `io_uring` lookup.
- **Native MTP (`1,543` linears, `1.340 GiB` in `model-mtp.safetensors`):**
  - `4` rotated `self_attn` projections (`head_dim=64`) + `3` rotated `mlp.shared_expert` projections (`head_dim=64`) + `1,536` rotated `group_size=64` routed MoE expert projections (`512` experts, `head_dim=64`).
- **Vision Tower (`110` rotated linears, `0.225 GiB` in `model-vision.safetensors`):**
  - `27` transformer blocks (`qkv`, `proj`, `linear_fc1`, `linear_fc2`) + `merger` (`linear_fc1`, `linear_fc2`) quantized with online block-Hadamard rotation + Cholesky GPTQ (`group_size=64`; `visual.blocks.*.mlp.linear_fc2` uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard `v_vision_fc2`).
- **End-to-End FP32-Master Norm, GDN, Hyper-Connection & PLE QAT:**
  - Post-quantization teacher-forced KL + hidden-MSE QAT over all 4-stream `hyper_connection` norms and injectors, QK norms, GDN norms/gates, and Layer-1 `ple.{norm_key,norm_query,norm_conv,conv1d}`.
- **Kept in BF16/FP32**:
  - `embed_tokens`, `hyper_connection_mixer`, `*.attn_hyper_connection.*`, `*.mlp_hyper_connection.*`, `*.linear_attn.{in_proj_a,in_proj_b,conv1d,A_log,dt_bias,norm}`, `*.self_attn.indexer.*`, `layers.1.ple.{key_proj,value_proj,norm_key,norm_query,norm_conv,conv1d}`, `*.mlp.{gate,shared_expert_gate}`, `mtp.{fc,pre_fc_norm_embedding,pre_fc_norm_hidden,norm,hyper_connection_mixer}`, `visual.patch_embed.proj`, norms, and biases.

## Shard Layout & Memory Tiering (`92.844 GiB` total)

| Component | Shards | Size | Format | Recommended Placement (`1x / 2x RTX 3090` or `1x 80–96 GB GPU`) |
|---|---:|---:|---|---|
| Embed + LM Head (`W8_g64`) + Mixer | `model-embed-head.safetensors` | `1.807 GiB` | BF16 + `W8_g64` | GPU VRAM resident |
| 48 Target Layers (Attn + Shared MLP + HC + Norms + PLE Proj) | `model-layer-00..47.safetensors` | `~3.85 GiB` | W4A16-HGPTQ (`g64-h64`) + BF16 | GPU VRAM resident |
| 48 Target Layers Routed Experts (`48 × 512` experts) | `model-layer-00..47.safetensors` | `~58.80 GiB` | W4A16-HGPTQ (`g64-h64`) | Hot cache in GPU VRAM + Cold experts in pinned DDR (`HostExpertStore` / UVA) |
| Layer-1 PLE N-Gram Table (`128` shards, `51.2B` params) | `model-ple-000..007.safetensors` | `26.822 GiB` | 4-bit `Custom-LUT4` (`g32`, `90 B/row`) | Linux page cache (`mmap`) / CPU offload |
| Native MTP Layer (`512` experts + Shared MLP + QSA Attn) | `model-mtp.safetensors` | `1.340 GiB` | W4A16-HGPTQ (`g64-h64`) | Hot cache in GPU VRAM + Cold experts in pinned DDR |
| Vision Tower (`27` blocks + merger) | `model-vision.safetensors` | `0.225 GiB` | W4A16-HGPTQ (`g64-h64` / `channel-h16`) | GPU VRAM resident (when multimodal enabled) |

## Static Expert Hot-Cache Rankings (`runtime/static_hot_cache_rankings.json`)

Both `runtime/static_hot_cache_rankings.json` and root `static_hot_cache_rankings.json` map layer indices `"0"` through `"47"` to global expert IDs (`0..511`) ordered from **most-frequently routed (hottest) to least-frequently routed**. Full per-layer activation counts, routing weight mass, and `mtp.0` rankings are provided in `runtime/expert_routing_profile.json` (`9,804,460` expert assignments per target layer):

| Layer | Active Experts | Top-5 Hottest Expert IDs | `top_32` Hit Rate | `top_64` Hit Rate | `top_88` Hit Rate | `top_128` Hit Rate | `top_256` Hit Rate |
|---|---:|---|---:|---:|---:|---:|---:|
| Layer `0` | `508 / 512` | `[306, 470, 279, 280, 266]` | `13.95%` | `23.96%` | `30.88%` | `41.38%` | `69.74%` |
| Layer `1` | `511 / 512` | `[130, 191, 308, 320, 58]` | `14.67%` | `25.36%` | `32.46%` | `43.24%` | `71.79%` |
| Layer `12` | `512 / 512` | `[34, 272, 139, 166, 63]` | `15.72%` | `26.94%` | `34.39%` | `45.31%` | `73.49%` |
| Layer `23` | `512 / 512` | `[499, 361, 242, 183, 84]` | `16.86%` | `28.21%` | `35.18%` | `45.30%` | `70.95%` |
| Layer `35` | `512 / 512` | `[189, 18, 479, 372, 199]` | `21.90%` | `34.78%` | `42.38%` | `52.99%` | `77.95%` |
| Layer `47` | `511 / 512` | `[242, 398, 157, 110, 55]` | `31.75%` | `45.89%` | `54.26%` | `65.54%` | `88.15%` |
| `mtp.0` | `512 / 512` | `[439, 433, 257, 410, 451]` | `26.94%` | `40.02%` | `47.71%` | `58.52%` | `81.33%` |
