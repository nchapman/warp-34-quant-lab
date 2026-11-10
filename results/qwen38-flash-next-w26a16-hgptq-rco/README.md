---
base_model: Qwen/Qwen3.8-Flash-Next
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
- ple
- rco
- compressed-tensors
- vllm
---

# Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO

Mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.6A16`, group size 64) and 4-bit `Custom-LUT4` Layer-1 PLE quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) built with Riemannian Constrained Optimization (`HGPTQ-RCO`). It compresses the 335.3 GiB (360 GB) BF16 checkpoint down to **71.75 GiB** total (4.67× smaller than BF16, saving 21.09 GiB vs. pure `W4A16`) and brings the 48-layer MoE trunk + Native MTP + Vision down to **44.93 GiB** so it fits on a single **48 GB GPU** (with the 26.82 GiB 4-bit PLE table `mmap`'d in system RAM) or 100% in VRAM on a single **80 GB GPU**.

- **VRAM & GPU fit:** 44.93 GiB for the 48-layer MoE trunk + Native MTP + Vision tower (~2.6 GiB headroom on a 48 GB GPU with Layer-1 PLE `mmap`'d in RAM) and 71.75 GiB total including the 26.82 GiB 4-bit PLE table (~7.5 GiB headroom all-in on an 80 GB GPU; for 96 GB+ setups, see [`Qwen3.8-Flash-Next-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ) at 92.84 GiB).
- **Quantization recipe:** Keeps 100% of Attention/GDN (156 linears), Shared Experts (144 linears), Native MTP (1,543 linears), and Vision (110 linears) in `W4_g64`, `lm_head` in `W8_g64`, and Layer-1 PLE in 4-bit `Custom-LUT4` with $W_{kv}$ MMSE compensation, while allocating the 12 highest-KL-sensitivity routed-expert layers (6,144 experts) to `W4_g64` and the remaining 36 routed-expert layers (18,432 experts) to `W2_g64`, polished with end-to-end FP32-master Norm/GDN/HC/PLE KL QAT.
- **Fidelity vs. BF16:** 0.1396 weighted KL (0.1695 mean KL, −5.97% PPL) on `probe-v4`, 0.3150 mean KL (0.4616 weighted KL, −3.81% PPL) on the `probe-guard` holdout, 95.41% / 81.55% isolated/E2E MTP top-1 agreement, and 97.02% / 90.73% isolated/E2E vision top-1 agreement (99.26% / 92.59% on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`Qwen/Qwen3.8-Flash-Next`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model and the `92.84 GiB` pure 4-bit [`Qwen3.8-Flash-Next-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ) checkpoint:

| Evaluation | BF16 Base (`Qwen/Qwen3.8-Flash-Next`) | `W4A16-HGPTQ` (`4.00 bpw` experts) | **`W2.6A16-HGPTQ-RCO` (`2.50 bpw` experts)** | Notes |
|---|---:|---:|---:|---|
| **Total Safetensors Size** | `335.28 GiB` (`360,000,192,888 B`) | `92.84 GiB` (`99,690,665,368 B`) | **`71.75 GiB` (`77,041,279,592 B`)** | **`4.67×` compression** (`−263.53 GiB` vs. BF16, `−21.09 GiB` vs. W4A16) |
| **Trunk + MTP + Vision Size** (excl. `mmap` PLE) | `239.91 GiB` (`257,599,900,928 B`) | `66.02 GiB` (`70,890,439,312 B`) | **`44.93 GiB` (`48,241,053,536 B`)** | Fits on a **48 GB GPU** (`26.82 GiB` 4-bit PLE `mmap`'d in host RAM) |
| **Text Perplexity** (378 prompts, 852k tokens) ↓ | `3.6085` | `3.5683` (`−1.11%`) | **`3.3931` (`−5.97%`)** | Math (`0.0679` KL), chat (`0.0780` KL), IF (`0.0803` KL), science (`0.0829` KL) |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | `0.0421` / `0.0798` | **`0.1396` / `0.1695`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 23k tokens) ↓ | `24.2483` | `22.8992` (`−5.56%`) | **`23.3233` (`−3.81%`)** | Token mean KL `0.3150` (weighted KL `0.4616`, `no_robots` KL `0.2076`) |
| **Native MTP (Speculative Drafter)** (12,339 positions) | `100.0%` Top-1 (`NLL 2.6008`) | `95.41%` Top-1 (`KL 0.0184`) | **`95.41%` Top-1 (`NLL 2.6186`)** | Isolated MTP KL `0.0184` (end-to-end Top-1 `81.55%`, KL `0.3472`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `0.9906` cos sim · `97.35%` Top-1 | **`0.9906` cos sim · `97.02%` Top-1** | Isolated ViT KL `0.0171` (end-to-end Top-1 `90.73%`, KL `0.2199`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `0.9930` cos sim · `99.26%` Top-1 | **`0.9930` cos sim · `99.26%` Top-1** | Isolated ViT KL `0.0088` (end-to-end Top-1 `92.59%`, KL `0.1265`) |

## What Is Quantized (`HGPTQ-RCO` 48 GB Trunk / 80 GB Total Allocation)

- **Strictly Standard Bit-Widths (`{2, 4, 8}` bits, `group_size=64`)**:
  - **Always-Active Backbone (`100%` in `W4_g64`, `300` linears)**: All `156` hybrid GatedDeltaNet + QSA-Indexed-Attention projections (`linear_attn`, `self_attn`) and all `144` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) across all 48 layers stay in `W4_g64` (`head_dim=64`, MSE scale search + 2-pass H-CD).
  - **Routed MoE Experts (`2.50 bpw` average across `73,728` linears, solved via Leave-One-Out End-to-End KL Sensitivity)**:
    - **`W4_g64` (`12` layers = `6,144` experts = `18,432` linears, `1.298 GiB/layer`)**: Layers `[3, 7, 15, 21, 22, 25, 29, 31, 34, 36, 46, 47]` — preserving the full-attention QSA anchor layers (`3, 7, 15, 31, 47`) plus the 7 highest-sensitivity GatedDeltaNet MoE layers.
    - **`W2_g64` (`36` layers = `18,432` experts = `55,296` linears, `0.712 GiB/layer`)**: Layers `[0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 23, 24, 26, 27, 28, 30, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45]` quantized with symmetric 2-bit Hadamard Cholesky GPTQ (`q in {-2, -1, 0, 1}`) + 2-pass Hessian Coordinate Descent.
  - **Untied Output Head (`W8_g64`, `1` linear)**: `lm_head` (`248320 × 2560`) quantized with unrotated 8-bit Cholesky GPTQ + MSE scale search (`group_size=64`).
  - **Layer-1 N-Gram Per-Layer Embedding (`51.2B` parameters → `26.822 GiB` across `model-ple-000..007.safetensors`)**: Kept in 4-bit `Custom-LUT4` (`group_size=32`, `90 B/row`, `45 rows/4 KiB page`) with $\operatorname{diag}(H_m)$-weighted MSE scale search + `W_kv`-HGPTQ + 2-pass H-CD + closed-form FP16 group scale refinement + population $W_{kv}$ MMSE compensation on `layers.1.ple.{key_proj,value_proj}`.
  - **Native MTP (`1,543` linears in `W4_g64`, `1.340 GiB`)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `1,536` routed MoE expert projections (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears in `W4_g64`, `0.225 GiB`)**: All 27 ViT blocks + 2 spatial merger projections (`visual.blocks.*.mlp.linear_fc2` uses per-channel GPTQ + `head_dim=16` Hadamard rotation `v_vision_fc2`; all other Vision linears use `group_size=64` + `head_dim=64`).
  - **End-to-End FP32-Master Norm, GDN, Hyper-Connection & PLE KL QAT**: Post-allocation teacher-forced KL + hidden-MSE QAT over all 4-stream `hyper_connection` norms and injectors, QK norms, GatedDeltaNet norms/gates, `mlp.shared_expert_gate`, and Layer-1 `ple.{norm_key,norm_query,norm_conv,conv1d}` — cutting held-out validation KL by `32.1%` (`0.1327 → 0.0901`).

## Shard Layout & Memory Tiering (`71.750 GiB` total / `44.928 GiB` Trunk+MTP+Vision)

| Component | Shards | Size | Format | Recommended Placement (`48 GB GPU` vs `80 GB GPU`) |
|---|---:|---:|---|---|
| Embed + LM Head (`W8_g64`) + Mixer | `model-embed-head.safetensors` | `1.807 GiB` | BF16 + `W8_g64` | GPU VRAM resident |
| 48 Target Layers (Attn + Shared MLP + HC + Norms + PLE Proj) | `model-layer-00..47.safetensors` | `~3.85 GiB` | W4A16-HGPTQ (`g64-h64`) + BF16 | GPU VRAM resident |
| 48 Target Layers Routed Experts (`12` `W4` + `36` `W2` layers) | `model-layer-00..47.safetensors` | `~37.71 GiB` | `W4_g64` (`12` layers) + `W2_g64` (`36` layers) | GPU VRAM resident on `48 GB` / `80 GB` GPUs |
| Layer-1 PLE N-Gram Table (`128` shards, `51.2B` params) | `model-ple-000..007.safetensors` | `26.822 GiB` | 4-bit `Custom-LUT4` (`g32`, `90 B/row`) | `mmap` in host RAM (`48 GB GPU`) or HBM resident (`80 GB GPU`) |
| Native MTP Layer (`512` experts + Shared MLP + QSA Attn) | `model-mtp.safetensors` | `1.340 GiB` | W4A16-HGPTQ (`g64-h64`) | GPU VRAM resident |
| Vision Tower (`27` blocks + merger) | `model-vision.safetensors` | `0.225 GiB` | W4A16-HGPTQ (`g64-h64` / `channel-h16`) | GPU VRAM resident |

## Static Expert Hot-Cache Rankings (`runtime/static_hot_cache_rankings.json`)

Both `runtime/static_hot_cache_rankings.json` and root `static_hot_cache_rankings.json` map layer indices `"0"` through `"47"` to global expert IDs (`0..511`) ordered from **most-frequently routed (hottest) to least-frequently routed**. Full per-layer activation counts, routing weight mass, and `mtp.0` rankings are provided in `runtime/expert_routing_profile.json` (`9,804,460` expert assignments per target layer):

| Layer | Precision | Active Experts | Top-5 Hottest Expert IDs | `top_32` Hit Rate | `top_64` Hit Rate | `top_88` Hit Rate | `top_128` Hit Rate | `top_256` Hit Rate |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Layer `0` | `W2_g64` | `508 / 512` | `[470, 306, 280, 279, 266]` | `13.82%` | `23.88%` | `30.82%` | `41.33%` | `69.83%` |
| Layer `1` | `W2_g64` | `511 / 512` | `[130, 308, 191, 320, 58]` | `14.61%` | `25.19%` | `32.27%` | `43.07%` | `71.83%` |
| Layer `12` | `W2_g64` | `512 / 512` | `[34, 139, 63, 272, 166]` | `15.65%` | `26.82%` | `34.27%` | `45.18%` | `73.27%` |
| Layer `23` | `W2_g64` | `512 / 512` | `[499, 361, 242, 183, 84]` | `16.65%` | `27.92%` | `34.87%` | `45.02%` | `70.63%` |
| Layer `35` | `W2_g64` | `512 / 512` | `[189, 479, 18, 372, 199]` | `22.03%` | `34.88%` | `42.43%` | `53.01%` | `77.90%` |
| Layer `47` | `W4_g64` | `510 / 512` | `[242, 398, 157, 110, 127]` | `31.81%` | `46.18%` | `54.51%` | `66.00%` | `88.52%` |
| `mtp.0` | `W4_g64` | `512 / 512` | `[439, 433, 410, 257, 451]` | `26.68%` | `39.88%` | `47.57%` | `58.41%` | `81.31%` |
