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

Full-model **W4A16 Cholesky GPTQ (`group_size=64`, `actorder="static"`) + online block-Hadamard input rotation (`head_dim=64`, `QuIPModifier` / $H_{64}$ + `HadaCore`) + 4-bit `Custom-LUT4` (`W_kv`-HGPTQ) Layer-1 PLE** quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters / `360 GB` BF16 $\to$ **`93.418 GiB`** across `59` safetensors shards):

- **Text Trunk (`48` hybrid GDN + QSA-Indexed-Attention + 512-expert MoE layers, `62.651 GiB`):**
  - `156` attention projections (`linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj`, `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`) quantized with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - `144` shared MoE expert projections (`48` layers $\times$ `3` projections: `mlp.shared_expert.{gate_proj,up_proj,down_proj}`) quantized with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, `actorder="static"`).
  - `73,728` routed MoE expert projections (`48` layers $\times$ `512` experts $\times$ `3` projections: `mlp.experts.0..511.{gate_proj,up_proj,down_proj}`) quantized with **online block-Hadamard input rotation (`head_dim=64` on both $K_{\text{in}}=2560$ and $K_{\text{down}}=640$) + Cholesky GPTQ (`group_size=64`, `actorder="static"`)**.
- **Layer-1 N-Gram Per-Layer Embedding (`51.2B` parameters / `102.4 GB` BF16 $\to$ `26.822 GiB` across `8` shards `model-ple-000..007.safetensors`):**
  - All `128` n-gram embedding shards (`320,000,456` rows $\times$ `160` dim) quantized to **4-bit non-uniform `Custom-LUT4` (`group_size=32`, `90 B/row`: `80 B` packed 4-bit nibbles + `10 B` FP16 scales)** using **exact downstream `W_kv`-HGPTQ error feedback (`H = W_k^T W_k + W_v^T W_v`) + 1-pass Coordinate Descent** so quantization error is orthogonal to the downstream `layers.1.ple.{key_proj,value_proj}` cross-attention subspaces.
  - Each shard stores linear tables (`weight_packed`, `weight_scale`, `weight_lut`, `weight_shape`), and `pack_ple_page_slotted_4k` packs `45` rows/page (`80 B` nibbles + `10 B` scales co-located inside a single `4096 B` OS page) for single-page-fault `mmap` / `io_uring` lookup.
- **Native MTP (`1,543` linears, `1.340 GiB` in `model-mtp.safetensors`):**
  - `4` rotated `self_attn` projections (`head_dim=64`) + `3` rotated `mlp.shared_expert` projections (`head_dim=64`) + `1,536` rotated `group_size=64` routed MoE expert projections (`512` experts, `head_dim=64`), calibrated on captured Layer-47 pre-mixer hidden states and shifted token embeddings.
- **Vision Tower (`110` rotated linears, `0.225 GiB` in `model-vision.safetensors`):**
  - `27` transformer blocks (`qkv`, `proj`, `linear_fc1`, `linear_fc2`) + `merger` (`linear_fc1`, `linear_fc2`) quantized with online block-Hadamard rotation + Cholesky GPTQ (`group_size=64`; `visual.blocks.*.mlp.linear_fc2` uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard `v_vision_fc2`).
- **Static Expert Hot-Cache Rankings (`runtime/static_hot_cache_rankings.json`, `static_hot_cache_rankings.json`, and `runtime/expert_routing_profile.json`):**
  - Pre-computed global expert rankings (`0..511` ordered hottest-first per layer `"0".."47"`, plus `"mtp.0"` in `expert_routing_profile.json`) captured across text, code, math, science, agentic tool use, multilingual, and multimodal VQA evaluation suites. Drop-in compatible with `VLLM_WNA16_STATIC_HOT_CACHE_FILE` and `warp-3090` `HostExpertStore`.

### Preserved BF16 Precision Tiers
`embed_tokens`, `lm_head`, `hyper_connection_mixer`, `*.attn_hyper_connection.*`, `*.mlp_hyper_connection.*`, `*.linear_attn.{in_proj_a,in_proj_b,conv1d,A_log,dt_bias,norm}`, `*.self_attn.indexer.*`, `layers.1.ple.{key_proj,value_proj,o_proj,q_norm,k_norm,gate_proj,post_ple_layernorm}`, `*.mlp.{gate,shared_expert_gate}`, `mtp.{fc,pre_fc_norm_embedding,pre_fc_norm_hidden,norm,hyper_connection_mixer}`, `visual.patch_embed.proj`, norms, and biases remain in BF16/FP32.

- **Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`7ecef55a2e549814c62143fcf8b5be457695e3b9`)
- **Text & MTP calibration:** `512` samples $\times$ `2048` tokens (`1,048,576` tokens) from [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) (`data/calib-cascade-qwenfinals.jsonl`) pinned in [warp-3090-quant-lab-frozen-data](https://huggingface.co/datasets/nchapman/warp-3090-quant-lab-frozen-data)
- **Vision calibration:** `120` multimodal calibration images (`data/vision-calib-v2`, strictly disjoint from `vision-probe-v2` and `vision-guard-v2`)
- **Chat template:** [froggeric Qwen-Fixed-Chat-Templates (v22.5)](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) ships as `chat_template.jinja` (`sha256: e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`)
- **Recipe and tooling:** [warp-3090-model-lab](https://github.com/nchapman/warp-3090-model-lab)

---

## Measured Fidelity (`probe_results.json` — v2 vs. v1)

| Suite / Metric | v1 (`experts` unrotated) | **v2 (this repo: $H_{64}$ on `experts` + `shared_expert`)** | Notes |
|---|---:|---:|---|
| **Total Safetensors Payload** | `93.418 GiB` (`100,306,498,784 B`) | **`93.418 GiB` (`100,306,498,784 B`)** | **`3.83×` compression** from `360 GB` BF16 (`59` shards) |
| `probe-v4` Perplexity (BF16 `3.6085` $\to$ Quant) ↓ | `3.7291` (`+3.342%`) | **`3.6264` (`+0.496%`, −6.7× PPL Δ)** | `math`: `2.3400 -> 2.3300`, `chat`: `3.7979 -> 3.8634` |
| `probe-v4` token mean KL / weightedKL ↓ | `0.092594` / `0.052489` | **`0.091579` / `0.053029`** | `chat`: `0.0185`, `if`: `0.0276`, `math`: `0.0293` (−11.3%), `science`: `0.0347` |
| `probe-guard` token mean KL / `no_robots` mean KL ↓ | `0.105677` / `0.070632` | **`0.105409` / `0.069307`** | `76` rows, `22,746` tokens (`no_robots` PPL `22.3109 -> 22.9307`) |
| **Native MTP** isolated mean KL / median KL / Top-1 ↑ | `0.020956` / `0.003483` / `94.08%` | **`0.019397` (−7.4%) / `0.003276` / `94.15%`** | `32` rows, `12,339` positions (`NLL 2.6008 -> 2.6120`) |
| **Native MTP** end-to-end mean KL / median KL / Top-1 ↑ | `0.172665` / `0.022108` / `86.78%` | **`0.167017` (−3.3%) / `0.020725` / `87.28%`** | Quantized Trunk + Quantized PLE + Quantized MTP |
| **Vision (`vision-probe-v2`)** encoder Rel-RMSE / Cos-Sim | `0.1783` / `0.9870` | **`0.1783` / `0.9870`** | `96` images across 12 VQA & OCR domains |
| **Vision (`vision-probe-v2`)** isolated KL / Top-1 ↑ | `0.019725` / `96.03%` | **`0.017803` (−9.7%) / `96.36%`** | `302` answer tokens (`NLL 0.3713 -> 0.3572`) |
| **Vision (`vision-probe-v2`)** end-to-end KL / Top-1 ↑ | `0.088678` / `94.70%` | **`0.085043` (−4.1%) / `95.03%`** | `302` answer tokens (`NLL 0.3713 -> 0.4340`) |
| **Vision Guard (`vision-guard-v2`)** encoder Rel-RMSE / Cos-Sim | `0.1352` / `0.9903` | **`0.1352` / `0.9903`** | `48` images (ScreenSpot + OCRBench artistic/math) |
| **Vision Guard (`vision-guard-v2`)** isolated KL / Top-1 ↑ | `0.019937` / `97.78%` | **`0.014864` (−25.4%) / `99.26%`** | `135` answer tokens (`134/135` Top-1 agreement) |
| **Vision Guard (`vision-guard-v2`)** end-to-end KL / Top-1 ↑ | `0.056267` / `97.04%` | **`0.049600` (−11.8%) / `97.78%`** | `135` answer tokens (`132/135` Top-1 agreement) |

---

## Shard Layout & Memory Tiering (`93.418 GiB` total)

| Component | Shards | Size | Format | Recommended Placement (`1x / 2x RTX 3090`) |
|---|---:|---:|---|---|
| Embed + LM Head + Mixer | `model-embed-head.safetensors` | `2.380 GiB` | BF16 | GPU VRAM resident |
| 48 Target Layers (Attn + Shared MLP + HC + Norms + PLE Proj) | `model-layer-00..47.safetensors` | `~3.85 GiB` | W4A16-HGPTQ (`g64-h64`) + BF16 | GPU VRAM resident |
| 48 Target Layers Routed Experts (`48 × 512` experts) | `model-layer-00..47.safetensors` | `~58.80 GiB` | W4A16-HGPTQ (`g64-h64`) | Hot cache in GPU VRAM + Cold experts in pinned DDR (`HostExpertStore` / UVA) |
| Layer-1 PLE N-Gram Table (`128` shards, `51.2B` params) | `model-ple-000..007.safetensors` | `26.822 GiB` | 4-bit `Custom-LUT4` (`g32`, `90 B/row`) | Linux page cache (`mmap`) / CPU offload |
| Native MTP Layer (`512` experts + Shared MLP + QSA Attn) | `model-mtp.safetensors` | `1.340 GiB` | W4A16-HGPTQ (`g64-h64`) | Hot cache in GPU VRAM + Cold experts in pinned DDR |
| Vision Tower (`27` blocks + merger) | `model-vision.safetensors` | `0.225 GiB` | W4A16-HGPTQ (`g64-h64` / `channel-h16`) | GPU VRAM resident (when multimodal enabled) |

---

## Static Expert Hot-Cache Rankings (`runtime/static_hot_cache_rankings.json`)

Both `runtime/static_hot_cache_rankings.json` and root `static_hot_cache_rankings.json` map layer indices `"0"` through `"47"` to global expert IDs (`0..511`) ordered from **most-frequently routed (hottest) to least-frequently routed**. Full per-layer activation counts, routing weight mass, and `mtp.0` rankings are provided in `runtime/expert_routing_profile.json` (`9,804,460` expert assignments per target layer):

| Layer | Active Experts | Top-5 Hottest Expert IDs | `top_32` Hit Rate | `top_64` Hit Rate | `top_88` Hit Rate | `top_128` Hit Rate | `top_256` Hit Rate |
|---|---:|---|---:|---:|---:|---:|---:|
| Layer `0` | `508 / 512` | `[306, 470, 279, 280, 266]` | `13.93%` | `23.96%` | `30.89%` | `41.40%` | `69.72%` |
| Layer `1` | `511 / 512` | `[130, 308, 320, 58, 383]` | `14.71%` | `25.45%` | `32.60%` | `43.50%` | `72.18%` |
| Layer `12` | `512 / 512` | `[34, 272, 139, 63, 166]` | `15.76%` | `27.02%` | `34.47%` | `45.42%` | `73.56%` |
| Layer `23` | `512 / 512` | `[499, 361, 242, 183, 84]` | `16.81%` | `28.20%` | `35.16%` | `45.27%` | `70.91%` |
| Layer `35` | `512 / 512` | `[189, 18, 479, 372, 199]` | `21.91%` | `34.77%` | `42.32%` | `52.90%` | `77.92%` |
| Layer `47` | `511 / 512` | `[242, 398, 157, 110, 55]` | `31.61%` | `45.71%` | `54.12%` | `65.45%` | `88.01%` |
| `mtp.0` | `512 / 512` | `[439, 433, 257, 410, 451]` | `26.86%` | `39.96%` | `47.72%` | `58.46%` | `81.34%` |
