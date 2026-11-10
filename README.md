# warp-34-quant-lab

**Full-model 4-bit (`W4A16-HGPTQ`) quantizations of the Qwen 3.5 / 3.6 / 3.8 family—including the Vision encoder and the built-in Multi-Token Prediction (MTP) speculative draft head.**

Most 4-bit checkpoints only quantize the main text layers. They either strip out the **MTP speculative draft head** and **Vision tower**, leave them bloated in 16-bit, or crush them with uncalibrated round-to-nearest (RTN) rounding.

This project takes a different approach: **every part of the model is calibrated and quantized to 4-bit (`~2.5×–3.6×` smaller overall)** so the entire multimodal + speculative model fits in a fraction of the memory while behaving almost identically to the original 16-bit weights.

### Why These Quants Are Cool

- **Vision and MTP are first-class citizens:**
  - **Calibrated 4-bit Vision:** Instead of leaving the vision encoder in 16-bit or blindly rounding it, we calibrate the vision tower on real charts, documents, UI screenshots, and OCR images—retaining **96%–99.7% top-1 token agreement** with the 16-bit vision encoder at ~`4×` smaller size.
  - **Calibrated 4-bit Native MTP (built-in speculative decoding):** Every Qwen 3.5 / 3.6 / 3.8 checkpoint includes a built-in Multi-Token Prediction head that drafts upcoming tokens for **1.3×–1.7× faster decoding**. We calibrate and quantize the MTP layers against the model's own hidden states so speculative decoding works out of the box with **90%–96% draft agreement**.
  - **Companion `DFlash2` block-diffusion drafters:** For `27B` and `Flash-Next`, we also ship 4-bit `DFlash2` drafters that propose **16 tokens in a single parallel pass** (`3.76×` less memory bandwidth per draft step) while keeping **99.7%–99.8%** of 16-bit acceptance length.
- **Rotation before quantization (`Hadamard + GPTQ`):**
  - Standard 4-bit quantizers struggle when a few weight or activation channels have huge outlier spikes. Before quantizing to 4-bit, we apply a fast mathematical rotation (a block Walsh–Hadamard transform) that spreads outliers evenly across 64-weight groups, then run Cholesky GPTQ to compensate for rounding error. That cuts overall divergence from 16-bit by **~30%** (and worst-case tail spikes by **up to 64%**) compared to standard 4-bit GPTQ.
- **Giant MoE & N-Gram models that actually fit on real hardware:**
  - On `Qwen3.6-35B-A3B` (`66 GB → 19.2 GiB`) and the 180B-class `Qwen3.8-Flash-Next` (`360 GB → 93.4 GiB`), we rotate and quantize all routed *and* shared MoE experts, compress `Flash-Next`’s massive `102.4 GB` N-gram lookup table down to `26.8 GiB` using a custom 4-bit lookup table tailored to its downstream projections, and bundle **pre-computed hot-expert rankings** (`static_hot_cache_rankings.json`) so inference engines know which experts to pin in GPU VRAM from the very first token.

---

## Published Models & Measured Fidelity

All 12 checkpoints are published on Hugging Face under [`nchapman`](https://huggingface.co/nchapman), with full evaluation receipts and model cards in [`results/`](results/):

| Model | Artifact (`nchapman/*`) | Size | `probe-v4` PPL (BF16 → Quant) | `probe-v4` `weighted_kl` | `probe-guard` `weighted_kl` (`mean_kl`) | MTP Iso `mean_kl` (Top-1) | Vision Iso `mean_kl` (Top-1) | Vision Guard Iso Top-1 |
|---|---|---:|---|---:|---|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | [`Qwen3.5-0.8B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-0.8B-W4A16-HGPTQ) ([card](results/dense-w4a16-hgptq/0.8b-g64-h64/README.md)) | `0.79 GiB` | `4.1695 → 4.3015` (`+3.166%`) | `0.073509` | `0.088739` (`0.070613`) | `0.028836` (`90.24%`) | `0.026580` (`94.04%`) | `97.04%` |
| `Qwen/Qwen3.5-2B` | [`Qwen3.5-2B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-2B-W4A16-HGPTQ) ([card](results/dense-w4a16-hgptq/g64-h64/README.md)) | `1.83 GiB` | `3.8302 → 3.7784` (`-1.352%`) | `0.059069` | `0.062987` (`0.050014`) | `0.020116` (`92.78%`) | `0.012995` (`97.35%`) | `99.26%` |
| `Qwen/Qwen3.5-4B` | [`Qwen3.5-4B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-4B-W4A16-HGPTQ) ([card](results/dense-w4a16-hgptq/4b-g64-h64/README.md)) | `3.19 GiB` | `4.0299 → 3.9537` (`-1.891%`) | `0.081090` | `0.074129` (`0.053346`) | `0.014983` (`94.26%`) | `0.004972` (`99.34%`) | `99.26%` |
| `Qwen/Qwen3.5-9B` | [`Qwen3.5-9B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-9B-W4A16-HGPTQ) ([card](results/dense-w4a16-hgptq/9b-g64-h64/README.md)) | `6.67 GiB` | `3.5754 → 3.4999` (`-2.112%`) | `0.063824` | `0.058614` (`0.039538`) | `0.010453` (`94.86%`) | `0.007192` (`99.34%`) | `98.52%` |
| `ornith-ai/Ornith-1.5-9B` | [`Ornith-1.5-9B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Ornith-1.5-9B-W4A16-HGPTQ) ([card](results/dense-w4a16-hgptq/9b-ornith-g64-h64/README.md)) | `6.67 GiB` | `6.6829 → 5.6579` (`-15.338%`) | `0.144025` | `0.133591` (`0.096463`) | `0.012652` (`94.68%`) | `0.020540` (`97.02%`) | `97.04%` |
| `Qwen/Qwen3.8-27B` | [`Qwen3.8-27B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ) ([card](results/dense-w4a16-hgptq/27b-g64-h64/README.md)) | `16.14 GiB` | `8.5332 → 8.1841` (`-4.091%`) | `0.100223` | `0.073907` (`0.052982`) | `0.008979` (`96.64%`) | `0.005603` (`97.68%`) | `99.26%` |
| `ukisai/Swift-1.5-Qwen3.8-27b` | [`Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ`](https://huggingface.co/nchapman/Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ) ([card](results/dense-w4a16-hgptq/27b-swift-g64-h64/README.md)) | `16.14 GiB` | `9.6265 → 9.0881` (`-5.593%`) | `0.107444` | `0.071995` (`0.055502`) | `0.009043` (`96.43%`) | `0.007063` (`98.01%`) | `99.26%` |
| `Qwen/Qwen3.6-35B-A3B` | [`Qwen3.6-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ) ([card](results/qwen36-35b-a3b-w4a16-hgptq/README.md)) | `18.78 GiB` | `4.0208 → 3.7522` (`-6.682%`) | `0.084694` | `0.050908` (`0.046230`) | `0.010490` (`95.80%`) | `0.007135` (`99.34%`) | `98.52%` |
| `ornith-ai/Ornith-1.5-35B-A3B` | [`Ornith-1.5-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ) ([card](results/ornith-15-35b-a3b-w4a16-hgptq/README.md)) | `18.78 GiB` | `10.8033 → 10.2627` (`-5.004%`) | `0.113922` | `0.228886` (`0.164721`) | `0.021504` (`91.44%`) | `0.030482` (`96.36%`) | `97.04%` |
| `Qwen/Qwen3.8-Flash-Next` | [`Qwen3.8-Flash-Next-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ) ([card](results/qwen38-flash-next-w4a16-hgptq/README.md)) | `93.42 GiB` | `3.6085 → 3.5963` (`-0.338%`) | `0.050367` | `0.130763` (`0.100696`) | `0.021747` (`95.01%`) | `0.017132` (`96.69%`) | `99.26%` |

### Sub-3-Bit Mixed-Precision Checkpoints (`HGPTQ-RCO` + `W8_g64` Head)

| Base Model | Artifact (`nchapman/*`) | Avg Bits | Full VLM+MTP (`Text-Only`) | Target GPU | `probe-v4` `weighted_kl` (`mean_kl`) | `probe-guard` `mean_kl` | MTP Iso / E2E Top-1 | Vision Iso / E2E Top-1 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `Qwen/Qwen3.5-9B` | [`Qwen3.5-9B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO) ([card](results/rco-9b-w2w4/README-W2.6A16-HGPTQ-RCO.md), [sweep](results/rco-9b-w2w4/README.md)) | `2.60` (`30%` `W4`) | `5.54 GiB` (`4.95 GiB`) | `8 GB` (full VLM + MTP) | `0.1753` (`0.1322`) | `0.1160` | `94.86%` / `84.63%` | `99.34%` / `93.38%` |
| `Qwen/Qwen3.8-27B` | [`Qwen3.8-27B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO) ([card](results/rco-27b-w2w4/README-W2.6A16-HGPTQ-RCO.md), [sweep](results/rco-27b-w2w4/README.md)) | `2.60` (`30%` `W4`) | `12.18 GiB` (`11.71 GiB`) | `16 GB` (full VLM + MTP) | `0.2717` (`0.4265`) | `0.3272` | `96.64%` / `74.47%` | `97.68%` / `88.74%` |
| `Qwen/Qwen3.8-27B` | [`Qwen3.8-27B-W2.2A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.2A16-HGPTQ-RCO) ([card](results/rco-27b-w2w4/README-W2.2A16-HGPTQ-RCO.md), [sweep](results/rco-27b-w2w4/README.md)) | `2.20` (`10%` `W4`) | `11.04 GiB` (`10.58 GiB`) | Headless `12 GB` / `16 GB` long-ctx | `0.3137` (`0.4597`) | `0.3835` | `96.64%` / `72.10%` | `97.68%` / `86.09%` |
| `Qwen/Qwen3.6-35B-A3B` | [`Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO) ([card](results/rco-35b-w2w4/README-W2.6A16-HGPTQ-RCO.md), [sweep](results/rco-35b-w2w4/README.md)) | `2.62` (`25%` `W4` exp) | `13.15 GiB` (`12.24 GiB`) | `16 GB` (full MoE VLM + MTP) | `0.2121` (`0.3885`) | `0.1943` | `95.80%` / `83.47%` | `99.34%` / `91.72%` |

### Companion DFlash2 Block-Diffusion Speculative Drafters (`W4A16-HGPTQ`)

| Base Drafter | Artifact (`nchapman/*`) | Size (BF16 → Quant) | Streamed Decoder Weights / Draft Step | Acceptance Length $\tau$ (BF16 → Quant) | `Recall@16` (BF16 → Quant) | Unary KL vs BF16 | Unary / Selector Top-1 Agreement |
|---|---|---|---|---|---|---|---|
| `incoai/Qwen3.8-27B-DFlash2` | [`Qwen3.8-27B-DFlash2-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ) ([card](results/dense-w4a16-hgptq/dflash2-27b-README.md)) | `3.58 GiB → 1.31 GiB` (`2.74×`) | `3,175 MiB → 843 MiB` (`3.76×`) | `2.540 → 2.535` (`99.8%`) | `53.88% → 53.88%` (`100.0%`) | `0.023087` | `92.63% / 92.03%` |
| `incoai/Qwen3.8-Flash-Next-DFlash2` | [`Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ) ([card](results/dense-w4a16-hgptq/dflash2-flash-next-README.md)) | `1.12 GiB → 0.52 GiB` (`2.17×`) | `843.8 MiB → 224.1 MiB` (`3.76×`) | `2.760 → 2.752` (`99.7%`) | `59.67% → 59.60%` (`99.9%`) | `0.027571` | `92.23% / 91.47%` |

---

## Reproducing the Quants (Quick Start)

Tooling is managed with [`mise`](.mise.toml) and [`uv`](pyproject.toml).

```bash
# 1. Create and sync .venv with uv
mise run setup

# 2. (GPU hosts) Compile fast-path GatedDeltaNet kernels (causal-conv1d + flash-linear-attention)
mise run setup-kernels

# 3. Verify frozen calibration & probe JSONL checksums
mise run verify-data

# 4. Run linter, formatter check, and CPU unit test suite
mise run check
```

### Materialize Vision Fixtures (`vision-calib-v2`, `vision-probe-v2`, `vision-guard-v2`)

```bash
mise run build-vision-data
```

Downloads pinned Hugging Face dataset Parquet shards and deterministically materializes three strictly disjoint image sets under `data/`:
- `data/vision-calib-v2` — 120 images across 12 VQA/OCR/chart/reasoning domains (`calibration=true`)
- `data/vision-probe-v2` — 96 held-out in-distribution evaluation images across 12 domains (`calibration=false`)
- `data/vision-guard-v2` — 48 held-out out-of-distribution UI/ScreenSpot/Math guard images (`guard=true`)

### Quantize & Probe

```bash
# List all registered model keys and default W4A16-HGPTQ recipes
mise run list-models

# Quantize Qwen3.6-35B-A3B (streaming layer-by-layer MoE + MTP + Vision + hot expert cache rankings)
mise run quantize -- \
  --model 35b \
  --output-dir ./out/Qwen3.6-35B-A3B-W4A16-HGPTQ

# Quantize Qwen3.8-Flash-Next (streaming 48-layer 512-expert MoE + 4-bit Custom-LUT4 PLE + MTP + Vision)
mise run quantize -- \
  --model flash-next \
  --output-dir ./out/Qwen3.8-Flash-Next-W4A16-HGPTQ

# Single-pass teacher-forced fidelity probe across probe-v4, probe-guard, MTP, and Vision v2
mise run probe -- \
  --model 35b \
  --quant ./out/Qwen3.6-35B-A3B-W4A16-HGPTQ \
  --eval-sets data/probe-v4.jsonl data/probe-guard.jsonl \
  --eval-mtp \
  --eval-vision data/vision-probe-v2 \
  --eval-vision-guard data/vision-guard-v2 \
  --output-json ./out/probe_summary.json

# Sub-3-bit & 4-bit H-RCO (H64 + Cholesky GPTQ + Gauss-Seidel H-CD + E2E Scale/Norm KL QAT + Riemannian RCO)
uv run --no-sync warp-quant rco \
  --model 9b \
  --work-dir ./out/rco-9b-w2w4 \
  --budgets 0.15 0.25 \
  --calib-seq-len 1024 \
  --steps 80 \
  --gumbel-samples 2 \
  --lr 0.25 \
  --assemble all_low 15pct_rco 25pct_rco all_high
```

---

## Under the Hood: Technical Deep Dive & Lessons Learned

### 1. Why `g64-h64` (`W4A16-HGPTQ`)
Right-multiplying each linear layer's weight matrix by a deterministic orthogonal block Walsh–Hadamard transform $W_{\text{rot}} = W H_{64}^\top / \sqrt{64}$ in FP64 prior to Cholesky GPTQ (`group_size=64`, `actorder="static"`, `dampening_frac=0.01`) spreads activation and weight channel outliers across 64-element blocks. At inference time, an online $H_{64}$ input transform ($x_{\text{rot}} = x H_{64} / \sqrt{64}$) cancels the weight rotation ($x_{\text{rot}} W_{\text{rot}}^\top = x W^\top$).
- Across the Qwen3.5 / 3.6 / 3.8 family, `g64-h64` cuts text `weighted_kl` by **26%–32%** and tail `p99.9` KL by **up to 64%** relative to unrotated `g128` GPTQ, while fitting comfortably inside the 24 GB RTX 3090 VRAM envelope for `0.8B`–`35B-A3B`.

### 2. Unanchored Prefix + Anchored Suffix Target Regexes
vLLM's `CompressedTensorsConfig` omits `apply_vllm_mapper` on `transform_config` targets, and passing `"Linear"` as a target string triggers a class-vs-module-name ambiguity in vLLM. Furthermore, Hugging Face and vLLM mount the same layers at different prefixes (`model.language_model.layers.*` vs `language_model.model.layers.*` vs `model.layers.*`, and `model.visual.*` vs `visual.*`).
- **Rule:** Never use `"Linear"` as a `config_groups` or `transform_config` target. Always use `TEXT_TARGET_RE`, `MTP_TARGET_RE`, `VISION_GROUP_TARGET_RE`, and `VISION_FC2_TARGET_RE` (`src/warp_quant/hadamard_gptq.py`), which use unanchored prefixes (`re:.*`) and anchored suffixes (`$`) with a negative lookbehind `(?<!mtp\.)layers\.\d+\.` so Text, MTP, and Vision targets never collide across HF or vLLM.
- **Strip `*.v_input.weight`:** `llmcompressor`'s `QuIPModifier` saves tied `*.v_input.weight` tensors for deterministic Hadamard matrices; both Transformers and vLLM `AutoWeightsLoader` reject those unexpected tensor keys unless stripped after `save_pretrained` (`_clean_and_augment_shards`).

### 3. Rotate and Quantize **Both** Routed Experts (`mlp.experts.\d+`) and Shared Experts (`mlp.shared_expert`)
In MoE architectures (`Qwen3.6-35B-A3B` and `Qwen3.8-Flash-Next`), leaving `mlp.shared_expert` unquantized or omitting $H_{64}$ rotation on routed/shared experts degrades both perplexity and tail KL:
- Applying $H_{64}$ rotation + W4A16 Cholesky GPTQ (`group_size=64`, `actorder="static"`) to **both** routed experts (`mlp.experts.\d+.(gate_proj|up_proj|down_proj)`) and shared experts (`mlp.shared_expert.(gate_proj|up_proj|down_proj)`) in the main model and Native MTP reduced `Qwen3.8-Flash-Next` perplexity degradation by **6.7×** (`+3.342%` → `+0.496%`) and reduced `Qwen3.6-35B-A3B` `probe-guard` `weighted_kl` by **6.1%** (`0.061686` → `0.057899`) while saving an additional `0.15–0.28 GiB`.
- Only the small routing gates (`mlp.gate`, `mlp.shared_expert_gate`), GDN per-head scalars (`linear_attn.in_proj_a`, `linear_attn.in_proj_b`, which have $N \in \{32, 48\}$ and would violate Marlin's $N \bmod 64 = 0$ packing invariant), embeddings, norms, and `lm_head` remain in BF16 (`BASE_IGNORE`).

### 4. Vision Tower (`model.visual.*`): Per-Channel `linear_fc2` + `head_dim=16` Hadamard
The Qwen3-VL vision tower has `intermediate_size = 4304` ($4304 = 16 \times 269$, not divisible by 32 or 64) on `visual.blocks.*.mlp.linear_fc2`:
- `visual.blocks.*.attn.{qkv,proj}`, `visual.blocks.*.mlp.linear_fc1`, and `visual.merger.linear_fc{1,2}` have $K \bmod 64 = 0$ and use `group_size=64` Cholesky GPTQ + `head_dim=64` Hadamard (`group_0` / `v`).
- `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses **per-channel** Cholesky GPTQ (`vision_fc2_channel`) paired with a separate `head_dim=16` online Hadamard transform group (`v_vision_fc2`).
- Calibrating Vision Hessians on `data/vision-calib-v2` (120 images across 12 domains) with length-equalized token weighting (`token_exp=0.25`, scaling activations by $(1024 / N_{\text{tok}})^{0.25}$) prevents high-resolution document images from drowning out chart/UI images and improves VisionKL by **33%–36%** over weight-only RTN.

### 5. Layer-1 PLE (`Qwen3.8-Flash-Next`): 4-Bit `Custom-LUT4` `W_kv`-HGPTQ + 4 KiB Page-Slotted Layout
`Qwen3.8-Flash-Next` includes a `51.2B`-parameter (`102.4 GB` / `95.37 GiB` BF16) Per-Layer Embedding (PLE) table at Layer 1 (`128` N-gram shards across `8` files `model-ple-000..007.safetensors`, `320,000,456` head-rows × `160` dim/head across `16` N-gram heads, `ple_embed_dim = 2560`) feeding only `layers.1.ple.{key_proj,value_proj}`. Full ablation details and rejected alternatives (`FP8`, `INT8`, `NVFP4`, `INT4`, `IQ4_NL`, low-rank SVD, orthogonal rotation, and page layouts) are documented in [`results/qwen38-flash-next-w4a16-hgptq/PLE-QUANTIZATION-DESIGN.md`](results/qwen38-flash-next-w4a16-hgptq/PLE-QUANTIZATION-DESIGN.md):
- **Why 4-bit (`26.82 GiB`) over 8-bit (`47.68 GiB` `FP8`):** At `47.7 GiB`, an 8-bit PLE table plus the `66.6 GiB` quantized backbone (`114.3 GiB` total) exceeds a `96 GB` GPU (`RTX PRO 6000` / `4× RTX 3090`) and overflows the Linux page cache when `HostExpertStore` pins `~45 GiB` of cold MoE experts on a 96–128 GB host. Compressing PLE to `26.82 GiB` brings the entire 180B checkpoint down to **`93.42 GiB`** and cuts random memory-controller sector / CPU cache-line reads per lookup by `40%` (`90 B/row` vs. `160 B/row`).
- **Why uniform INT4 and `NVFP4` fail on PLE (`PLE_CUSTOM_LUT4`):** Normalized PLE groups are super-Gaussian ($\kappa \approx 4.6$); uniform `INT4` wastes outer bins on sparse tails, while `NVFP4` (`E2M1`) has coarse steps near the core. A fixed 16-entry non-uniform lookup table (`PLE_CUSTOM_LUT4`, 16 FP16 centroids fitting in a single warp register via `__shfl_sync`) matches the empirical PLE density.
- **Rank-1 Alignment & Analytical `W_kv`-HGPTQ + 1-Pass Coordinate Descent:** Because finite calibration sets touch `<0.1%` of the 320M hash rows, activation-sampled GPTQ cannot calibrate the table. However, SVD of each head's downstream `key_proj` slice $W_{k,h} \in \mathbb{R}^{10240 \times 160}$ reveals a massive rank-1 dominant singular direction ($\sigma_1 = 5.66\text{–}6.88$ vs. $\sigma_2 \approx 2.2$, a `25×–47×` energy amplification) aligned at `0.9501` cosine similarity with $E_h$'s top singular vector. Solving Cholesky GPTQ (`group_size=32`, `actorder="static"`) + 1-pass Coordinate Descent scale refinement (`ple_cd_passes=1`) against the exact calibration-free downstream Hessian:
  $$H_h = \frac{1}{2}\left(\frac{W_{k,h}^\top W_{k,h}}{\|W_{k,h}\|_F^2 / 160} + \frac{W_{v,h}^\top W_{v,h}}{\|W_{v,h}\|_F^2 / 160}\right) + \lambda \operatorname{diag}(\dots) \in \mathbb{R}^{160 \times 160}$$
  steers quantization error into the low-gain subspace of $(W_{k,h}, W_{v,h})$ across all 320M rows (`0.98s`/shard on H200), cutting `key_proj(E)` error by **`29.6%` vs. `NVFP4 g16`** and **`22.3%` vs. `32 GB` `INT4 g16`**:

| 4-Bit PLE Scheme (`320M × 160`) | Effective `bpw` | Total PLE Size | `E` Rel-RMSE | `key_proj(E)` Rel-RMSE ↓ | `value_proj(E)` Rel-RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| NVIDIA `NVFP4 E2M1 g16` (`FP4-Local`) | `4.50` | `28.80 GB` (`26.82 GiB`) | `9.534%` | `9.449%` | `9.528%` |
| Uniform `INT4 sym g32` (RTN) | `4.50` | `28.80 GB` (`26.82 GiB`) | `9.424%` | `9.371%` | `9.423%` |
| Primitive AI `INT4 sym g16` (`ples_int4`) | `5.00` | `32.00 GB` (`29.80 GiB`) | `8.672%` | `8.567%` | `8.676%` |
| ISTA-DASLab `IQ4_NL g32` (Non-uniform LUT4, RTN) | `4.50` | `28.80 GB` (`26.82 GiB`) | `8.287%` | `7.434%` | `8.285%` |
| **`Custom-LUT4 + W_kv-HGPTQ + CD (g=32)` (Ours)** | **`4.50`** | **`28.80 GB` (`26.82 GiB`)** | **`7.990%`** | **`6.655%`** | **`7.930%`** |

- **4 KiB Page-Slotted NVMe Layout (`pack_ple_page_slotted_4k`):** Each token performs `16` independent random head lookups; with unaligned contiguous rows, `39.8%` of tokens at 8-bit (`160 B/row`) and `29.9%` at 4-bit (`90 B/row`) suffer at least one head straddling two `4,096 B` OS pages. `pack_ple_page_slotted_4k` / `unpack_ple_page_slotted_4k` packs `45` complete PLE head-rows (`80 B` nibbles + `10 B` FP16 scales = `90 B/row`) into each `4,096 B` page (`4,050 / 4,096 B` = `98.88%` page density, **`0.000%` page-straddle rate**) for single-page-fault `mmap` / `io_uring` / `O_DIRECT` lookup.

### 6. Static Hot Expert Cache Rankings (`runtime/static_hot_cache_rankings.json`)
During streaming MoE quantization and probing (`src/warp_quant/streaming_moe.py`), exact top-$k$ router hits and routing probability mass are accumulated across all $L$ MoE layers and Native MTP (`9,804,460` expert assignments/layer for `flash-next`, `6,979,616` for `35b`) and exported to:
- `runtime/static_hot_cache_rankings.json` (and root `static_hot_cache_rankings.json`) — maps target layer indices (`"0".."39"` or `"0".."47"`) to global expert IDs ordered hottest-first (`active_ranked_expert_ids`), enabling zero-warmup static hot-expert pinning in VRAM at server startup (`VLLM_WNA16_STATIC_HOT_CACHE_FILE` / `HostExpertStore`).
- `runtime/expert_routing_profile.json` — full per-layer `ranked_expert_ids`, `active_ranked_expert_ids`, `hit_counts`, `weight_sums`, and `top_k_coverage` (`top_8..top_256`), including `"mtp.0"`.

### 7. Sub-3-Bit & 4-Bit Optimization (`H-CD` + `E2E Scale/Norm KL QAT` + Riemannian `HGPTQ-RCO` + `W8_g64` Head)
`src/warp_quant/rco/` (`warp-quant rco`) extends `g64-h64` down to `2.00–2.60` bits (`W2`/`W4` mixed precision + unrotated `W8_g64` untied head/MTP `fc`) and sharpens pure `4.00`-bit `W4A16-HGPTQ` checkpoints without changing the `compressed-tensors` format (`pack_quantized` + `W2A16`/`W4A16`/`W8A16` config groups + online $H_{64}$ transforms). Full receipts live in [`results/rco-2b-w2w4/`](results/rco-2b-w2w4/README.md), [`results/rco-9b-w2w4/`](results/rco-9b-w2w4/README.md), and [`results/rco-27b-w2w4/`](results/rco-27b-w2w4/README.md):
- **Gauss–Seidel Hessian Coordinate Descent (`refine_hessian_cd` in `src/warp_quant/rco/options.py`):** Standard GPTQ freezes group scales (`amax / qmax`) before error propagation. Alternating closed-form Hessian-weighted least-squares scale updates $s_g^\star = \sum (R H)_j Q_j / \sum (Q H)_j Q_j$ with exact columnwise Gauss–Seidel integer code updates $\Delta q_j = \operatorname{round}(\dots) - q_j$ reduces layerwise $H$-weighted reconstruction error and cuts 4-bit validation KL by `10.8%` (`0.0415 → 0.0370` on `2B`).
- **End-to-End Continuous Scale + FP32-Master `RMSNorm`/GDN-Gate KL Tuning (`run_e2e_scale_qat` in `src/warp_quant/rco/scale_qat.py`):** Parameterizes 2-bit `weight_scale` tensors in FP32 as $s = s_0 \cdot \exp(\delta)$ (`train_scales_max_bits=2`, keeping 4-bit `H-CD` scales frozen to avoid BF16 ULP jitter) while maintaining `RMSNorm` and GatedDeltaNet recurrence gates (`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`) as FP32 master weights with straight-through `bfloat16` rounding. Training against the BF16 teacher's logits cuts pure 2-bit (`2.00b`) `probe-v4` `weighted_kl` by **`46.5%`** on `2B` (`1.1794 → 0.6313`, PPL `+92.0% → +27.1%`) and cuts pure 4-bit (`4.00b`) `probe-v4` `mean_kl` by **`21.0%`** vs. shipped `2B W4A16-HGPTQ` (`0.2543 → 0.2010`, `MTP E2E` KL `-28.7%`).
- **Riemannian Constrained Optimization (`HGPTQ-RCO`, `src/warp_quant/rco/search.py`):** Solves discrete per-module `[w2_g64, w4_g64]` allocations under exact byte budgets via Gumbel-Softmax + Straight-Through Estimator + exact Newton-bisection KL projection, followed by a post-search scale/norm polish:
  - **`Qwen3.5-2B` ([receipts](results/rco-2b-w2w4/README.md)):** `25%` (`2.50b`, `449 MiB` text) and `30%` (`2.60b`, `466 MiB` text, **`-33%` text bytes vs. `W4`**) reach **`+7.81%` / `+6.22%` `probe-v4` PPL** (`4.137` / `4.076` vs. `3.837` BF16) and outperform shipped `W4A16-HGPTQ` on long-context `terminal_agent` (`0.6981` vs. `0.7395`) and `swe` (`1.2384` vs. `1.9242`).
  - **`Qwen3.5-9B` ([receipts](results/rco-9b-w2w4/README.md)):** Sub-3-bit fidelity scales sharply with model size—at `25%` `HGPTQ-RCO` (`2.50b`, `2,266 MiB` text, **`-35%` text bytes vs. `W4`**), `math` KL drops in half vs. `2B` (`0.2583 → 0.1320`), held-out `probe-guard` perplexity is within **`+3.45%` of BF16** (`10.929` vs. `10.564`), and `MTP E2E` speculative draft agreement reaches **`75.09%`** (`72.17%` even at pure `2.00`-bit `All-W2`).
  - **`Qwen3.8-27B` ([receipts](results/rco-27b-w2w4/README.md)):** At `30%` `HGPTQ-RCO` (`2.60b`, [`Qwen3.8-27B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO), `12.18 GiB` full VLM+MTP with `W8_g64` `lm_head` + `mtp.fc`), a `51.8 GiB` dense model fits comfortably on a `16 GB` GPU (`~3.8 GiB` KV/graph headroom) with `0.2717` `probe-v4` `weighted_kl`, `88.74%` E2E Vision Top-1, and `74.47%` E2E Native MTP Top-1, while `All-W4` (`16.14 GiB`) cuts `probe-v4` `weighted_kl` by `13.6%` (`0.1160 → 0.1002`) and `swe` KL by `20.7%` (`0.8918 → 0.7075`) while saving `1.47 GiB`.

