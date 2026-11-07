# warp-34-quant-lab

**W4A16-HGPTQ** (`g64-h64`, block Walsh–Hadamard input rotation + Cholesky GPTQ + 4-bit `Custom-LUT4` `W_kv`-HGPTQ PLE) quantization and teacher-forced fidelity probing for the Qwen3.5 / Qwen3.6 / Qwen3.8 model family (`0.8B` → `27B` dense, `35B-A3B` MoE, and `Qwen3.8-Flash-Next` 512-expert MoE + Layer-1 PLE).

Managed with [`mise`](.mise.toml) and [`uv`](pyproject.toml).

---

## Quick Start

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
```

---

## Published W4A16-HGPTQ Results

Full receipts and Hugging Face model cards are stored in [`results/`](results/):

| Model | Artifact (`nchapman/*`) | Size | `probe-v4` PPL (BF16 → Quant) | `probe-v4` `weighted_kl` | `probe-guard` `weighted_kl` (`mean_kl`) | MTP Iso `mean_kl` (Top-1) | Vision Iso `mean_kl` (Top-1) | Vision Guard Iso Top-1 |
|---|---|---:|---|---:|---|---:|---:|---:|
| `Qwen/Qwen3.5-2B` | [`Qwen3.5-2B-W4A16-HGPTQ`](results/dense-w4a16-hgptq/g64-h64/README.md) | `1.84 GiB` | `3.8302 → 3.8942` (`+1.669%`) | `0.072782` | `0.072536` (`0.057225`) | `0.025525` (`91.87%`) | `0.015169` (`97.45%`) | — |
| `Qwen/Qwen3.5-9B` | [`Qwen3.5-9B-W4A16-HGPTQ`](results/dense-w4a16-hgptq/9b-g64-h64/README.md) | `7.62 GiB` | `3.5754 → 3.6428` (`+1.887%`) | `0.072561` | `0.056641` (`0.039547`) | `0.012801` (`94.80%`) | `0.007640` (`98.09%`) | — |
| `Qwen/Qwen3.8-27B` | [`Qwen3.8-27B-W4A16-HGPTQ`](results/dense-w4a16-hgptq/27b-g64-h64/README.md) | `17.33 GiB` | `8.5438 → 8.5189` (`-0.292%`) | `0.115968` | `0.081058` (`0.063923`) | `0.010930` (`96.22%`) | `0.015734` (`97.45%`) | — |
| `Qwen/Qwen3.6-35B-A3B` | [`Qwen3.6-35B-A3B-W4A16-HGPTQ`](results/qwen36-35b-a3b-w4a16-hgptq/README.md) | `19.24 GiB` | `4.0208 → 3.9020` (`-2.956%`) | `0.090070` | `0.057899` (`0.048598`) | `0.011066` (`95.93%`) | `0.007872` (`98.34%`) | `99.26%` |
| `Qwen/Qwen3.8-Flash-Next` | [`Qwen3.8-Flash-Next-W4A16-HGPTQ`](results/qwen38-flash-next-w4a16-hgptq/README.md) | `93.42 GiB` | `3.6085 → 3.6264` (`+0.496%`) | `0.053029` | `0.153487` (`0.105409`) | `0.019397` (`94.15%`) | `0.017803` (`96.36%`) | `99.26%` |

---

## Architecture & Engineering Lessons Learned

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
`Qwen3.8-Flash-Next` includes a `51.2B`-parameter (`102.4 GB` / `95.37 GiB` BF16) Per-Layer Embedding (PLE) table at Layer 1 (`128` N-gram shards across `8` files `model-ple-000..007.safetensors`, `320,000,456` head-rows × `160` dim/head across `16` N-gram heads, `ple_embed_dim = 2560`) feeding downstream `key_proj` and `value_proj`:
- **Why uniform INT4 fails on PLE:** PLE rows are super-Gaussian ($\kappa \approx 4.6$); uniform INT4 wastes outer levels or clips tails. A fixed 16-entry non-uniform lookup table (`PLE_CUSTOM_LUT4`, 16 FP32 centroids in `[-1, 1]`) aligned to the empirical PLE distribution cuts reconstruction error by **>2×**.
- **Analytical `W_kv`-HGPTQ + 1-Pass Coordinate Descent:** Because PLE rows are sparsely indexed by N-grams, activation sampling misses the tail of the 320M-row table. Instead, the exact downstream linear map is known analytically:
  $$H_h = W_{k,h}^\top W_{k,h} + W_{v,h}^\top W_{v,h} + \lambda \operatorname{diag}(\dots) \in \mathbb{R}^{160 \times 160}$$
  Solving Cholesky GPTQ against $H_h$ per head (`group_size=32`) followed by 1 pass of per-group coordinate-descent scale refinement (`ple_cd_passes=1`) compresses PLE `3.56×` (`102.4 GB` / `95.37 GiB` → `26.822 GiB`, `90 B/row`: `80 B` packed 4-bit nibbles + `10 B` FP16 scales) with near-zero perplexity impact.
- **4 KiB Page-Slotted NVMe Layout (`pack_ple_page_slotted_4k`):** `pack_ple_page_slotted_4k` / `unpack_ple_page_slotted_4k` packs `45` PLE head-rows (`80 B` nibbles + `10 B` FP16 scales = `90 B/row`) into each `4096 B` OS/NVMe page (`98.9%` density, zero page-boundary straddling) for single-page-fault `mmap` / `io_uring` / `O_DIRECT` lookup.

### 6. Static Hot Expert Cache Rankings (`runtime/static_hot_cache_rankings.json`)
During streaming MoE quantization and probing (`src/warp_quant/streaming_moe.py`), exact top-$k$ router hits and routing probability mass are accumulated across all $L$ MoE layers and Native MTP (`9,804,460` expert assignments/layer for `flash-next`, `6,979,616` for `35b`) and exported to:
- `runtime/static_hot_cache_rankings.json` (and root `static_hot_cache_rankings.json`) — maps target layer indices (`"0".."39"` or `"0".."47"`) to global expert IDs ordered hottest-first (`active_ranked_expert_ids`), enabling zero-warmup static hot-expert pinning in VRAM at server startup (`VLLM_WNA16_STATIC_HOT_CACHE_FILE` / `HostExpertStore`).
- `runtime/expert_routing_profile.json` — full per-layer `ranked_expert_ids`, `active_ranked_expert_ids`, `hit_counts`, `weight_sums`, and `top_k_coverage` (`top_8..top_256`), including `"mtp.0"`.
