# Layer-1 N-Gram Per-Layer Embedding (PLE) Quantization Design (`Qwen3.8-Flash-Next`)

This document records the empirical analysis, alternatives evaluated, and design rationale behind the **4-bit `Custom-LUT4` (`group_size=32`) + Analytical Downstream `W_kv`-HGPTQ + 1-Pass Coordinate Descent + 4 KiB Page-Slotted Layout** used for the `51.2B`-parameter Layer-1 Per-Layer Embedding (PLE) table in [`nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ`](README.md).

---

## 1. Problem Geometry & Why Standard Quantization Methods Fail

In `Qwen/Qwen3.8-Flash-Next`, Layer 1 (`layers.1.ple`) includes a **`51.20B`-parameter (`102.4 GB` / `95.37 GiB` BF16)** N-gram hash-lookup table (`layers.1.ple.ple_embedding.ngram_embedding`) split across `128` shards (`shard_0..127`, totaling `320,000,456` head-rows $\times$ `160` columns).

At runtime (`ngram_size = 3`, `heads_per_ngram = 8`, `ple_embed_dim = 2560`):
1. For each token, `16` independent hash heads (`8` bigram heads + `8` trigram heads, `160` dimensions per head) are gathered from the table and concatenated into a `2560`-dimensional embedding vector $e \in \mathbb{R}^{2560}$.
2. That embedding vector $e$ is consumed **only** by two linear projections in Layer 1:
   - `layers.1.ple.key_proj` ($W_k \in \mathbb{R}^{10240 \times 2560}$, bias=`False`)
   - `layers.1.ple.value_proj` ($W_v \in \mathbb{R}^{2560 \times 2560}$, bias=`False`)

Direct inspection of `shard_0.weight` (`[2,500,012, 160]`), `key_proj.weight`, and `value_proj.weight` ruled out four standard compression approaches immediately:

1. **Row pruning / sparsity is inapplicable (`0.0%` dead rows):**
   - `0.0%` of hash-bucket rows have L2 norm `< 0.01`; row L2 norms sit tightly in `[0.044, 0.199]` (median `0.093`). Every hash bucket carries trained signal.
2. **Low-rank SVD truncation fails (`flat singular spectrum`):**
   - Across the `160` columns of $E$, singular values are nearly flat (`max = 1.88`, `median = 0.95`, `min = 0.84`). Truncating even `16` of the `160` columns (`rank 144/160`, a `10%` dimension reduction) discards `5.66%` of total energy (`23.8%` Rel-RMSE).
3. **Per-head orthogonal rotation ($E' = E Q_{160}$) alone gives negligible gain:**
   - Although any per-head orthogonal matrix $Q_{160 \times 160}$ folds 100% offline into $W_k$ and $W_v$, PLE rows are already outlier-free (`crest_factor` $= \max |x_i| / \operatorname{RMS}(x) = 2.89$, below i.i.d. Gaussian $\sqrt{2\ln 160} \approx 3.18$). Rotating $E$ before round-to-nearest (RTN) quantization barely changes reconstruction error because there are no channel outliers to smooth.
4. **Activation-sampled GPTQ cannot calibrate the table:**
   - Because the table has `320,000,456` rows indexed by N-gram hashes, any practical calibration set (`500K–1M` tokens) touches `<0.1%` of the rows, leaving `>99.9%` of rows with zero empirical activation count.

---

## 2. Ecosystem Survey & Why 8-Bit PLE (`~48–51 GiB`) Hits a Systems Cliff

We evaluated all existing PLE quantization formats used across the ecosystem (vLLM, ModelOpt, Primitive AI, MLX, ExLlamaV3, and GGUF) alongside our own measurements on `shard_0.weight` + `layers.1.ple.{key_proj,value_proj}`:

| Scheme / Ecosystem Reference | PLE Table Size | `E` Rel-RMSE | `key_proj(E)` Rel-RMSE | `value_proj(E)` Rel-RMSE | `value_proj(E)` Cosine | Systems & Accuracy Tradeoff |
|---|---:|---:|---:|---:|---:|---|
| **FP8 `E4M3FN` per-tensor** (`Qwen/...-FP8`, `RadixArk`, `albucino/...-FP8PLE`) | `51.20 GB` (`47.68 GiB`) | `2.653%` | `2.643%` | `2.653%` | `0.999754` | **Fails the 96–128 GB cliff (`+20.9 GiB` over 4-bit):** Backbone (`66.1 GiB`) + FP8 PLE (`47.7 GiB`) = **`113.8 GiB`**, exceeding a `96 GB` GPU (`RTX PRO 6000` / `4× RTX 3090`) and thrashing NVMe when `HostExpertStore` pins `~45 GiB` of cold MoE experts on a 96–128 GB host. |
| **FP8 `E4M3FN` per-row** (`primitive-ai` `ples_fp8`) | `51.84 GB` | `2.593%` | `2.567%` | `2.592%` | `0.999770` | Same `~48 GiB` memory cliff; 3-bit FP8 mantissa provides almost no gain over per-tensor FP8. |
| **INT8 per-row** (FP16 scale per 160-dim row) | `51.84 GB` | `0.659%` | `0.651%` | `0.659%` | `0.999978` | `4×` lower RMSE than FP8 (`7` vs `3` mantissa bits) at the same size, but still `~48 GiB`. |
| **INT8 `g32`** (`arnomatic/...-PLE8`, GGUF `Q8_0`) | `54.40 GB` | `0.532%` | `0.528%` | `0.532%` | `0.999986` | `54.4 GB` footprint for a single layer's hash table. |
| **INT4 sym `g16`** (`primitive-ai` `ples_int4`) | `32.00 GB` (`29.80 GiB`) | `8.672%` | `8.567%` | `8.676%` | `0.996337` | Uniform symmetric bins waste levels on tails; ignores downstream $W_k, W_v$ sensitivity. |
| **NVFP4 `E2M1` `g16`** (`nvidia/...-FP4-Local`, `ples_nvfp4`) | `28.80 GB` (`26.82 GiB`) | `9.534%` | `9.449%` | `9.528%` | `0.995569` | Highest error among 4-bit formats (`9.45%` `key_proj` RMSE) due to coarse `E2M1` (`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`) steps near the Gaussian core. |

### Why 4-Bit (`26.82 GiB`) vs. 8-Bit (`47.68 GiB`) Matters in Practice

While a single token only reads `16` rows, shrinking PLE from `8-bit` to `4-bit` improves real-world inference across three hardware regimes:
1. **Crossing the 96 GB Zero-Offload VRAM Threshold (`RTX PRO 6000 Blackwell` or `4× RTX 3090`):**
   - The rest of the quantized 180B model (`48` target layers + `1` MTP layer + `Vision` in `W4A16-HGPTQ g64-h64`) is **`66.6 GiB`**.
   - With `FP8` PLE (`47.7 GiB`), total weights are **`114.3 GiB`** — forcing CPU/PCIe offload.
   - With `4-bit` PLE (`26.8 GiB`), total weights drop to **`93.42 GiB`** — fitting 100% inside `96 GB` of GPU VRAM with zero PCIe offload.
2. **Eliminating NVMe Page-Cache Thrashing on 96–128 GB Host RAM Tiered Setups:**
   - When serving on `1×` or `2× RTX 3090` with `HostExpertStore` pinning `~40–45 GiB` of cold MoE experts in host DDR, an `8-bit` (`47.7 GiB`) mmap PLE table overflows the remaining Linux page cache on a `96 GB` host and thrashes NVMe continuously. At `26.8 GiB`, the entire 4-bit PLE table stays resident in the OS page cache with zero NVMe I/O after warmup.
3. **`+20.9 GiB` More VRAM for KV Cache, GDN Recurrent State, and MTP Speculative Decoding on `121–141 GB` GPUs (`GB10` / `H200`):**
   - Every active sequence in `Qwen3.8-Flash-Next` requires `36` GatedDeltaNet recurrent states (`[16, 128, 128]` FP32 $\approx 38\text{ MB}$/seq plus speculative rollback slots) and `13` full-attention KV caches (`12` target + `1` MTP). Saving `20.9 GiB` on PLE goes directly into the KV + GDN/MTP state pool, supporting `2×–3×` higher batch concurrency or longer context lengths.
4. **40% Fewer Random Cache-Line / Memory-Sector Reads During Long-Context Prefill:**
   - A 32K-token prefill performs `32,768 × 16 = 524,288` cold random row lookups across `320M` rows (`500×` larger than GPU L2 cache). Shrinking each row from `160 B` (`FP8`) to `90 B` (`4-bit g32`) cuts per-lookup random memory transactions from `5 GPU sectors / 3 CPU cache lines` down to **`3 GPU sectors / 2 CPU cache lines`**.

---

## 3. How `Custom-LUT4` + Analytical `W_kv`-HGPTQ Cuts 4-Bit Error by ~30%

To reach 4-bit (`4.50 bpw`, `26.82 GiB`) without suffering the `8.6%–9.5%` projection error of `INT4` and `NVFP4`, we combined two mathematical properties of `layers.1.ple`:

### 3.1 Non-Uniform Empirical Centroids (`PLE_CUSTOM_LUT4`)
Within each 32-element group, normalized PLE weights follow a heavy-tailed / super-Gaussian distribution ($\kappa \approx 4.6$) with most probability mass concentrated inside `[-0.55, +0.55]`.
- Uniform `INT4` spaces all 16 levels equally across `[-1, 1]`, wasting bins in the sparse outer tails.
- `NVFP4` (`E2M1`) places half its non-zero dynamic range at `{±2, ±3, ±4, ±6}`.
- Our fixed 16-entry symmetric non-uniform lookup table (`PLE_CUSTOM_LUT4` in `src/warp_quant/streaming_moe.py`):
  ```python
  PLE_CUSTOM_LUT4 = [
      -1.0000, -0.7229, -0.5250, -0.3959, -0.2844, -0.1848, -0.0911, 0.0000,
       0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0000,
  ]
  ```
  places dense resolution across the central core while preserving exact `-1.0`, `0.0`, and `+1.0` levels. In a CUDA kernel, all 16 FP16 entries fit inside a **single 32-lane warp register** and dequantize via a single `__shfl_sync` instruction with zero shared-memory bank conflicts.

### 3.2 The Rank-1 Alignment Between $E_h$ and $W_{k,h}$ (Analytical `W_kv`-HGPTQ)
When we computed the SVD of each head's downstream key projection slice $W_{k,h} \in \mathbb{R}^{10240 \times 160}$ ($h \in \{0,\dots,15\}$), we uncovered a striking structural property:
- $W_{k,h}$ has **one dominant singular value**:
  - Bigram heads (`h = 0..7`): $\sigma_1 = 5.66$ vs. $\sigma_2 \approx 2.2, \dots, \sigma_{160} \approx 1.1$ (**`25×` energy amplification** along a single direction in $\mathbb{R}^{160}$).
  - Trigram heads (`h = 8..15`): $\sigma_1 = 6.88$ vs. $\sigma_2 \approx 2.2, \dots, \sigma_{160} \approx 1.1$ (**`47×` energy amplification** along a single direction in $\mathbb{R}^{160}$).
- Furthermore, the top right singular vector of $W_{k,0}$ and the top right singular vector of the embedding table $E_0$ have **cosine similarity `0.9501`**.

Round-to-nearest (RTN) quantizers (`NVFP4`, `INT4`, `IQ4_NL`) minimize isotropic weight error $\|E_h - \hat{E}_h\|_F^2$, treating all 160 columns equally and injecting quantization noise directly into $W_{k,h}$'s `25×–47×` amplified singular direction.

Because head $h$ of `ngram_embedding` **only** feeds $W_{k,h}$ and $W_{v,h}$, the exact downstream quadratic error $\|E_h W_{k,h}^\top - \hat{E}_h W_{k,h}^\top\|_F^2 + \|E_h W_{v,h}^\top - \hat{E}_h W_{v,h}^\top\|_F^2$ has a **closed-form, calibration-free $160 \times 160$ Hessian** for every head $h \in \{0,\dots,15\}$:

$$H_h = \frac{1}{2}\left(\frac{W_{k,h}^\top W_{k,h}}{\|W_{k,h}\|_F^2 / 160} + \frac{W_{v,h}^\top W_{v,h}}{\|W_{v,h}\|_F^2 / 160}\right) + \lambda \operatorname{diag}(\dots) \in \mathbb{R}^{160 \times 160}$$

For each of the `128` PLE shards (`2,500,012 × 160` rows):
1. We permute the `160` columns in descending order of $\operatorname{diag}(H_h)$ (`actorder="static"`) and compute the upper Cholesky inverse factor $U_h = \operatorname{chol}(H_h^{-1})$.
2. We run sequential **Cholesky GPTQ** across all `160` columns using `PLE_CUSTOM_LUT4` (`group_size=32`), propagating each column's quantization residual into the remaining unquantized columns so error is steered into the low-gain directions of $(W_{k,h}, W_{v,h})$.
3. We finish with **1 pass of exact coordinate-descent (`CD`) per-group scale refinement** ($s_g^\star = \frac{\langle w_g, c_g \rangle}{\langle c_g, c_g \rangle}$) and re-run the Cholesky GPTQ error feedback with the refined scales (`0.98s` per shard on H200; `2.7 min` for all `128` shards).
4. Finally, during Layer-1 linear quantization, `layers.1.ple.key_proj` and `layers.1.ple.value_proj` (`W4A16-HGPTQ g64-h64`) are calibrated on the **already-dequantized** 4-bit PLE embeddings gathered across the calibration corpus, allowing $W_k$ and $W_v$ to absorb any remaining PLE quantization bias.

### 3.3 Measured 4-Bit Quantization Ablation (`shard_0.weight` + `key_proj` + `value_proj`)

| 4-Bit PLE Quantization Scheme | Effective `bpw` | Total PLE Size | `E` Rel-RMSE | **`key_proj(E)` Rel-RMSE** ↓ | **`value_proj(E)` Rel-RMSE** ↓ | Relative `key_proj` Error vs. Chosen |
|---|---:|---:|---:|---:|---:|---|
| **NVIDIA `NVFP4 E2M1 g16`** (`FP4-Local` / `ples_nvfp4`) | `4.50` | `28.80 GB` (`26.82 GiB`) | `9.534%` | `9.449%` | `9.528%` | `+42.0%` higher error |
| **Uniform `INT4 sym g32`** (RTN) | `4.50` | `28.80 GB` (`26.82 GiB`) | `9.424%` | `9.371%` | `9.423%` | `+40.8%` higher error |
| **Primitive AI `INT4 sym g16`** (`ples_int4`, RTN) | `5.00` | `32.00 GB` (`29.80 GiB`) | `8.672%` | `8.567%` | `8.676%` | `+28.7%` higher error (`+3.2 GB` larger) |
| **ISTA-DASLab `IQ4_NL g32`** (Non-uniform LUT4, RTN) | `4.50` | `28.80 GB` (`26.82 GiB`) | `8.287%` | `7.434%` | `8.285%` | `+11.7%` higher error |
| **Ours: `Custom-LUT4 + W_kv-HGPTQ + CD` (`g=160`, 1 scale/row)** | `4.10` | `26.24 GB` (`24.44 GiB`) | `9.450%` | `7.869%` | `9.375%` | Beats `32 GB` `INT4 g16` with 1 scale/row |
| **Ours: `Custom-LUT4 + W_kv-HGPTQ + CD` (`g=32`) — Chosen** | **`4.50`** | **`28.80 GB` (`26.82 GiB`)** | **`7.990%`** | **`6.655%`** | **`7.930%`** | **Baseline (`-29.6%` vs `NVFP4`, `-22.3%` vs `32 GB INT4 g16`)** |
| **Ours: `Custom-LUT4 + W_kv-HGPTQ + CD` (`g=16`)** | `5.00` | `32.00 GB` (`29.80 GiB`) | `7.146%` | `5.955%` | `7.092%` | `-0.70 pp` RMSE, but costs `+3.2 GB` (`+11.1%` size) |

---

## 4. Eliminating OS/NVMe Page-Boundary Straddles (`4 KiB` Page-Slotted Layout)

Each token performs **`16` independent random row lookups** (`8` bigram + `8` trigram heads, `160` cols each). When the PLE table is served from `mmap` / Linux page cache or NVMe (`io_uring` / `O_DIRECT`), storing rows contiguously causes rows to straddle `4,096 B` OS page boundaries:
- **Contiguous 8-bit (`160 B/row`):** `4096 / 160 = 25.6` rows/page (`3.125%` page-straddle rate per head $\implies 1 - (1 - 0.03125)^{16} = \mathbf{39.8\%}$ probability per token that at least one head faults across two `4 KiB` pages).
- **Contiguous 4-bit (`90 B/row`):** `4096 / 90 = 45.51` rows/page (`2.20%` page-straddle rate per head $\implies \mathbf{29.9\%}$ probability per token of an extra page fault).

Instead of unaligned contiguous rows, `pack_ple_page_slotted_4k` (`src/warp_quant/streaming_moe.py`) packs an exact integer number of rows ($R_{\text{page}}$) into every `4,096 B` page:

$$\text{byte\_offset}(\text{row}) = (\text{row} \mathbin{/\!/} R_{\text{page}}) \times 4096 + (\text{row} \bmod R_{\text{page}}) \times \text{row\_bytes}$$

We evaluated four page-slotted candidate layouts under `Custom-LUT4 + W_kv-HGPTQ + CD`:

| 4-Bit Page-Slotted Layout (`160` cols/row) | Bytes / Row | Rows / `4 KiB` Page | Page Fill Efficiency | Total Paged Size | `key_proj(E)` Rel-RMSE | `value_proj(E)` Rel-RMSE | Page-Straddle Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| *Reference: Contiguous `NVFP4 g16` (unaligned)* | `90 B` | `45.51` | — | `26.82 GiB` | `9.45%` | `9.53%` | `2.20%`/head (`29.9%`/tok) |
| *Reference: Contiguous `IQ4_NL g32` (unaligned)* | `90 B` | `45.51` | — | `26.82 GiB` | `7.43%` | `8.29%` | `2.20%`/head (`29.9%`/tok) |
| **Option A:** Split resident `g=80` FP16 scale (`80 B` nibbles in mmap + `4 B` scale in RAM) | `80 B` *(+`4 B`)* | `51` | `99.61%` (`4080/4096`) | `23.93 GiB` *(+`1.19 GiB`)* | `7.41%` | `8.83%` | **`0.000%`** |
| **Option B:** Inline `g=32` `FP8 E4M3` scales (`80 B` nibbles + `5 × 1 B` FP8 scales) | `85 B` | `48` | `99.61%` (`4080/4096`) | `25.43 GiB` | `6.99%` | `8.33%` | **`0.000%`** |
| **Option C:** Inline `g=40` `FP16` scales (`80 B` nibbles + `4 × 2 B` FP16 scales) | `88 B` | `46` | `98.83%` (`4048/4096`) | `26.54 GiB` | `6.86%` | `8.17%` | **`0.000%`** |
| **Option D (Chosen):** Inline `g=32` `FP16` scales (`80 B` nibbles + `5 × 2 B` FP16 scales) | **`90 B`** | **`45`** | **`98.88%` (`4050/4096`)** | **`27.13 GiB`** (`26.82 GiB` linear) | **`6.66%`** | **`7.93%`** | **`0.000%`** |

### Why Option D (`g=32` FP16 Scales, `90 B/row`, `45 rows/page`) Won
1. **Lowest downstream projection error (`6.66%` `key_proj`, `7.93%` `value_proj`):** Avoids `FP8` scale quantization noise (`Option B`) and wider `g=40`/`g=80` groups (`Options A & C`).
2. **Co-located nibbles + scales (`90 B/row`, `45 rows/page`):** Each row's `80 B` packed nibbles and `10 B` FP16 scales sit side-by-side inside a single `90 B` slot (`2` CPU cache lines / `3` GPU sectors), packing `45` complete rows into `4,050 B` (`98.88%` of every `4,096 B` OS/NVMe page, with `46` slack bytes and **`0.000%` page-straddle faults**).
3. **Dual compatibility:** In `model-ple-000..007.safetensors`, we store the linear `weight_packed` (`[2500012, 80]`), `weight_scale` (`[2500012, 5]`), `weight_lut` (`[16]`), and `weight_shape` tensors (`26.822 GiB` total), and `pack_ple_page_slotted_4k` / `unpack_ple_page_slotted_4k` deterministically converts between linear and `4 KiB` page-slotted (`[ceil(N/45), 4096]`) layouts for direct `mmap` / `io_uring` engines.

---

## 5. Runtime Execution Note: Zero-Critical-Path Lookahead Prefetch

In `Qwen3.8-Flash-Next`, the entire PLE lookup and projection path in `layers.1.ple`:
```python
embeddings = self.ple_embedding(input_ids, ...)   # [T, 2560] — depends ONLY on input_ids!
key   = self.norm_key(self.key_proj(embeddings))  # [T, 4, 2560] — ALSO independent of hidden_states!
value = self.value_proj(embeddings)               # [T, 2560] — ALSO independent of hidden_states!
```
depends **exclusively on `input_ids` $(w_t, w_{t-1}, w_{t-2})$** — `hidden_states` is not touched until the elementwise gate inside Layer 1 after `layers.0` (GDN + 512-expert MoE) and `layers.1` (GDN + 512-expert MoE) have already executed.

Consequently, an inference engine can launch the 64-bit SplitMix/XOR N-gram hash, `16`-row `90 B` gather (`1.44 KB`/token), `Custom-LUT4` warp-shuffle dequantization, and fused `[key_proj; value_proj]` projection **at the very beginning of the forward pass** on a secondary CUDA stream (or async host-pinned DMA), hiding 100% of the PLE gather + projection latency behind `embed_tokens` + `layers.0`.
