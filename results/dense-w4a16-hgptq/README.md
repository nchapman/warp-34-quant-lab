# Online block-Hadamard All-W4 Cholesky GPTQ (`Qwen3.5-2B`) — 2026-11-06

**Full 3×3 factorial (`g128, g64, g32` × `H=0, 64, 128`) across all 255 quantizable linear layers (`150` language, `7` Native MTP, `98` Vision; `248` in `model.safetensors` + `7` in `model_mtp.safetensors`) using Cholesky GPTQ and deterministic online Walsh-Hadamard rotation (`QuIPModifier` / $R_4$-style + HadaCore).**
Online block-Hadamard rotation (`H=64`) adds **zero weight storage overhead** (`0` bytes in safetensors shards; deterministic Hadamard matrices are reconstructed at load time) while reducing Language (`probe-v4`, `probe-guard`), Native MTP, and end-to-end Vision teacher-forced KL across all three group sizes, and reducing isolated ViT KL at `g64` and `g32` (at `g128`, rotation reduces ViT feature RMSE `0.4080 → 0.3515` and end-to-end VLM KL `0.0748 → 0.0673`, while isolated ViT KL rises `0.0136 → 0.0174`):
- **Language (`probe-v4`, 378 rows / 840,187 tokens)**: `H=64` lowers blend-weighted KL by **23.9% at g128** (`0.1038 → 0.0790`), **16.3% at g64** (`0.0869 → 0.0728`), and **17.0% at g32** (`0.0774 → 0.0642`), winning **360/378, 347/378, and 351/378 rows**, respectively. Notably, **`g128-h64` beats unrotated `g64-h0` and `g32-h0` on token-mean KL (`0.2680` vs `0.2874` / `0.2759`) and perplexity delta (`+1.897%` vs `+4.917%` / `+5.464%`) while saving 26.1–78.3 MiB of weight storage**.
- **Guard (`probe-guard`, 76 rows / 19,534 tokens)**: `H=64` cuts token-mean KL by **39.6% at g128** (`0.1030 → 0.0622`, `74/76` rows), **31.8% at g64** (`0.0839 → 0.0572`, `75/76` rows), and **36.6% at g32** (`0.0702 → 0.0445`, `74/76` rows).
- **Native MTP (`7` W4A16 projections, 32 rows / 11,583 positions)**: Cholesky GPTQ cuts unrotated W4A16 isolated MTP KL by **64.3%** vs RTN (`0.1156 → 0.0413`), and combining Cholesky GPTQ with `H=64` cuts it further to **`0.0255` at g64 (`91.87%` top-1 agreement)** and **`0.0208` at g32 (`92.84%` top-1 agreement)**.
- **Vision (`98` W4A16 ViT + merger projections, 48 images / 157 answer tokens)**: Cholesky GPTQ calibrated on a strictly disjoint 64-image ChartQA+TextVQA set (`0` image overlap with the 48-image evaluation set) combined with `H=64`/`H=128` cuts isolated W4A16 ViT teacher-forced KL to **`0.0152` (`97.45%` top-1 agreement) at g64-h64**, **`0.0104` (`97.45%` top-1) at g64-h128**, and **`0.0085` (`96.18%` top-1) at g32-h64**—matching the `97.45%` (153/157) top-1 agreement previously measured for W8A16-g64 Vision while cutting Vision+MTP checkpoint storage by an additional **~327 MiB**.

All quantization and teacher-forced fidelity evaluations ran on an H200 SXM pod (`ssh-1`) acting as the quality gate. **No RTX 3090 throughput or latency claim is made here**; SM86 runtime speed and speculative decoding acceptance rate ($\tau$) on the deployment target remain a separate measurement.

---

## Scope and architecture boundaries

- **Base model**: `Qwen/Qwen3.5-2B` at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- **Quantized modules (`255` Linear layers total: `248` in `model.safetensors` + `7` in `model_mtp.safetensors`)**:
  - **Language (`150` linears)**: All 24 hybrid GDN + full-attention blocks (`18 × 6 = 108` GDN layers + `6 × 7 = 42` full-attention layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`).
  - **Native MTP (`7` linears)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj`.
  - **Vision (`98` linears)**: All 24 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`).
- **Kept in BF16 (Marlin $N \pmod{64} = 0$ and boundary invariants)**:
  - `linear_attn.in_proj_a`, `linear_attn.in_proj_b` ($N = 16$, incompatible with Marlin's $N \pmod{64} = 0$ tile constraint)
  - `model.language_model.embed_tokens`, `lm_head` (tied embeddings)
  - `mtp.fc` (fusion projection), `visual.patch_embed.proj` (3D patch conv), norms, and biases.
- **Online Hadamard rotation (`QuIPModifier` / $R_4$-style)**:
  - Block-diagonal deterministic Walsh-Hadamard transform $H_b \in \mathbb{R}^{b \times b}$ ($b \in \{64, 128\}$) folded offline into weight columns ($W' = W H_b$) prior to Cholesky GPTQ quantization and undone online on activations ($x' = x H_b^T$) via `hadacore` kernels (`location: "input"`).
  - Exact $K \pmod b = 0$ divisibility holds across all 255 quantized layers in `Qwen3.5-2B`: language ($K \in \{2048, 4096, 6144\}$), MTP ($K \in \{2048, 6144\}$), and Vision ($K \in \{1024, 4096\}$).
  - Materialized transform weight tensors are stripped from `model-*.safetensors` after calibration (`explicit_weight: false`), so `H=0`, `H=64`, and `H=128` produce **byte-for-byte identical safetensors shard sizes** at each group size.

---

## Calibration and evaluation protocol

- **Language & MTP calibration**: `999` rows from `data/calib-cascade-v1.jsonl` at `max_seq_length = 4096` (`4,091,904` tokens), `actorder="static"`, `dampening_frac=0.01`, executed with `--sequential-offload-device cuda` on H200. During the post-quantization language pass, hidden states after `model.language_model.norm` and shifted token embeddings on the first 64 calibration sequences (`512` tokens each) are captured to build the exact input Hessians $H = \frac{2}{N} X^T X$ for the 7 MTP projections.
- **Vision calibration (strictly disjoint from evaluation)**:
  - `quants/component_probe_data.py --calibration` selects `64` unique images (`32` ChartQA + `32` TextVQA) ranked by hash seed `vision-calib-1` (`receipts/vision-calib-manifest.json`, `rows_sha256: 05b705bda00067dc366d12375d38eb7e00e327ff826b09541c6604feb1b3893a`) after excluding all `48` evaluation image digests selected by `vision-probe-0` (`receipts/vision-data-manifest.json`, `rows_sha256: 0854474a436a6163486bb01c5e2c564de2d9c2187773f726d48890d1989b31f0`), guaranteeing **0 overlapping images** (`112` distinct image SHA-256 digests).
  - Input Hessians for all 98 Vision linears are accumulated in FP32 with `torch.amp.autocast("cuda", enabled=False)` inside the forward hook so high-magnitude ViT MLP activations (`blocks.23.mlp.linear_fc2`) never overflow BF16 during $X^T X$ accumulation (`0/98` Cholesky fallbacks across all runs).
- **Evaluation instruments**:
  1. **`probe-v4` (`378` rows / `840,187` scored tokens)** and **`probe-guard` (`76` rows / `19,534` scored tokens)** via `quants/probe.py` at `max_tokens = 4096` using exact dequantization + inverse Hadamard transform for teacher-forced full-vocabulary FP32 KL.
  2. **Native MTP probe (`32` rows / `11,583` scored positions, 2 rows × 16 `probe-v4` domains)**:
     - *Isolated MTP*: feeds identical BF16 target hidden states $H_{\text{BF16}}[t]$ and embeddings $E_{\text{BF16}}[x_{t+1}]$ into the quantized MTP block to isolate MTP weight error from main-model drift.
     - *End-to-end MTP*: feeds the quantized language model's own hidden states $H_{\text{quant}}[t]$ into the quantized MTP block, measuring compound target + drafter divergence against BF16 MTP.
  3. **Vision probe (`48` held-out images / `157` reference answer tokens)**:
     - *Encoder feature fidelity*: relative RMSE and cosine similarity of post-merger visual tokens (`visual(pixel_values, grid_thw)`) vs BF16.
     - *Isolated ViT*: injects quantized ViT visual embeddings into the **BF16 language model** to measure answer-token KL and top-1 agreement caused purely by ViT quantization.
     - *End-to-end VLM*: runs both quantized ViT and quantized language model together.

---

## Component ablation: RTN vs Cholesky GPTQ for W4A16 MTP and Vision (`g64`)

Before launching the 9-arm sweep, we ablated Round-to-Nearest (`--component-quant-mode rtn`, 128 calibration rows for text) against Hessian-weighted Cholesky GPTQ (`--component-quant-mode gptq`) on the 7 MTP and 98 Vision linears at `g64`:

| Arm (`g64`) | Component mode | Hadamard $H$ | MTP Isolated KL ↓ | MTP Iso Top-1 ↑ | MTP E2E KL ↓ | ViT Feature Rel-RMSE ↓ | ViT Feature CosSim ↑ | ViT Isolated KL ↓ | ViT Iso Top-1 ↑ | ViT E2E KL ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `g64-h0-rtn` | RTN | 0 | 0.115629 | 83.62% | 0.541826 | 0.5612 | 0.862023 | 0.032019 | 93.63% | 0.126987 |
| `g64-h64-rtn` | RTN | 64 | 0.080516 | 86.01% | 0.476617 | 0.5113 | 0.882777 | 0.047646 | 95.54% | 0.113728 |
| `g64-h128-rtn` | RTN | 128 | 0.077170 | 86.62% | 0.457684 | 0.5001 | 0.883346 | 0.036678 | 94.27% | 0.092440 |
| `g64-h0-gptq` | **Cholesky GPTQ** | 0 | 0.041291 | 89.65% | 0.459451 | 0.3757 | 0.940996 | 0.020636 | 96.18% | 0.066950 |
| `g64-h64-gptq` | **Cholesky GPTQ** | **64** | **0.024907** | **92.23%** | **0.411620** | 0.3305 | 0.954395 | 0.015169 | **97.45%** | 0.079726 |
| `g64-h128-gptq` | **Cholesky GPTQ** | **128** | 0.025660 | **92.26%** | 0.426465 | **0.3251** | **0.955061** | **0.010387** | **97.45%** | **0.064707** |

**Takeaway**: Naive W4A16 RTN severely degrades both MTP (`0.1156` KL, `83.62%` top-1) and Vision (`0.5612` feature RMSE). Switching to Cholesky GPTQ + `H=64`/`H=128` reduces isolated MTP KL by **78.5%** (`0.1156 → 0.0249`, `+8.6 pp` top-1 agreement) and reduces isolated ViT KL by **52.6%–67.6%** (`0.0320 → 0.0152 / 0.0104`, reaching `97.45%` top-1 agreement). All 9 main-sweep checkpoints therefore use Cholesky GPTQ across all 255 linears.

---

## Full 9-arm factorial results (`999` calibration samples × `4096` tokens)

### 1. Language model fidelity (`probe-v4` & `probe-guard`) and checkpoint size

| Arm | Group size | Hadamard $H$ | Safetensors bytes (GiB) | `probe-v4` weightedKL ↓ | `probe-v4` token mean KL ↓ | `probe-v4` p99.9 KL ↓ | `probe-v4` PPL Δ% vs BF16 ↓ | `probe-v4` row wins vs $H=0$ | `probe-guard` token mean KL ↓ | `probe-guard` p99.9 KL ↓ | `probe-guard` row wins vs $H=0$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `g128-h0` | 128 | 0 | 1,948,961,744 (1.8151) | 0.103798 | 0.315542 | 15.9908 | +5.363% | — | 0.103019 | 1.5858 | — |
| **`g128-h64`** | **128** | **64** | **1,948,961,744 (1.8151)** | **0.078976 (−23.9%)** | **0.267985 (−15.1%)** | **14.9547** | **+1.897%** | **360/378** | **0.062211 (−39.6%)** | **1.4529** | **74/76** |
| `g128-h128` | 128 | 128 | 1,948,961,744 (1.8151) | 0.081394 (−21.6%) | 0.283912 (−10.0%) | 15.4031 | +3.062% | 361/378 | 0.066824 (−35.1%) | 1.6567 | 74/76 |
| `g64-h0` | 64 | 0 | 1,976,323,096 (1.8406) | 0.086938 | 0.287402 | 15.4200 | +4.917% | — | 0.083856 | 1.6155 | — |
| **`g64-h64`** | **64** | **64** | **1,976,323,096 (1.8406)** | **0.072782 (−16.3%)** | **0.254294 (−11.5%)** | **14.7976** | **+1.669%** | **347/378** | **0.057225 (−31.8%)** | 1.1966 | **75/76** |
| `g64-h128` | 64 | 128 | 1,976,323,096 (1.8406) | 0.073930 (−15.0%) | 0.273544 (−4.8%) | 15.2287 | +6.347% | 351/378 | 0.057225 (−31.8%) | **1.1230** | 74/76 |
| `g32-h0` | 32 | 0 | 2,031,045,704 (1.8916) | 0.077392 | 0.275858 | 15.3572 | +5.464% | — | 0.070166 | 1.5205 | — |
| **`g32-h64`** | **32** | **64** | **2,031,045,704 (1.8916)** | 0.064233 (−17.0%) | **0.245936 (−10.8%)** | **14.5040** | **+2.718%** | 351/378 | 0.044460 (−36.6%) | **0.8584** | **74/76** |
| `g32-h128` | 32 | 128 | 2,031,045,704 (1.8916) | **0.063804 (−17.6%)** | 0.255295 (−7.5%) | 15.8788 | +4.581% | **368/378** | **0.043667 (−37.8%)** | 0.9078 | 70/76 |

#### Key findings on Language

1. **`H=64` strictly dominates `H=0` and outperforms `H=128` on token-mean KL, p99.9 KL, and PPL Δ% across all three group sizes**:
   - Why `H=64` beats `H=128` for $g \in \{64, 128\}$: A smaller rotation block ($b = 64$) mixes outliers over a 64-channel neighborhood that aligns tightly with the `g64` scale grid (1:1) and splits each `g128` scale group into two balanced 64-channel sub-blocks without spreading a single extreme activation channel across 128 columns.
2. **Rotation substitutes for finer group sizes at zero byte cost**:
   - **`g128-h64` (`1.815 GiB`) beats unrotated `g64-h0` (`1.841 GiB`)** on every text metric: `weightedKL` (`0.0790` vs `0.0869`), `mean_kl` (`0.2680` vs `0.2874`, winning `261/378` v4 rows), `guard_kl` (`0.0622` vs `0.0839`, winning `66/76` guard rows), and `ppl_delta` (`+1.897%` vs `+4.917%`).
   - **`g64-h64` (`1.841 GiB`) beats unrotated `g32-h0` (`1.892 GiB`)** on every text metric: `weightedKL` (`0.0728` vs `0.0774`), `mean_kl` (`0.2543` vs `0.2759`, winning `258/378` v4 rows), `guard_kl` (`0.0572` vs `0.0702`, winning `59/76` guard rows), and `ppl_delta` (`+1.669%` vs `+5.464%`).
   - Compared against the earlier [2B RCO-15% mixed W4/W8 artifact](../rco-artifact-2b-20261105/README.md) (`2,295,410,352` bytes, `weightedKL = 0.0763`, `mean_kl = 0.2477`, `guard_kl = 0.0489`), pure W4A16 **`g64-h64` (`1,976,323,096` bytes) achieves lower `weightedKL` (`0.0728` vs `0.0763`) while saving `304.3 MiB`**, and **`g32-h64` (`2,031,045,704` bytes) outperforms RCO-15% on every metric (`weightedKL = 0.0642`, `mean_kl = 0.2459`, `guard_kl = 0.0445`) while still saving `252.1 MiB`**.

---

### 2. Native MTP and Vision fidelity across all 9 arms

| Arm | MTP Isolated KL ↓ | MTP Iso Top-1 ↑ | MTP E2E KL ↓ | MTP E2E Top-1 ↑ | ViT Feature Rel-RMSE ↓ | ViT Feature CosSim ↑ | ViT Isolated KL ↓ | ViT Iso Top-1 ↑ | ViT E2E KL ↓ | ViT E2E Top-1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `g128-h0` | 0.053702 | 89.14% | 0.538390 | 74.98% | 0.4080 | 0.930208 | 0.013550 | 95.54% | 0.074758 | 91.72% |
| **`g128-h64`** | 0.029814 (−44.5%) | **91.69%** | **0.441359 (−18.0%)** | **79.63%** | 0.3515 | 0.948272 | 0.017441 | 94.27% | **0.067283** | 92.36% |
| `g128-h128` | **0.029730 (−44.6%)** | 91.64% | 0.446424 (−17.1%) | 79.20% | **0.3426** | **0.950268** | 0.017742 | **96.82%** | 0.073249 | **94.27%** |
| `g64-h0` | 0.039735 | 89.75% | 0.480333 | 77.33% | 0.3757 | 0.940996 | 0.020636 | 96.18% | 0.090502 | 91.72% |
| **`g64-h64`** | 0.025525 (−35.8%) | 91.87% | **0.443014 (−7.8%)** | **80.49%** | 0.3305 | 0.954395 | 0.015169 | **97.45%** | **0.065351** | **94.90%** |
| `g64-h128` | **0.024922 (−37.3%)** | **92.14%** | 0.479118 (−0.3%) | 79.37% | **0.3251** | **0.955061** | **0.010387** | **97.45%** | 0.084821 | 92.99% |
| `g32-h0` | 0.030267 | 91.26% | 0.448589 | 79.24% | 0.3504 | 0.948779 | 0.011957 | 96.82% | 0.086144 | 91.72% |
| **`g32-h64`** | **0.020755 (−31.4%)** | **92.84%** | **0.387267 (−13.7%)** | **81.66%** | **0.3059** | **0.961051** | **0.008503** | 96.18% | 0.082508 | 91.08% |
| `g32-h128` | 0.020902 (−30.9%) | 92.38% | 0.419198 (−6.6%) | 81.12% | 0.3099 | 0.959220 | 0.010743 | **97.45%** | **0.050043** | **95.54%** |

#### Key findings on Native MTP and Vision

1. **Native MTP (`7` linears)**:
   - Online Hadamard rotation (`H=64`) slashes isolated W4A16 MTP KL by **31.4%–44.5%** at every group size (`0.0537 → 0.0298` at `g128`, `0.0397 → 0.0255` at `g64`, `0.0303 → 0.0208` at `g32`), raising top-1 draft token agreement with BF16 MTP by `+1.58` to `+2.55` percentage points.
   - End-to-end MTP KL (where the quantized MTP block receives hidden states from the quantized main model) drops from `0.5384` (`74.98%` top-1) at `g128-h0` to **`0.3873` (`81.66%` top-1) at `g32-h64`**.
2. **Vision (`98` linears)**:
   - Post-merger visual feature relative RMSE drops monotonically with both finer group size and Hadamard rotation, from `0.4080` (`cos = 0.9302`) at `g128-h0` down to **`0.3305` (`cos = 0.9544`) at `g64-h64`** and **`0.3059` (`cos = 0.9611`) at `g32-h64`**.
   - On the held-out 48-image / 157-token fixture, W4A16 `g64-h64` (`0.0152` KL, `153/157 = 97.45%` top-1), `g64-h128` (`0.0104` KL, `97.45%` top-1), and `g32-h128` (`0.0107` KL, `97.45%` top-1) match the `97.45%` top-1 agreement of W8A16-g64 Vision (`0.0021` KL) while cutting Vision weight bytes in half.

---

## vLLM V1 serving & multimodal MTP verification

We verified end-to-end loading and generation in **vLLM V1 (`v0.31.0`)** on H200 with:
- `MarlinLinearKernel` (`CompressedTensorsWNA16`) + `HadaCore` online block-Hadamard transform (`CompressedTensorsLinearTransformMethod`)
- Native MTP speculative decoding enabled (`speculative_config={"method": "mtp", "num_speculative_tokens": 3}`)
- Multimodal VQA inference across 8 ChartQA test images (`receipts/vllm_verification.json`).

All four tested checkpoints (`g64-h0`, `g64-h64`, `g32-h0`, `g32-h64`) loaded cleanly (`1.84 GiB` / `1.89 GiB` checkpoint size, `3.63–3.66 GiB` total weight + runtime footprint) and generated coherent text and VQA answers (`5/8`–`6/8` exact string matches on raw ChartQA reference answers without post-processing).

### Upstream vLLM bug findings for Hadamard-transformed Vision layers

When loading `transform_config` on `Qwen3_VisionTransformer` in `vllm 0.31.0`, we identified and verified a 2-line fix in `vllm/model_executor/layers/quantization/compressed_tensors/transform/linear.py`:
1. **Input-only transforms with linear bias (`line 159`)**: `CompressedTensorsLinearTransformMethod.apply` asserted `assert bias is None`. Because an input-only transform computes $y = (x H) (W H)^T + b$, the additive bias $b$ is applied *after* the inner product and does not interact with the input transform; only output transforms fail to commute with bias. Relaxing the assertion to `assert bias is None or self.output_transform is None` enables biased Vision linears (`qkv`, `proj`, `linear_fc1`, `linear_fc2`).
2. **Unfused `QKVParallelLinear` partition validation (`line 177`)**: `Qwen2_5_VisionAttention.qkv` is an unfused `QKVParallelLinear` (`num_partitions = 3`, `len(self.input_tfms) == 1` at key `0`). While `_validate_tfm_schemes` already iterated over `self.output_tfms.values()`, its input check iterated over `range(num_partitions)`, raising `KeyError: 1`. Changing `for part_index in range(num_partitions):` to `for part_index in self.input_tfms:` resolves the bug.

---

## Scaling the winning combo (`HGPTQ-g64-h64`) to `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.8-27B`, and `Qwen/Qwen3.6-35B-A3B` (`MoE`)

Following the 2B sweep, we applied the winning `g64-h64` All-W4 Cholesky GPTQ + online block-Hadamard recipe across the rest of the family:
- **`Qwen/Qwen3.5-9B`** (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, `317` linears: `200` Language, `7` Native MTP, `110` Vision)
- **`Qwen/Qwen3.8-27B`** (`8c0b3d30f3ae96f5c7f774de704bf0ada83e422a`, `517` linears: `400` Language, `7` Native MTP, `110` Vision)
- **`Qwen/Qwen3.6-35B-A3B` (`MoE`)** (`995ad96eacd98c81ed38be0c5b274b04031597b0`, `31,732` linears: `30,850` Language = `130` rotated `self_attn` / `linear_attn` + `30,720` `g64` routed experts, `772` Native MTP = `4` rotated `self_attn` + `768` `g64` routed experts, and `110` rotated Vision linears; routers `mlp.gate`, shared experts `mlp.shared_expert.*`, and `mlp.shared_expert_gate` stay BF16). In `35B-A3B`, Hadamard rotation (`transform_config`) is restricted to `LinearBase` layers (`TEXT_HAD_TARGET_RE`, `MTP_HAD_TARGET_RE`, `VISION_TARGET_RE`) while the `31,488` routed expert linears use unrotated `group_size=64` Cholesky GPTQ so they execute directly inside vLLM V1's fused `CompressedTensorsWNA16MoEMethod` (`MarlinExperts` / `MarlinMoE`) kernel.

Because the 9B/27B/35B ViT MLP has `intermediate_size = 4304 = 269 × 16`, the 27 `visual.blocks.*.mlp.linear_fc2` layers (`K = 4304`) automatically use per-channel Cholesky GPTQ (`vision_fc2_channel`) paired with `head_dim = 16` online Hadamard rotation (`v_vision_fc2`), while all other quantized linears use `group_size = 64` + `head_dim = 64`.

| Metric | `Qwen3.5-9B-W4A16-HGPTQ-g64` | `Qwen3.8-27B-W4A16-HGPTQ-g64` | `Qwen3.6-35B-A3B-W4A16-HGPTQ-g64` (`MoE`) |
|---|---:|---:|---:|
| Safetensors payload | **7.6175 GiB (8,179,266,352 B)** (−826.6 MiB vs old HF) | **17.3342 GiB (18,612,467,944 B)** (−801.4 MiB vs old HF) | **19.4191 GiB (20,851,089,304 B)** (**−1,289.5 MiB** vs old HF `20.6784 GiB`) |
| `probe-v4` weightedKL (`378` rows) ↓ | **0.072561** (vs `0.094080` old HF, **−22.9%**) | **0.115968** (vs `0.1369` old HF, **−15.3%**) | **0.088640** *(legacy: **0.076917** vs `0.111` old HF, **−30.7%**)* |
| `probe-v4` token mean KL / p99.9 ↓ | **0.219906 / 18.9820** (vs `0.2759 / 20.42`) | **0.301646 / 24.2052** (vs `0.357 / 24.66`) | **0.230016 / 17.2924** (vs `0.269 / 18.06` old HF, **−14.5%**) |
| `probe-v4` PPL Δ% vs BF16 ↓ | **+1.887%** (vs `+7.411%` old HF) | **−0.292%** (vs `+1.00%` old HF) | **+0.924%** (`4.0230 → 4.0601`, vs `+1.33%` old HF) |
| `probe-guard` token mean KL / PPL Δ% (`76` rows) ↓ | **0.039547 / +1.388%** (vs `0.0645` old HF, **−38.7%**) | **0.063923 / −6.828%** (vs `0.0867` old HF, **−26.3%**) | **0.051031 / +4.296%** (`weightedKL 0.061633`; unmeasured on old HF) |
| Native MTP isolated KL / Top-1 agreement ↑ | **0.012801 / 94.80% (10,981/11,583)** | **0.010930 / 96.22% (11,873/12,339)** | **0.011587 / 95.93% (11,112/11,583)** (`0.916 GiB` vs `1.573 GiB` old HF) |
| Native MTP end-to-end KL / Top-1 agreement ↑ | **0.204901 / 86.81%** | **0.213082 / 87.14%** | **0.122539 / 89.17%** |
| Vision ViT feature Rel-RMSE / Cosine similarity (`48` images) | **0.2114 / 0.9813** | **0.1143 / 0.9940** | **0.2157 / 0.9802** |
| Vision isolated answer KL / Top-1 agreement (`157` tokens) ↑ | **0.007640 / 98.09% (154/157)** | **0.015734 / 97.45% (153/157)** | **0.021799 / 97.45% (153/157)** (`ChartQA 0.002169 / 98.67%`) |
| Vision end-to-end answer KL / Top-1 agreement ↑ | **0.038995 / 94.27%** | **0.267053 / 89.81%** | **0.052438 / 96.82%** (`ChartQA 0.010593 / 100.0%`) |
| vLLM V1 (`v0.31.0`) load memory / VQA smoke (`k=3` MTP) | **7.90 GiB model (9.60 GiB total) / 6/8 exact match** | **17.89 GiB model (19.68 GiB total) / 6/8 exact match** | **19.62 GiB model (22.25 GiB total) / 6/8 exact match (`302.1 toks/s`)** |

### Published Hugging Face Checkpoints

- **2B**: [`nchapman/Qwen3.5-2B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-2B-W4A16-HGPTQ) (`SHA256SUMS` digest: `b6dc3bcf2638bdb998ee922ce755ad570be1f03fb41a3b5587d6ed3cb51e74a4`; see `receipts/g64-h64/hub-SHA256SUMS.txt`)
- **9B**: [`nchapman/Qwen3.5-9B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-9B-W4A16-HGPTQ) (`SHA256SUMS` digest: `0589647f11e5145a54caba42efb54a5aafd82b1652802141b59f4a172ffa3c07`; see `receipts/9b-g64-h64/hub-SHA256SUMS.txt`)
- **27B**: [`nchapman/Qwen3.8-27B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ) (`SHA256SUMS` digest: `bdda0b679feddb5344fe15d624e28503db736b5924a8bc3ebadad2ef2e03806e`; see `receipts/27b-g64-h64/hub-SHA256SUMS.txt`)
- **27B DFlash2 Speculative Drafter**: [`nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ) (`1,338.9 MiB` / `1.31 GiB`; see `dflash2-27b-w4-ablation.json`)
- **35B-A3B (`MoE`)**: [`nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ) (`SHA256SUMS` digest: `cb98ab7a9b41ae09f81673181b7d1f61f1b96c756add13099a51af8745d6a778`; see `receipts/35b-g64-h64/hub-SHA256SUMS.txt`)

See also [`blog-4bit-vs-8bit-and-gguf.md`](blog-4bit-vs-8bit-and-gguf.md) for the companion writeup comparing `HGPTQ-g64` against off-the-shelf 4-bit, mixed 4/8-bit (`RCO-15%`), and Unsloth `Q4_K_M` / `UD-Q4_K_XL` GGUFs.

---

## Reproduction and receipts

All implementation code lives in `quants/rot_experiment.py` and `quants/component_probe_data.py`, covered by 12 unit tests in `quants/test_rot_experiment.py`:

```bash
# Run CPU unit tests
CUDA_VISIBLE_DEVICES="" .venv-quant/bin/python -m unittest quants/test_rot_experiment.py -v

# Materialize disjoint 48-image eval and 64-image calibration fixtures
.venv-quant/bin/python quants/component_probe_data.py \
  --output runs/rot-2b-20261106/vision-data
.venv-quant/bin/python quants/component_probe_data.py \
  --calibration --output runs/rot-2b-20261106/vision-calib

# Quantize All-W4 Cholesky GPTQ + Hadamard (example: g64-h64)
.venv-quant/bin/python quants/rot_experiment.py quantize \
  --model 2b \
  --group-size 64 \
  --hadamard-block-size 64 \
  --quantize-mtp \
  --quantize-vision \
  --component-quant-mode gptq \
  --vision-calib-dir runs/rot-2b-20261106/vision-calib \
  --num-samples 999 \
  --max-seq-length 4096 \
  --sequential-offload-device cuda \
  --output runs/rot-2b-20261106/ckpt-g64-h64

# Run full probe-v4, probe-guard, Native MTP, and Vision evaluation
.venv-quant/bin/python quants/rot_experiment.py probe \
  --model 2b \
  --checkpoint runs/rot-2b-20261106/ckpt-g64-h64 \
  --probe-mtp \
  --probe-vision \
  --vision-data runs/rot-2b-20261106/vision-data \
  --output runs/rot-2b-20261106/receipts/g64-h64/probe.json \
  --rows-dir runs/rot-2b-20261106/receipts/g64-h64/rows
```

Verify all 79 receipt files from `experiments/quants/records/hadamard-w4-2b-20261106/receipts/`:

```bash
(cd experiments/quants/records/hadamard-w4-2b-20261106/receipts && sha256sum -c SHA256SUMS)
```

---

## 9. Follow-Up: Multi-Domain Vision Suite `v2` (12 Domains + 6 OOD Guard Domains + $\sqrt{N_{\text{tokens}}}$ Hessian Weighting)

While `v1` Vision fixtures covered 3 domains (`ChartQA` human/machine + `TextVQA`), `--suite v2` in `quants/component_probe_data.py` expands Vision evaluation and calibration to a 3-split benchmark mirroring the text `calib-cascade-v1` + `probe-v4` + `probe-guard` structure (with zero image SHA-256 overlap across splits):

- **`v2` Primary Probe (`96` images = 12 domains × 8 images)**: `chartqa_human`, `chartqa_machine`, `textvqa`, `docvqa` (full-page scanned documents), `infographic_vqa` (posters/infographics), `funsd_forms` (scanned forms), `kie_receipts_labels` (`SROIE` receipts + `POIE` nutrition labels), `ocrvqa_scene` (book covers + multilingual scene text), and `MMStar` (`mmstar_perception`, `mmstar_reasoning`, `mmstar_math`, `mmstar_science`).
- **`v2` Calibration (`--calibration`, `120` images = 12 domains × 10 images)**: disjoint calibration images from the same 12 domains (`251,312` total ViT patch tokens).
- **`v2` Out-of-Distribution Guard (`--guard`, `48` images = 6 OOD domains × 8 images)**: `ScreenSpot` desktop/mobile GUI screenshots (`screenspot_macos_text`, `screenspot_macos_icon`, `screenspot_windows_text`, `screenspot_windows_icon`, `screenspot_ios`) and `OCRBench` `WordArt` + `HME100k` handwritten/printed math expressions (`ocrbench_artistic_math`).

### Multi-Resolution $\sqrt{N_{\text{tokens}}}$ Hessian Weighting
In `Qwen3_5VisionModel`, each image's patches are passed to `nn.Linear` as a 2D ragged tensor `[N_tokens, K]`. Because standard `accumulate_hessian` weights 2D inputs proportionally to `N_tokens`, a high-resolution `2000×1700` `DocVQA` scan (`~4,000` patches, mostly blank page margin) receives **6× the Hessian weight** of a `512×384` `MMStar` math or reasoning diagram (`~650` patches). Scaling each calibration image's activation tensor by $(1024 / N_{\text{tokens}})^{0.25}$ (`token_exp=0.25`, i.e. $\sqrt{N_{\text{tokens}}}$ per-image Hessian weight) prevents giant document pages from dominating the Hessian while still giving larger pages proportionally higher weight than small crops.

### Measured `Qwen/Qwen3.5-2B` Vision Tower (`98` linears, `W4 g64-h64`) Results (`vision-v2-ablation-2b.json`)

| Eval Split | Vision W4 Arm (`98` linears) | ViT Feature Rel-RMSE ↓ | ViT Feature Cosine ↑ | Isolated Answer Mean KL ↓ | Median KL ↓ | Tail `p99` KL ↓ | Top-1 Agreement ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| **`v2_probe`** (`96` imgs, 12 domains) | `W4_RTN g64-h0` (unrotated) | 0.5354 | 0.8705 | 0.040692 | 0.001571 | 0.9651 | 95.70% |
| | `W4_RTN g64-h64` (data-free) | 0.4907 | 0.8913 | 0.032159 | 0.001421 | 0.5698 | 97.35% |
| | `W4_HGPTQ v1-calib` (64 imgs, ChartQA+TextVQA) | 0.3223 | 0.9525 | 0.031012 | 0.000623 | 0.6517 | 96.36% |
| | `W4_HGPTQ v2-calib` (120 imgs, unweighted) | 0.3112 | 0.9565 | 0.014248 | 0.000445 | 0.2612 | 96.36% |
| | **`W4_HGPTQ v2-calib-sqrt` (120 imgs, $\sqrt{N}$ weight)** | **0.3085** | **0.9570** | **0.010229 (−67.0%)** | **0.000305 (−51.0%)** | **0.1082 (−83.4%)** | **97.35% (294/302)** |
| **`v2_guard`** (`48` OOD UI/Math imgs) | `W4_RTN g64-h0` (unrotated) | 0.4712 | 0.8867 | 0.024853 | 0.000519 | 0.1748 | 97.04% |
| | `W4_HGPTQ v1-calib` (64 imgs) | 0.2734 | 0.9627 | 0.007081 | 0.000144 | 0.1315 | 97.78% |
| | **`W4_HGPTQ v2-calib-sqrt` (120 imgs, $\sqrt{N}$ weight)** | 0.2707 | **0.9637** | **0.004906 (−30.7%)** | 0.000245 | **0.0702 (−46.6%)** | 97.78% |
| | **`W4_HGPTQ v2-calib-eqimg` (120 imgs, equal/img)** | **0.2673** | **0.9638** | **0.002741 (−61.3%)** | **0.000137** | **0.0379 (−71.2%)** | **98.52% (133/135)** |
| **`v1_probe`** (`48` legacy ChartQA+TextVQA) | `W4_HGPTQ v1-calib` (64 imgs) | **0.3332** | **0.9546** | 0.012115 | **0.000496** | **0.1434** | 96.18% |
| | **`W4_HGPTQ v2-calib-sqrt` (120 imgs, $\sqrt{N}$ weight)** | 0.3369 | 0.9521 | 0.012160 | 0.000613 | 0.1544 | **96.82%** |

---

## 10. Follow-Up: Quantizing the 5-Layer `incoai/Qwen3.8-27B-DFlash2` Speculative Drafter (`dflash2-27b-w4-ablation.json`)

We applied the same `W4A16-HGPTQ-g64` recipe to [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (`3,670.5 MiB` / `3.584 GiB` in BF16; 5 `Qwen3DFlash2DecoderLayer` blocks at `hidden_size=5120`, `intermediate_size=17408`, `block_size=8`, `target_layer_ids=[5, 19, 33, 47, 61]`, `GroupedDynamicCausalConv` + bilinear `CandidateSelector`).

- **Calibration**: 96 sequences (`512` tokens each) from `data/calib-cascade-v1.jsonl` using 5-tap hidden states (`[6, 20, 34, 48, 62]`) extracted from `Qwen/Qwen3.8-27B` (`90.6s` calibration time across all 47 `DFlash2` linear modules, including dual-input `k_proj`/`v_proj` hooks over both `target_hidden` and `hidden_states`).
- **Evaluation**: 32 sequences (`512` tokens each, `2,048` 8-token draft blocks = `14,336` drafted tokens) from `data/probe-v4.jsonl`.

| `Qwen3.8-27B-DFlash2` Arm | Size (MiB) | Unary KL vs BF16 ↓ | Median KL ↓ | `p99` KL ↓ | Unary Top-1 Agree ↑ | Selector Top-1 Agree ↑ | Verifier Accept $\tau$ ↑ | Recall@16 ↑ | Pos-1 / Pos-2 / Pos-3 Acc ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`BF16_Reference`** (unquantized) | 3,670.5 | 0.00000 | 0.00000 | 0.0000 | 100.00% | 100.00% | **2.540** | **53.88%** | 56.0% / 38.5% / 29.4% |
| `W4_RTN g64-h0` (45 layer linears) | 1,338.9 | 0.05452 | 0.02412 | 0.4949 | 88.57% | 87.60% | 2.543 | 53.66% | 55.7% / 38.4% / 29.1% |
| `W4_RTN g64-h64` (data-free Hadamard) | 1,338.9 | 0.08127 | 0.05444 | 0.5160 | 88.51% | 87.44% | 2.518 | 53.52% | 56.0% / 38.9% / 29.4% |
| **`W4_HGPTQ g64-h64` (35 linears: `attn+mlp`)** | 1,430.7 | **0.02278 (−58.2%)** | **0.01047** | **0.1872 (−62.2%)** | **92.63%** | **92.06%** | **2.535** | **53.86%** | 56.2% / 39.0% / 29.7% |
| **`W4_HGPTQ g64-h64` (45 linears: `+conv`, sweet spot)** | **1,338.9** | **0.02309 (−57.6%)** | **0.01074** | **0.1876 (−62.1%)** | **92.63%** | **92.03%** | **2.535** | **53.88%** | **56.5% / 38.9% / 29.4%** |
| **`W4_HGPTQ g64-h64` (46 linears: `+conv+fc`)** | **1,155.3** | 0.02546 | 0.01210 | 0.2148 | 92.05% | 91.73% | **2.535** | 53.84% | 56.1% / 38.8% / 29.4% |
| `W4_HGPTQ g64-h64` (47 linears: `+sel_proj`) | 1,153.4 | 0.02546 | 0.01210 | 0.2148 | 92.05% | 91.60% | 2.535 | 53.84% | 56.1% / 38.8% / 29.4% |


