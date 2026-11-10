# Qwen3.8-27B Sub-3-Bit & 4-Bit `H-RCO` + `W8_g64` Head Pareto Sweep (`0%`, `10%–30%` `W2/W4`, and `100%` `All-W4`)

Full teacher-forced fidelity receipts (`probe-v4`, `probe-guard`, Native `MTP`, `vision-probe-v2`, and `vision-guard-v2`) for `Qwen/Qwen3.8-27B` calibrated on `data/calib-cascade-v1.jsonl` (`512×4096` bank, `935`-sequence E2E Scale-QAT, `64×2048` Riemannian search, `data/vision-calib-v2` Vision, and unrotated `W8_g64` `GPTQ+MSE` on untied `lm_head` + `mtp.fc`):

```bash
warp-quant rco \
  --model 27b \
  --work-dir out/rco-27b-h200 \
  --options w2_g64 w4_g64 \
  --budgets 0.10 0.20 0.25 0.30 \
  --calib-samples 512 \
  --calib-seq-len 4096 \
  --search-samples 64 \
  --search-seq-len 2048 \
  --steps 100 \
  --gumbel-samples 2 \
  --lr 0.25 \
  --quantize-mtp \
  --quantize-vision \
  --quantize-lm-head-w8 \
  --vision-calib-dir data/vision-calib-v2 \
  --scale-qat \
  --scale-qat-samples 935 \
  --assemble all_low 10pct_rco 20pct_rco 25pct_rco 30pct_rco all_high
```

---

## Summary Table (`Qwen/Qwen3.8-27B`, `512×4096` `calib-cascade-v1`)

| Allocation | Avg Bits | Text Size (`MiB`) | Total Checkpoint (`GiB` / `GB`) | Val KL $\downarrow$ | `probe-v4` `mean_kl` $\downarrow$ | `probe-v4` `weighted_kl` $\downarrow$ | `probe-guard` `mean_kl` $\downarrow$ | `probe-guard` PPL (`61.08` base) | `MTP E2E` `mean_kl` (Top-1) | `Vision E2E` `mean_kl` (Top-1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`0%` `All-W2` (`+H-CD+Scale/Norm QAT+W8 Head`)** | **`2.00`** | **`6,525.0`** | **`10.48` / `11.25`** | **`0.3827`** *(from `0.9224`)* | **`0.4920`** | **`0.3477`** | **`0.4150`** | **`26.46 (-56.68%)`** | **`0.7701` (`71.13%`)** | **`0.5982` (`84.44%`)** |
| **`10%` `H-RCO` (`+Scale/Norm Polish+W8 Head`)** | **`2.20`** | **`7,105.0`** | **`11.04` / `11.86`** | **`0.3539`** *(sens `0.3732`)* | **`0.4597`** | **`0.3137`** | **`0.3835`** | **`24.51 (-59.88%)`** | **`0.7131` (`72.10%`)** | **`0.5599` (`86.09%`)** |
| **`20%` `H-RCO` (`+Scale/Norm Polish+W8 Head`)** | **`2.40`** | **`7,685.0`** | **`11.61` / `12.46`** | **`0.3343`** *(sens `0.3688`)* | **`0.4411`** | **`0.2931`** | **`0.3758`** | **`23.67 (-61.25%)`** | **`0.6823` (`73.17%`)** | **`0.5090` (`88.74%`)** |
| **`25%` `H-RCO` (`+Scale/Norm Polish+W8 Head`)** | **`2.50`** | **`7,975.0`** | **`11.89` / `12.77`** | **`0.3266`** *(sens `0.3669`)* | **`0.4349`** | **`0.2827`** | **`0.3409`** | **`23.87 (-60.91%)`** | **`0.6797` (`73.43%`)** | **`0.5448` (`88.74%`)** |
| **`30%` `H-RCO` (`+Scale/Norm Polish+W8 Head`)** | **`2.60`** | **`8,265.0`** | **`12.18` / `13.07`** | **`0.3214`** *(sens `0.3798`)* | **`0.4265`** | **`0.2717`** | **`0.3272`** | **`23.33 (-61.81%)`** | **`0.6399` (`74.47%`)** | **`0.4969` (`88.74%`)** |
| **`100%` `All-W4` (`+MSE+H-CD+FP32 Norm QAT+W8 Head`)** | **`4.00`** | **`12,325.0`** | **`16.14` / `17.33`** | **`0.2083`** *(post-QAT `0.2293`)* | **`0.2583`** | **`0.1002`** | **`0.0530`** | **`56.67 (-7.23%)`** | **`0.1782` (`89.16%`)** | **`0.1808` (`92.05%`)** |
| `100%` Shipped `27B W4A16-HGPTQ` (`BF16` head) | `4.00` | `12,328.0` | `17.61` / `18.91` | — | `0.3016` | `0.1160` | `0.0639` | `56.20 (-6.83%)` | `0.2131` (`87.14%`) | `0.2671` (`89.81%`) |

---

## Per-Domain `probe-v4` `mean_kl` Breakdown (`Qwen/Qwen3.8-27B`)

| Allocation | `math` (`303K`) | `instruction_following` (`40K`) | `science` (`65K`) | `chat` (`218K`) | `conversational_agent` (`73K`) | `terminal_agent` (`55K`) | `swe` (`66K`) | `belebele` (`9`-lang macro) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0%` `All-W2` (`2.00b`) | `0.2304` | `0.2402` | `0.2450` | `0.2583` | `2.3590` | `0.3637` | `0.9172` | `0.5032` |
| `10%` `H-RCO` (`2.20b`) | `0.2051` | `0.2112` | `0.2150` | `0.2213` | `2.3177` | `0.3453` | `0.8656` | `0.4647` |
| **`20%` `H-RCO` (`2.40b`)** | **`0.1896`** | **`0.1904`** | **`0.1929`** | **`0.1955`** | **`2.2880`** | **`0.3413`** | **`0.8542`** | **`0.4645`** |
| **`25%` `H-RCO` (`2.50b`)** | **`0.1859`** | **`0.1893`** | **`0.1871`** | **`0.1873`** | **`2.2993`** | **`0.3354`** | **`0.8372`** | **`0.4151`** |
| **`30%` `H-RCO` (`2.60b`)** | **`0.1795`** | **`0.1764`** | **`0.1803`** | **`0.1744`** | **`2.2584`** | **`0.3437`** | **`0.8563`** | **`0.4070`** |
| **`100%` `All-W4` (`+MSE+H-CD+Norm QAT+W8 Head`)** | **`0.0718`** | **`0.0354`** | **`0.0607`** | **`0.0223`** | **`1.6902`** | **`0.2833`** | **`0.7075`** | **`0.0628`** |
| `100%` Shipped `27B W4A16-HGPTQ` | `0.0812` | `0.0411` | `0.0710` | `0.0262` | `1.8860` | `0.3785` | `0.8918` | `0.0815` |

---

## Multimodal & Speculative Decoding Receipts (`MTP` + `Vision`)

| Allocation | `MTP` Isolated `KL` / Top-1 | `MTP` E2E `KL` / Top-1 | `vision-probe-v2` Isolated `KL` / Top-1 | `vision-probe-v2` E2E `KL` / Top-1 | `vision-guard-v2` E2E `KL` / Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0%` `All-W2` (`2.00b`) | `0.0090` / `96.64%` | `0.7701` / `71.13%` | `0.0056` / `97.68%` | `0.5982` / `84.44%` | `0.6342` / `85.19%` |
| `10%` `H-RCO` (`2.20b`) | `0.0090` / `96.64%` | `0.7131` / `72.10%` | `0.0056` / `97.68%` | `0.5599` / `86.09%` | `0.5783` / `85.19%` |
| **`20%` `H-RCO` (`2.40b`)** | **`0.0090` / `96.64%`** | **`0.6823` / `73.17%`** | **`0.0056` / `97.68%`** | **`0.5090` / `88.74%`** | **`0.5642` / `85.93%`** |
| **`25%` `H-RCO` (`2.50b`)** | **`0.0090` / `96.64%`** | **`0.6797` / `73.43%`** | **`0.0056` / `97.68%`** | **`0.5448` / `88.74%`** | **`0.5613` / `87.41%`** |
| **`30%` `H-RCO` (`2.60b`)** | **`0.0090` / `96.64%`** | **`0.6399` / `74.47%`** | **`0.0056` / `97.68%`** | **`0.4969` / `88.74%`** | **`0.5311` / `88.15%`** |
| **`100%` `All-W4` (`4.00b`)** | **`0.0090` / `96.64%`** | **`0.1782` / `89.16%`** | **`0.0056` / `97.68%`** | **`0.1808` / `92.05%`** | **`0.0856` / `94.81%`** |

---

## Structural `H-RCO` Allocation Pattern (`256` Fused vLLM Linear Groups)

| Budget | Realized Bits | Group Swaps vs Sensitivity | Full-Attention Groups (`32`) | GDN Linear-Attention Groups (`96`) | MLP Groups (`128`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `10%` | `2.200b` | `54` | `13 / 32` (`40.6%`) | `20 / 96` (`20.8%`) | `14 / 128` (`10.9%`) |
| **`20%`** | **`2.400b`** | **`85`** | **`15 / 32` (`46.9%`)** | **`32 / 96` (`33.3%`)** | **`28 / 128` (`21.9%`)** |
| **`25%`** | **`2.500b`** | **`98`** | **`21 / 32` (`65.6%`)** | **`36 / 96` (`37.5%`)** | **`33 / 128` (`25.8%`)** |
| **`30%`** | **`2.600b`** | **`110`** | **`21 / 32` (`65.6%`)** | **`38 / 96` (`39.6%`)** | **`39 / 128` (`30.5%`)** |

- **`W8_g64` (`GPTQ+MSE`) head ablation**: Quantizing untied `lm_head` (`[248320, 5120]`) and `mtp.fc` (`[5120, 10240]`) from `BF16` to unrotated `W8_g64` saves **`1.202 GiB` (`1,230.8 MiB`) per checkpoint** while shifting `all_low` `probe-v4` `weighted_kl` by **`+0.000011`** (`0.347698 → 0.347709`) and `probe-guard` `mean_kl` by **`+0.000025`** (`0.414982 → 0.415007`).
- **`All-W4` (`16.14 GiB` vs `17.61 GiB` shipped)**: Combining MSE-initialized GPTQ + 2-pass Gauss–Seidel H-CD + FP32-master RMSNorm QAT + `W8_g64` head reduces total checkpoint footprint by **`1.47 GiB`** while cutting `probe-v4` `weighted_kl` by **`13.6%`** (`0.1160 → 0.1002`), `probe-guard` `mean_kl` by **`17.1%`** (`0.0639 → 0.0530`), `swe` KL by **`20.7%`** (`0.8918 → 0.7075`), and lifting E2E `MTP` (`87.14% → 89.16%`) and `Vision` (`89.81% → 92.05%`) Top-1 agreement.
- **Consumer VRAM fit**: **`30% H-RCO` (`2.60b`, `12.18 GiB` full VLM+MTP)** fits `16 GiB` consumer GPUs (`RTX 4060 Ti 16GB` / `4070 Ti Super` / `5060 Ti 16GB`) with `~3.8 GiB` headroom for KV cache + CUDA graphs, while **`10%–20% H-RCO` (`10.56–11.12 GiB` text-only)** fits `12 GiB` GPUs.
