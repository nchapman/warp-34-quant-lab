# Qwen3.5-2B Sub-3-Bit & 4-Bit Optimization (`All-W2`, `All-W4`, and `10%–30%` `H-RCO`)

Full teacher-forced fidelity receipts (`probe-v4`, `probe-guard`, and Native `MTP`) for `Qwen/Qwen3.5-2B` across:
1. **Pure 2-Bit (`All-W2-g64-h64`, `2.00` bits, `367.9 MiB` text body):**
   - `all_w2_hcd`: $H_{64}$ block Walsh–Hadamard rotation + MSE grid initialization + Cholesky GPTQ (`actorder="static"`, `group_size=64`) + 4-pass Gauss–Seidel Hessian Coordinate Descent (`refine_hessian_cd`).
   - `all_w2_scale_qat`: Adds 2 epochs (`935` non-overlapping calibration sequences $\times$ `1024` tokens) of end-to-end continuous `weight_scale` + FP32-master `RMSNorm`/GDN-gate teacher-forced KL tuning (`run_e2e_scale_qat`) with integer grids frozen.
2. **Pure 4-Bit (`All-W4-g64-h64`, `4.00` bits, `694.9 MiB` text body):**
   - `all_w4_hcd`: $H_{64}$ + MSE init + Cholesky GPTQ + 2-pass Gauss–Seidel `H-CD`.
   - `all_w4_scale_qat`: Adds 1 epoch (`935` sequences) of FP32-master `RMSNorm` KL tuning (`train_scales_max_bits=2`, keeping 4-bit `H-CD` scales frozen to avoid BF16 ULP jitter).
3. **`10%–30%` Mixed `W2/W4` Riemannian Constrained Optimization (`H-RCO`, `2.20–2.60` bits, `400.4–465.9 MiB` text body):**
   - Discrete `[w2_g64, w4_g64]` allocation solved via Gumbel-Softmax + Straight-Through Estimator + exact Newton-bisection KL projection (`run_rco_search_on_bank`) on `bank_opt`, followed by a 1-epoch (`512`-sequence) post-search scale/norm polish.

---

## Summary Table (`Qwen/Qwen3.5-2B`)

| Allocation | Avg Bits | Text Size (`MiB`) | Val KL $\downarrow$ | `probe-v4` `mean_kl` $\downarrow$ | `probe-v4` `weighted_kl` $\downarrow$ | `probe-v4` PPL (`3.837` base) | `probe-guard` `mean_kl` $\downarrow$ | `probe-guard` PPL (`12.784` base) | `MTP E2E` `mean_kl` (Top-1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0%` `All-W2` (`H-CD` only) | `2.00` | `367.88` | `1.0151` | `1.1151` | `1.1794` | `7.3674 (+92.00%)` | `1.8165` | `59.4063 (+364.7%)` | `1.2073` (`56.09%`) |
| **`0%` `All-W2` (`+Scale/Norm QAT`)** | **`2.00`** | **`367.88`** | **`0.5397`** | **`0.6199`** | **`0.6313`** | **`4.8780 (+27.13%)`** | **`0.9449`** | **`23.3827 (+82.91%)`** | **`0.8931` (`64.54%`)** |
| `10%` `H-RCO` (`+Polish`) | `2.20` | `400.38` | `0.4222` | `0.5257` | `0.5160` | `4.5003 (+17.28%)` | `0.7026` | `20.8919 (+63.42%)` | `0.7347` (`66.23%`) |
| `15%` `H-RCO` (`+Polish`) | `2.30` | `416.88` | `0.4024` | `0.5067` | `0.4922` | `4.3880 (+14.36%)` | `0.6675` | `19.4573 (+52.20%)` | `0.6951` (`67.03%`) |
| `20%` `H-RCO` (`+Polish`) | `2.40` | `432.88` | `0.3638` | `0.4785` | `0.4525` | `4.2635 (+11.11%)` | `0.5923` | `18.3081 (+43.21%)` | `0.6673` (`67.79%`) |
| **`25%` `H-RCO` (`+Polish`)** | **`2.50`** | **`449.38`** | **`0.3389`** | **`0.4540`** | **`0.4157`** | **`4.1368 (+7.81%)`** | **`0.5626`** | **`18.8913 (+47.77%)`** | **`0.6435` (`68.39%`)** |
| **`30%` `H-RCO` (`+Polish`)** | **`2.60`** | **`465.88`** | **`0.3118`** | **`0.4286`** | **`0.3834`** | **`4.0760 (+6.22%)`** | **`0.4969`** | **`18.0182 (+40.94%)`** | **`0.5946` (`69.14%`)** |
| `100%` Shipped `W4A16-HGPTQ` | `4.00` | `694.88` | `0.0415` | `0.2543` | `0.0728` | `3.8942 (+1.67%)` | `0.0572` | `12.9914 (+1.64%)` | `0.4430` (`80.49%`) |
| `100%` `All-W4` (`+2-pass H-CD`) | `4.00` | `694.88` | `0.0370` | `0.2547` | `0.0672` | `4.0438 (+5.38%)` | `0.0493` | `13.0451 (+2.04%)` | `0.3791` (`81.21%`) |
| **`100%` `All-W4` (`+H-CD+Norm QAT`)** | **`4.00`** | **`694.88`** | **`0.0348`** | **`0.2010`** | **`0.0581`** | **`3.7694 (-1.77%)`** | **`0.0482`** | **`12.9352 (+1.18%)`** | **`0.3158` (`81.52%`)** |

---

## Per-Domain `probe-v4` `mean_kl` Breakdown

| Allocation | `math` (`303K`) | `instruction_following` (`40K`) | `science` (`65K`) | `chat` (`218K`) | `conversational_agent` (`72K`) | `terminal_agent` (`53K`) | `swe` (`66K`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0%` `All-W2` (`H-CD` only) | `0.7469` | `0.9001` | `0.8588` | `1.1243` | `1.4529` | `1.5400` | `1.8358` |
| **`0%` `All-W2` (`+Scale/Norm QAT`)** | **`0.3916`** | **`0.4658`** | **`0.4652`** | **`0.6120`** | **`0.7859`** | **`0.9712`** | **`1.3115`** |
| `10%` `H-RCO` (`2.20b`) | `0.3282` | `0.3788` | `0.3772` | `0.4879` | `0.6695` | `0.8531` | `1.2019` |
| `15%` `H-RCO` (`2.30b`) | `0.3059` | `0.3554` | `0.3638` | `0.4648` | `0.6754` | `0.8224` | `1.2255` |
| `20%` `H-RCO` (`2.40b`) | `0.2810` | `0.3292` | `0.3358` | `0.4316` | `0.6303` | `0.7878` | `1.2442` |
| `25%` `H-RCO` (`2.50b`) | `0.2583` | `0.3079` | `0.3100` | `0.4068` | `0.6020` | `0.7534` | `1.2416` |
| **`30%` `H-RCO` (`2.60b`)** | **`0.2404`** | **`0.2900`** | **`0.2847`** | **`0.3800`** | **`0.5694`** | **`0.6981`** | **`1.2384`** |
| `100%` Shipped `W4A16-HGPTQ` | `0.0303` | `0.0321` | `0.0367` | `0.0399` | `0.3240` | `0.7395` | `1.9242` |
| `100%` `All-W4` (`+2-pass H-CD`) | `0.0246` | `0.0289` | `0.0327` | `0.0369` | `0.2896` | `0.6683` | `2.0731` |
| **`100%` `All-W4` (`+H-CD+Norm QAT`)** | **`0.0229`** | **`0.0274`** | **`0.0307`** | **`0.0347`** | **`0.2729`** | **`0.4358`** | **`1.6174`** |
