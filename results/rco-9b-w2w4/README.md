# Qwen3.5-9B Sub-3-Bit & 4-Bit Optimization (`All-W2`, `15%–25%` `H-RCO`, and `All-W4`)

Full teacher-forced fidelity receipts (`probe-v4`, `probe-guard`, and Native `MTP`) for `Qwen/Qwen3.5-9B` produced via `warp-quant rco --model 9b`:

```bash
uv run --no-sync warp-quant rco \
  --model 9b \
  --work-dir out/rco-9b-w2w4 \
  --budgets 0.15 0.25 \
  --calib-seq-len 1024 \
  --steps 80 \
  --gumbel-samples 2 \
  --lr 0.25 \
  --assemble all_low 15pct_rco 25pct_rco all_high
```

---

## Summary Table (`Qwen/Qwen3.5-9B`)

| Allocation | Avg Bits | Text Size (`MiB`) | Val KL $\downarrow$ | `probe-v4` `mean_kl` $\downarrow$ | `probe-v4` `weighted_kl` $\downarrow$ | `probe-guard` `mean_kl` $\downarrow$ | `probe-guard` `weighted_kl` $\downarrow$ | `probe-guard` PPL (`10.564` base) | `MTP E2E` `mean_kl` (Top-1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`0%` `All-W2` (`+H-CD+Scale/Norm QAT`)** | **`2.00`** | **`1,854.0`** | **`0.3686`** *(from `0.7625`)* | **`0.5839`** | **`0.4586`** | **`0.5749`** | **`0.7351`** | **`12.3506 (+16.91%)`** | **`0.5626` (`72.17%`)** |
| **`15%` `H-RCO` (`+Scale/Norm Polish`)** | **`2.30`** | **`2,100.0`** | **`0.2868`** *(sens `0.3196`)* | **`0.5311`** | **`0.3788`** | **`0.4737`** | **`0.6091`** | **`11.3191 (+7.15%)`** | **`0.4994` (`73.89%`)** |
| **`25%` `H-RCO` (`+Scale/Norm Polish`)** | **`2.50`** | **`2,266.0`** | **`0.2556`** *(sens `0.3127`)* | **`0.5007`** | **`0.3441`** | **`0.4228`** | **`0.5354`** | **`10.9289 (+3.45%)`** | **`0.4679` (`75.09%`)** |
| **`100%` `All-W4` (`+H-CD+FP32 Norm QAT`)** | **`4.00`** | **`3,502.0`** | **`0.0271`** *(from `0.0329`)* | **`0.2152`** | **`0.0736`** | **`0.0415`** | **`0.0619`** | **`10.5314 (-0.31%)`** | **`0.1700` (`87.78%`)** |
| `100%` Shipped `9B W4A16-HGPTQ` | `4.00` | `3,502.0` | — | `0.2199` | `0.0726` | `0.0395` | `0.0566` | `10.7110 (+1.39%)` | `0.2049` (`86.81%`) |

---

## Per-Domain `probe-v4` `mean_kl` Breakdown (`Qwen/Qwen3.5-9B`)

| Allocation | `math` (`303K`) | `instruction_following` (`40K`) | `science` (`65K`) | `chat` (`218K`) | `conversational_agent` (`73K`) | `terminal_agent` (`55K`) | `swe` (`66K`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0%` `All-W2` (`2.00b`) | `0.2093` | `0.3291` | `0.3007` | `0.3986` | `2.1644` | `1.0454` | `1.1684` |
| `15%` `H-RCO` (`2.30b`) | `0.1517` | `0.2549` | `0.2322` | `0.3058` | `2.2497` | `0.9994` | `1.1633` |
| **`25%` `H-RCO` (`2.50b`)** | **`0.1320`** | **`0.2228`** | **`0.2111`** | **`0.2739`** | **`2.1994`** | **`0.9381`** | **`1.1448`** |
| **`100%` `All-W4` (`+H-CD+Norm QAT`)** | **`0.0160`** | **`0.0230`** | **`0.0345`** | **`0.0221`** | **`1.1438`** | **`0.4707`** | **`0.8739`** |
| `100%` Shipped `9B W4A16-HGPTQ` | `0.0160` | `0.0259` | `0.0383` | `0.0230` | `1.0133` | `0.5261` | `1.0248` |
