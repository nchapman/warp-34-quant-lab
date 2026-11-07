---
license: apache-2.0
library_name: compressed-tensors
pipeline_tag: text-generation
base_model:
  - incoai/Qwen3.8-27B-DFlash2
  - Qwen/Qwen3.8-27B
inference: false
tags:
  - dflash2
  - speculative-decoding
  - block-diffusion
  - draft-model
  - compressed-tensors
  - w4a16
  - gptq
  - hadamard
---

# Qwen3.8-27B-DFlash2-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (the 5-layer block-diffusion speculative drafter for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and [`nchapman/Qwen3.8-27B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ)).

## Quality vs. BF16 Base Model (`incoai/Qwen3.8-27B-DFlash2`)

Evaluated across `2,048` 8-token draft blocks (`14,336` drafted tokens) against the BF16 drafter and target verifier:

| Metric | BF16 Base (`incoai/Qwen3.8-27B-DFlash2`) | **`Qwen3.8-27B-DFlash2-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `3.58 GiB` (`3,670.5 MiB`) | **`1.31 GiB` (`1,338.9 MiB`)** | **`2.74×` smaller** (`−2.27 GiB` VRAM saved) |
| **Streamed Decoder Weights / Draft Step** | `3,175 MiB` | **`843 MiB`** | **`3.76×` less memory bandwidth** per 8-token draft step |
| **Verifier Acceptance Length ($\tau$)** ↑ | `2.540` | **`2.535`** | **99.8%** of BF16 acceptance length preserved |
| **Top-16 Candidate Recall (`Recall@16`)** ↑ | `53.88%` | **`53.88%`** | **100.0%** of BF16 candidate recall preserved |
| **Position-1 / Position-2 / Position-3 Accuracy** ↑ | `56.0% / 38.5% / 29.4%` | **`56.5% / 38.9% / 29.4%`** | Matches BF16 across draft positions |
| **Draft Token Agreement vs. BF16** (Unary / Selector Top-1) ↑ | `100.0% / 100.0%` | **`92.63% / 92.03%`** | Unary mean KL `0.0231` (median KL `0.0107`) |

## What Is Quantized

- **Quantized (`45` linear layers, `W4A16-HGPTQ`, `group_size=64`, `hadamard_block_size=64`, `actorder="static"`)**:
  - `layers.{0..4}.self_attn.{q_proj,k_proj,v_proj,o_proj}` (`20` matrices)
  - `layers.{0..4}.mlp.{gate_proj,up_proj,down_proj}` (`15` matrices)
  - `layers.{0..4}.{attention_conv,mlp_conv}.kernel_projection` (`10` matrices)
- **Kept in BF16**:
  - `fc.weight` (`[5120, 25600]`, 5-tap target hidden state fusion projection)
  - `candidate_selector.{predecessor_codebook,successor_codebook,hidden_projection.weight}`
  - `layers.{0..4}.{attention_conv,mlp_conv}.base_kernel` (`[2, 2, 5120]`) and all RMSNorm weights (`hidden_norm`, `norm`, `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`)

## Attribution & Citation

Derived from [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (Apache-2.0). Please cite the original DFlash 2 and DFlash works:

```bibtex
@misc{inco2026dflash2,
  title  = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}

@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
