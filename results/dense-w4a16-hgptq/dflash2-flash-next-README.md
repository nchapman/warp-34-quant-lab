---
license: apache-2.0
library_name: compressed-tensors
pipeline_tag: text-generation
base_model:
  - incoai/Qwen3.8-Flash-Next-DFlash2
  - Qwen/Qwen3.8-Flash-Next
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

# Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (the 5-layer block-diffusion speculative drafter for [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and [`nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ)).

> **Summary:** Shrinks the 5-layer `DFlash2` block-diffusion drafter `2.2×` (`1,150 MiB → 531 MiB`) and cuts the decoder weights read from memory on every 8-token draft step by `3.8×` (`844 MiB → 224 MiB`). Preserves `99.7%` of the BF16 drafter's acceptance length (`2.760 → 2.752` tokens/step) and `99.9%` of its top-16 candidate recall (`59.67% → 59.60%`).

## Quality vs. BF16 Base Model (`incoai/Qwen3.8-Flash-Next-DFlash2`)

Evaluated across `2,048` 8-token draft blocks (`14,336` drafted tokens) against the BF16 drafter and target verifier:

| Metric | BF16 Base (`incoai/Qwen3.8-Flash-Next-DFlash2`) | **`Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `1.12 GiB` (`1,150.3 MiB`) | **`0.52 GiB` (`530.6 MiB`)** | **`2.17×` smaller** (`−619.6 MiB` VRAM saved) |
| **Streamed Decoder Weights / Draft Step** | `843.8 MiB` | **`224.1 MiB`** | **`3.76×` less memory bandwidth** per 8-token draft step |
| **Verifier Acceptance Length ($\tau$)** ↑ | `2.760` | **`2.752`** | **99.7%** of BF16 acceptance length preserved |
| **Top-16 Candidate Recall (`Recall@16`)** ↑ | `59.67%` | **`59.60%`** | **99.9%** of BF16 candidate recall preserved |
| **Position-1 / Position-2 / Position-3 Accuracy** ↑ | `61.7% / 45.6% / 36.5%` | **`61.8% / 45.6% / 36.5%`** | Matches BF16 across draft positions |
| **Draft Token Agreement vs. BF16** (Unary / Selector Top-1) ↑ | `100.0% / 100.0%` | **`92.23% / 91.47%`** | Unary mean KL `0.0276` (median KL `0.0121`) |

## What Is Quantized

- **Quantized (`45` linear layers, `W4A16-HGPTQ`, `group_size=64`, `hadamard_block_size=64`, `actorder="static"`)**:
  - `layers.{0..4}.self_attn.{q_proj,k_proj,v_proj,o_proj}` (`20` matrices)
  - `layers.{0..4}.mlp.{gate_proj,up_proj,down_proj}` (`15` matrices)
  - `layers.{0..4}.{attention_conv,mlp_conv}.kernel_projection` (`10` matrices)
- **Kept in BF16**:
  - `fc.weight` (`[2560, 12800]`, 5-tap target hidden state fusion projection over `target_layer_ids=[5, 15, 25, 35, 45]` averaged across the 4 hyper-connection streams)
  - `candidate_selector.{predecessor_codebook,successor_codebook,hidden_projection.weight}`
  - `layers.{0..4}.{attention_conv,mlp_conv}.base_kernel` (`[2, 2, 2560]`) and all RMSNorm weights (`hidden_norm`, `norm`, `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`)

## Attribution & Citation

Derived from [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (Apache-2.0). Please cite the original DFlash 2 and DFlash works:

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
