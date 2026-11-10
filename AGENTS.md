# warp-34-quant-lab

`W4A16-HGPTQ` (`g64-h64` Walsh–Hadamard rotation + Cholesky GPTQ + 4-bit `Custom-LUT4` `W_kv`-HGPTQ PLE) quantization and teacher-forced fidelity probing for Qwen3.5 / Qwen3.6 / Qwen3.8 (`0.8B`–`27B` dense, `35B-A3B` MoE, `Qwen3.8-Flash-Next`). Deeper architecture notes and measured receipts live in `README.md` and `results/`.

## Commands

Tooling is managed by `mise` (`.mise.toml`) and `uv` (`pyproject.toml`):

```bash
mise run setup                                       # sync .venv with uv (--prerelease=allow)
mise run setup-kernels                               # compile GPU GDN kernels (causal-conv1d + flash-linear-attention)
mise run check                                       # ruff format --check + ruff check + pytest
mise run fmt                                         # format src/ and tests/ with ruff
mise run test                                        # run full pytest suite
uv run --no-sync pytest tests/test_hadamard_gptq.py -k test_target_regexes  # single test
mise run verify-data                                 # check data/SHA256SUMS
mise run list-models                                 # print model keys, Hub pins, and recipes
mise run build-vision-data                           # materialize data/vision-{calib,probe,guard}-v2
mise run quantize -- --model 35b --output-dir ./out/Qwen3.6-35B-A3B-W4A16-HGPTQ
mise run probe -- --model 35b --quant ./out/Qwen3.6-35B-A3B-W4A16-HGPTQ --eval-mtp \
  --eval-vision data/vision-probe-v2 --eval-vision-guard data/vision-guard-v2
uv run --no-sync warp-quant rco --model 9b --work-dir ./out/rco-9b-w2w4 --budgets 0.15 0.25
```

## Conventions

- **Unified dispatch**: `warp-quant quantize` and `warp-quant probe` (`src/warp_quant/cli.py`) automatically dispatch dense `Qwen3_5ForConditionalGeneration` models (`0.8b`–`27b`) to `src/warp_quant/hadamard_gptq.py` and MoE models (`35b`, `flash-next`) to the streaming layer-by-layer pipeline in `src/warp_quant/streaming_moe.py`.
- **Target regexes (`src/warp_quant/hadamard_gptq.py`)**: Never use `"Linear"` as a `compressed-tensors` target or anchor prefixes to `model.language_model.`. Always use `TEXT_TARGET_RE`, `MTP_TARGET_RE`, `VISION_GROUP_TARGET_RE`, `VISION_FC2_TARGET_RE`, `LM_HEAD_TARGET_RE`, and `MTP_FC_TARGET_RE` so checkpoints load identically in Hugging Face and vLLM (`CompressedTensorsConfig` does not run `apply_vllm_mapper` on `transform_config`).
- **MoE expert & W8 head coverage**: Both routed experts (`mlp.experts.\d+`) and shared experts (`mlp.shared_expert`) in the main model and Native MTP are rotated with $H_{64}$ and quantized with W4A16 Cholesky GPTQ (`group_size=64`, `actorder="static"`). Untied `lm_head` (`tie_word_embeddings=False`) and `mtp.fc` are quantized to unrotated `W8_g64` (`w8_head`, GPTQ+MSE, `actorder=None`) by default (`--quantize-lm-head-w8`). Router gates (`mlp.gate`, `mlp.shared_expert_gate`), GDN scalars (`linear_attn.in_proj_a`, `linear_attn.in_proj_b`), embeddings, norms, and tied `lm_head` stay in BF16.
- **Vision `linear_fc2` (`K=4304`)**: Because $4304 = 16 \times 269$ is not divisible by 32 or 64, `visual.blocks.*.mlp.linear_fc2` uses per-channel Cholesky GPTQ (`vision_fc2_channel`) and a separate `head_dim=16` online Hadamard group (`v_vision_fc2`).
- **Marlin packing invariant**: `_assert_packed_alignment` enforces $N \bmod 64 = 0, K \bmod 16 = 0$ on all packed Text/MTP tensors and $N \bmod 16 = 0, K \bmod 16 = 0$ on Vision tensors before a checkpoint is stamped.
- **GDN GPU fast-path**: Always run `mise run setup-kernels` on a GPU host before calibrating or probing; without `causal_conv1d` and `fla`, GatedDeltaNet layers fall back to a ~15× slower Python loop.

## Boundaries

- **Completion gate**: `mise run check` and `mise run verify-data` must exit `0` before committing.
- **Frozen datasets**: Never modify `data/*.jsonl` or `data/SHA256SUMS`, and never calibrate Vision on `vision-probe-v2` (`calibration=false`) or evaluate Vision on `vision-calib-v2` (`calibration=true`) — the pipeline enforces these manifest guards at runtime.
- **Untracked outputs**: Never commit `.venv/`, `runs/`, `out/`, materialized image blobs (`data/vision-*-v2/`), or `.safetensors` weights.
