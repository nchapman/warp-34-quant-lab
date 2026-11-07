#!/usr/bin/env bash
# Build and install GPU fast-path GatedDeltaNet kernels into .venv:
#   - causal-conv1d==1.6.1.post4 (CUDA extension; built with --no-build-isolation against .venv torch)
#   - flash-linear-attention==0.4.2 (Triton kernels for chunk_gated_delta_rule)
# Without these two packages, Qwen3.5 / 3.6 / 3.8 GDN layers fall back to a
# Python token loop that is ~15x slower during 999x4096 GPTQ calibration.
set -euo pipefail

export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-2}"

uv pip install --python .venv/bin/python --no-build-isolation \
    "causal-conv1d==1.6.1.post4" \
    "flash-linear-attention==0.4.2"

.venv/bin/python -c "import causal_conv1d, fla; print('GDN GPU kernels OK:', causal_conv1d.__file__, fla.__file__)"
