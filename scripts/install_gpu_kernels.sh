#!/usr/bin/env bash
# Build and install GPU fast-path GatedDeltaNet kernels into .venv:
#   - flash-linear-attention==0.6.0 + fla-core==0.6.0 (Triton kernels for chunk_gated_delta_rule)
#   - causal-conv1d==1.7.0 (CUDA extension; built with --no-build-isolation against .venv torch)
#
# Upstream causal-conv1d setup.py hardcodes 9 SM targets (sm_53..sm_121) and ignores
# TORCH_CUDA_ARCH_LIST. When a CUDA GPU is present, we patch setup.py in a temporary
# sdist checkout to compile only for the active GPU's SM architecture (~90s vs ~20m).
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-2}"

uv pip install --python "$PYTHON" "flash-linear-attention==0.6.0" "fla-core==0.6.0" "ninja>=1.11" "packaging>=24"

TMP_BUILD="$(mktemp -d)"
trap 'rm -rf "$TMP_BUILD"' EXIT

"$PYTHON" - "$TMP_BUILD" <<'PY'
import pathlib, re, subprocess, sys, tarfile, urllib.request, json, torch

tmp_dir = pathlib.Path(sys.argv[1])
url = "https://pypi.org/pypi/causal-conv1d/1.7.0/json"
with urllib.request.urlopen(url, timeout=30) as resp:
    meta = json.load(resp)
sdist = next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist")
tar_path = tmp_dir / "causal_conv1d.tar.gz"
urllib.request.urlretrieve(sdist, tar_path)
with tarfile.open(tar_path) as tf:
    tf.extractall(tmp_dir)

src_dir = next(p for p in tmp_dir.iterdir() if p.is_dir() and p.name.startswith("causal_conv1d"))
setup_py = src_dir / "setup.py"
text = setup_py.read_text()

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"{major}{minor}"
    hook = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"
    patch = (
        f'    if not HIP_BUILD:\n'
        f'        cc_flag = ["-gencode", "arch=compute_{sm},code=sm_{sm}"]\n\n'
        f"{hook}"
    )
    if hook not in text:
        raise RuntimeError("Unexpected setup.py structure in causal-conv1d==1.7.0")
    text = (
        "import torch.utils.cpp_extension as _ce\n"
        "_ce._check_cuda_version = lambda *a, **kw: None\n"
        + text.replace(hook, patch, 1).replace(
            "check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)", "None"
        )
    )
    setup_py.write_text(text)
    print(f"Patched causal-conv1d==1.7.0 setup.py for single target sm_{sm}", flush=True)

subprocess.check_call(
    ["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", str(src_dir)]
)
PY

"$PYTHON" - <<'PY'
import causal_conv1d
import fla
import torch
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

print("GDN GPU kernels imported:", causal_conv1d.__file__, fla.__file__)
if torch.cuda.is_available():
    x = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16)
    y = causal_conv1d_fn(x, w, None, activation="silu")
    q = torch.randn(1, 64, 2, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 64, 2, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 64, 2, 16, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, 64, 2, device="cuda", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(1, 64, 2, device="cuda", dtype=torch.bfloat16))
    out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False)
    assert torch.isfinite(y).all() and torch.isfinite(out).all()
    print(f"GDN GPU dispatch verified on {torch.cuda.get_device_name(0)} (causal_conv1d + fla chunk_gated_delta_rule)")
PY
