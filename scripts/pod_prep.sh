#!/usr/bin/env bash
# scripts/pod_prep.sh — Prepare a remote RunPod GPU pod for warp-34-quant-lab.
#
# Usage:
#   bash scripts/pod_prep.sh <ssh_target> [remote_dir]
#   SSH_PORT=12345 bash scripts/pod_prep.sh root@1.2.3.4 /workspace/warp-34-quant-lab
#
# Steps (idempotent):
#   1. Survey GPU, CUDA, disk, RAM, and install uv if missing
#   2. Ship HF_TOKEN via stdin to /root/.cache/huggingface/token (0600)
#   3. Rsync repo (excluding .venv, runs, out, .git, caches, weights)
#   4. Verify frozen data checksums (data/SHA256SUMS)
#   5. Sync .venv (uv sync --prerelease=allow) + compile GDN GPU kernels
#   6. Run pytest + CUDA BF16 matmul + GDN dispatch smoke test
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <ssh_target> [remote_dir]" >&2
    exit 1
fi

SSH_TARGET="$1"
REMOTE_DIR="${2:-/workspace/warp-34-quant-lab}"
SSH_PORT="${SSH_PORT:-}"
SSH_KEY="${SSH_KEY:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
if [[ -n "$SSH_PORT" ]]; then
    SSH_OPTS+=(-p "$SSH_PORT")
fi
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS+=(-i "$SSH_KEY")
fi

rssh() {
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"
}

echo "=== [1/6] Surveying $SSH_TARGET ==="
rssh "bash -se" "$REMOTE_DIR" <<'REMOTE'
set -euo pipefail
REMOTE_DIR="$1"
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
mkdir -p "$REMOTE_DIR"
df -h "$REMOTE_DIR"
REMOTE

echo "=== [2/6] Shipping HF_TOKEN via stdin ==="
HF_TOKEN_VAL="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN_VAL" && -f "$HOME/.config/secrets.env" ]]; then
    HF_TOKEN_VAL="$(grep -E '^HF_TOKEN=' "$HOME/.config/secrets.env" | head -n1 | cut -d= -f2- || true)"
fi
if [[ -z "$HF_TOKEN_VAL" && -f "$HOME/.cache/huggingface/token" ]]; then
    HF_TOKEN_VAL="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
fi
if [[ -n "$HF_TOKEN_VAL" ]]; then
    printf '%s' "$HF_TOKEN_VAL" | rssh "umask 077 && mkdir -p ~/.cache/huggingface && cat > ~/.cache/huggingface/token"
    echo "HF_TOKEN installed at ~/.cache/huggingface/token (0600)"
else
    echo "WARNING: No local HF_TOKEN found; skipping token upload" >&2
fi

echo "=== [3/6] Syncing warp-34-quant-lab -> $SSH_TARGET:$REMOTE_DIR ==="
RSYNC_RSH="ssh ${SSH_OPTS[*]}"
rsync -az --delete \
    -e "$RSYNC_RSH" \
    --exclude='.venv/' \
    --exclude='runs/' \
    --exclude='out/' \
    --exclude='.git/' \
    --exclude='.pi/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='*.safetensors' \
    --exclude='data/vision-*-v2/' \
    "$REPO_ROOT/" "$SSH_TARGET:$REMOTE_DIR/"

echo "=== [4/6] Verifying frozen dataset SHA-256 checksums ==="
rssh "cd '$REMOTE_DIR/data' && sha256sum -c SHA256SUMS"

echo "=== [5/6] Syncing .venv & compiling GDN GPU kernels ==="
rssh "bash -se" "$REMOTE_DIR" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
cd "$1"
uv sync --prerelease=allow
bash scripts/install_gpu_kernels.sh
REMOTE

echo "=== [6/6] Running remote test suite & GPU smoke test ==="
rssh "bash -se" "$REMOTE_DIR" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
cd "$1"
uv run --no-sync pytest -q
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
c = a @ a.T
torch.cuda.synchronize()
assert torch.isfinite(c).all()
print(f"Pod ready: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GiB VRAM)")
PY
REMOTE
