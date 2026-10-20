#!/usr/bin/env bash
# LEGACY: build the Python environment for Azure App Service on Linux.
#
# The deployment target is now Hugging Face Spaces (see the "Deploy
# (free)" section of README.md). This script is kept for reference only.
#
# The Free F1 plan gives 60 CPU minutes a day, and an Oryx build on the
# instance (pip installing onnxruntime, numpy, langgraph ...) burns most of
# that and can exceed the 10 minute deploy window. So build the venv here
# (in a Linux container) and ship it inside the deployment package.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/.deploy/venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
IMAGE="python:${PYTHON_VERSION}-slim"

if [ ! -f "$ROOT/apps/api/requirements-azure.txt" ]; then
  echo "ERROR: apps/api/requirements-azure.txt is missing." >&2
  exit 1
fi

echo "==> Building the venv with $IMAGE (linux/amd64)"
rm -rf "$OUT"
mkdir -p "$OUT"

docker run --rm --platform linux/amd64 \
  -v "$ROOT/apps/api/requirements-azure.txt:/tmp/requirements.txt:ro" \
  -v "$OUT:/out" \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    # --copies so bin/python is a real file, not a symlink into the base image.
    python -m venv --copies /tmp/venv
    /tmp/venv/bin/pip install --no-cache-dir --upgrade pip >/dev/null
    /tmp/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
    # pip is not needed at runtime.
    rm -rf /tmp/venv/lib/python3.12/site-packages/pip \
           /tmp/venv/lib/python3.12/site-packages/pip-*.dist-info
    # Ship libpython with the venv so it does not depend on the base image.
    cp -a /usr/local/lib/libpython3.12.so* /tmp/venv/lib/ 2>/dev/null || true
    # -a keeps symlinks (lib64 -> lib), otherwise the venv doubles in size.
    cp -a /tmp/venv/. /out/venv/
  '

echo "==> venv size: $(du -sh "$OUT" | cut -f1)"
echo "==> architecture: $(file "$OUT/venv/bin/python" | cut -d: -f2- | cut -c1-80)"
