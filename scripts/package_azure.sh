#!/usr/bin/env bash
# Build the Azure App Service deployment package for the crop-advisor API.
#
# The package is a zip that Oryx builds on the server:
#   - apps/, agents/, mcp_servers/, rag/, speech/, strings/  (the code)
#   - ml/export_edge/artifacts/                            (the ONNX model)
#   - requirements.txt                                     (the Azure variant)
#
# No torch/torchvision: the classifier runs on ONNX Runtime, which fits the
# 1GB memory of a Free F1 plan.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-$ROOT/.deploy/stage}"
ZIP="${ZIP:-$ROOT/.deploy/crop-advisor-api.zip}"
ONNX_DIR="$ROOT/ml/export_edge/artifacts"

if [ ! -f "$ONNX_DIR/classifier.onnx" ]; then
  echo "ERROR: $ONNX_DIR/classifier.onnx is missing." >&2
  echo "       Run ml/export_edge/export_onnx.py first." >&2
  exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/apps/api" "$STAGE/ml/export_edge/artifacts"

# Code
cp -R "$ROOT/agents" "$STAGE/agents"
cp -R "$ROOT/mcp_servers" "$STAGE/mcp_servers"
cp -R "$ROOT/rag" "$STAGE/rag"
cp -R "$ROOT/speech" "$STAGE/speech"
cp -R "$ROOT/strings" "$STAGE/strings"
cp "$ROOT/apps/api/main.py" "$ROOT/apps/api/config.py" "$ROOT/apps/api/__init__.py" "$STAGE/apps/api/"

# Model (ONNX - no torch needed at runtime)
cp "$ONNX_DIR/classifier.onnx" "$ONNX_DIR/classes.json" "$STAGE/ml/export_edge/artifacts/"

# Azure dependency set
cp "$ROOT/apps/api/requirements-azure.txt" "$STAGE/requirements.txt"

# App Service runs the app from the package root, so expose the ASGI app in a
# top-level module for the Oryx default startup command.
cat > "$STAGE/application.py" <<'PY'
"""App Service entry point: the default Linux startup command runs
'gunicorn ... application:app'."""

from apps.api.main import app

__all__ = ["app"]
PY

# Startup script: run the app with whichever Python is available - the
# pre-built venv we shipped, or the one Oryx built on the instance.
cat > "$STAGE/startup.sh" <<'SH'
#!/bin/bash
cd /home/site/wwwroot || exit 1

if [ -x /home/site/wwwroot/venv/bin/python ]; then
    # Pre-built venv shipped in the package.
    export LD_LIBRARY_PATH=/home/site/wwwroot/venv/lib:$LD_LIBRARY_PATH
    PY=/home/site/wwwroot/venv/bin/python
elif [ -x /usr/local/bin/python3.12 ]; then
    # Oryx built the environment on the instance.
    PY=/usr/local/bin/python3.12
elif [ -x /usr/local/bin/python ]; then
    PY=/usr/local/bin/python
else
    PY=$(command -v python3)
fi

echo "Starting with $PY"
exec "$PY" -m uvicorn application:app --host 0.0.0.0 --port 8000
SH
chmod +x "$STAGE/startup.sh"

# Pre-built Linux venv (see scripts/build_azure_venv.sh). Shipping it means
# App Service does not build the environment on the instance, which burns the
# Free tier's 60 CPU minutes a day. It also makes the upload ~176MB, which is
# slow on a Free instance, so it is opt-in:
#   INCLUDE_VENV=1 ./scripts/package_azure.sh
if [ "${INCLUDE_VENV:-0}" = "1" ] && [ -d "$ROOT/.deploy/venv/venv" ]; then
  echo "Including the pre-built venv"
  rm -rf "$STAGE/venv"
  cp -a "$ROOT/.deploy/venv/venv" "$STAGE/venv"
else
  echo "Not including a pre-built venv - App Service will build the" \
       "environment on the instance (set INCLUDE_VENV=1 to ship one)."
fi

# Drop caches and anything that must not ship
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true
rm -rf "$STAGE/mcp_servers"/*/__pycache__ 2>/dev/null || true

mkdir -p "$(dirname "$ZIP")"
rm -f "$ZIP"
(cd "$STAGE" && zip -qr "$ZIP" .)

echo "Built $ZIP ($(du -h "$ZIP" | cut -f1))"
echo "Contents:"
(cd "$STAGE" && find . -type f | sed 's|^\./||' | sort | head -40)
