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
