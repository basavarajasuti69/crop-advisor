# Docker image for the crop-advisor API.
#
# Build from the repository root:
#   docker build -t crop-advisor-local .
#
# The image needs the shared packages (agents, mcp_servers, rag, speech,
# strings) and the ONNX classifier, so the build context is the repo root
# and the Dockerfile lives at the root.
#
# CPU only: no torch. The classifier runs on ONNX Runtime (~50MB instead
# of ~2.5GB of CUDA wheels) and starts in seconds.
FROM python:3.12-slim

WORKDIR /app

# CPU-only dependency set: the classifier runs on ONNX Runtime.
COPY apps/api/requirements-cpu.txt ./requirements-cpu.txt
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Application code.
COPY agents/ ./agents/
COPY mcp_servers/ ./mcp_servers/
COPY rag/ ./rag/
COPY speech/ ./speech/
COPY strings/ ./strings/
COPY apps/api/ ./apps/api/

# The ONNX classifier: ml/export_edge/artifacts/classifier.onnx
# (committed with Git LFS) and classes.json.
COPY ml/export_edge/artifacts/ ./ml/export_edge/artifacts/

# Run as a non-root user.
RUN set -eux; \
    (getent group 1000 || groupadd --gid 1000 app); \
    (getent passwd 1000 || useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash app); \
    chown -R 1000:1000 /app
USER 1000:1000

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    CA_REGION=ap-south-1

# The host sets $PORT (Render uses 10000); the default keeps a local
# `docker run -p 10000:10000` working.
EXPOSE 10000
CMD uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-10000}
