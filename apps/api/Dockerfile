# Build context is the repository root:
#   docker build -f apps/api/Dockerfile -t crop-advisor-api .
# The image needs the shared packages (agents, rag, speech, strings) as well
# as the API itself, so the context cannot be apps/api alone.
FROM python:3.11-slim

WORKDIR /app

# Dependencies first so the layer caches across code changes.
# The default PyPI torch wheels bundle CUDA (~2.5GB); the classifier runs on
# CPU here and is served from SageMaker in production, so use the CPU wheels.
COPY apps/api/requirements.txt ./apps/api/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r apps/api/requirements.txt

# Application code.
COPY agents/ ./agents/
COPY mcp_servers/ ./mcp_servers/
COPY rag/ ./rag/
COPY speech/ ./speech/
COPY strings/ ./strings/
COPY apps/api/ ./apps/api/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    CA_REGION=ap-south-1

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
