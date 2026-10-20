# Deploying to Azure App Service (Linux)

The API runs on Azure App Service. The classifier runs through **ONNX Runtime**,
not PyTorch: a Free F1 plan has 1 GB of memory and 60 CPU minutes a day, and
the CUDA torch wheels are ~2.5 GB.

## One-time setup

```bash
az login
az account set --subscription "Azure subscription 1"

APP=crop-advisor-yourname-api
RG=crop-advisor-yourname-api

az webapp create \
  --name $APP \
  --resource-group $RG \
  --runtime "PYTHON|3.12" \
  --os-type Linux \
  --region "Central India" \
  --plan <the Free F1 plan> \
  --sku F1
```

## Deploying

```bash
APP_NAME=crop-advisor-yourname-api \
RESOURCE_GROUP=crop-advisor-yourname-api \
./scripts/deploy_azure.sh
```

This builds the package with `scripts/package_azure.sh`, sets the app settings,
deploys the zip (Oryx builds on the server) and waits for `/health`.

## What is in the package

| Path | Why |
|---|---|
| `apps/api/` | the FastAPI app |
| `agents/` | the LangGraph graph |
| `mcp_servers/` | weather, crop_knowledge, officer_directory |
| `rag/` | the knowledge base index and glossary |
| `speech/` | TTS / STT adapters |
| `strings/` | en / hi / kn farmer-facing strings |
| `ml/export_edge/artifacts/` | `classifier.onnx` + `classes.json` |
| `requirements.txt` | the Azure dependency set (no torch) |

## The model

`classifier.onnx` is a build artifact and is **not in git** (`*.onnx` is
gitignored). Export it with:

```bash
python ml/export_edge/export_onnx.py \
    --weights ml/training/artifacts/model.pt \
    --metrics ml/training/artifacts/metrics.json \
    --output ml/export_edge/artifacts
```

The ONNX preprocessing must match the training transform exactly
(`Resize(256)` + `CenterCrop(224)` + ImageNet normalisation). `agents/graph.py`
implements this in numpy and `tests/test_graph.py` asserts it matches the
PyTorch transform to within 1e-6.

## App settings

| Setting | Value | Why |
|---|---|---|
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` | Oryx builds the venv from `requirements.txt` |
| `ENABLE_ORYX_BUILD` | `true` | same |
| `PYTHONUNBUFFERED` | `1` | logs appear immediately in App Service |
| `CA_CORS_ORIGINS` | your frontend origin | the web app calls this API |
| `CA_SAGEMAKER_ENDPOINT` | endpoint name | optional: serve the classifier from SageMaker instead |

## Logs

```bash
az webapp log tail --name $APP --resource-group $RG
```

## Known limits of the Free F1 plan

- 60 CPU minutes a day, 1 GB RAM, no Always On (the app sleeps when idle, so
  the first request after a pause is slow)
- `torch` cannot be installed: use the ONNX path
- For real traffic, move to Azure Container Apps, which has a much larger
  monthly free grant
