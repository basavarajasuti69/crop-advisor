---
title: Crop Disease Advisor API
emoji: 🌾
colorFrom: green
colorTo: lime
sdk: docker
app_port: 7860
pinned: false
---

# Crop Disease Advisor

AI crop-disease advisor — languages **English, Hindi, Kannada**, crops **rice, wheat, cotton, maize, tomato**.

This repository is the API for a **Hugging Face Space** (`sdk: docker`, port 7860). The web app is deployed separately to Netlify.

```
crop-advisor/
├── apps/          api/ (FastAPI) · web/ (Next.js PWA) · expert-review/
├── agents/        LangGraph state + node definitions
├── mcp_servers/   crop_knowledge, weather, officer_directory
├── ml/            data, training, evaluation, export_edge, pipelines
├── rag/           ingest, retriever, glossary
├── speech/        stt, tts_adapter
├── safety/        guardrails, ood, policies
├── evals/         vision, rag, language, safety test suites
├── infra/         Terraform (network, ecs, sagemaker, data, monitoring)
├── .github/       CI/CD workflows
└── docs/          architecture, cost-report, failure-cases
```

## Phase plan
| Phase | Weeks | Deliverable |
|---|---|---|
| 1: Core | 1–2 | Classifier, quality check, English advice with RAG, CI/CD |
| 2: Voice & language | 3–4 | Hindi answers, TTS, audio-first PWA, glossary |
| 3: Agents & MCP | 5–6 | LangGraph, weather MCP, critic, follow-up reminders |
| 4: MLOps & safety | 7–8 | SageMaker Pipeline, registry, drift monitor, guardrails, eval gates |
| 5: Extras | 9+ | Expert review queue, offline edge model, outbreak dashboard |

## Getting started
```bash
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Terraform is not installed in this environment. Install it, then:
```bash
cd infra/terraform
terraform init
terraform plan -var region=ap-south-1
```

## Deploy (free)

Both parts run on free tiers with no credit card:

| Piece | Where | Cost |
|---|---|---|
| API (FastAPI + ONNX classifier) | Hugging Face Spaces (`sdk: docker`, port 7860) | free CPU |
| Web app (Next.js static export) | Netlify | free tier |

The API image is CPU-only: no torch, the classifier runs on ONNX Runtime
(~50MB instead of ~2.5GB of CUDA wheels).

### 1. API on Hugging Face Spaces

The Space uses the **root `Dockerfile`** in this repository, so this repo
is the Space repo. It listens on `$PORT` (default 7860) and runs as
UID 1000.

The ONNX classifier must be in the repo at
`ml/export_edge/artifacts/classifier.onnx` - it is stored with Git LFS
(`.gitattributes`) because it is ~16MB.

```bash
# once, if LFS is not installed
brew install git-lfs && git lfs install

# the model is a build artifact, so it is not in git history yet
git lfs track "*.onnx"          # .gitattributes already has this line
git add .gitattributes .gitignore ml/export_edge/artifacts/classifier.onnx
git commit -m "feat: add the ONNX classifier for the Space"
git push origin main
```

Then create the Space and push:

```bash
hf auth login
# The Space must be PUBLIC: the Netlify frontend calls it from the browser,
# so a private Space would be rejected by CORS.
hf repo create <your-user>/<space-name> --repo-type space --space-sdk docker
git remote add space https://huggingface.co/spaces/<your-user>/<space-name>
git push space main              # the LFS file goes with it
```

The Space's README front matter (already in this README) sets
`sdk: docker` and `app_port: 7860`.

> **If the push is rejected with "contains files larger than 10 MiB"**
>
> Hugging Face runs a pre-receive hook that rejects any pushed commit
> holding a file over 10 MiB that is **not** stored with LFS. The model is
> 16 MB, so it must be an LFS object.
>
> - *If the model has never been committed* (it has not been - the
>   history is clean): install LFS, `git lfs track "*.onnx"`, then
>   `git add` / `git commit` / `git push` the model. The pointer file
>   that goes into git is a few hundred bytes, and the real file is
>   uploaded to the LFS store. `git lfs ls-files` should list
>   `ml/export_edge/artifacts/classifier.onnx`.
> - *If the model is already in the history as a raw blob*:
>   `git rm --cached` does **not** help - it only affects future
>   commits, and the old blob is still in the history, so the hook
>   still rejects the push. The fix is to rewrite the history so the
>   blob becomes an LFS pointer:
>
>   ```bash
>   git lfs install
>   git lfs migrate import --include="ml/export_edge/artifacts/classifier.onnx" --everything
>   git push --force origin main
>   ```
>
>   That rewrites history and force-pushes, so run it only if you are
>   sure - it changes every commit hash.
>
> For a Space, `hf lfs-enable-largefiles .` also enables the large-file
> support on the Hub side (needed for files over 5 GB).

Check it: `https://<your-user>-<space-name>.hf.space/health`

**Space environment variables**

| Name | Required | Default | What it does |
|---|---|---|---|
| `CA_CORS_ORIGINS` | for the browser | `http://localhost:3000` | comma-separated allowed origins. Set it to your Netlify URL. |
| `CA_REGION` | no | `ap-south-1` | region label reported by `/health` |
| `PORT` | no | `7860` | port the app listens on (set by Spaces) |
| `CA_LLM_BASE_URL` | no | empty | OpenAI-compatible LLM endpoint. If any LLM variable is missing the model is never called and the plain template is used. |
| `CA_LLM_API_KEY` | no | empty | LLM API key |
| `CA_LLM_MODEL_NAME` | no | empty | LLM model name |
| `CA_VISION_SECOND_OPINION` | no | `false` | sends the image to the vision model. Leave off. |
| `CA_SARVAM_API_KEY` | no | empty | text-to-speech (Hindi, Kannada). Without it `/speak` returns 503 and the web app shows "Audio is not available right now." |
| `GOOGLE_APPLICATION_CREDENTIALS` | no | empty | Google Cloud TTS / STT fallback |
| `CA_SAGEMAKER_ENDPOINT` | no | empty | serve the classifier from SageMaker instead of the bundled ONNX model |

No secret is required: with none of these set the API is fully
self-contained (ONNX + the local knowledge base + the local MCP servers)
and makes no outbound paid calls.

### 2. Web app on Netlify

`netlify.toml` at the repository root has the build settings. Import the
repository in Netlify, then set the environment variable
**before** the first deploy:

| Key | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://<your-user>-<space-name>.hf.space` |

`NEXT_PUBLIC_*` is inlined into the bundle at **build** time, so changing
it needs a rebuild (Netlify: Triggers -> Clear build cache and deploy).

### Order of steps

1. Commit the ONNX model (with LFS) and push to GitHub.
2. Create the **public** Space, push this repo to it, confirm `/health`
   responds.
3. Set `CA_CORS_ORIGINS` to your Netlify URL in the Space settings and
   restart the Space.
4. Import the repo in Netlify, set `NEXT_PUBLIC_API_URL`, deploy.
5. Test both sample photos in English, Hindi and Kannada.

The Space is public on purpose - the browser has to call it. It is
stateless: photos are processed in memory and nothing is stored.

## Legacy

The AWS deployment path is kept for reference but is no longer the
target:

- `infra/terraform/` - legacy (AWS)
