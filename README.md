# Crop Disease Advisor

AI crop-disease advisor — languages **English, Hindi, Kannada**, crops **rice, wheat, cotton, maize, tomato**.

This repository holds the API (a Docker image built from the root
`Dockerfile`) and the web app. The API is deployed to Render and the web
app to Netlify.

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
├── infra/         Terraform (legacy AWS path: network, ecs, sagemaker, data)
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

The web app:
```bash
cd apps/web
npm install && npm run dev
```

The legacy AWS infrastructure is not installed in this environment.
Install Terraform if you want to look at it:
```bash
cd infra/terraform
terraform init
terraform plan -var region=ap-south-1
```

## Architecture

```
Next.js (static export)  ->  FastAPI  ->  ONNX classifier
   on Netlify               on Render      + RAG (rag/index.jsonl)
                                            + MCP servers (weather, crop_knowledge, officer_directory)
```

The browser talks to the API directly. The API is stateless: photos are
processed in memory and nothing is stored.

## Deploy (free)

Both parts run on free tiers with no credit card:

| Piece | Where | Cost |
|---|---|---|
| API (FastAPI + ONNX classifier) | Render (Docker, free web service) | free CPU |
| Web app (Next.js static export) | Netlify | free tier |

The API image is CPU-only: no torch, the classifier runs on ONNX Runtime
(~50MB instead of ~2.5GB of CUDA wheels).

### 1. Deploy the API on Render

`render.yaml` at the repository root declares the service: one Docker
web service on the free plan in Singapore, with `/health` as the health
check.

1. Commit the ONNX model (with LFS) and push to GitHub:

   ```bash
   # once, if LFS is not installed
   brew install git-lfs && git lfs install

   # the model is a build artifact, so it is not in git history yet
   git lfs track "*.onnx"          # .gitattributes already has this line
   git add .gitattributes .gitignore ml/export_edge/artifacts/classifier.onnx
   git commit -m "feat: add the ONNX classifier"
   git push origin main
   ```

2. In Render: **New -> Blueprint**, point it at this repository and
   accept the service from `render.yaml`.

3. Check it: `https://crop-advisor-rkat.onrender.com/health` should
   return `{"status":"ok","region":"ap-south-1"}`.

The API is built from the **root `Dockerfile`**, so the build context
is the repository root. It listens on `$PORT` (Render sets it) and runs
as UID 1000.

**Environment variables**

| Name | Required | Default | What it does |
|---|---|---|---|
| `CA_CORS_ORIGINS` | for the browser | `http://localhost:3000` | comma-separated allowed origins, **no trailing slashes**. Set it to your Netlify URL. |
| `CA_REGION` | no | `ap-south-1` | region label reported by `/health` |
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

### 2. Deploy the web app on Netlify

`netlify.toml` at the repository root has the build settings. Import the
repository in Netlify, then set the environment variable
**before** the first deploy:

| Key | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://crop-advisor-rkat.onrender.com` |

`NEXT_PUBLIC_*` is inlined into the bundle at **build** time, so changing
it needs a rebuild (Netlify: Triggers -> Clear build cache and deploy).

### Order of steps

1. Commit the ONNX model (with LFS) and push to GitHub.
2. **Deploy the API on Render.** Wait for the build, then confirm
   `https://crop-advisor-rkat.onrender.com/health` responds.
3. **Set `CA_CORS_ORIGINS` on Render** (Render: Environment ->
   CA_CORS_ORIGINS). Comma-separated, **no trailing slash**:

   ```
   http://localhost:3000,https://crop-adviser.netlify.app
   ```

   Save it - Render restarts the service automatically.
4. **Set `NEXT_PUBLIC_API_URL` in Netlify** (Site settings ->
   Environment variables):
   `https://crop-advisor-rkat.onrender.com`
5. **Clear the Netlify build cache and redeploy** (Triggers -> Clear
   build cache and deploy site). The variable is baked into the
   bundle, so the first build after setting it is what matters.
6. Test both sample photos in English, Hindi and Kannada, and check the
   browser console for CORS errors.

## Known limitations

- **The model is weak.** It was trained on a small synthetic set (40
  images, two classes: `healthy` and `leaf_blight`). Real photos often
  score below the 0.70 confidence threshold, so the app answers "retake
  the photo" instead of naming a disease. That is the intended
  behaviour for a low-confidence result, not a crash - but it means the
  app mostly asks for a better photo until the model is retrained on
  real field data.
- **The free API sleeps.** Render's free plan sleeps after 15 minutes
  of inactivity, so the first request after a pause can take up to about
  a minute to wake up. The web app shows a "Waking up the server" note
  after 5 seconds.
- **Audio needs a key.** `/speak` returns 503 until a TTS provider is
  configured (`CA_SARVAM_API_KEY` for Hindi and Kannada). The web app
  disables the "Listen to advice" button with a clear label when that
  happens.
- **Only three languages and five crops** are wired up: English, Hindi
  and Kannada; rice, wheat, cotton, maize and tomato.

## Legacy

The AWS deployment path is kept for reference but is no longer the
target:

- `infra/terraform/` - legacy (AWS)
