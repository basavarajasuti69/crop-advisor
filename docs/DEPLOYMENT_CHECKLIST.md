# Deployment Checklist — Crop Disease Advisor

## Netlify (Frontend)

### Build settings
- Base directory: `apps/web`
- Build command: `npm ci && NEXT_OUTPUT=export npm run build`
- Publish directory: `out`
- Node version: 20

### Required environment variables
- `NEXT_PUBLIC_API_URL` — URL of the Render backend (e.g. `https://crop-advisor-api.onrender.com`)
  - This is inlined at build time; changing it requires a new deploy.

### Steps
1. Connect the GitHub repository to Netlify.
2. Set the build settings above.
3. Set `NEXT_PUBLIC_API_URL` in Site settings → Environment variables.
4. Deploy. Verify the site loads and the language selector works.
5. Test image upload against the deployed backend.

## Render (Backend)

### Service configuration
- Type: Web (Docker)
- Dockerfile: `./Dockerfile`
- Docker context: repository root
- Health check path: `/health`
- Plan: Free (sleeps after 15 min inactivity; first request after sleep can take ~60s)

### Required environment variables
- `CA_CORS_ORIGINS` — comma-separated frontend origins (e.g. `https://crop-adviser.netlify.app`)
- `CA_REGION` — AWS region (default: `ap-south-1`)
- `CA_SARVAM_API_KEY` — Sarvam TTS API key (for Hindi/Kannada speech)
- `GOOGLE_APPLICATION_CREDENTIALS` — path to Google Cloud service account JSON (optional)
- `CA_LLM_BASE_URL` — LLM endpoint (optional, for advice generation)
- `CA_LLM_API_KEY` — LLM API key (optional)
- `CA_LLM_MODEL_NAME` — LLM model name (optional)

### Model artifacts
- The Dockerfile copies `ml/export_edge/artifacts/` into the image.
- For V11: export the checkpoint first:
  ```bash
  python ml/export_edge/export_plantdoc.py \
      --experiment plantdoc_27class_v11 \
      --output ml/export_edge/artifacts_plantdoc_v11
  ```
- Set `CA_MODEL_DIR=ml/export_edge/artifacts_plantdoc_v11` in Render env vars,
  or copy the export into `ml/export_edge/artifacts/` before deploying.
- Verify `/ready` returns `"status": "ready"` after deploy.

### Steps
1. Create a Blueprint service from `render.yaml`.
2. Set environment variables in the Render dashboard.
3. Deploy and verify `/health` returns `{"status": "ok"}`.
4. Verify `/ready` returns `"status": "ready"`.
5. Test `/advice/image` with a sample leaf photo.

## V11 Model Integration (when training completes)

1. Verify `best_model.pt`, `classes.json`, and `metrics.json` exist in
   `ml/training/experiments/plantdoc_27class_v11/`.
2. Verify class ordering matches the V5 dataset's sorted class list.
3. Export to ONNX:
   ```bash
   python ml/export_edge/export_plantdoc.py \
       --experiment plantdoc_27class_v11 \
       --output ml/export_edge/artifacts_plantdoc_v11
   ```
4. Set `CA_MODEL_DIR` to the new export directory.
5. Verify `/model` endpoint reports 27 classes and `trained_on: "PlantDoc"`.
6. Run the full test suite.

## Known limitations
- Free Render tier sleeps after 15 min inactivity (cold start ~60s).
- Kannada/Hindi TTS requires Sarvam API key (free credits on signup).
- Model confidence is uncalibrated (not a disease probability).
- 27-class model covers 13 crops; knowledge base covers 5 (rice, wheat,
  cotton, maize, tomato). Other classes receive an officer referral.
- Test accuracy ~67-69% on 225 PlantDoc images; not evidence of field
  reliability.
