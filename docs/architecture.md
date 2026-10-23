# Architecture

```
Next.js PWA (static export)   ->   FastAPI (Docker)      ->   ONNX classifier
   on Netlify                       on Render                 + RAG (rag/index.jsonl)
                                                              + MCP servers
```

The browser calls the API directly. Nothing runs on a server between
the two.

## Request flow

`POST /advice/image` (multipart) or `POST /advice` (JSON, base64
image). The LangGraph pipeline in `agents/graph.py` runs:

| Node | What it does |
|---|---|
| `triage` | Checks an image is present. Without one it answers "upload a photo" in the farmer's language. |
| `classify` | ONNX Runtime (falls back to the PyTorch checkpoint). Low confidence or a `healthy` result below `HEALTHY_MIN` produces the "retake the photo" answer instead of a diagnosis. |
| `vision_second_opinion` | Off by default. When `CA_VISION_SECOND_OPINION=true` it sends the image (EXIF/GPS stripped, resized to 512px) and compares the model's diseases with the classifier's. |
| `context_weather` | The `weather` MCP server (open-meteo, keyless). |
| `retrieve` | The `crop_knowledge` MCP server. It splits the knowledge base on `## ` headings and matches token by token, so `<disease> <crop>` finds the `<Crop> - <Disease>` section. |
| `draft_advice` | Builds the English answer from the retrieved chunk. |
| `localize_advice` | The diagnosis and the safety line in the farmer's script. The English passage is not embedded, so the answer is never mixed script. |
| `llm_explain` | Optional. When `CA_LLM_BASE_URL`, `CA_LLM_API_KEY` and `CA_LLM_MODEL_NAME` are set, a language model writes the explanation in the target language from the English chunks. Its output is validated (script, no headings, no invented dose or chemical) and the fixed safety line and helpline are appended by the code. Without a key this node does nothing. |
| `critic_safety` | Dose verification and the script check. |
| `find_officer` | The `officer_directory` MCP server. |
| `follow_up` | Schedules a follow-up for a low-confidence case, and appends the officer details or the helpline. |

## Data flow

1. The farmer uploads a leaf photo from the PWA.
2. The photo is processed in memory. Nothing is stored.
3. The response carries the advice, the confidence, the citations and
   the `safe` flag.
4. A low-confidence case is posted to the expert review queue
   (`apps/expert-review`, on `localhost:8001`) if that service is
   running. If it is not, the case is still escalated in the graph.

## What is bundled in the image

| Path | What |
|---|---|
| `apps/api/` | the FastAPI service |
| `agents/` | the graph, the LLM client and the vision client |
| `mcp_servers/` | weather, crop_knowledge, officer_directory |
| `rag/` | the knowledge base index and the hi/kn glossary |
| `speech/` | the TTS and STT adapters |
| `strings/` | the en / hi / kn farmer-facing strings |
| `ml/export_edge/artifacts/` | `classifier.onnx` and `classes.json` |

The image has no torch: the classifier runs on ONNX Runtime, which is
about 50MB instead of 2.5GB of CUDA wheels.

## Language and safety rules

- The safety line, the helpline and the "retake the photo" text come
  from `strings/<lang>.json`. The code never writes farmer-facing
  English into a Hindi or Kannada answer.
- A Hindi or Kannada answer that carries a letter from another script
  is rejected by the critic and replaced with the plain template.
- The language model may only use the retrieved chunks, and may not
  invent a dose or a chemical name. The safety line is never written by
  the model.
- Below `CONFIDENCE_MIN` (0.70) the API says "retake the photo" rather
  than guessing.

## Optional integrations

All of them are off unless the environment variable is set, and the
app is fully self-contained without any of them.

| Variable | Effect |
|---|---|
| `CA_LLM_BASE_URL` / `CA_LLM_API_KEY` / `CA_LLM_MODEL_NAME` | The language model writes the explanation. Any OpenAI-compatible endpoint works. |
| `CA_SARVAM_API_KEY` | Text-to-speech in Hindi and Kannada. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud TTS / STT. |
| `CA_VISION_SECOND_OPINION` | Send the image to the vision model. |

## Operational notes

- The free Render plan sleeps after 15 minutes, so the first request
  can take up to a minute. The web app shows a "Waking up the server"
  message after 5 seconds.
- The CORS allow-list comes from `CA_CORS_ORIGINS` (comma-separated,
  no trailing slashes). Set it to the Netlify URL.
- `NEXT_PUBLIC_API_URL` is inlined into the web bundle at build time,
  so changing it needs a Netlify rebuild.
