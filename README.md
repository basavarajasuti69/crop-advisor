# Crop Disease Advisor

AI crop-disease advisor on AWS — Phase 1 defaults: **English + Hindi**, crops **rice, wheat, cotton, maize, tomato**, region **ap-south-1**.

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
