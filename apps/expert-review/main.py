"""Expert Review Queue — FastAPI app for agronomists to review escalated cases.

Agronomists review low-confidence / high-risk cases, provide corrected
diagnosis, and their feedback becomes labeled training data for retraining.

Run:
    uvicorn main:app --reload --port 8001
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Expert Review Queue", version="0.1.0")

REVIEW_DIR = Path("apps/expert-review/data")
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
CASES_FILE = REVIEW_DIR / "cases.jsonl"
FEEDBACK_FILE = REVIEW_DIR / "feedback.jsonl"


class CaseReview(BaseModel):
    case_id: str
    image_url: str | None = None
    predicted_label: str
    predicted_score: float
    crop: str
    language: str
    location: str | None = None
    confidence: float
    escalate_reason: str | None = None
    answer: str | None = None
    created_at: str


class ExpertFeedback(BaseModel):
    case_id: str
    expert_id: str
    corrected_label: str
    corrected_disease: str | None = None
    notes: str | None = None
    approved: bool  # True = diagnosis correct, False = needs correction
    reviewed_at: str


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Simple HTML dashboard listing pending reviews."""
    cases = _load_cases()
    pending = [c for c in cases if c.get("escalate") or c.get("confidence", 1.0) < 0.8]
    rows = ""
    for c in pending:
        rows += (
            f"<tr>"
            f"<td>{c['case_id'][:8]}...</td>"
            f"<td>{c.get('crop','')}</td>"
            f"<td>{c.get('predicted_label','')}</td>"
            f"<td>{c.get('confidence',0):.0%}</td>"
            f"<td>{c.get('escalate_reason','')}</td>"
            f"<td><a href='/review/{c['case_id']}'>Review</a></td>"
            f"</tr>"
        )
    return f"""
    <html><body style='font-family:sans-serif;padding:2rem'>
    <h1>🌾 Expert Review Queue</h1>
    <p>Pending reviews: {len(pending)}</p>
    <table border='1' cellpadding='8'>
    <tr><th>Case ID</th><th>Crop</th><th>Predicted</th><th>Confidence</th><th>Reason</th><th>Action</th></tr>
    {rows}
    </table>
    <p><a href='/feedback'>View all expert feedback</a></p>
    </body></html>
    """


@app.get("/cases", response_model=list[CaseReview])
async def list_cases() -> list[CaseReview]:
    return [CaseReview.model_validate(case) for case in _load_cases()]


@app.get("/review/{case_id}", response_class=HTMLResponse)
async def review_page(case_id: str) -> str:
    cases = {c["case_id"]: c for c in _load_cases()}
    case = cases.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return f"""
    <html><body style='font-family:sans-serif;padding:2rem'>
    <h1>Review Case: {case_id[:8]}...</h1>
    <p><strong>Crop:</strong> {case.get('crop','')}</p>
    <p><strong>Predicted:</strong> {case.get('predicted_label','')} ({case.get('confidence',0):.0%})</p>
    <p><strong>Reason:</strong> {case.get('escalate_reason','n/a')}</p>
    <p><strong>AI Advice:</strong> {case.get('answer','')}</p>
    <img src="{case.get('image_url','')}" alt="Leaf photo" style='max-width:400px'/>
    <form method='post' action='/feedback'>
      <input type='hidden' name='case_id' value='{case_id}'/>
      <input type='hidden' name='image_url' value='{case.get('image_url','')}'/>
      <label>Expert ID: <input name='expert_id' required/></label><br/><br/>
      <label>Corrected label:
        <select name='corrected_label'>
          <option value='healthy'>Healthy</option>
          <option value='leaf_blight'>Leaf Blight</option>
          <option value='leaf_rust'>Leaf Rust</option>
          <option value='leaf_curl_virus'>Leaf Curl Virus</option>
        </select>
      </label><br/><br/>
      <label>Approved? <input type='checkbox' name='approved' value='true'/></label><br/><br/>
      <label>Notes: <textarea name='notes'></textarea></label><br/><br/>
      <button type='submit'>Submit Feedback</button>
    </form>
    <p><a href='/'>Back to queue</a></p>
    </body></html>
    """


@app.post("/feedback")
async def submit_feedback(
    case_id: str,
    expert_id: str,
    corrected_label: str,
    corrected_disease: str | None = None,
    notes: str | None = None,
    approved: bool = False,
) -> dict[str, str]:
    feedback = ExpertFeedback(
        case_id=case_id,
        expert_id=expert_id,
        corrected_label=corrected_label,
        corrected_disease=corrected_disease,
        notes=notes,
        approved=approved,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )
    with open(FEEDBACK_FILE, "a") as f:
        f.write(feedback.model_dump_json() + "\n")
    return {"status": "saved", "case_id": case_id}


@app.get("/feedback", response_class=HTMLResponse)
async def list_feedback() -> str:
    feedbacks = _load_feedback()
    rows = ""
    for f in feedbacks:
        rows += f"<tr><td>{f['case_id'][:8]}...</td><td>{f['expert_id']}</td><td>{f['corrected_label']}</td><td>{'✅' if f['approved'] else '❌'}</td><td>{f.get('notes','')}</td></tr>"
    return f"""
    <html><body style='font-family:sans-serif;padding:2rem'>
    <h1>📋 Expert Feedback History</h1>
    <table border='1' cellpadding='8'>
    <tr><th>Case</th><th>Expert</th><th>Corrected Label</th><th>Approved</th><th>Notes</th></tr>
    {rows}
    </table>
    <p><a href='/'>Back to queue</a></p>
    </body></html>
    """


def _load_cases() -> list[dict[str, Any]]:
    if not CASES_FILE.exists():
        return []
    return [json.loads(line) for line in CASES_FILE.read_text().splitlines() if line.strip()]


def _load_feedback() -> list[dict[str, Any]]:
    if not FEEDBACK_FILE.exists():
        return []
    return [json.loads(line) for line in FEEDBACK_FILE.read_text().splitlines() if line.strip()]


def ingest_case(case: dict[str, Any]) -> None:
    """Called by the LangGraph graph when a case is escalated."""
    case["received_at"] = datetime.now(timezone.utc).isoformat()
    with open(CASES_FILE, "a") as f:
        f.write(json.dumps(case) + "\n")


@app.post("/ingest")
async def ingest(case: dict[str, Any]) -> dict[str, str]:
    ingest_case(case)
    return {"status": "ingested", "case_id": str(case.get("case_id", ""))}
