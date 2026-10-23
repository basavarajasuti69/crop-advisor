"""Build a self-contained, read-only HTML review page for the V10 audit.

Reads the consolidated tracker and the V3 dataset images, embeds them as
base64 data URIs, and writes a single HTML file the reviewer can open in
any browser. Decisions are recorded in-browser (localStorage) and can be
exported as a CSV that matches the tracker schema. Nothing is written
back to the dataset or to the original tracker.

Usage:
    python ml/training/review_tools/build_review_html.py
    open ml/training/experiments/plantdoc_v9_audit/V10_REVIEW.html
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "ml/training/experiments/plantdoc_v9_audit"
V3 = ROOT / "ml/training/experiments/plantdoc_deduplicated_v3/data"
TRACKER = BASE / "V10_CONSOLIDATED_REVIEW.csv"
OUT_HTML = BASE / "V10_REVIEW.html"

PRI_LABELS = {
    "1": "Missing file (investigate)",
    "2": "Multi-panel / unsuitable",
    "3": "Possible duplicate",
    "4": "Possible label mismatch",
    "5": "Low quality / ambiguous",
    "6": "No obvious issue",
}


def b64_image(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def main() -> None:
    rows = list(csv.DictReader(open(TRACKER)))
    entries = []
    for r in rows:
        rel = r["file"].strip()
        img_path = (V3 / Path(rel)).resolve()
        try:
            img_path.relative_to(V3.resolve())
            inside = True
        except ValueError:
            inside = False
        exists = inside and img_path.is_file()
        thumb = b64_image(img_path) if exists else None
        entries.append({
            "split": r["split"],
            "current_label": r["current_label"],
            "file": rel,
            "suggested_label": r["suggested_label"],
            "reason": r["reason"],
            "reviewer_notes": r["reviewer_notes"],
            "resolved_source_path": r["resolved_source_path"],
            "source_exists": r["source_exists"],
            "review_priority": r["review_priority"],
            "priority_label": PRI_LABELS.get(r["review_priority"], r["review_priority"]),
            "proposed_action": r["proposed_action"],
            "proposed_target_label": r["proposed_target_label"],
            "final_user_decision": r["final_user_decision"],
            "decision_evidence": r["decision_evidence"],
            "review_status": r["review_status"],
            "image_inside_dataset": inside,
            "image_exists": exists,
            "image_b64": thumb,
        })

    payload = json.dumps(entries, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>V10 PlantDoc Label Review</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #f6f7f9; color: #222; }}
h1 {{ margin-bottom: 4px; }}
.sub {{ color: #555; font-size: 14px; margin-bottom: 16px; }}
.controls {{ background: #fff; padding: 12px; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
.controls label {{ font-size: 13px; }}
.controls select, .controls input {{ padding: 4px 8px; }}
button {{ padding: 6px 12px; cursor: pointer; border: 1px solid #ccc; background: #fff; border-radius: 4px; }}
button.primary {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
.stats {{ font-size: 13px; color: #444; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }}
.card {{ background: #fff; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.card img {{ width: 100%; height: 220px; object-fit: contain; background: #eee; border-radius: 4px; }}
.card .missing {{ height: 220px; display: flex; align-items: center; justify-content: center; background: #fee; color: #900; font-weight: bold; border-radius: 4px; }}
.lbl {{ font-weight: 600; font-size: 13px; margin-top: 8px; }}
.meta {{ font-size: 12px; color: #555; margin: 4px 0; }}
.notes {{ font-size: 12px; color: #333; background: #f3f4f6; padding: 6px; border-radius: 4px; margin: 6px 0; max-height: 90px; overflow: auto; }}
select.decision {{ width: 100%; padding: 6px; margin-top: 6px; }}
textarea.evidence {{ width: 100%; height: 52px; font-size: 12px; margin-top: 4px; box-sizing: border-box; }}
.pri {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; color: #fff; }}
.p1 {{ background: #dc2626; }} .p2 {{ background: #ea580c; }} .p3 {{ background: #ca8a04; }} .p4 {{ background: #2563eb; }} .p5 {{ background: #6b7280; }} .p6 {{ background: #9ca3af; }}
.path {{ font-size: 11px; color: #888; word-break: break-all; }}
</style>
</head>
<body>
<h1>V10 PlantDoc Label Review</h1>
<div class="sub">Read-only with respect to the dataset. Decisions are stored in your browser and exported as CSV. Original tracker and images are never modified.</div>

<div class="controls">
  <label>Filter priority:
    <select id="priFilter"><option value="">All</option>
      <option value="1">1 - Missing file</option>
      <option value="2">2 - Multi-panel</option>
      <option value="3">3 - Possible duplicate</option>
      <option value="4">4 - Possible mismatch</option>
      <option value="5">5 - Low quality</option>
      <option value="6">6 - No obvious issue</option>
    </select>
  </label>
  <label>Filter label:
    <input id="labelFilter" placeholder="current label" style="width:140px">
  </label>
  <label>Filter decision:
    <select id="decFilter"><option value="">All</option><option>PENDING</option><option>KEEP</option><option>RELABEL</option><option>REMOVE</option></select>
  </label>
  <button class="primary" onclick="exportCsv()">Export decisions CSV</button>
  <button onclick="clearAll()">Clear all decisions</button>
  <span class="stats" id="stats"></span>
</div>

<div class="grid" id="grid"></div>

<script>
const DATA = {payload};
const STORAGE_KEY = "v10_review_decisions";

function loadDecisions() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }} catch(e) {{ return {{}}; }}
}}
function saveDecisions(d) {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(d)); }}

let decisions = loadDecisions();

function keyOf(e) {{ return e.file; }}

function render() {{
  const pri = document.getElementById('priFilter').value;
  const lab = document.getElementById('labelFilter').value.toLowerCase();
  const dec = document.getElementById('decFilter').value;
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  let shown = 0, decided = 0;
  DATA.forEach(e => {{
    const k = keyOf(e);
    const d = decisions[k] || {{decision: e.final_user_decision, evidence: e.decision_evidence}};
    if (pri && e.review_priority !== pri) return;
    if (lab && !e.current_label.toLowerCase().includes(lab)) return;
    if (dec && d.decision !== dec) return;
    shown++;
    if (d.decision && d.decision !== 'PENDING') decided++;
    const card = document.createElement('div');
    card.className = 'card';
    const imgHtml = e.image_b64
      ? `<img src="${{e.image_b64}}" alt="">`
      : `<div class="missing">${{e.image_exists ? 'UNREADABLE' : 'MISSING FILE'}}</div>`;
    card.innerHTML = `
      <span class="pri p${{e.review_priority}}">P${{e.review_priority}} ${{e.priority_label}}</span>
      <div class="lbl">${{e.current_label}}  <span style="color:#888;font-weight:400">→ ${{e.proposed_target_label}}</span></div>
      <div class="meta">proposed: ${{e.proposed_action}} &nbsp;|&nbsp; source: ${{e.source_exists}}</div>
      ${{imgHtml}}
      <div class="path">${{e.file}}</div>
      <div class="notes"><b>reason:</b> ${{e.reason}}<br><b>notes:</b> ${{e.reviewer_notes}}</div>
      <select class="decision" data-key="${{k}}">
        <option value="PENDING">PENDING</option>
        <option value="KEEP">KEEP</option>
        <option value="RELABEL">RELABEL</option>
        <option value="REMOVE">REMOVE</option>
      </select>
      <textarea class="evidence" data-key="${{k}}" placeholder="decision_evidence (required for RELABEL/REMOVE)"></textarea>
    `;
    const sel = card.querySelector('select');
    const ta = card.querySelector('textarea');
    sel.value = d.decision || 'PENDING';
    ta.value = d.evidence || '';
    sel.addEventListener('change', () => {{
      decisions[k] = decisions[k] || {{}};
      decisions[k].decision = sel.value;
      decisions[k].evidence = ta.value;
      saveDecisions(decisions);
      render();
    }});
    ta.addEventListener('input', () => {{
      decisions[k] = decisions[k] || {{}};
      decisions[k].decision = sel.value;
      decisions[k].evidence = ta.value;
      saveDecisions(decisions);
    }});
    grid.appendChild(card);
  }});
  document.getElementById('stats').textContent = `showing ${{shown}} of ${{DATA.length}} | decided: ${{decided}}`;
}}

function exportCsv() {{
  const cols = ['split','current_label','file','suggested_label','reason','reviewer_notes','resolved_source_path','source_exists','review_priority','proposed_action','proposed_target_label','final_user_decision','decision_evidence','review_status'];
  let csv = cols.join(',') + '\\n';
  DATA.forEach(e => {{
    const k = e.file;
    const d = decisions[k] || {{decision: e.final_user_decision, evidence: e.decision_evidence}};
    const decision = d.decision || 'PENDING';
    const evidence = (d.evidence || '').replace(/"/g,'""');
    const row = [
      e.split, e.current_label, e.file, e.suggested_label, e.reason, e.reviewer_notes,
      e.resolved_source_path, e.source_exists, e.review_priority, e.proposed_action,
      e.proposed_target_label, decision, '"' + evidence + '"', 'PENDING'
    ].map(v => (typeof v==='string' && v.includes(',') && !v.startsWith('"')) ? '"'+v.replace(/"/g,'""')+'"' : v);
    csv += row.join(',') + '\\n';
  }});
  const blob = new Blob([csv], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'V10_REVIEW_DECISIONS.csv';
  a.click();
}}

function clearAll() {{
  if (confirm('Clear all recorded decisions?')) {{
    decisions = {{}};
    saveDecisions(decisions);
    render();
  }}
}}

document.getElementById('priFilter').addEventListener('change', render);
document.getElementById('labelFilter').addEventListener('input', render);
document.getElementById('decFilter').addEventListener('change', render);
render();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(html)
    print(f"Wrote {OUT_HTML} ({OUT_HTML.stat().st_size} bytes)")
    print(f"Embedded {len(entries)} entries")
    missing = sum(1 for e in entries if not e['image_exists'])
    print(f"Missing images: {missing}")


if __name__ == "__main__":
    main()
