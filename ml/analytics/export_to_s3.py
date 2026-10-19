"""Analytics pipeline: stream escalated cases to S3 for Athena/QuickSight.

Runs periodically (e.g., via EventBridge) to export expert review queue
cases and all diagnosed cases to S3 as Parquet for Athena querying.

Run:
    python ml/analytics/export_to_s3.py --bucket <bucket> --prefix cases/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import boto3


def export_cases(bucket: str, prefix: str = "cases/") -> int:
    """Read all cases from expert review queue and write to S3 as JSONL."""
    cases_file = Path("apps/expert-review/data/cases.jsonl")
    feedback_file = Path("apps/expert-review/data/feedback.jsonl")

    if not cases_file.exists():
        print("No cases file found")
        return 0

    cases = []
    for line in cases_file.read_text().splitlines():
        if line.strip():
            cases.append(json.loads(line))

    # Enrich with feedback
    feedback_by_case = {}
    if feedback_file.exists():
        for line in feedback_file.read_text().splitlines():
            if line.strip():
                fb = json.loads(line)
                feedback_by_case[fb["case_id"]] = fb

    for c in cases:
        fb = feedback_by_case.get(c["case_id"], {})
        c["corrected_label"] = fb.get("corrected_label")
        c["approved"] = fb.get("approved")
        c["expert_notes"] = fb.get("notes")

    if not cases:
        print("No cases to export")
        return 0

    # Write as JSONL to S3 (Athena can query JSONL with a Glue table)
    key = f"{prefix}cases-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    body = "\n".join(json.dumps(c) for c in cases)

    s3 = boto3.client("s3", region_name="ap-south-1")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    print(f"Exported {len(cases)} cases to s3://{bucket}/{key}")
    return len(cases)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="S3 analytics bucket")
    ap.add_argument("--prefix", default="cases/", help="S3 key prefix")
    args = ap.parse_args()
    export_cases(args.bucket, args.prefix)


if __name__ == "__main__":
    main()
