# V10 Consolidated Review

Consolidated human-review tracker for the proposed PlantDoc training-data
corrections. No dataset files, labels, or checkpoints were changed.

## Training-change entries

- Total consolidated training entries: **56**
  (9 prior candidates + 47 pending entries, deduplicated by file path)
- Duplicate paths found: **0**
- Unapproved decisions (still PENDING): **56**

## Review priority counts

- Priority 1 (Missing file): 2
- Priority 2 (Multi-panel/unsuitable): 3
- Priority 3 (Possible duplicate): 5
- Priority 4 (Possible label mismatch): 46

## Test-set entries (separate)

- 12 pending test-set entries preserved in `V10_TEST_REVIEW.csv`.
- These are NOT mixed into the training-change list.
- Test split is byte-for-byte identical to V3 (225 images, hash set unchanged).

## Dataset-integrity warning (V4)

The V4 dataset contains a byte-identical duplicate pair with conflicting
labels:

- `train/Potato leaf early blight/18028_1.jpg`
- `train/Potato leaf late blight/24064_1.jpg`

Both files have the same SHA-256 hash but different class folders. V3
already removed both copies. **Do not use V4 unchanged.** Neither file
was added to the V3 proposed changes.

## Candidate vs verified mismatch

A *candidate label mismatch* means the filename or manifest suggests a
different label than the current folder. This is a clue, not proof.
A *verified labeling error* requires visual confirmation of the leaf
morphology and disease symptoms, and ideally source documentation.

Of the 56 entries, 46 are
flagged as possible label mismatches. Only the 9 candidates with
RELABEL/REMOVE decisions have any visual review notes; the remaining
entries have not been visually verified.

## Files

- `V10_CONSOLIDATED_REVIEW.csv` - consolidated training-change tracker
- `V10_TEST_REVIEW.csv` - 12 test-set entries (separate)
- `v10_pending_training_review.csv` - source 47 pending entries
- `v10_label_review_worksheet.csv` - source 9 candidates

## Validation results

- Row counts: training 56, test 12
- Duplicate paths: 0
- All unapproved decisions PENDING: True
- Test split unchanged: yes (225 images, hash-identical to V3)
- No dataset files, checkpoints, or models modified: yes

## Next step

Human review of each entry. Set `final_user_decision` to KEEP, RELABEL,
or REMOVE with supporting `decision_evidence`. Do not build a corrected
dataset or train V10 without explicit approval.
