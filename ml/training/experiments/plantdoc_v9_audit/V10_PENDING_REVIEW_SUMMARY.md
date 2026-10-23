# V10 Pending Training-Entry Review

Read-only audit of the 47 remaining `PENDING` training entries in the
reviewed manifest. No dataset files, labels, or checkpoints were changed.

## Counts

- Total remaining pending training entries: **47**
- Missing files: **2** (the byte-identical duplicate pair already
  removed in V3: `18028_1.jpg`, `24064_1.jpg`)
- Unreadable files: **0**

## Review categories (entries may appear in multiple)

- **LOW_QUALITY_OR_AMBIGUOUS**: 1
- **MISSING_FILE**: 2
- **MULTI_PANEL_OR_UNSUITABLE**: 1
- **POSSIBLE_DUPLICATE**: 5
- **POSSIBLE_LABEL_MISMATCH**: 45

## By current label

- Potato leaf early blight: 18
- Potato leaf late blight: 10
- Corn leaf blight: 6
- Tomato Septoria leaf spot: 5
- Bell_pepper leaf spot: 4
- Tomato mold leaf: 2
- Tomato leaf: 1
- Tomato leaf late blight: 1

## Notes

- `POSSIBLE_LABEL_MISMATCH` (45 entries) indicates the filename-derived
  `suggested_label` differs from `current_label`. This is a hint from
  filename tokens, not proof of mislabeling.
- `POSSIBLE_DUPLICATE` (5 entries) are perceptual near-duplicates
  (Hamming distance <= 8 on a 16x16 hash). Verify visually.
- `MULTI_PANEL_OR_UNSUITABLE` (1 entry): a comparison figure.
- `LOW_QUALITY_OR_AMBIGUOUS` (1 entry): 150x150 thumbnail.
- `MISSING_FILE` (2 entries): the duplicate pair removed in V3.

## Generated files

- `v10_pending_training_review.csv` - full worksheet (47 rows)
- `pending_review_contact_sheets/` - 8 contact sheets grouped by current label
- `v10_pending_review_summary.json` - machine-readable counts

## Safety

All operations were read-only. No V2/V3/V4 files were modified. No
test-set entries were changed. No model was trained. All
`final_user_decision` values are `PENDING`.

## Next step

Human review of the contact sheets and worksheet. Approve or reject each
entry. Do not build a corrected dataset or train V10 without explicit
approval.
