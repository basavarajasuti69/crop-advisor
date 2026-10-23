# V10 Label Review Guide

This guide supports manual review of the nine proposed training-data
changes before any dataset is built or any model is trained.

## How to review each candidate

1. Open the image at `original_path` in an image viewer at full resolution.
2. Compare the visible symptoms with the `current_label` and the
   `proposed_target_label`.
3. Read `existing_review_notes` for the contact-sheet observation.
4. Record your decision in `final_user_decision`:
   - `KEEP` - leave the image under its current label.
   - `RELABEL` - move it to `proposed_target_label` (must be an existing
     class; fill `decision_evidence` with the visual reason).
   - `REMOVE` - exclude it from training (multi-panel or unidentifiable
     image).
   - `PENDING` - not yet decided.
5. Fill `decision_evidence` with the concrete visual evidence (lesion
   shape, colour, vein pattern, leaf morphology) or the source/expert
   reference that supports the decision.

## Important caveats

- Visual symptoms alone may not establish the exact disease or plant
  species. A leaf with concentric rings could be early blight,
  Septoria, or another Alternaria disease. A yellow, curled leaf could
  be TYLC, nutrient deficiency, or herbicide injury.
- Filename and folder names are hints, not ground truth. The current
  labels were assigned by the original dataset curators and may be
  wrong, but they may also be right.
- Do not approve a relabel unless you can state the visual evidence.
  When in doubt, keep the image (`KEEP`) or exclude it (`REMOVE`).
- Multi-panel comparison figures (the two REMOVE candidates) should
  generally be excluded because they contain more than one disease or
  a captioned comparison, which confuses single-label training.
- Do not modify test-set images or labels. The 12 test-set candidates
  are held for separate review and are not part of this worksheet.

## Candidate summary

- `Corn-gray-leaf-spot-8-14-15-F1.jpg` (Corn leaf blight -> Corn Gray leaf spot) [RELABEL]
- `tylc.jpg` (Tomato mold leaf -> Tomato leaf yellow virus) [RELABEL]
- `corn_gray_leaf_spot.jpg` (Corn leaf blight -> Corn Gray leaf spot) [RELABEL]
- `early_blight.jpg` (Potato leaf late blight -> Potato leaf early blight) [RELABEL]
- `1573852-Early-Blight-e1515624612559.jpg` (Potato leaf early blight -> Tomato Early blight leaf) [RELABEL]
- `gray-leaf-spot-lesions-F4.jpg` (Corn leaf blight -> Corn Gray leaf spot) [RELABEL]
- `gray-leaf-spot-severe--.jpg` (Corn leaf blight -> Corn Gray leaf spot) [RELABEL]
- `potato_ocw_botrytis-lateblight.jpg` (Potato leaf late blight -> (excluded from training)) [REMOVE]
- `potato_ocw_lateblight-earlyblight.jpg` (Potato leaf late blight -> (excluded from training)) [REMOVE]

## Files

- Worksheet: `v10_label_review_worksheet.csv`
- Audit report: `v10_training_change_review.csv`
- Manifest: `v9_cleaning_manifest_claude_reviewed.csv`

## After review

Return the completed worksheet. Only after explicit approval will the
cleaned dataset be built and V10 training configured.
