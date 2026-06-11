# Apply v0.8 Reviewed Delta Outputs

Run this after you finish:

```text
03_raw_nontarget_context_REVIEW.csv
04_holdout_nontarget_context_REVIEW.csv
06_lawyer_new_samples_REVIEW.csv
```

## Run

```powershell
$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline"
$HCBBRoot = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline"

python scripts\apply_v08_reviewed_delta_outputs.py `
  --v06_root "$V06Root" `
  --out_root "$HCBBRoot" `
  --labels_json "configs\human_talk_10label_schema.json"
```

Strict version:

```powershell
python scripts\apply_v08_reviewed_delta_outputs.py `
  --v06_root "$V06Root" `
  --out_root "$HCBBRoot" `
  --labels_json "configs\human_talk_10label_schema.json" `
  --require_reviewed
```

## Main outputs

```text
corrected_training_inputs/
  02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_v08_context_checked.csv
  06_lawyer_new_samples_REVIEWED_v08.csv
  hybrid_accepted_with_warning_PLUS_REVIEWED_LAWYER_NEW_v08.csv

corrected_holdout/
  01_raw_final_holdout_GROUND_TRUTH_FINAL_v08_context_checked.csv
```
