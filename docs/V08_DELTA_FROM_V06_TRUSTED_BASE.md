# v0.8 Delta Queue from v0.6 Trusted Base

This package adds:

```text
scripts/prepare_v08_delta_from_v06_trusted_base.py
```

## Purpose

Use v0.6 as trusted base and create a small review queue:

```text
v0.6 trusted data
+ LAWYER changed-label suggestions only
+ LAWYER new accepted/warning samples only
+ raw non-target background/event correction
+ holdout non-target background/event correction
```

This avoids re-reviewing all old v0.6 rows.

## Run

```powershell
$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline"
$V08Root = "human_talk_workspace\tata_v0.8_raw_pipeline"
$HCBBRoot = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline"

# Optional: archive old full v0.8 queue if it exists.
if (Test-Path "$HCBBRoot\manual_review_queue") {
  Rename-Item "$HCBBRoot\manual_review_queue" "manual_review_queue_AUTO_FULL_OLD_DO_NOT_EDIT"
}

python scripts\prepare_v08_delta_from_v06_trusted_base.py `
  --v06_root "$V06Root" `
  --v08_root "$V08Root" `
  --out_root "$HCBBRoot" `
  --labels_json "configs\human_talk_10label_schema.json" `
  --lawyer_config "configs\lawyer_v08_human_talk.json"
```

## Output files

```text
manual_review_queue/
  00_README_DELTA_REVIEW_INSTRUCTIONS.md
  00_review_queue_summary.json
  00_v06_raw_hybrid_corrected_TRUSTED_COPY_DO_NOT_EDIT.csv
  00_v06_holdout_TRUSTED_COPY_DO_NOT_EDIT.csv
  00_lawyer_v08_parent_labels_all_AUTO_COPY_DO_NOT_EDIT.csv
  01_v06_trusted_base_index.csv
  03_raw_nontarget_context_REVIEW.csv
  04_holdout_nontarget_context_REVIEW.csv
  05_lawyer_delta_changed_labels_REVIEW.csv
  06_lawyer_new_samples_REVIEW.csv
  07_training_delta_master_REVIEW.csv
```

## What to edit

### Edit first

```text
03_raw_nontarget_context_REVIEW.csv
04_holdout_nontarget_context_REVIEW.csv
```

For non-target speakers, keep target identity fixed and edit only:

```text
music_present
audience_reaction_present
silence_present
```

### Edit next

```text
05_lawyer_delta_changed_labels_REVIEW.csv
```

This is prefilled from v0.6. Change labels only if LAWYER is correct.

### Edit last

```text
06_lawyer_new_samples_REVIEW.csv
```

These are not in v0.6 trusted base, so review before training.

### Use for training later

```text
07_training_delta_master_REVIEW.csv
```

Only after you finish manual correction.
