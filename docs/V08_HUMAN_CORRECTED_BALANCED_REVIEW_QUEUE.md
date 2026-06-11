# v0.8-human-corrected-balanced Review Queue Preparation

Adds:

```text
scripts/prepare_v08_human_corrected_balanced_review_queue.py
```

Run:

```powershell
$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline"
$V08Root = "human_talk_workspace\tata_v0.8_raw_pipeline"
$HCBBRoot = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline"

python scripts\prepare_v08_human_corrected_balanced_review_queue.py `
  --v06_root "$V06Root" `
  --v08_root "$V08Root" `
  --out_root "$HCBBRoot" `
  --labels_json "configs\human_talk_10label_schema.json" `
  --lawyer_config "configs\lawyer_v08_human_talk.json"
```

Output folder:

```text
human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\manual_review_queue\
```

Main editable files:

```text
01_lawyer_needs_review_FULL.csv
02_raw_hybrid_nontarget_context_review.csv
03_holdout_nontarget_context_check.csv
04_lawyer_event_positive_review.csv
05_lawyer_nontarget_context_review.csv
06_v08_training_review_master_deduplicated.csv
07_holdout_context_review_master.csv
```

Do not edit frozen reference files under:

```text
frozen_reference\lawyer_v08_auto\
```
