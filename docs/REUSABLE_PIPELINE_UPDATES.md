# Reusable Pipeline Updates

This package makes the remaining hard-coded scripts reusable.

## 1. `scripts/build_tata_holdout_segments.py`

Old issue:

```text
LABELS = [...]
```

was hard-coded inside the script.

New version:

```powershell
python scripts\build_tata_holdout_segments.py `
  --holdout_csv "..." `
  --labels_json "configs\human_talk_10label_schema.json" `
  --out_dir "..."
```

For a new dataset, create a new labels JSON and pass it through `--labels_json`.

## 2. `scripts/filter_tata_v07_remove_nontarget_sources.py`

Old issue:

```text
EXCLUDE_CLASSES = {...}
files = [...]
```

were hard-coded.

New version:

```powershell
python scripts\filter_tata_v07_remove_nontarget_sources.py `
  --config "configs\filter_v07_target_only.json" `
  --src_root "human_talk_workspace\tata_v0.6_raw_pipeline" `
  --dst_root "human_talk_workspace\tata_v0.7_raw_pipeline"
```

For a new dataset or ablation, create a new filter config.

## 3. `scripts/lawyer_refine_weak_labels_v08.py`

This is already config-driven after the previous update. It should use:

```powershell
python scripts\lawyer_refine_weak_labels_v08.py `
  --config "configs\lawyer_v08_human_talk.json" `
  --segment_predictions_csv "..." `
  --out_dir "..."
```

## Files to copy

```text
scripts/build_tata_holdout_segments.py
scripts/filter_tata_v07_remove_nontarget_sources.py
configs/human_talk_10label_schema.json
configs/filter_v07_target_only.json
docs/REUSABLE_PIPELINE_UPDATES.md
```
