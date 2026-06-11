# LAWYER v0.8 Config-Driven Implementation

LAWYER means **Label-Aware Weak-label Yield Estimation and Refinement**.

This version removes dataset-specific hard-coding from the Python script. The script now reads labels, label groups, thresholds, source-matching rules, and non-target source classes from:

```text
configs/lawyer_v08_human_talk.json
```

## Files

```text
scripts/lawyer_refine_weak_labels_v08.py
scripts/run_lawyer_v08_refinement.ps1
configs/lawyer_v08_human_talk.json
docs/LAWYER_v08_config_driven.md
```

## Why this change matters

The previous prototype directly hard-coded labels such as:

```text
Brene_Brown
other_speaker_present
audience_reaction_present
silence_present
Les_Brown
```

That was fine for a fast prototype, but it reduces reusability. This config-driven version keeps the LAWYER algorithm generic and moves dataset-specific information into JSON.

## Run

```powershell
$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline"
$V08Root = "human_talk_workspace\tata_v0.8_raw_pipeline"

powershell -ExecutionPolicy Bypass -File scripts\run_lawyer_v08_refinement.ps1 `
  -V06Root "$V06Root" `
  -V08Root "$V08Root" `
  -Config "configs\lawyer_v08_human_talk.json" `
  -SegmentPredictionsCsv "$V06Root\raw_tata_pseudo_routing\raw_segment_predictions.csv" `
  -ParentCsv "$V06Root\raw_tata_pseudo_routing\hybrid\hybrid_parent_predictions_all.csv"
```

## Expected important check

After the run, the summary should show:

```text
known_non_target_rows_with_any_target_active = 0
```

This confirms that known non-target speakers are no longer mislabeled as one of the six target speakers.

## Reuse for a new dataset

For a new dataset, create a new config file, for example:

```text
configs/lawyer_v08_new_dataset.json
```

Change:

```text
labels
target_labels
open_set_label
event_labels
focus_labels
known_non_target_source_classes
thresholds
```

The Python script should not need to change.
