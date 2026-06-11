# LAWYER v0.8 Silence Signal Completion

This update finishes the pending part:

```text
silence_present = signal rule + TATA probability
```

## New file

```text
scripts/add_lawyer_silence_signal_features_v08.py
```

It reads:

```text
raw_segment_predictions.csv
```

and writes:

```text
raw_segment_predictions_with_silence_signal.csv
```

with these new columns:

```text
silence_rms_dbfs
silence_peak_dbfs
silence_zcr
silence_speech_activity_ratio
silence_is_acoustic_silent
silence_signal_error
```

## Updated files

```text
scripts/lawyer_refine_weak_labels_v08.py
scripts/run_lawyer_v08_refinement.ps1
configs/lawyer_v08_human_talk.json
```

## Run

```powershell
$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline"
$V08Root = "human_talk_workspace\tata_v0.8_raw_pipeline"

powershell -ExecutionPolicy Bypass -File scripts\run_lawyer_v08_refinement.ps1 `
  -V06Root "$V06Root" `
  -V08Root "$V08Root" `
  -Config "configs\lawyer_v08_human_talk.json" `
  -SegmentPredictionsCsv "$V06Root\raw_tata_pseudo_routing\raw_segment_predictions.csv" `
  -ParentCsv "$V06Root\raw_tata_pseudo_routing\hybrid\hybrid_parent_predictions_all.csv" `
  -AddSilenceSignalFeatures
```

## Interpretation

LAWYER now computes:

```text
silence_score = max(TATA_silence_score, acoustic_silence_score)
```

The acoustic silence flag is based on:

```text
RMS <= -45 dBFS
AND speech_activity_ratio <= 0.15
```

This is safer than using TATA alone.
