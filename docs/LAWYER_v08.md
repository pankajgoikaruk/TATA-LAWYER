# LAWYER v0.8 — Label-Aware Weak-label Yield Estimation and Refinement

LAWYER is the v0.8 research direction for improving weak labels in the ASHADIP / NeuroAccuExit TATA-assisted audio pipeline.

## Motivation

The weak labels are not equally difficult. Speaker identity labels, open-set speaker labels, audience reactions, and silence events have different temporal and semantic behaviour. Therefore, a single confidence threshold is not ideal.

LAWYER applies label-specific refinement rules for:

```text
other_speaker_present
audience_reaction_present
silence_present
```

while preserving the existing 10-label schema.

## Label-specific rules

### 1. Known speaker identity

For each target speaker label `l`:

```text
score_l = alpha * mean_t p_t(l) + (1 - alpha) * max_t p_t(l)
```

Default:

```text
alpha = 0.70
threshold = 0.50
```

Rationale: speaker identity should be stable across the parent clip, so mean confidence is important, while max confidence still helps when only part of the clip is highly discriminative.

### 2. Open-set `other_speaker_present`

LAWYER estimates other-speaker presence using direct and open-set evidence.

```text
other_direct = max_t p_t(other_speaker_present)
speech_like = max(other_direct, max_target_segment_probability)
open_set_score = speech_like * (1 - known_target_score)
```

Then:

```text
other_speaker_present = 1
if source class is known non-target
or other_direct >= threshold_direct
or (speech_like >= threshold_speech and known_target_score <= threshold_known)
```

This is designed for rows from:

```text
Les_Brown
Mel_Robbins
Oprah_Winfrey
Rabin_Sharma
Simon_Sinek
```

and future unknown speakers.

### 3. Transient `audience_reaction_present`

Audience reactions are short burst events. Mean aggregation can hide them.

LAWYER uses top-k aggregation:

```text
score_audience = mean(top-k segment probabilities)
```

Default:

```text
k = 2
threshold = 0.50
uncertain zone = [0.35, 0.65]
```

### 4. `silence_present`

Silence can be inferred from neural predictions and optional signal features.

Default neural rule:

```text
score_silence = max_t p_t(silence_present)
```

Optional acoustic rule:

```text
silent_segment = RMS <= threshold_energy or VAD <= threshold_vad
```

Final:

```text
score_silence = max(TATA_silence_score, acoustic_silence_score)
```

## Main script

```text
scripts/lawyer_refine_weak_labels_v08.py
```

## Runner

```text
scripts/run_lawyer_v08_refinement.ps1
```

## Example command

```powershell
$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline"
$V08Root = "human_talk_workspace\tata_v0.8_raw_pipeline"

powershell -ExecutionPolicy Bypass -File scripts\run_lawyer_v08_refinement.ps1 `
  -V06Root "$V06Root" `
  -V08Root "$V08Root" `
  -SegmentPredictionsCsv "$V06Root\raw_tata_pseudo_routing\raw_segment_predictions.csv" `
  -ParentCsv "$V06Root\raw_tata_pseudo_routing\hybrid\hybrid_parent_predictions_all.csv"
```

## Outputs

```text
human_talk_workspace\tata_v0.8_raw_pipeline\raw_tata_pseudo_routing\lawyer_v08\
```

Important files:

```text
lawyer_v08_accepted.csv
lawyer_v08_accepted_with_warning.csv
lawyer_v08_needs_review.csv
lawyer_v08_rejected.csv
lawyer_v08_parent_labels_all.csv
lawyer_v08_manual_review_prefill.csv
lawyer_v08_summary.md
```

## Rebuild the final expanded manifest

Use the existing final-manifest builder, but pass LAWYER outputs:

```powershell
$ScratchRoot = "human_talk_workspace\tata_v0.6_scratch"
$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline"
$V08Root = "human_talk_workspace\tata_v0.8_raw_pipeline"
$LawyerRoot = "$V08Root\raw_tata_pseudo_routing\lawyer_v08"

python scripts\build_tata_v06_final_expanded_training_manifest.py `
  --seed_feature_manifest "$ScratchRoot\feature_cache\metadata\multilabel_features_manifest.csv" `
  --seed_features_root "$ScratchRoot\feature_cache\features" `
  --raw_feature_manifest "$V06Root\raw_pseudo_pool_feature_cache\metadata\multilabel_features_manifest.csv" `
  --raw_features_root "$V06Root\raw_pseudo_pool_feature_cache\features" `
  --hybrid_accepted_csv "$LawyerRoot\lawyer_v08_accepted.csv" `
  --hybrid_warning_csv "$LawyerRoot\lawyer_v08_accepted_with_warning.csv" `
  --corrected_needs_review_csv "$LawyerRoot\lawyer_v08_manual_review_prefill.csv" `
  --labels_json "$ScratchRoot\metadata\tata_v06_labels.json" `
  --out_root "$V08Root\final_expanded_training_dataset"
```

Important: before using `lawyer_v08_manual_review_prefill.csv` as corrected needs-review, manually inspect/correct the `needs_review` rows if you want a strict human-in-the-loop experiment. Otherwise, report it as an automatic weak-label refinement ablation.

## Evaluation plan

Compare:

| Version | Meaning |
|---|---|
| v0.6 | original full TATA weak-label pipeline |
| v0.7 | filtered six-target-speaker ablation |
| v0.8 LAWYER | full 10-label label-aware weak-label refinement |

Primary focus labels:

```text
other_speaker_present
audience_reaction_present
silence_present
```

Use final raw holdout parent-level mean aggregation as the headline evaluation.
