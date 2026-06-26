# Evaluate v0.2 tuned NeuroAccuExit on untouched human ground-truth holdout

Run these commands from:

```text
C:\Users\wwwsa\PycharmProjects\TATA-LAWYER
```

## 0. Save the evaluator script

Save the downloaded script as:

```text
scripts\evaluate_neuroaccuexit_v03_on_holdout.py
```

No GitHub pull/push is required.

---

## 1. Shared setup

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$RepoRoot = "C:\Users\wwwsa\PycharmProjects\TATA-LAWYER"

$V09Root = "$RepoRoot\human_talk_workspace\tata_v0.9_pipeline"
$RecoveredRoot = "$V09Root\tata_triage_model\silence_recovered_v09"

$HybridRoot = "$RepoRoot\human_talk_workspace\neuroaccuexit_hybrid_weaklabels"

$RunV02 = "$HybridRoot\runs\neuroaccuexit_hybrid_weaklabels_v02_posweight5_20260625_202246"
$TuneV02 = "$RunV02\threshold_tuning_strict_validation"
$ThresholdsV02 = "$TuneV02\per_label_thresholds.json"

$LabelsJson = "$V09Root\shared\human_talk_10label_schema.json"

$HoldoutRoot = "$HybridRoot\untouched_ground_truth_holdout_v03"
$HoldoutSegmentsRoot = "$HoldoutRoot\segments"
$HoldoutFeatureCache = "$HoldoutRoot\feature_cache"
$HoldoutEvalOut = "$HoldoutRoot\evaluation_v02_tuned"

New-Item -ItemType Directory -Path $HoldoutRoot -Force | Out-Null
```

---

## 2. Find the untouched holdout CSV

First search:

```powershell
Get-ChildItem "$RepoRoot\human_talk_workspace" -Recurse -Filter "*GROUND_TRUTH_FINAL*.csv" |
  Select-Object FullName, LastWriteTime, Length |
  Sort-Object LastWriteTime -Descending |
  Format-Table -AutoSize
```

You are looking for:

```text
01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv
```

Then set:

```powershell
$HoldoutCsv = "PASTE_FULL_PATH_TO_01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv"
```

Example format:

```powershell
$HoldoutCsv = "C:\Users\wwwsa\PycharmProjects\TATA-LAWYER\human_talk_workspace\...\01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv"
```

---

## 3. Preflight checks

```powershell
Test-Path $HoldoutCsv
Test-Path $LabelsJson
Test-Path "$RunV02\ckpt\best.pt"
Test-Path "$RunV02\config_used.json"
Test-Path $ThresholdsV02
Test-Path scripts\build_tata_holdout_segments.py
Test-Path scripts\extract_multilabel_features.py
Test-Path scripts\evaluate_neuroaccuexit_v03_on_holdout.py
```

All should return `True`.

---

## 4. Build holdout 1-second segment manifest

```powershell
python scripts\build_tata_holdout_segments.py `
  --holdout_csv "$HoldoutCsv" `
  --labels_json "$LabelsJson" `
  --out_dir "$HoldoutSegmentsRoot" `
  --sample_rate 16000 `
  --segment_sec 1.0 `
  --hop_sec 1.0 `
  --include_tail `
  --split_name "test" `
  --task_name "untouched_human_ground_truth_holdout_v03" `
  --labeling_level "untouched_human_ground_truth"
```

Expected output files:

```text
$HoldoutSegmentsRoot\metadata\final_holdout_segment_manifest.csv
$HoldoutSegmentsRoot\metadata\final_holdout_segment_summary.json
$HoldoutSegmentsRoot\segment_wavs\
```

---

## 5. Extract log-mel features for holdout segments

```powershell
$HoldoutSegmentManifest = "$HoldoutSegmentsRoot\metadata\final_holdout_segment_manifest.csv"
$HoldoutSegmentLabelsJson = "$HoldoutSegmentsRoot\metadata\labels.json"

python scripts\extract_multilabel_features.py `
  --manifest "$HoldoutSegmentManifest" `
  --labels_json "$HoldoutSegmentLabelsJson" `
  --out_cache "$HoldoutFeatureCache" `
  --sample_rate 16000 `
  --clip_sec 1.0 `
  --n_mels 64 `
  --n_fft 1024 `
  --win_ms 25 `
  --hop_ms 10 `
  --cmvn `
  --progress_every 200
```

Expected output files:

```text
$HoldoutFeatureCache\metadata\multilabel_features_manifest.csv
$HoldoutFeatureCache\features\test\*.npy
```

---

## 6. Evaluate v0.2 tuned on untouched holdout

```powershell
$HoldoutFeatureManifest = "$HoldoutFeatureCache\metadata\multilabel_features_manifest.csv"
$HoldoutFeaturesRoot = "$HoldoutFeatureCache\features"

python scripts\evaluate_neuroaccuexit_v03_on_holdout.py `
  --run_dir "$RunV02" `
  --feature_manifest "$HoldoutFeatureManifest" `
  --features_root "$HoldoutFeaturesRoot" `
  --labels_json "$HoldoutSegmentLabelsJson" `
  --thresholds_json "$ThresholdsV02" `
  --output_dir "$HoldoutEvalOut" `
  --device cpu `
  --batch_size 64 `
  --num_workers 0 `
  --fixed_threshold 0.50 `
  --parent_col "parent_clip_id"
```

---

## 7. Read final result

```powershell
$m = Get-Content "$HoldoutEvalOut\holdout_v03_eval_summary.json" -Raw | ConvertFrom-Json

$m.segment_final_exit | Select-Object rows, macro_f1, micro_f1, samples_f1, exact_match, hamming_loss

$m.parent_reports.mean | Select-Object rows, macro_f1, micro_f1, samples_f1, exact_match, hamming_loss

$m.parent_reports.max | Select-Object rows, macro_f1, micro_f1, samples_f1, exact_match, hamming_loss
```

Per-label results:

```powershell
Import-Csv "$HoldoutEvalOut\holdout_v03_per_label_segment.csv" | Format-Table -AutoSize
Import-Csv "$HoldoutEvalOut\holdout_v03_per_label_parent_mean.csv" | Format-Table -AutoSize
Import-Csv "$HoldoutEvalOut\holdout_v03_per_label_parent_max.csv" | Format-Table -AutoSize
```

---

## 8. What to paste back

Paste:

```text
Final-exit holdout results
Segment-level
Parent-mean
Parent-max
Per-label parent_mean table
Per-label parent_max table
```
