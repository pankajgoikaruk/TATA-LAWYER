# NeuroAccuExit Hybrid Weak-Label Experiments

This document runs the first three downstream experiments using the final TATA-LAWYER v0.10.1 domain-aware hybrid weak-label manifest.

## Experiment ladder

| Experiment | Training loss / calibration | Purpose |
|---|---|---|
| v0.1 | Plain masked BCE, no positive weighting, fixed threshold 0.50 | Clean downstream baseline |
| v0.2 | Masked BCE with capped `pos_weight`, max 5.0, fixed threshold 0.50 | Improve rare-label recall without extreme reweighting |
| v0.3 | Per-label threshold tuning on validation for v0.1 and v0.2 checkpoints | Calibrate the trained models |

Do not rebuild TATA-LAWYER or alter the frozen v0.10.1 hybrid policy during these runs.

---

## 0. Local file placement

Save this file as:

```text
docs\COMMANDS_NEUROACCUEXIT_HYBRID_WEAKLABELS.md
```

Since Git pull is giving you issues, you can place it manually in the `docs` folder.

---

## 1. Shared path setup

Run this once per PowerShell session from the repository root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$RepoRoot = "C:\Users\wwwsa\PycharmProjects\TATA-LAWYER"

$V09Root = "$RepoRoot\human_talk_workspace\tata_v0.9_pipeline"
$RecoveredRoot = "$V09Root\tata_triage_model\silence_recovered_v09"

$HybridRoot = "$RepoRoot\human_talk_workspace\neuroaccuexit_hybrid_weaklabels"
$HybridManifest = "$HybridRoot\metadata\final_hybrid_weaklabel_manifest.csv"
$HybridReports = "$HybridRoot\reports"
$HybridRunsRoot = "$HybridRoot\runs"

$FeaturesRoot = "$RecoveredRoot\feature_cache\features"
$LabelsJson = "$V09Root\shared\human_talk_10label_schema.json"

New-Item -ItemType Directory -Path $HybridRunsRoot -Force | Out-Null
```

Preflight checks:

```powershell
Test-Path $HybridManifest
Test-Path $FeaturesRoot
Test-Path $LabelsJson
Test-Path training\train_multilabel_masked.py
Test-Path scripts\tune_v09_masked_per_label_thresholds.py
```

All should return `True`.

---

## 2. Experiment v0.1 — plain BCE baseline

```powershell
python -m training.train_multilabel_masked `
  --manifest "$HybridManifest" `
  --features_root "$FeaturesRoot" `
  --labels_json "$LabelsJson" `
  --runs_root "$HybridRunsRoot" `
  --variant "neuroaccuexit_hybrid_weaklabels_v01_plain_bce" `
  --tap_blocks "1,3" `
  --n_mels 64 `
  --epochs 40 `
  --batch_size 64 `
  --num_workers 0 `
  --log_every 25 `
  --lr 0.001 `
  --weight_decay 0.0 `
  --seed 42 `
  --threshold 0.5 `
  --loss_weights "0.3,0.3,1.0" `
  --device cpu
```

After training finishes, copy the printed run directory:

```powershell
$RunV01 = "PASTE_V01_RUN_DIRECTORY_HERE"
```

Check:

```powershell
Test-Path "$RunV01\metrics.json"
Test-Path "$RunV01\ckpt\best.pt"
```

---

## 3. Experiment v0.2 — capped positive weighting

```powershell
python -m training.train_multilabel_masked `
  --manifest "$HybridManifest" `
  --features_root "$FeaturesRoot" `
  --labels_json "$LabelsJson" `
  --runs_root "$HybridRunsRoot" `
  --variant "neuroaccuexit_hybrid_weaklabels_v02_posweight5" `
  --tap_blocks "1,3" `
  --n_mels 64 `
  --epochs 40 `
  --batch_size 64 `
  --num_workers 0 `
  --log_every 25 `
  --lr 0.001 `
  --weight_decay 0.0 `
  --seed 42 `
  --threshold 0.5 `
  --loss_weights "0.3,0.3,1.0" `
  --use_pos_weight `
  --pos_weight_max 5.0 `
  --device cpu
```

After training finishes, copy the printed run directory:

```powershell
$RunV02 = "PASTE_V02_RUN_DIRECTORY_HERE"
```

Check:

```powershell
Test-Path "$RunV02\metrics.json"
Test-Path "$RunV02\ckpt\best.pt"
```

---

## 4. Read v0.1 and v0.2 summary metrics

```powershell
function Show-FinalExitSummary($RunDir) {
  $m = Get-Content "$RunDir\metrics.json" -Raw | ConvertFrom-Json

  Write-Host "`nRUN: $RunDir"
  Write-Host "Best epoch:" $m.best.epoch
  Write-Host "Best strict-val Macro-F1:" $m.best.score

  foreach ($name in @("test_strict", "test_all_masked", "test_recovered_masked")) {
    $finalExit = $m.$name.exit_metrics[-1]
    [PSCustomObject]@{
      report     = $name
      rows       = $m.$name.rows
      macroF1    = $finalExit.macro_f1_known_labels
      microF1    = $finalExit.micro_f1_known_decisions
      samplesF1  = $finalExit.samples_f1_known_labels
      exactFull  = $finalExit.exact_match_fully_known_rows
      hamming    = $finalExit.masked_hamming_loss
    }
  }
}

Show-FinalExitSummary $RunV01 | Format-Table -AutoSize
Show-FinalExitSummary $RunV02 | Format-Table -AutoSize
```

---

## 5. Experiment v0.3 — per-label threshold tuning

Tune thresholds for v0.1:

```powershell
$TuneV01 = "$RunV01\threshold_tuning_strict_validation"

python scripts\tune_v09_masked_per_label_thresholds.py `
  --run_dir "$RunV01" `
  --manifest "$HybridManifest" `
  --features_root "$FeaturesRoot" `
  --labels_json "$LabelsJson" `
  --output_dir "$TuneV01" `
  --device cpu `
  --batch_size 64 `
  --num_workers 0 `
  --grid_start 0.05 `
  --grid_end 0.95 `
  --grid_step 0.01 `
  --fixed_threshold 0.50
```

Tune thresholds for v0.2:

```powershell
$TuneV02 = "$RunV02\threshold_tuning_strict_validation"

python scripts\tune_v09_masked_per_label_thresholds.py `
  --run_dir "$RunV02" `
  --manifest "$HybridManifest" `
  --features_root "$FeaturesRoot" `
  --labels_json "$LabelsJson" `
  --output_dir "$TuneV02" `
  --device cpu `
  --batch_size 64 `
  --num_workers 0 `
  --grid_start 0.05 `
  --grid_end 0.95 `
  --grid_step 0.01 `
  --fixed_threshold 0.50
```

Each tuning run writes:

```text
per_label_thresholds.json
threshold_tuning_by_label.csv
threshold_tuning_metrics.json
final_exit_probabilities_and_targets.npz
```

---

## 6. Compare tuned outputs

```powershell
function Show-TuningSummary($TuneDir) {
  $m = Get-Content "$TuneDir\threshold_tuning_metrics.json" -Raw | ConvertFrom-Json
  $fixed = $m.test_strict_fixed_0_5
  $tuned = $m.test_strict_tuned

  Write-Host "`nTUNING: $TuneDir"

  [PSCustomObject]@{ metric = "Macro-F1"; fixed = $fixed.macro_f1; tuned = $tuned.macro_f1; change = $tuned.macro_f1 - $fixed.macro_f1 }
  [PSCustomObject]@{ metric = "Micro-F1"; fixed = $fixed.micro_f1; tuned = $tuned.micro_f1; change = $tuned.micro_f1 - $fixed.micro_f1 }
  [PSCustomObject]@{ metric = "Samples-F1"; fixed = $fixed.samples_f1; tuned = $tuned.samples_f1; change = $tuned.samples_f1 - $fixed.samples_f1 }
  [PSCustomObject]@{ metric = "Exact Match"; fixed = $fixed.exact_match; tuned = $tuned.exact_match; change = $tuned.exact_match - $fixed.exact_match }
  [PSCustomObject]@{ metric = "Hamming Loss"; fixed = $fixed.hamming_loss; tuned = $tuned.hamming_loss; change = $tuned.hamming_loss - $fixed.hamming_loss }
}

Show-TuningSummary $TuneV01 | Format-Table -AutoSize
Show-TuningSummary $TuneV02 | Format-Table -AutoSize
```

Inspect per-label tuning:

```powershell
Import-Csv "$TuneV01\threshold_tuning_by_label.csv" | Format-Table -AutoSize
Import-Csv "$TuneV02\threshold_tuning_by_label.csv" | Format-Table -AutoSize
```

---

## 7. Decision rule

| Result pattern | Selected downstream model |
|---|---|
| v0.1 fixed is strongest | Plain BCE baseline |
| v0.1 tuned is strongest | Plain BCE + per-label thresholds |
| v0.2 fixed is strongest | Capped pos_weight model |
| v0.2 tuned is strongest | Capped pos_weight + per-label thresholds |

Prefer the model that improves Macro-F1 and Samples-F1 without damaging Exact Match and Hamming Loss too much.

---

## 8. Current status

```text
Hybrid weak-label manifest: created locally
v0.1 plain BCE: not run yet
v0.2 pos_weight<=5: not run yet
v0.3 threshold tuning: not run yet
```
