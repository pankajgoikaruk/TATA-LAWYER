# TATA v0.9 PowerShell Command History

This document records the reproducible PowerShell/Python command sequence used for the TATA v0.9 seed audit, feature-cache reconstruction, low-energy recovery, training, and manual-review preparation.

Run commands from:

```text
C:\Users\wwwsa\PycharmProjects\TATA-LAWYER
```

Activate the environment first:

```powershell
conda activate ASHADIP_V0
```

---

## 0. Common environment variables

**Purpose:** make local modules importable and avoid the Windows OpenMP duplicate-library error.

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

Canonical v0.9 roots:

```powershell
$V09Root = "human_talk_workspace\tata_v0.9_pipeline"
$TataRoot = "$V09Root\tata_triage_model"
$LabelsJson = "$V09Root\shared\human_talk_10label_schema.json"
```

---

## 1. Create the non-destructive v0.9 workspace

**Purpose:** create the canonical v0.9 directory structure and copy immutable baseline manifests without modifying v0.6/v0.8 assets.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_v09_workspace_step1.ps1
```

Expected baseline assets:

```text
tata2_parent_manifest_12label_2074_BASELINE.csv
tata_seed_features_manifest_10label_12469_BASELINE.csv
human_talk_10label_schema.json
```

Expected inventory:

```text
human_talk_workspace\tata_v0.9_pipeline\shared\correction_ledgers\v09_baseline_inventory.csv
```

---

## 2. Prepare the existing silence-review queue

**Purpose:** extract the old seed silence-related parent rows and review audio without changing the baseline manifest.

Default-path invocation used by the workflow:

```powershell
python scripts\prepare_v09_existing_silence_review.py
```

The review stage produced files such as:

```text
silence_existing_review_sheet_v09.csv
silence_existing_missing_or_ambiguous_v09.csv
```

### Repair unsupported/mixed audio extensions

**Purpose:** repair or export review audio whose original extension was not directly playable.

```powershell
python scripts\repair_v09_silence_review_audio_extensions.py
```

> Historical note: the exact CLI subcommands used to apply the completed existing-silence and audience-reaction review sheets were not present in the supplied terminal logs. The resulting manifests are preserved in the v0.9 metadata directory and documented in the experiment log. Do not reconstruct missing subcommands by guessing; use the local script help if this stage must be rerun:
>
> ```powershell
> python scripts\v09_seed_review_workflow.py --help
> ```

---

## 3. Rename the 27 verified silence files safely

**Purpose:** rename new silence clips after the existing maximum index so they cannot overwrite existing files.

```powershell
$HCBBRoot = "human_talk_workspace\tata_v0.9_pipeline"

python scripts\rename_wavs_by_class.py `
  --root "dataset\new_verified_rare_event_audio" `
  --separator "__" `
  --reference_root "dataset\human_talk_tata_seed_dataset" `
  --reference_manifest "$HCBBRoot\tata_triage_model\metadata\tata_seed_parent_manifest_v09_RARE_EVENTS_CORRECTED.csv" `
  --manifest "$HCBBRoot\shared\correction_ledgers\v09_new_silence_rename_manifest.csv" `
  --apply
```

Expected naming range:

```text
silence__0088
...
silence__0114
```

Expected files renamed: `27`.

---

## 4. Add the 27 verified silence parents

**Purpose:** append the verified silence parents to the reviewed v0.9 parent manifest without editing the baseline.

```powershell
python scripts\v09_seed_review_workflow.py add-new-silence `
  --new_silence_root "dataset\new_verified_rare_event_audio\silence"
```

Expected final parent count:

```text
2,074 + 27 = 2,101
```

---

## 5. Locate the legacy 12,469-feature cache

**Purpose:** reuse the verified legacy `.npy` arrays rather than recomputing unchanged data.

```powershell
$OldFeatures = "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP\human_talk_workspace\tata_v0.6_scratch\feature_cache\features"

Test-Path $OldFeatures
(Get-ChildItem $OldFeatures -Recurse -Filter *.npy -File).Count
```

Expected:

```text
True
12469
```

---

## 6. Build the initial v0.9 feature cache — dry run

**Purpose:** reuse 12,469 legacy features, extract the 27 new silence parents, validate metadata, and preview counts without writing files.

```powershell
python scripts\build_v09_tata_feature_cache.py `
  --old_features_root "$OldFeatures"
```

Expected core counts:

```text
Final parents:               2,101
Old segment rows:           12,469
New silence segment rows:      120
Combined segment rows:      12,589
Unique parents:              2,101
```

---

## 7. Build the initial v0.9 feature cache — apply

**Purpose:** create the audited v0.9 feature cache and final manifest.

```powershell
python scripts\build_v09_tata_feature_cache.py `
  --old_features_root "$OldFeatures" `
  --apply `
  --overwrite
```

Canonical output:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\feature_cache\metadata\multilabel_features_manifest_v09_FINAL.csv
```

---

## 8. Validate the initial v0.9 feature manifest

**Purpose:** confirm row counts, parent counts, missing feature files, duplicate paths, and split safety.

```powershell
$V09 = "human_talk_workspace\tata_v0.9_pipeline"
$Manifest = "$V09\tata_triage_model\feature_cache\metadata\multilabel_features_manifest_v09_FINAL.csv"
$FeaturesRoot = "$V09\tata_triage_model\feature_cache\features"
$LabelsJson = "$V09\shared\human_talk_10label_schema.json"

$Rows = Import-Csv $Manifest

"Manifest rows: $($Rows.Count)"
"Labels: $((Get-Content $LabelsJson | ConvertFrom-Json).labels.Count)"

$Missing = @(
    $Rows | Where-Object {
        $FeaturePath = Join-Path $FeaturesRoot ($_.feat_relpath -replace "/", "\")
        -not (Test-Path $FeaturePath)
    }
).Count

"Missing features: $Missing"
```

Expected:

```text
Manifest rows: 12589
Labels: 10
Missing features: 0
```

---

## 9. Train the original v0.9 three-exit TATA model

**Purpose:** establish the clean v0.9 segment-level ten-label baseline using the audited 12,589-row manifest.

```powershell
$V09 = "human_talk_workspace\tata_v0.9_pipeline"

$Manifest = "$V09\tata_triage_model\feature_cache\metadata\multilabel_features_manifest_v09_FINAL.csv"
$FeaturesRoot = "$V09\tata_triage_model\feature_cache\features"
$LabelsJson = "$V09\shared\human_talk_10label_schema.json"
$RunsRoot = "$V09\tata_triage_model\runs"

python -m training.train_multilabel `
  --manifest "$Manifest" `
  --features_root "$FeaturesRoot" `
  --labels_json "$LabelsJson" `
  --runs_root "$RunsRoot" `
  --variant "tata_v09_human_corrected_3exit" `
  --tap_blocks "1,3" `
  --epochs 40 `
  --batch_size 64 `
  --num_workers 0 `
  --log_every 25 `
  --lr 0.001 `
  --threshold 0.5 `
  --device cpu
```

Recorded final-exit result:

```text
Macro-F1  = 0.8195
Micro-F1  = 0.8226
Samples-F1= 0.8226
Exact     = 0.6527
Hamming   = 0.0483
```

---

## 10. Audit missing low-energy windows — dry run

**Purpose:** rediscover low-energy windows from the original parent audio without changing any manifest, features, labels, or audio.

```powershell
python scripts\audit_v09_low_energy_silence_candidates.py
```

Recorded audit:

```text
Parents scanned/resolved:        2,074 / 2,074
Grid windows scanned:           22,391
Already represented:            12,164
Missing low-energy before cap:   1,178
Review candidates retained:      1,018
```

---

## 11. Export the low-energy review queue and one-second WAVs

**Purpose:** write the candidate CSV and review audio after the dry-run counts were accepted.

```powershell
python scripts\audit_v09_low_energy_silence_candidates.py `
  --apply `
  --export_audio
```

Outputs:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\manual_review\low_energy_recovery_v09\
  low_energy_silence_review_queue_v09.csv
  missing_or_ambiguous_parent_audio_v09.csv
  low_energy_silence_audit_summary_v09.json
  audio\
```

Manual review outcome:

```text
271 silence positives
747 silence negatives
1,018 reviewed total
```

---

## 12. Apply reviewed low-energy recovery — dry run

**Purpose:** validate the reviewed queue, timeline collisions, expected feature counts, and parent-level consecutive-silence logic.

```powershell
python scripts\apply_v09_low_energy_silence_recovery.py
```

The final script must print:

```text
[script] version: 3.0-windows-safe-finalisation
```

Expected preview:

```text
Existing feature rows:             12,589
Reviewed rows to incorporate:       1,018
Existing rows updated in place:         1
Missing rows to append:             1,017
New feature-manifest rows:         13,606
```

---

## 13. Apply reviewed low-energy recovery

**Purpose:** create a self-contained recovered feature cache while preserving the original v0.9 files.

```powershell
python scripts\apply_v09_low_energy_silence_recovery.py --apply
```

Canonical output root:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09
```

Recorded outcome:

```text
Final folder promotion: Succeeded
Existing v0.9 files modified: No
```

---

## 14. Validate the recovered cache

**Purpose:** confirm the complete recovered cache before training.

```powershell
$RecoveredRoot = "human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09"

$Manifest = "$RecoveredRoot\feature_cache\metadata\multilabel_features_manifest_v09_SILENCE_RECOVERED.csv"
$Features = "$RecoveredRoot\feature_cache\features"
$ParentManifest = "$RecoveredRoot\metadata\tata_seed_parent_manifest_v09_SILENCE_RECOVERED.csv"

python -c "
import pandas as pd
from pathlib import Path

manifest = pd.read_csv(r'$Manifest', low_memory=False)
parents = pd.read_csv(r'$ParentManifest', low_memory=False)
root = Path(r'$Features')

missing = [
    x for x in manifest['feat_relpath'].astype(str)
    if not (root / x).is_file()
]

print('Feature rows:', len(manifest))
print('Parent rows:', len(parents))
print('Split counts:', manifest['split'].value_counts().to_dict())
print('Unique parents:', manifest['clip_id'].astype(str).nunique())
print('Missing feature files:', len(missing))
print('Duplicate feat_relpath:', manifest['feat_relpath'].astype(str).duplicated().sum())
print('Silence positives:', int(manifest['silence_present'].sum()))
print('Recovered rows:', int((manifest['v09_data_origin'] == 'recovered_low_energy_human_reviewed').sum()))
print('Recovered silence positives:', int(((manifest['v09_data_origin'] == 'recovered_low_energy_human_reviewed') & (manifest['silence_present'] == 1)).sum()))
print('Recovered hard negatives:', int(((manifest['v09_data_origin'] == 'recovered_low_energy_human_reviewed') & (manifest['silence_present'] == 0)).sum()))
"
```

Expected core values:

```text
Feature rows: 13606
Parent rows: 2101
Missing feature files: 0
Duplicate feat_relpath: 0
```

---

## 15. Train the full low-energy recovery model

**Purpose:** test the effect of including all 271 confirmed silence positives and 747 low-energy non-silence hard negatives.

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$V09Root = "human_talk_workspace\tata_v0.9_pipeline"
$RecoveredRoot = "$V09Root\tata_triage_model\silence_recovered_v09"

$Manifest = "$RecoveredRoot\feature_cache\metadata\multilabel_features_manifest_v09_SILENCE_RECOVERED.csv"
$FeaturesRoot = "$RecoveredRoot\feature_cache\features"
$LabelsJson = "$V09Root\shared\human_talk_10label_schema.json"
$RunsRoot = "$RecoveredRoot\runs"

if (-not (Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }
if (-not (Test-Path $FeaturesRoot)) { throw "Features root not found: $FeaturesRoot" }
if (-not (Test-Path $LabelsJson)) { throw "Labels JSON not found: $LabelsJson" }

New-Item -ItemType Directory -Path $RunsRoot -Force | Out-Null

python -m training.train_multilabel `
  --manifest "$Manifest" `
  --features_root "$FeaturesRoot" `
  --labels_json "$LabelsJson" `
  --runs_root "$RunsRoot" `
  --variant "tata_v09_silence_recovered_3exit" `
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

Recorded final-exit result:

```text
Macro-F1   = 0.8064
Micro-F1   = 0.8064
Samples-F1 = 0.7968
Exact      = 0.6022
Hamming    = 0.0539
Silence F1 = 0.7875
```

---

## 16. Create the silence-positive-only ablation manifest

**Purpose:** remove all 747 manually reviewed low-energy non-silence rows from train, validation, and test while retaining all 271 confirmed silence positives.

```powershell
$RecoveredRoot = "human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09"

$SourceManifest = "$RecoveredRoot\feature_cache\metadata\multilabel_features_manifest_v09_SILENCE_RECOVERED.csv"
$FilteredManifest = "$RecoveredRoot\feature_cache\metadata\multilabel_features_manifest_v09_ONLY_RECOVERED_SILENCE_POSITIVE.csv"

python -c "
import pandas as pd

src = r'$SourceManifest'
dst = r'$FilteredManifest'

df = pd.read_csv(src, low_memory=False)

origin = df['v09_data_origin'].fillna('').astype(str).str.strip()
candidate = df['recovery_candidate_id'].fillna('').astype(str).str.strip()
silence = pd.to_numeric(
    df['silence_present'],
    errors='coerce'
).fillna(0).astype(int)

human_reviewed_recovery = (
    origin.eq('recovered_low_energy_human_reviewed') |
    candidate.ne('')
)

remove = human_reviewed_recovery & silence.eq(0)

print('Original rows:', len(df))
print('Reviewed recovery rows:', int(human_reviewed_recovery.sum()))
print('Non-silent reviewed rows removed:', int(remove.sum()))
print('Recovered silence positives retained:',
      int((human_reviewed_recovery & silence.eq(1)).sum()))

if int(human_reviewed_recovery.sum()) != 1018:
    raise RuntimeError(
        f'Expected 1018 reviewed rows, found {int(human_reviewed_recovery.sum())}'
    )

if int(remove.sum()) != 747:
    raise RuntimeError(
        f'Expected 747 non-silent rows, found {int(remove.sum())}'
    )

filtered = df.loc[~remove].copy()
filtered.to_csv(dst, index=False)

print('Rows retained:', len(filtered))
print('Split counts:', filtered['split'].value_counts().to_dict())
print('Saved:', dst)
"
```

Expected:

```text
Original rows: 13606
Reviewed recovery rows: 1018
Non-silent reviewed rows removed: 747
Recovered silence positives retained: 271
Rows retained: 12859
Split counts: {'train': 8938, 'test': 1995, 'val': 1926}
```

Verify:

```powershell
Test-Path $FilteredManifest
```

Expected: `True`.

---

## 17. Train the silence-positive-only ablation

**Purpose:** determine whether the 747 non-silence recovered rows were responsible for global performance degradation.

```powershell
$RecoveredRoot = "human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09"

$FilteredManifest = "$RecoveredRoot\feature_cache\metadata\multilabel_features_manifest_v09_ONLY_RECOVERED_SILENCE_POSITIVE.csv"
$FeaturesRoot = "$RecoveredRoot\feature_cache\features"
$LabelsJson = "human_talk_workspace\tata_v0.9_pipeline\shared\human_talk_10label_schema.json"
$RunsRoot = "$RecoveredRoot\runs"

$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

if (-not (Test-Path $FilteredManifest)) { throw "Manifest missing: $FilteredManifest" }
if (-not (Test-Path $FeaturesRoot)) { throw "Features missing: $FeaturesRoot" }
if (-not (Test-Path $LabelsJson)) { throw "Labels JSON missing: $LabelsJson" }

New-Item -ItemType Directory -Path $RunsRoot -Force | Out-Null

python -m training.train_multilabel `
  --manifest "$FilteredManifest" `
  --features_root "$FeaturesRoot" `
  --labels_json "$LabelsJson" `
  --runs_root "$RunsRoot" `
  --variant "tata_v09_recovered_silence_positive_only_3exit" `
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

Recorded final-exit result:

```text
Macro-F1   = 0.8199
Micro-F1   = 0.8120
Samples-F1 = 0.8075
Exact      = 0.6110
Hamming    = 0.0537
Silence F1 = 0.7355
```

---

## 18. Build the nine-label tri-state manual-review CSV

**Purpose:** preserve the trusted silence decision and prepare editable review columns for the remaining nine labels.

```powershell
python scripts\build_v09_9label_manual_review_csv.py
```

Output:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\manual_review\low_energy_recovery_v09\low_energy_9label_manual_review_v09.csv
```

Annotation values:

```text
 1 = present
 0 = absent
-1 = reviewed but uncertain
blank = not reviewed
```

After completing a row:

```text
review_9label_status = reviewed
```

The CSV is ordered test, validation, then training so trusted evaluation labels can be completed first.

---

## 19. Optional file and run checks

### List current recovered runs

```powershell
Get-ChildItem "$RecoveredRoot\runs" -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object Name, LastWriteTime
```

### Count recovered feature files

```powershell
(Get-ChildItem "$RecoveredRoot\feature_cache\features" -Recurse -Filter *.npy -File).Count
```

### Confirm the command-documented manifests

```powershell
Test-Path "$RecoveredRoot\feature_cache\metadata\multilabel_features_manifest_v09_SILENCE_RECOVERED.csv"
Test-Path "$RecoveredRoot\feature_cache\metadata\multilabel_features_manifest_v09_ONLY_RECOVERED_SILENCE_POSITIVE.csv"
```

---

## 20. Experiment interpretation

| Run variant | Purpose | Status |
|---|---|---|
| `tata_v09_human_corrected_3exit` | Original audited ten-label v0.9 baseline | Current general baseline |
| `tata_v09_silence_recovered_3exit` | All 1,018 recovered rows | Silence diagnostic; inherited-label uncertainty |
| `tata_v09_recovered_silence_positive_only_3exit` | Keep only 271 recovered silence positives | Ablation |
| Future masked-loss run | Use all reviewed rows and mask `-1` labels | Planned final v0.9 model |

Do not compare these segment-level TATA metrics directly with the v0.8 parent-level corrected-holdout metrics.


---

## 20. Build the v0.10 human-reviewed masked manifest

**Purpose:** integrate the final manually reviewed low-energy labels using tri-state supervision masks without re-extracting features or overwriting v0.9 files.

Manual review input:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\manual_review\low_energy_recovery_v09\low_energy_9label_manual_review_v09.csv
```

Source recovered manifest:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09\feature_cache\metadata\multilabel_features_manifest_v09_SILENCE_RECOVERED.csv
```

Command:

```powershell
$RecoveredRoot = "human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09"

$SourceManifest = "$RecoveredRoot\feature_cache\metadata\multilabel_features_manifest_v09_SILENCE_RECOVERED.csv"

$ReviewCsv = "human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\manual_review\low_energy_recovery_v09\low_energy_9label_manual_review_v09.csv"

$MaskedRoot = "$RecoveredRoot\human_reviewed_masked_v09"

$MaskedManifest = "$MaskedRoot\feature_cache\metadata\multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv"

$ReportsDir = "$MaskedRoot\reports"

python scripts\build_v09_human_reviewed_masked_manifest.py `
  --source_manifest "$SourceManifest" `
  --review_csv "$ReviewCsv" `
  --output_manifest "$MaskedManifest" `
  --reports_dir "$ReportsDir" `
  --overwrite
```

Recorded output:

```text
Source rows:                   13,606
Output rows:                   13,606
Reviewed rows matched:          1,018
Fully known reviewed rows:        966
Partially known reviewed rows:      52
Strict checkpoint val rows:      1,883
Strict standard test rows:       1,961
Candidate-ID matches:            1,018
Fallback segment matches:            0
Source files modified:              No
```

Generated reports:

```text
$ReportsDir\v09_masked_manifest_summary.json
$ReportsDir\v09_masked_reviewed_label_summary.csv
$ReportsDir\v09_masked_review_match_audit.csv
```

---

## 21. Install the v0.10 masked training modules

**Purpose:** add new mask-aware loader and trainer without modifying the original v0.9 trainer.

Extract the provided package into the repository root:

```text
C:\Users\wwwsa\PycharmProjects\TATA-LAWYER
```

Expected new files:

```text
data\datasets_multilabel_masked.py
training\train_multilabel_masked.py
```

These are additive files. They do not replace:

```text
data\datasets_multilabel.py
training\train_multilabel.py
```

---

## 22. Train the v0.10 masked three-exit model

**Purpose:** train on all rows using known-label supervision while masking `-1` labels.

```powershell
$RecoveredRoot = "human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09"
$MaskedRoot = "$RecoveredRoot\human_reviewed_masked_v09"

$MaskedManifest = "$MaskedRoot\feature_cache\metadata\multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv"
$FeaturesRoot = "$RecoveredRoot\feature_cache\features"
$LabelsJson = "human_talk_workspace\tata_v0.9_pipeline\shared\human_talk_10label_schema.json"
$RunsRoot = "$MaskedRoot\runs"

$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

python -m training.train_multilabel_masked `
  --manifest "$MaskedManifest" `
  --features_root "$FeaturesRoot" `
  --labels_json "$LabelsJson" `
  --runs_root "$RunsRoot" `
  --variant "tata_v09_human_reviewed_masked_3exit" `
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

Recorded dataset loading:

```text
train: 9,445 rows
val_strict: 1,883 rows
test_strict: 1,961 rows
val_all_masked: 2,042 rows
test_all_masked: 2,119 rows
val_recovered_masked: 159 rows
test_recovered_masked: 158 rows
```

Recorded best checkpoint:

```text
Best epoch: 38
Best strict validation Macro-F1: 0.7771
```

Recorded strict original-test result:

```text
Exit 3:
Macro-F1       = 0.7950
Micro-F1       = 0.7952
Samples-F1     = 0.7655
Exact fully known = 0.5926
Masked Hamming = 0.0552
```

Recorded all-test masked result:

```text
Exit 3:
Macro-F1       = 0.7991
Micro-F1       = 0.7958
Samples-F1     = 0.7624
Exact fully known = 0.5988
Masked Hamming = 0.0539
```

Recorded recovered-only masked test result:

```text
Exit 3:
Rows           = 158
Macro-F1       = 0.5095
Micro-F1       = 0.8065
Samples-F1     = 0.7236
Exact fully known = 0.6828
Masked Hamming = 0.0384
```

Strict final-exit per-label test result:

```text
Brene_Brown:                P=0.7939, R=0.8733, F1=0.8317
Eckhart_Tolle:              P=0.9915, R=0.8593, F1=0.9206
Eric_Thomas:                P=0.8785, R=0.6963, F1=0.7769
Gary_Vee:                   P=0.9922, R=0.6737, F1=0.8025
Jay_Shetty:                 P=0.9417, R=0.8077, F1=0.8696
Nick_Vujicic:               P=0.7933, R=0.7933, F1=0.7933
other_speaker_present:      P=0.6797, R=0.6145, F1=0.6454
music_present:              P=0.8947, R=0.8608, F1=0.8774
audience_reaction_present:  P=0.9253, R=0.6716, F1=0.7783
silence_present:            P=0.6140, R=0.7000, F1=0.6542
```

---

## 23. v0.10 interpretation command notes

The v0.10 manifest did not re-extract features. It reused:

```text
$RecoveredRoot\feature_cache\features
```

Only targets and masks changed in:

```text
multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv
```

The v0.10 fixed-threshold model is scientifically cleaner but did not improve the strict original-test benchmark. It became conservative for several labels, especially:

```text
audience_reaction_present
Gary_Vee
Eric_Thomas
```

The next command sequence should implement per-label threshold tuning on `val_strict` and evaluate once on `test_strict`.
