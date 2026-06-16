# Multi-Label Experiment Log

---

### 11. Global max aggregation diagnostic

After the official parent-level mean corrected-holdout evaluation, global max aggregation was tested as a diagnostic.

Command section:

```text
13_eval_v08_global_max_parent_fixed
```

Result:

| Aggregation | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| Parent mean | **0.7801** | **0.9332** | **0.9406** | **0.8397** | **0.0194** | 1.4302 |
| Global max | 0.7251 | 0.8203 | 0.8423 | 0.5121 | 0.0630 | 2.0346 |

Decision:

```text
Do not use global max as the final aggregation strategy.
```

Reason: global max over-predicts parent labels, increasing false positives and worsening Exact Match and Hamming Loss.

### 12. Weak-label improvement under max aggregation

The global max diagnostic showed that the two weak transient labels improved:

| Label | Parent mean F1 | Global max F1 |
|---|---:|---:|
| `audience_reaction_present` | 0.1250 | **0.4706** |
| `silence_present` | 0.0000 | **0.1739** |

Interpretation:

```text
The model has some segment-level evidence for weak transient labels,
but parent-level mean aggregation dilutes that evidence.
```

This motivated a label-aware parent aggregation rule.

### 13. Label-aware aggregation experiment

Command section:

```text
15_compute_v08_label_aware_parent_aggregation
```

Rule:

```text
mean aggregation for 8 stable labels:
  Brene_Brown
  Eckhart_Tolle
  Eric_Thomas
  Gary_Vee
  Jay_Shetty
  Nick_Vujicic
  other_speaker_present
  music_present

max aggregation for 2 transient labels:
  audience_reaction_present
  silence_present
```

Result:

| Strategy | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| Parent mean official | 0.7801 | **0.9332** | **0.9406** | **0.8397** | **0.0194** | 1.4302 |
| Label-aware mean/max | **0.8320** | 0.9285 | 0.9375 | 0.8235 | 0.0211 | 1.4844 |

Finding:

```text
Label-aware aggregation improved Macro-F1 from 0.7801 to 0.8320
without retraining the model.
```

### 14. Updated final decision

The final reporting strategy is:

| Reporting role | Method | Reason |
|---|---|---|
| Main official overall result | Parent mean, fixed threshold 0.5, Exit 3 | Best Micro-F1, Samples-F1, Exact Match, and Hamming Loss. |
| Research/ablation contribution | Label-aware mean/max, fixed threshold 0.5, Exit 3 | Best Macro-F1 and better handling of weak transient labels. |
| Diagnostic only | Global max, fixed threshold 0.5, Exit 3 | Helps transient labels but damages overall multi-label prediction. |

Official result remains:

```text
v0.8-HCB
parent-level mean
fixed threshold 0.5
Exit 3
Macro-F1=0.7801
Micro-F1=0.9332
Samples-F1=0.9406
Exact=0.8397
Hamming=0.0194
```

Additional label-aware research finding:

```text
v0.8-HCB
label-aware parent aggregation
mean for stable labels
max for transient labels
Exit 3
Macro-F1=0.8320
Micro-F1=0.9285
Samples-F1=0.9375
Exact=0.8235
Hamming=0.0211
```

### 15. Updated limitation and future work

The v0.8-HCB experiment shows that weak labels should not be treated only as a training-data problem. Parent-level aggregation also matters.

Remaining limitations:

- `audience_reaction_present` and `silence_present` remain difficult under parent mean.
- Global max is too aggressive for stable labels.
- Label-aware aggregation was post-hoc; future work should integrate this rule into a formal evaluation script or learn label-specific pooling automatically.
- Future early-exit policy should consider label type: stable labels may exit based on accumulated mean confidence, while transient labels may require max/event-detection evidence.

Updated future direction:

```text
Develop label-type-aware early-exit inference:
  stable labels -> evidence accumulation / mean confidence
  transient labels -> event-triggered max confidence
```

---

### 16. v0.9 workspace creation

A non-destructive v0.9 workspace was created under:

```text
human_talk_workspace\tata_v0.9_pipeline
```

Baseline assets were copied with `_BASELINE` names and marked as do-not-edit. The TATA triage model and downstream NeuroAccuExit main model were separated into independent directories.

Decision:

```text
Preserve all v0.6/v0.8 manifests and caches.
Perform v0.9 changes only in new versioned files.
```

### 17. Existing seed review and new silence data

Existing seed silence labels and rare-event labels were manually reviewed. Twenty-seven externally collected silence clips were verified, renamed beginning after the existing maximum silence index, and added without overwriting existing files.

Final parent lineage:

```text
2,074 original reviewed parents
+27 new verified silence parents
=2,101 parents
```

### 18. Initial v0.9 feature-cache build

The v0.9 builder reused the 12,469 legacy feature arrays and extracted 120 new segments from the 27 new silence parents.

| Item | Count |
|---|---:|
| Final parents | 2,101 |
| Reused feature rows | 12,469 |
| New silence feature rows | 120 |
| Final feature rows | 12,589 |
| Missing required new metadata | 0 |
| Duplicate feature paths | 0 |
| Parent split leakage | 0 |

### 19. Original v0.9 TATA training

Run variant:

```text
tata_v09_human_corrected_3exit
```

Final-exit result:

| Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---:|---:|---:|---:|---:|
| 0.8195 | 0.8226 | 0.8226 | 0.6527 | 0.0483 |

Decision:

```text
Use original v0.9 as the current general ten-label TATA baseline.
```

### 20. Low-energy preprocessing audit

The old feature cache was confirmed to contain only windows retained after legacy preprocessing. A raw-audio audit was run to find low-energy windows missing from the feature manifest.

| Audit item | Count |
|---|---:|
| Parents scanned/resolved | 2,074 / 2,074 |
| Grid windows scanned | 22,391 |
| Existing windows | 12,164 |
| Missing low-energy windows before cap | 1,178 |
| Review queue | 1,018 |

### 21. Manual silence review

All 1,018 candidate one-second clips were manually checked.

| Decision | Count |
|---|---:|
| Genuine silence | 271 |
| Low-energy non-silence | 747 |
| Unreviewed | 0 |

Finding:

```text
Low energy is not equivalent to silence.
Both positive and hard-negative examples are valuable.
```

### 22. Full low-energy recovery cache

Run script version:

```text
3.0-windows-safe-finalisation
```

One reviewed row collided with an existing timeline location and was updated in place. The other 1,017 rows were appended.

| Item | Count |
|---|---:|
| Existing rows | 12,589 |
| Existing rows updated | 1 |
| New rows appended | 1,017 |
| Final rows | 13,606 |
| Affected parents | 522 |
| Parent silence 0-to-1 changes | 0 |

### 23. Full recovery training

Run variant:

```text
tata_v09_silence_recovered_3exit
```

Best epoch: 33. Best validation final-exit Macro-F1: 0.7835.

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| 1 | 0.1960 | 0.3489 | 0.2755 | 0.1888 | 0.1220 |
| 2 | 0.5113 | 0.6342 | 0.5894 | 0.4011 | 0.0861 |
| 3 | **0.8064** | **0.8064** | **0.7968** | **0.6022** | **0.0539** |

`silence_present` improved to F1 0.7875.

### 24. Positive-only recovery ablation

All 747 reviewed non-silence rows were excluded from train, validation, and test. The 271 confirmed silence-positive rows were retained.

Manifest:

```text
multilabel_features_manifest_v09_ONLY_RECOVERED_SILENCE_POSITIVE.csv
```

Run variant:

```text
tata_v09_recovered_silence_positive_only_3exit
```

Best epoch: 37. Best validation final-exit Macro-F1: 0.7993.

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| 1 | 0.1969 | 0.3536 | 0.2810 | 0.1845 | 0.1230 |
| 2 | 0.5635 | 0.6572 | 0.6149 | 0.4005 | 0.0863 |
| 3 | **0.8199** | **0.8120** | **0.8075** | **0.6110** | **0.0537** |

`silence_present` F1 was 0.7355.

### 25. Recovery interpretation

Comparison:

| Experiment | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming | Silence F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original v0.9 | 0.8195 | **0.8226** | **0.8226** | **0.6527** | **0.0483** | 0.6667 |
| Full recovery | 0.8064 | 0.8064 | 0.7968 | 0.6022 | 0.0539 | **0.7875** |
| Positive-only | **0.8199** | 0.8120 | 0.8075 | 0.6110 | 0.0537 | 0.7355 |

Decision:

```text
Do not discard the 747 hard negatives.
Do not train them with unverified inherited labels.
Review or mask the remaining nine labels.
```

### 26. Nine-label tri-state review

A new review CSV was prepared for all 1,018 recovered clips. `review_silence_present` remains trusted. The other nine labels use:

```text
1  = present
0  = absent
-1 = reviewed but unknown
blank = not reviewed
```

Review order:

```text
test first
validation second
training last
```

### 27. Planned masked-loss model

Future recovered-row supervision:

```text
known labels 0/1 -> included in BCE loss
unknown label -1 -> excluded from BCE loss
```

Evaluation decision:

- keep the original 1,961 test rows for like-for-like ten-label comparison;
- evaluate recovered low-energy rows separately for silence;
- use masked metrics for partially labelled rows;
- do not calculate ordinary Exact Match on rows containing unknown labels.


---

## v0.10 — Human-reviewed masked low-energy manifest and fixed-threshold experiment

### Motivation

The v0.9 full-recovery experiment showed that low-energy recovery improved `silence_present`, but parent-inherited labels were unsafe for one-second recovered clips. The positive-only ablation further showed that removing the 747 low-energy non-silence examples restored many global metrics but reduced silence performance. Therefore, v0.10 introduced a human-reviewed tri-state annotation and mask-aware training procedure.

### Manual-review decision

All 1,018 recovered low-energy clips were reviewed for all nine non-silence labels using:

```text
1  = confidently present
0  = confidently absent
-1 = reviewed but uncertain / unknown
```

The already reviewed `review_silence_present` remained a final 0/1 label. A `previous_review_silence_present` column preserved first-pass silence decisions.

Final review summary:

```text
Reviewed rows:                 1,018
Fully known reviewed rows:       966
Partially known reviewed rows:    52
Final silence positives:         277
Final silence negatives:         741
Silence revisions 0->1:           44
Silence revisions 1->0:           39
```

### Masked-manifest build

Command:

```powershell
python scripts\build_v09_human_reviewed_masked_manifest.py `
  --source_manifest "$SourceManifest" `
  --review_csv "$ReviewCsv" `
  --output_manifest "$MaskedManifest" `
  --reports_dir "$ReportsDir" `
  --overwrite
```

Output:

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
```

### Training setting

```text
Model:          TinyAudioCNN + ExitNet
Exits:          3
tap_blocks:     1,3
loss_weights:   0.3,0.3,1.0
epochs:         40
batch_size:     64
lr:             0.001
threshold:      0.5
seed:           42
device:         CPU
```

Training data:

```text
train: 9,445
val_strict: 1,883
test_strict: 1,961
val_all_masked: 2,042
test_all_masked: 2,119
val_recovered_masked: 159
test_recovered_masked: 158
```

Checkpoint:

```text
Best epoch: 38
Best strict validation Macro-F1: 0.7771
```

### Strict original-test result

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| Original v0.9 | **0.8195** | **0.8226** | **0.8226** | **0.6527** | **0.0483** |
| v0.10 masked strict | 0.7950 | 0.7952 | 0.7655 | 0.5926 | 0.0552 |
| Delta | -0.0245 | -0.0274 | -0.0571 | -0.0601 | +0.0069 worse |

### Per-label strict-test comparison

| Label | Original v0.9 F1 | v0.10 masked F1 | Delta |
|---|---:|---:|---:|
| Brene_Brown | 0.8393 | 0.8317 | -0.0076 |
| Eckhart_Tolle | 0.9191 | 0.9206 | +0.0015 |
| Eric_Thomas | 0.8458 | 0.7769 | -0.0689 |
| Gary_Vee | 0.8927 | 0.8025 | -0.0902 |
| Jay_Shetty | 0.8845 | 0.8696 | -0.0149 |
| Nick_Vujicic | 0.7799 | 0.7933 | +0.0134 |
| other_speaker_present | 0.6253 | 0.6454 | +0.0201 |
| music_present | 0.8420 | 0.8774 | +0.0354 |
| audience_reaction_present | 0.8993 | 0.7783 | -0.1210 |
| silence_present | 0.6667 | 0.6542 | -0.0125 |

### Secondary evaluations

| Subset | Rows | Macro-F1 | Micro-F1 | Samples-F1 | Exact/Fully Known | Hamming |
|---|---:|---:|---:|---:|---:|---:|
| test_strict | 1,961 | 0.7950 | 0.7952 | 0.7655 | 0.5926 | 0.0552 |
| test_all_masked | 2,119 | 0.7991 | 0.7958 | 0.7624 | 0.5988 | 0.0539 |
| test_recovered_masked | 158 | 0.5095 | 0.8065 | 0.7236 | 0.6828 | 0.0384 |

### Interpretation

The v0.10 manifest is the cleanest annotation artifact so far, but fixed-threshold performance on the original strict test set decreased. The model became conservative: several labels had high precision but low recall. This particularly affected `audience_reaction_present`, `Gary_Vee`, and `Eric_Thomas`.

The next experiment should tune one threshold per label on the strict original validation set, then evaluate once on the strict original test set. Original v0.9 should also be evaluated on the same 158 recovered human-reviewed test rows to establish whether v0.10 improves the low-energy domain.
