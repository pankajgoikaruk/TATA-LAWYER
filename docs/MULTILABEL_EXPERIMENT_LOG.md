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

---

## 28. Downstream NeuroAccuExit hybrid weak-label branch

A new downstream branch was created to use the final TATA-LAWYER v0.10.1 domain-aware hybrid system as a weak-label generator for the main NeuroAccuExit model.

Branch/workspace:

```text
neuroaccuexit_hybrid_weaklabels
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/
```

The downstream weak-label manifest uses domain-aware routing:

| Audio domain | Weak-label source |
|---|---|
| Normal/original audio | Original v0.9 TATA model, fixed threshold 0.50 |
| Recovered low-energy audio | Human-reviewed masked v0.10 model, recovered-domain thresholds |

The generated manifest contains:

| Split/domain | Rows |
|---|---:|
| train, normal/original | 8,744 |
| train, recovered low-energy | 701 |
| val, normal/original | 1,883 |
| val, recovered low-energy | 159 |
| test, normal/original | 1,961 |
| test, recovered low-energy | 158 |
| **Total** | **13,606** |

Interpretation:

```text
The downstream model is trained to learn the final TATA-LAWYER hybrid weak-label policy.
Weak-label test performance measures policy distillation, not direct semantic ground-truth accuracy.
```

---

## 29. Downstream v0.1/v0.2 training

Two downstream NeuroAccuExit models were trained on the hybrid weak-label manifest.

| Version | Loss setting | Purpose |
|---|---|---|
| v0.1 | Plain BCE | Conservative baseline |
| v0.2 | BCE with capped positive weight 5.0 | Improve recall and shallow-exit learning |

Strict weak-label test result at final exit:

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| v0.1 plain BCE | **0.8912** | **0.8946** | 0.8537 | **0.7766** | **0.0260** |
| v0.2 pos_weight5 | 0.8800 | 0.8793 | **0.8691** | 0.7374 | 0.0328 |

Although v0.1 was stronger at the final exit, v0.2 improved shallow-exit behaviour:

| Exit | v0.1 Macro-F1 | v0.2 Macro-F1 |
|---|---:|---:|
| Exit 1 | 0.2165 | **0.4013** |
| Exit 2 | 0.5734 | **0.6949** |
| Exit 3 | **0.8912** | 0.8800 |

Decision:

```text
Keep v0.2 as the preferred downstream candidate because NeuroAccuExit requires useful shallow exits, not only the strongest final-exit classifier.
```

---

## 30. Downstream v0.3 threshold-tuned weak-label model

Per-label threshold tuning was applied on the strict validation split and evaluated on the strict weak-label test split.

| Candidate | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| v0.1 fixed | 0.8912 | 0.8946 | 0.8537 | 0.7766 | 0.0260 |
| v0.1 tuned | **0.9058** | 0.9020 | 0.8711 | 0.7879 | 0.0250 |
| v0.2 fixed | 0.8800 | 0.8793 | 0.8691 | 0.7374 | 0.0328 |
| **v0.2 tuned** | 0.9040 | **0.9073** | **0.8814** | **0.7930** | **0.0239** |

Selected downstream weak-label model:

```text
NeuroAccuExit hybrid weak-label v0.3
= v0.2 pos_weight5 model
+ per-label tuned thresholds
```

Selected v0.3 thresholds:

| Label | Threshold |
|---|---:|
| Brene_Brown | 0.60 |
| Eckhart_Tolle | 0.46 |
| Eric_Thomas | 0.68 |
| Gary_Vee | 0.95 |
| Jay_Shetty | 0.95 |
| Nick_Vujicic | 0.50 |
| other_speaker_present | 0.38 |
| music_present | 0.74 |
| audience_reaction_present | 0.69 |
| silence_present | 0.88 |

Important limitation:

```text
The v0.3 weak-label test result shows that the downstream model learned the TATA-LAWYER hybrid policy well.
It does not prove human ground-truth generalisation.
```

---

## 31. Human context-checked holdout evaluation

The selected v0.3 model was evaluated on the human context-checked final holdout from the v0.8 human-corrected-balanced pipeline.

Holdout source:

```text
human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/corrected_holdout/
01_raw_final_holdout_GROUND_TRUTH_FINAL_v08_context_checked.csv
```

Evaluation outputs:

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/
human_context_checked_holdout_v03/evaluation_v02_tuned/
```

Initial holdout result:

| Level/aggregation | Rows | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|---:|
| Segment-level | 4,335 | 0.5522 | 0.6568 | 0.6082 | 0.4076 | 0.0917 |
| Parent mean | 867 | 0.5379 | **0.6816** | 0.6191 | **0.4498** | **0.0752** |
| Parent max | 867 | **0.5715** | 0.6553 | **0.6742** | 0.2641 | 0.1309 |

Interpretation:

```text
Parent mean is safest overall, but parent max recovers transient/bursty labels at the cost of many false positives.
This confirms that a single global parent aggregation rule is suboptimal.
```

---

## 32. v0.4/v0.4b label-specific parent aggregation diagnostics

A diagnostic experiment compared parent-level aggregation rules without retraining the model.

| Method | Rows | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean | 867 | 0.5379 | 0.6816 | 0.6191 | 0.4498 | **0.0752** | 0.8927 |
| max | 867 | 0.5715 | 0.6553 | 0.6742 | 0.2641 | 0.1309 | 2.3287 |
| top2mean | 867 | 0.6125 | 0.7278 | 0.7310 | 0.4268 | 0.0870 | 1.7255 |
| **v0.4b labelwise** | 867 | **0.6647** | **0.7527** | **0.7560** | **0.4591** | 0.0762 | 1.6136 |

v0.4b diagnostic aggregation map:

| Label | Aggregation |
|---|---|
| Brene_Brown | mean |
| Eckhart_Tolle | mean |
| Eric_Thomas | top2mean |
| Gary_Vee | mean |
| Jay_Shetty | top2mean |
| Nick_Vujicic | top2mean |
| other_speaker_present | max |
| music_present | top2mean |
| audience_reaction_present | top2mean |
| silence_present | max |

Interpretation:

```text
Label-specific parent aggregation improves human context-checked holdout performance without retraining.
However, v0.4b is diagnostic because the aggregation map was chosen after observing holdout behaviour.
```

---

## 33. v0.5 calibration-selected label-wise aggregation

To avoid directly selecting the aggregation map on the full holdout, the 867 parent clips were split into calibration and evaluation halves.

Selection was performed on the calibration split only, then evaluated on the evaluation split.

Selected v0.5 aggregation map:

| Label | Selected aggregation |
|---|---|
| Brene_Brown | mean |
| Eckhart_Tolle | mean |
| Eric_Thomas | top2mean |
| Gary_Vee | mean |
| Jay_Shetty | top2mean |
| Nick_Vujicic | max |
| other_speaker_present | max |
| music_present | top2mean |
| audience_reaction_present | top2mean |
| silence_present | max |

Evaluation split result:

| Method | Rows | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|---:|
| mean | 434 | 0.5551 | 0.7084 | 0.6462 | **0.4862** | **0.0687** |
| max | 434 | 0.5427 | 0.6533 | 0.6793 | 0.2765 | 0.1313 |
| top2mean | 434 | 0.6129 | 0.7335 | 0.7368 | 0.4447 | 0.0839 |
| **v0.5 calibration-selected labelwise** | 434 | **0.6355** | **0.7491** | **0.7521** | 0.4447 | 0.0772 |

Interpretation:

```text
Calibration-selected label-wise aggregation improves Macro-F1, Micro-F1, and Samples-F1 over all global aggregation rules on the evaluation split, while mean remains slightly better for Exact Match and Hamming Loss.
```

---

## 34. v0.6 repeated calibration/evaluation stability

The calibration/evaluation experiment was repeated across 20 random seeds to test stability.

| Method | Macro-F1 mean | Macro-F1 std | Micro-F1 mean | Micro-F1 std | Samples-F1 mean | Samples-F1 std | Exact Match mean | Hamming Loss mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 0.5345 | 0.0130 | 0.6776 | 0.0073 | 0.6138 | 0.0127 | 0.4471 | **0.0759** |
| max | 0.5605 | 0.0177 | 0.6519 | 0.0087 | 0.6709 | 0.0106 | 0.2589 | 0.1328 |
| top2mean | 0.6108 | 0.0096 | 0.7252 | 0.0074 | 0.7283 | 0.0087 | 0.4233 | 0.0879 |
| **calibration-selected labelwise** | **0.6377** | 0.0140 | **0.7470** | 0.0161 | **0.7508** | 0.0151 | **0.4520** | 0.0785 |

Selection frequency across 20 seeds:

| Label | Most frequent selected aggregation | Count |
|---|---|---:|
| Brene_Brown | mean | 20/20 |
| Eckhart_Tolle | mean | 20/20 |
| Eric_Thomas | top2mean | 18/20 |
| Gary_Vee | mean | 17/20 |
| Jay_Shetty | top2mean | 20/20 |
| Nick_Vujicic | top2mean | 18/20 |
| other_speaker_present | max | 20/20 |
| music_present | top2mean | 20/20 |
| audience_reaction_present | top2mean | 15/20 |
| silence_present | max | 16/20 |

Confirmed v0.6 finding:

```text
Across 20 repeated calibration/evaluation splits, calibration-selected label-specific aggregation consistently outperformed global mean, max, and top2mean aggregation in Macro-F1, Micro-F1, and Samples-F1. It also slightly improved Exact Match over global mean, while Hamming Loss remained only marginally worse than mean.
```

This finding is now frozen as the confirmed downstream NeuroAccuExit result before any v0.7 threshold-calibration experiment.

---

## 35. v0.7 repeated aggregation + threshold calibration

After v0.6 was frozen as the confirmed aggregation-only finding, v0.7 tested whether per-label thresholds should also be selected on calibration splits.

The v0.7 diagnostic used:

```text
20 repeated random 50/50 parent-level splits
calibration-selected aggregation method per label
calibration-selected threshold per label
coarse threshold grid: 0.10, 0.15, ..., 0.95
held-out evaluation half for reporting
```

No model retraining was performed. The experiment reused the saved segment-level prediction file from the human context-checked holdout evaluation.

Result:

| Method | Macro-F1 mean | Macro-F1 std | Micro-F1 mean | Micro-F1 std | Samples-F1 mean | Samples-F1 std | Exact Match mean | Hamming Loss mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean fixed thresholds | 0.5345 | 0.0130 | 0.6776 | 0.0073 | 0.6138 | 0.0127 | 0.4471 | 0.0759 |
| max fixed thresholds | 0.5605 | 0.0177 | 0.6519 | 0.0087 | 0.6709 | 0.0106 | 0.2589 | 0.1328 |
| top2mean fixed thresholds | 0.6108 | 0.0096 | 0.7252 | 0.0074 | 0.7283 | 0.0087 | 0.4233 | 0.0879 |
| v0.7 aggregation + threshold calibrated | **0.6637** | 0.0204 | **0.7725** | 0.0095 | **0.7861** | 0.0099 | **0.4809** | **0.0702** |

Comparison with frozen v0.6:

| Metric | v0.6 aggregation only | v0.7 aggregation + threshold calibration | Change |
|---|---:|---:|---:|
| Macro-F1 | 0.6377 | **0.6637** | +0.0260 |
| Micro-F1 | 0.7470 | **0.7725** | +0.0255 |
| Samples-F1 | 0.7508 | **0.7861** | +0.0353 |
| Exact Match | 0.4520 | **0.4809** | +0.0289 |
| Hamming Loss | 0.0785 | **0.0702** | -0.0083 |

Aggregation selection frequency:

| Label | Most frequent aggregation after threshold calibration | Frequency |
|---|---|---:|
| `Brene_Brown` | mean | 13/20 |
| `Eckhart_Tolle` | top2mean | 11/20 |
| `Eric_Thomas` | mean/top2mean tie | 10/20 each |
| `Gary_Vee` | mean | 20/20 |
| `Jay_Shetty` | mean | 17/20 |
| `Nick_Vujicic` | mean/top2mean tie | 10/20 each |
| `other_speaker_present` | max | 11/20 |
| `music_present` | mean | 12/20 |
| `audience_reaction_present` | top2mean | 15/20 |
| `silence_present` | max | 19/20 |

Threshold summary:

| Label | Mean threshold | Std | Min | Max | Mean calibration support |
|---|---:|---:|---:|---:|---:|
| `Brene_Brown` | 0.7375 | 0.1685 | 0.55 | 0.95 | 37.45 |
| `Eckhart_Tolle` | 0.6625 | 0.1621 | 0.35 | 0.85 | 42.10 |
| `Eric_Thomas` | 0.5750 | 0.1594 | 0.30 | 0.80 | 33.85 |
| `Gary_Vee` | 0.8450 | 0.0536 | 0.75 | 0.95 | 33.35 |
| `Jay_Shetty` | 0.7350 | 0.0947 | 0.65 | 0.95 | 45.15 |
| `Nick_Vujicic` | 0.3725 | 0.1118 | 0.20 | 0.70 | 23.85 |
| `other_speaker_present` | 0.1725 | 0.0343 | 0.10 | 0.20 | 228.60 |
| `music_present` | 0.6750 | 0.1509 | 0.50 | 0.95 | 171.40 |
| `audience_reaction_present` | 0.9050 | 0.0776 | 0.65 | 0.95 | 13.15 |
| `silence_present` | 0.5075 | 0.2944 | 0.10 | 0.95 | 6.65 |

Interpretation:

```text
v0.7 is the best calibrated diagnostic result. It improves all aggregate metrics over the frozen v0.6 aggregation-only result and over the global aggregation baselines.
```

The result also shows that aggregation and threshold choice interact: labels that used mean/top2mean/max in v0.6 may choose different aggregation rules after threshold calibration is allowed.

---

## 36. Current downstream stopping decision

Stop diagnostic experiments here.

Do not retrain the downstream model for this finding.

Current selected diagnostic result:

```text
main_model_v0.7:
v0.2 pos_weight5 checkpoint
+ v0.3 tuned weak-label model
+ v0.7 repeated calibration-selected aggregation
+ v0.7 repeated per-label threshold calibration
```

Confirmed finding:

```text
One global parent aggregation rule and one inherited threshold profile are suboptimal for multi-label audio.
Repeated calibration-selected label-specific aggregation and threshold calibration substantially improve human context-checked holdout performance without retraining.
```

Limitations:

```text
This is a calibrated diagnostic over the human context-checked holdout collection, not an external unbiased test result.
```

The next phase should be writing the method/results section, not running more diagnostics.
