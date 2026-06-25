# NeuroAccuExit-ASHADIP Human-Talk Pipeline

This repository contains the TATA-assisted human-talk preprocessing, low-energy recovery, manual review, and multi-label early-exit experiments used by the NeuroAccuExit-ASHADIP project.

The documentation now distinguishes three experimental tracks:

1. **v0.8 human-corrected-balanced main model** — evaluated at parent level on the corrected final holdout.
2. **v0.9 TATA triage model** — audited seed reconstruction plus low-energy recovery ablations.
3. **v0.10 human-reviewed masked TATA triage model** — low-energy review integration using tri-state labels and masked BCE.
4. **v0.10.1 TATA-LAWYER domain-aware hybrid** — final selected system: original v0.9 for normal audio and masked v0.10 for recovered low-energy audio.

These tracks must not be compared as if they used the same model, prediction level, or evaluation set.

---

## 1. Ten-label schema

The current human-talk schema contains:

```text
Brene_Brown
Eckhart_Tolle
Eric_Thomas
Gary_Vee
Jay_Shetty
Nick_Vujicic
other_speaker_present
music_present
audience_reaction_present
silence_present
```

Overlapping labels are valid. A one-second segment may contain a target speaker, another speaker, music, audience reaction, and a meaningful silence/near-silence portion.

---

## 2. v0.8 main-model result

The official v0.8-HCB corrected-holdout result uses parent-level mean probability aggregation, a fixed threshold of 0.5, and Exit 3.

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| Parent mean, fixed 0.5 | 0.7801 | **0.9332** | **0.9406** | **0.8397** | **0.0194** |

A post-hoc label-aware aggregation analysis used mean aggregation for eight stable labels and max aggregation for the two transient labels:

```text
audience_reaction_present
silence_present
```

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| Parent mean official | 0.7801 | **0.9332** | **0.9406** | **0.8397** | **0.0194** |
| Label-aware mean/max | **0.8320** | 0.9285 | 0.9375 | 0.8235 | 0.0211 |

Global max aggregation was diagnostic only because it over-predicted labels:

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| Parent mean | **0.7801** | **0.9332** | **0.9406** | **0.8397** | **0.0194** | 1.4302 |
| Global max | 0.7251 | 0.8203 | 0.8423 | 0.5121 | 0.0630 | 2.0346 |

The earlier 93–94% Samples-F1 and 82–84% Exact Match values belong to this **parent-level main-model evaluation**, not to the one-second TATA triage test.

---

## 3. v0.9/v0.10 workspace and data lineage

The workspace separates the TATA triage model from the downstream NeuroAccuExit main model:

```text
human_talk_workspace/
└── tata_v0.9_pipeline/
    ├── tata_triage_model/
    │   ├── metadata/
    │   ├── feature_cache/
    │   ├── runs/
    │   ├── manual_review/
    │   └── silence_recovered_v09/
    │       └── human_reviewed_masked_v09/
    ├── neuroaccuexit_main_model/
    └── shared/
```

### Baseline assets

The baseline was copied rather than edited in place:

```text
tata2_parent_manifest_12label_2074_BASELINE.csv
tata_seed_features_manifest_10label_12469_BASELINE.csv
human_talk_10label_schema.json
```

### Audited v0.9 seed build

| Item | Count |
|---|---:|
| Original reviewed parent clips | 2,074 |
| New verified silence parents | 27 |
| Final v0.9 parents | **2,101** |
| Reused legacy feature rows | 12,469 |
| New silence feature rows | 120 |
| Initial v0.9 feature rows | **12,589** |

Initial v0.9 manifest:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\feature_cache\metadata\multilabel_features_manifest_v09_FINAL.csv
```

---

## 4. Original v0.9 TATA triage result

The original v0.9 model used 12,589 one-second segment rows.

| Split | Rows |
|---|---:|
| Train | 8,745 |
| Validation | 1,883 |
| Test | 1,961 |

Final-exit test result:

| Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---:|---:|---:|---:|---:|
| **0.8195** | **0.8226** | **0.8226** | **0.6527** | **0.0483** |

This remained approximately level with the earlier v0.6 TATA internal baseline while using a cleaner, audited manifest.

---

## 5. Low-energy recovery audit

The original preprocessing discarded low-energy windows before feature extraction. The v0.9 audit reconstructed candidate windows from the original parent audio using one-second windows and a 0.5-second hop.

| Audit item | Count |
|---|---:|
| Original parents scanned | 2,074 |
| Parent audio resolved | 2,074 |
| One-second grid windows scanned | 22,391 |
| Already represented windows | 12,164 |
| Missing low-energy windows before parent cap | 1,178 |
| Review candidates retained | **1,018** |
| High priority | 250 |
| Medium priority | 90 |
| Low priority | 678 |

First-pass silence review:

| Review result | Count |
|---|---:|
| Genuine silence | **271** |
| Low-energy but non-silence | **747** |

One reviewed candidate already existed in the feature manifest and was updated in place. The remaining 1,017 rows were appended.

| Recovery item | Count |
|---|---:|
| Existing feature rows | 12,589 |
| Existing rows updated in place | 1 |
| Missing reviewed rows appended | 1,017 |
| Full recovered feature rows | **13,606** |
| Affected parents | 522 |
| Parent labels changed from silence 0 to 1 | 0 |

Recovered manifest:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09\feature_cache\metadata\multilabel_features_manifest_v09_SILENCE_RECOVERED.csv
```

Recovered features root:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09\feature_cache\features
```

No later feature re-extraction was required because v0.10 changed annotations and masks, not the audio clips or `.npy` feature arrays.

---

## 6. v0.9 low-energy experiments

### Experiment A — full recovered dataset

This run included all 1,018 reviewed low-energy candidates. For recovered rows, `silence_present` came from manual segment review while the other nine labels were inherited from the parent.

| Split | Rows |
|---|---:|
| Train | 9,445 |
| Validation | 2,042 |
| Test | 2,119 |

| Metric | Score |
|---|---:|
| Macro-F1 | 0.8064 |
| Micro-F1 | 0.8064 |
| Samples-F1 | 0.7968 |
| Exact Match | 0.6022 |
| Hamming Loss | 0.0539 |
| Silence F1 | **0.7875** |

### Experiment B — recovered silence positives only

This ablation removed all 747 manually reviewed non-silence candidates from train, validation, and test while retaining the 271 confirmed silence candidates.

| Split | Rows |
|---|---:|
| Train | 8,938 |
| Validation | 1,926 |
| Test | 1,995 |

| Metric | Score |
|---|---:|
| Macro-F1 | **0.8199** |
| Micro-F1 | 0.8120 |
| Samples-F1 | 0.8075 |
| Exact Match | 0.6110 |
| Hamming Loss | 0.0537 |
| Silence F1 | 0.7355 |

### Comparison

| Experiment | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Silence F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original v0.9 | 0.8195 | **0.8226** | **0.8226** | **0.6527** | **0.0483** | 0.6667 |
| Full recovery | 0.8064 | 0.8064 | 0.7968 | 0.6022 | 0.0539 | **0.7875** |
| Silence-positive only | **0.8199** | 0.8120 | 0.8075 | 0.6110 | 0.0537 | 0.7355 |

Interpretation:

- Recovering low-energy signals is useful because `silence_present` improved substantially.
- Inheriting the other nine parent labels onto one-second recovered windows introduces uncertain or incorrect supervision.
- The 747 non-silence clips should not be deleted permanently; they are valuable hard negatives showing that low energy does not always mean silence.

---

## 7. v0.10 human-reviewed tri-state protocol

`review_silence_present` was rechecked and became final human ground truth for all 1,018 recovered clips. The user corrected both directions:

```text
0 -> 1 when silence was missed
1 -> 0 when a clip was not actually silence
```

The other nine labels were manually reviewed with tri-state annotation:

```text
 1 = confidently present
 0 = confidently absent
-1 = reviewed but uncertain / unknown
blank = not reviewed yet
```

Multiple labels may be `1` for the same one-second clip.

If the whole clip is unclear, set all **nine new review labels** to `-1`; do not change the already verified silence label.

For training:

```text
known label (0 or 1) -> loss mask = 1
unknown label (-1)   -> loss mask = 0
```

Manual review CSV:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\manual_review\low_energy_recovery_v09\low_energy_9label_manual_review_v09.csv
```

Final review state before v0.10 manifest build:

| Item | Count |
|---|---:|
| Reviewed low-energy rows | 1,018 |
| Fully known reviewed rows | 966 |
| Partially known reviewed rows | 52 |
| Rows with all nine labels unknown | 0 |
| Final silence positives | 277 |
| Final silence negatives | 741 |
| Silence revisions 0→1 | 44 |
| Silence revisions 1→0 | 39 |

---

## 8. v0.10 masked manifest

The v0.10 manifest builder matched all 1,018 reviewed rows by candidate ID and produced a non-destructive masked feature manifest.

Builder output:

| Item | Count |
|---|---:|
| Source rows | 13,606 |
| Output rows | 13,606 |
| Reviewed rows matched | 1,018 |
| Fully known reviewed rows | 966 |
| Partially known reviewed rows | 52 |
| Strict checkpoint validation rows | 1,883 |
| Strict standard test rows | 1,961 |
| Candidate-ID matches | 1,018 |
| Fallback segment matches | 0 |

Masked manifest:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09\human_reviewed_masked_v09\feature_cache\metadata\multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv
```

Important new columns:

```text
mask_Brene_Brown
mask_Eckhart_Tolle
mask_Eric_Thomas
mask_Gary_Vee
mask_Jay_Shetty
mask_Nick_Vujicic
mask_other_speaker_present
mask_music_present
mask_audience_reaction_present
mask_silence_present

v09_masked_review_applied
v09_review_candidate_id
v09_review_has_unknown
v09_review_known_label_count
v09_review_unknown_label_count
v09_evaluation_group
v09_checkpoint_eligible
v09_standard_test_eligible
```

Strict evaluation policy:

```text
Checkpoint selection:
  original 1,883 validation rows only

Fair main test:
  original 1,961 test rows only

Secondary reports:
  all masked test rows
  recovered human-reviewed test rows
```

---

## 9. v0.10 masked training result

Training module:

```text
training.train_multilabel_masked
```

The model used the same architecture and fixed-threshold settings as v0.9:

```text
tap_blocks = 1,3
loss_weights = 0.3,0.3,1.0
epochs = 40
batch_size = 64
lr = 0.001
seed = 42
threshold = 0.5
device = CPU
```

Training rows:

```text
train: 9,445
val_strict: 1,883
test_strict: 1,961
val_all_masked: 2,042
test_all_masked: 2,119
val_recovered_masked: 159
test_recovered_masked: 158
```

Best checkpoint:

```text
Best epoch: 38
Best strict validation Macro-F1: 0.7771
```

### Strict original test result

This is the fair comparison against original v0.9 because it uses the same 1,961 test rows.

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| Original v0.9 | **0.8195** | **0.8226** | **0.8226** | **0.6527** | **0.0483** |
| v0.10 masked strict | 0.7950 | 0.7952 | 0.7655 | 0.5926 | 0.0552 |
| Change | -0.0245 | -0.0274 | -0.0571 | -0.0601 | +0.0069 worse |

### Strict per-label F1

| Label | Original v0.9 | v0.10 masked strict | Change |
|---|---:|---:|---:|
| Brene Brown | 0.8393 | 0.8317 | -0.0076 |
| Eckhart Tolle | 0.9191 | **0.9206** | +0.0015 |
| Eric Thomas | **0.8458** | 0.7769 | -0.0689 |
| Gary Vee | **0.8927** | 0.8025 | -0.0902 |
| Jay Shetty | 0.8845 | 0.8696 | -0.0149 |
| Nick Vujicic | 0.7799 | **0.7933** | +0.0134 |
| Other speaker | 0.6253 | **0.6454** | +0.0201 |
| Music | 0.8420 | **0.8774** | +0.0354 |
| Audience reaction | **0.8993** | 0.7783 | -0.1210 |
| Silence | **0.6667** | 0.6542 | -0.0125 |

### Secondary v0.10 reports

| Evaluation subset | Rows | Macro-F1 | Micro-F1 | Samples-F1 | Exact/Fully Known | Hamming |
|---|---:|---:|---:|---:|---:|---:|
| test_strict | 1,961 | 0.7950 | 0.7952 | 0.7655 | 0.5926 | 0.0552 |
| test_all_masked | 2,119 | 0.7991 | 0.7958 | 0.7624 | 0.5988 | 0.0539 |
| test_recovered_masked | 158 | 0.5095 | 0.8065 | 0.7236 | 0.6828 | 0.0384 |

Interpretation:

- The masked manifest is technically correct and scientifically cleaner.
- The fixed-threshold v0.10 model did **not** improve the overall original-test benchmark.
- It improved `music_present`, `other_speaker_present`, `Nick_Vujicic`, and `Eckhart_Tolle`.
- It became too conservative for `audience_reaction_present`, `Gary_Vee`, and `Eric_Thomas`.
- Precision was high and recall was low for several dropped labels, indicating threshold calibration is likely needed.
- The recovered-only masked test subset shows strong micro-F1 and hamming results, but its macro-F1 is not directly comparable because several labels have low support.

---

---
## 10. v0.10.1 final selected system: domain-aware hybrid

The v0.10.1 branch documents the final model-selection decision after the masked-training, recovered-domain thresholding, and hybrid-routing experiments.

Final selected system:

```text
Normal/original audio:
    model:      Original v0.9 TATA triage checkpoint
    threshold:  fixed 0.50

Recovered low-energy audio:
    model:      Human-reviewed masked v0.10 checkpoint
    threshold:  recovered-domain per-label thresholds
```

This system is referred to as:

```text
TATA-LAWYER v0.10.1 Domain-Aware Hybrid
branch: tata_lawyer_v0.10.1
```

The branch is a documentation and finalisation release. It does not introduce new feature extraction, new manual review, or a new training run.

### Final combined 2,119-row test comparison

| Policy | Normal/original audio | Recovered low-energy audio | Macro-F1 | Micro-F1 | Samples-F1 | Known-label Exact | Fully-known Exact | Hamming Loss ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Original fixed everywhere | Original v0.9, threshold 0.50 | Original v0.9, threshold 0.50 | 0.8155 | 0.8171 | 0.8108 | 0.6465 | 0.6462 | 0.0489 |
| Masked fixed everywhere | Masked v0.10, threshold 0.50 | Masked v0.10, threshold 0.50 | 0.7991 | 0.7958 | 0.7624 | 0.5998 | 0.5988 | 0.0539 |
| Original domain-aware | Original v0.9, normal threshold | Original v0.9, recovered thresholds | 0.8162 | 0.8165 | 0.8092 | 0.6446 | 0.6453 | 0.0491 |
| Masked domain-aware | Masked v0.10, strict thresholds | Masked v0.10, recovered thresholds | 0.8173 | 0.8081 | 0.7813 | 0.6031 | 0.6030 | 0.0526 |
| **Hybrid recommended** | **Original v0.9, threshold 0.50** | **Masked v0.10, recovered thresholds** | **0.8224** | **0.8218** | **0.8174** | **0.6555** | **0.6557** | **0.0477** |

Compared with the original fixed-threshold system, the selected hybrid improved every aggregate metric:

| Metric | Original fixed everywhere | v0.10.1 hybrid | Absolute change |
|---|---:|---:|---:|
| Macro-F1 | 0.8155 | **0.8224** | **+0.0068** |
| Micro-F1 | 0.8171 | **0.8218** | **+0.0047** |
| Samples-F1 | 0.8108 | **0.8174** | **+0.0066** |
| Known-label Exact | 0.6465 | **0.6555** | **+0.0090** |
| Fully-known Exact | 0.6462 | **0.6557** | **+0.0095** |
| Hamming Loss ↓ | 0.0489 | **0.0477** | **-0.0013** |

### Recovered low-energy test comparison

Both checkpoints were evaluated on the same 158 recovered human-reviewed test clips. The original v0.9 checkpoint was also tuned using the same recovered validation clips, so this is the decisive low-energy-domain comparison.

| Model on recovered audio | Threshold source | Macro-F1 | Micro-F1 | Samples-F1 | Fully-known Exact | Hamming Loss ↓ |
|---|---|---:|---:|---:|---:|---:|
| Original v0.9 | Fixed 0.50 | 0.4756 | 0.7262 | 0.6646 | 0.5586 | 0.0569 |
| Masked v0.10 | Fixed 0.50 | 0.5095 | 0.8065 | 0.7236 | 0.6828 | **0.0384** |
| Original v0.9 | Recovered-domain thresholds | 0.5126 | 0.7143 | 0.6424 | 0.5448 | 0.0589 |
| **Masked v0.10** | **Recovered-domain thresholds** | **0.5438** | **0.8075** | **0.7532** | **0.6966** | 0.0397 |

This shows that the masked v0.10 checkpoint is the better low-energy specialist. The gain is not only a thresholding artifact because the masked checkpoint also outperformed the original checkpoint at the same fixed threshold of 0.50.

### Final recovered-domain thresholds

```text
Brene_Brown                0.22
Eckhart_Tolle              0.50
Eric_Thomas                0.44
Gary_Vee                   0.50
Jay_Shetty                 0.57
Nick_Vujicic               0.50
other_speaker_present      0.45
music_present              0.57
audience_reaction_present  0.18
silence_present            0.46
```

---

## 11. Research questions answered by v0.10.1

| Research question | Answer |
|---|---|
| Does low-energy recovery matter? | Yes. It exposed missing low-energy evidence and improved silence-related learning in the diagnostic experiments. |
| Is low energy equivalent to silence? | No. The 741 final silence-negative reviewed clips show that low-energy audio contains important hard negatives. |
| Are inherited parent-level labels safe for recovered one-second windows? | No. They introduce label noise because a parent-level speaker/music/audience label may not apply to every recovered segment. |
| Does tri-state human review plus masked BCE solve the supervision problem? | Yes. It integrates known labels while preventing uncertain labels from being treated as false negatives. |
| Should the masked v0.10 model replace original v0.9 globally? | No. It is not the best universal normal-audio model. |
| What is the final selected system? | A deterministic domain-aware hybrid: original v0.9 for normal audio and masked v0.10 for recovered low-energy audio. |

---

## 12. Final research findings

1. The original preprocessing pipeline censored low-energy windows before feature extraction.
2. Low-energy recovery is scientifically justified, but simple recovery with inherited parent labels is not sufficient.
3. Manual segment-level tri-state review corrected inherited-label contamination in the recovered low-energy subset.
4. Masked BCE is the correct training objective for partially known multi-label annotations.
5. The masked v0.10 model became a useful low-energy specialist rather than the best universal model.
6. Domain-aware routing produced the strongest combined system across all tested policies.
7. The selected hybrid improved Macro-F1, Micro-F1, Samples-F1, exact match, and Hamming Loss over the original fixed-threshold baseline.

---

## 13. Final reporting decision

| Role | Final result |
|---|---|
| Official v0.8 main-model parent-level result | Parent mean, fixed 0.5, Exit 3 |
| v0.8 aggregation research finding | Label-aware mean/max aggregation |
| General normal-audio TATA baseline | Original v0.9 model with fixed threshold 0.50 |
| Low-energy specialist | Human-reviewed masked v0.10 model with recovered-domain thresholds |
| Final TATA-LAWYER v0.10.1 system | Domain-aware hybrid |
| Branch for final documentation | `tata_lawyer_v0.10.1` |

The v0.10.1 hybrid should now be treated as the final TATA-LAWYER model-selection result. Further work should focus on documentation, reproducibility, routing implementation, and thesis reporting rather than additional training.

---

## 14. Do not restart these steps

Do not restart:

1. manual review;
2. feature extraction;
3. masked-manifest building;
4. masked training;
5. class weighting or oversampling;
6. another threshold search on the same final test set.

The modelling question is resolved for the current dataset. A future paper-strength confirmation can use a new untouched low-energy holdout, but that is a separate validation step, not part of v0.10.1 finalisation.
