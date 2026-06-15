# NeuroAccuExit-ASHADIP Human-Talk Pipeline

This repository contains the TATA-assisted human-talk preprocessing and multi-label early-exit experiments used by the NeuroAccuExit-ASHADIP project.

The documentation currently distinguishes two separate experimental tracks:

1. **v0.8 human-corrected-balanced main model** — evaluated at parent level on the corrected final holdout.
2. **v0.9 TATA triage model** — rebuilt from the audited seed data and extended with low-energy silence recovery experiments.

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

Overlapping labels are valid. A one-second segment may contain a target speaker, another speaker, music, and audience reaction simultaneously.

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

## 3. v0.9 workspace and data lineage

The v0.9 workspace separates the TATA triage model from the downstream NeuroAccuExit main model:

```text
human_talk_workspace/
└── tata_v0.9_pipeline/
    ├── tata_triage_model/
    │   ├── metadata/
    │   ├── feature_cache/
    │   ├── runs/
    │   ├── manual_review/
    │   └── silence_recovered_v09/
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

The initial v0.9 manifest was saved as:

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

All 1,018 candidates were manually reviewed for `silence_present`:

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

### Interpretation

The experiment shows that recovering low-energy signals is useful because `silence_present` improved substantially. However, inheriting the other nine parent labels onto one-second recovered windows introduces uncertain supervision.

The 747 non-silence clips should not be deleted permanently. They are valuable hard negatives showing that low energy does not always mean silence. Their other nine labels must be manually reviewed or masked during training.

---

## 7. Current manual-review protocol

`review_silence_present` is already trusted human ground truth for all 1,018 recovered clips.

The other nine labels use tri-state annotation:

```text
 1 = confidently present
 0 = confidently absent
-1 = reviewed but uncertain / unknown
blank = not reviewed yet
```

Multiple labels may be `1` for the same one-second clip.

If the whole clip is unclear, set all **nine new review labels** to `-1`; do not change the already verified silence label.

For future training:

```text
known label (0 or 1) -> loss mask = 1
unknown label (-1)   -> loss mask = 0
```

The review CSV is generated as:

```text
human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\manual_review\low_energy_recovery_v09\low_energy_9label_manual_review_v09.csv
```

---

## 8. Current reporting decisions

| Role | Recommended result |
|---|---|
| Official v0.8 main-model parent-level result | Parent mean, fixed 0.5, Exit 3 |
| v0.8 aggregation research finding | Label-aware mean/max |
| General v0.9 TATA ten-label baseline | Original v0.9 model |
| Silence-focused diagnostic | Full low-energy recovery model |
| Recovery ablation | Silence-positive-only model |
| Final v0.9 target | Fully reviewed or masked nine-label recovered dataset |

The full-recovery and positive-only results are research ablations. Neither should replace the original v0.9 general ten-label baseline until the nine uncertain labels have been manually reviewed or handled with masked loss.

---

## 9. Documentation map

```text
docs/DOC_STRUCTURE.md
docs/README.md
docs/APPENDIX.md
docs/MULTILABEL_EXPERIMENT_LOG.md
docs/COMMANDS_V08.md
docs/COMMANDS_V09.md
```

`COMMANDS_V09.md` contains the reproducible PowerShell command history and the purpose of each command.
