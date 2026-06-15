# Methodology Appendix

---

## O. Post-hoc label-aware aggregation analysis

After the official corrected-holdout parent-level mean evaluation, an additional aggregation diagnostic was performed to test whether transient labels were being diluted by mean probability aggregation.

The motivation was that not all labels have the same temporal behaviour:

| Label type | Labels | Parent aggregation assumption |
|---|---|---|
| Stable identity/background labels | `Brene_Brown`, `Eckhart_Tolle`, `Eric_Thomas`, `Gary_Vee`, `Jay_Shetty`, `Nick_Vujicic`, `other_speaker_present`, `music_present` | Evidence should be consistent across several segments, so mean aggregation suppresses noisy false positives. |
| Transient/bursty event labels | `audience_reaction_present`, `silence_present` | Evidence may appear in only one or two short segments, so max aggregation can recover events diluted by mean aggregation. |

Three parent-level aggregation strategies were compared on the same corrected holdout set:

```text
1. Parent mean aggregation for all labels
2. Global max aggregation for all labels
3. Label-aware aggregation:
   - mean for 8 stable labels
   - max for audience_reaction_present and silence_present
```

## P. Global max diagnostic

Global max aggregation was tested because it can recover short events. However, applying max to every label over-predicted labels and created too many false positives.

| Metric | Parent mean official | Global max diagnostic |
|---|---:|---:|
| Macro-F1 | **0.7801** | 0.7251 |
| Micro-F1 | **0.9332** | 0.8203 |
| Samples-F1 | **0.9406** | 0.8423 |
| Exact Match | **0.8397** | 0.5121 |
| Hamming Loss | **0.0194** | 0.0630 |
| Avg Pred Labels | 1.4302 | 2.0346 |

The global max result shows that max aggregation should not be used as the final overall parent-level strategy.

## Q. Weak/transient label behaviour

Although global max damaged the overall result, it improved the two rare transient labels.

| Label | Parent mean F1 | Global max F1 |
|---|---:|---:|
| `audience_reaction_present` | 0.1250 | **0.4706** |
| `silence_present` | 0.0000 | **0.1739** |

This supports the hypothesis that weak transient labels are not necessarily absent from the model predictions; instead, their probabilities are diluted when averaged across the whole parent clip.

## R. Label-aware aggregation result

The final post-hoc aggregation rule used:

```text
mean for 8 stable labels:
  Brene_Brown
  Eckhart_Tolle
  Eric_Thomas
  Gary_Vee
  Jay_Shetty
  Nick_Vujicic
  other_speaker_present
  music_present

max for 2 transient labels:
  audience_reaction_present
  silence_present
```

The resulting corrected-holdout Exit-3 performance was:

| Metric | Parent mean official | Label-aware mean/max |
|---|---:|---:|
| Macro-F1 | 0.7801 | **0.8320** |
| Micro-F1 | **0.9332** | 0.9285 |
| Samples-F1 | **0.9406** | 0.9375 |
| Exact Match | **0.8397** | 0.8235 |
| Hamming Loss | **0.0194** | 0.0211 |
| Avg Pred Labels | 1.4302 | 1.4844 |

The label-aware method produced a large Macro-F1 improvement:

```text
0.7801 -> 0.8320
absolute gain = +0.0519
```

with only a small reduction in Micro-F1, Samples-F1, Exact Match, and Hamming Loss.

## S. Updated final interpretation

The official headline result remains:

```text
v0.8-human-corrected-balanced
parent-level mean aggregation
fixed threshold 0.5
Exit 3
Micro-F1=0.9332
Samples-F1=0.9406
Exact Match=0.8397
Hamming Loss=0.0194
```

The label-aware aggregation result should be reported as an additional research finding:

```text
v0.8-human-corrected-balanced
post-hoc label-aware aggregation
mean for stable labels
max for transient labels
Exit 3
Macro-F1=0.8320
```

This finding supports the thesis claim that label-specific parent aggregation can improve rare transient labels without retraining the model.

## T. Updated thesis-ready conclusion with label-aware finding

On the corrected parent-level holdout set containing 867 parent clips and 4,335 one-second segments, the v0.8-human-corrected-balanced 3-exit model achieved the strongest overall final-exit performance under mean probability aggregation and a fixed 0.5 threshold. Compared with the previous v0.6 3-exit model re-evaluated on the same corrected holdout, it improved Macro-F1 from 0.7537 to 0.7801, Micro-F1 from 0.8865 to 0.9332, Samples-F1 from 0.8992 to 0.9406, and Exact Match from 0.7497 to 0.8397, while reducing Hamming Loss from 0.0315 to 0.0194.

A further post-hoc aggregation analysis showed that rare transient labels were diluted by parent-level mean aggregation. Global max aggregation improved `audience_reaction_present` and `silence_present`, but degraded the overall result by over-predicting labels. A label-aware aggregation rule, using mean aggregation for eight stable labels and max aggregation only for the two transient labels, improved Macro-F1 from 0.7801 to 0.8320 while maintaining high Micro-F1 of 0.9285 and Samples-F1 of 0.9375. This indicates that label-specific aggregation can recover rare event labels without changing the trained model.

## U. Additional recommended thesis figures

| Figure | File | Purpose |
|---|---|---|
| Aggregation strategy line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_aggregation_strategy_lineplot.png` | Compares parent mean, global max, and label-aware aggregation across Macro-F1, Micro-F1, Samples-F1, and Exact Match. |
| Aggregation Hamming loss line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_aggregation_hamming_loss_lineplot.png` | Shows that global max increases false positives and label error. |
| Weak-label F1 line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_weak_label_f1_lineplot.png` | Shows improvement for `audience_reaction_present` and `silence_present` under max/label-aware aggregation. |
| Macro-Hamming trade-off plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_macro_hamming_tradeoff_bar.png` | Shows the trade-off between Macro-F1 improvement and small Hamming-loss increase for label-aware aggregation. |

---

## V. v0.9 TATA-triage workspace and preservation policy

The v0.9 experiment was created as a separate workspace so that the original TATA-2/v0.6 assets and the v0.8 main-model assets remained unchanged.

```text
human_talk_workspace/
└── tata_v0.9_pipeline/
    ├── tata_triage_model/
    ├── neuroaccuexit_main_model/
    └── shared/
```

The baseline parent and feature manifests were copied with `_BASELINE` names and treated as immutable:

```text
tata2_parent_manifest_12label_2074_BASELINE.csv
tata_seed_features_manifest_10label_12469_BASELINE.csv
human_talk_10label_schema.json
```

The TATA triage manifest remained separate from the downstream main-model manifest. This separation prevents pseudo-label generation, triage experiments, and low-energy ablations from silently modifying the main-model training data.

## W. Audited v0.9 seed reconstruction

The original reviewed seed contained 2,074 parent clips and 12,469 one-second feature rows. Existing silence and rare-event labels were reviewed, and 27 additional verified silence clips were renamed without colliding with existing filenames.

The final audited parent count was:

```text
2,074 original parents
+   27 new verified silence parents
=2,101 final parents
```

The initial v0.9 feature cache reused the 12,469 legacy features and extracted 120 additional segments from the 27 new silence parents:

```text
12,469 reused feature rows
+  120 new silence feature rows
=12,589 v0.9 feature rows
```

The split remained parent-safe:

| Split | Rows |
|---|---:|
| Train | 8,745 |
| Validation | 1,883 |
| Test | 1,961 |

The v0.9 builder also repaired missing metadata, recomputed `labels` and `num_active_labels`, cleared stale fine-audience metadata, normalised paths, and retained one-second segments with a 0.5-second hop for the new silence data.

## X. Original v0.9 TATA result

The original v0.9 3-exit model used the same architecture and training settings as the previous TATA comparison:

```text
tap blocks = 1,3
loss weights = 0.3,0.3,1.0
epochs = 40
batch size = 64
learning rate = 0.001
threshold = 0.5
seed = 42
device = CPU
```

Final-exit internal test performance was:

| Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---:|---:|---:|---:|---:|
| 0.8195 | 0.8226 | 0.8226 | 0.6527 | 0.0483 |

This was approximately level with the earlier v0.6 internal TATA result while using a cleaner and more auditable manifest.

## Y. Low-energy filtering diagnosis

The legacy preprocessing pipeline generated features only for windows that survived the energy filter. Therefore, missing low-energy windows could not be recovered from the 12,469-row feature manifest alone; they had to be rediscovered from the original parent audio.

A non-destructive audit used:

```text
window length = 1.0 second
window hop = 0.5 second
raw RMS measured before peak normalisation
```

Audit result:

| Item | Count |
|---|---:|
| Original parents scanned | 2,074 |
| Parent audio resolved | 2,074 |
| Grid windows scanned | 22,391 |
| Already represented windows | 12,164 |
| Missing low-energy windows before parent cap | 1,178 |
| Review candidates retained | 1,018 |
| High priority | 250 |
| Medium priority | 90 |
| Low priority | 678 |

Manual silence review produced:

| Human silence decision | Count |
|---|---:|
| `silence_present = 1` | 271 |
| `silence_present = 0` | 747 |

The 747 negative examples are scientifically important because they demonstrate that low energy is not equivalent to silence.

## Z. Recovered timeline and feature cache

All 1,018 reviewed candidates were incorporated into the recovered timeline. One candidate already matched an existing feature-manifest location and was updated in place; 1,017 missing candidates received new features.

| Recovery item | Count |
|---|---:|
| Existing feature rows | 12,589 |
| Existing reviewed rows updated in place | 1 |
| Reviewed rows appended | 1,017 |
| Full recovered rows | 13,606 |
| Affected parents | 522 |
| Parent silence labels changed 0 to 1 | 0 |

The parent-level consecutive-silence rule was:

```text
at least two reviewed silent one-second windows
with start times separated by one 0.5-second hop
```

With one-second windows and a 0.5-second hop, two consecutive positives represent approximately 1.5 seconds of continuous silence evidence.

The lack of parent-label changes does not invalidate the recovery. It means the newly recovered silence windows occurred inside parents already labelled silence-positive, while the major benefit was improved segment-level evidence.

## AA. Low-energy recovery experiments

### AA.1 Full recovery

The full recovery experiment included all 271 silence-positive and 747 silence-negative reviewed candidates.

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

### AA.2 Silence-positive-only ablation

The positive-only ablation removed all 747 reviewed non-silence candidates from train, validation, and test while retaining the 271 confirmed silence candidates.

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

### AA.3 Comparison with original v0.9

| Experiment | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Silence F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original v0.9 | 0.8195 | **0.8226** | **0.8226** | **0.6527** | **0.0483** | 0.6667 |
| Full recovery | 0.8064 | 0.8064 | 0.7968 | 0.6022 | 0.0539 | **0.7875** |
| Silence-positive only | **0.8199** | 0.8120 | 0.8075 | 0.6110 | 0.0537 | 0.7355 |

The full recovery model produced the strongest silence classification, while the positive-only ablation restored most of the global Macro-F1 lost under full recovery.

## AB. Interpretation of inherited-label uncertainty

Only `silence_present` was manually verified for the recovered one-second clips. The other nine labels were initially inherited from the parent clip. Parent labels may describe another temporal region and are not guaranteed to be true for the recovered second.

For example, a five-second parent may contain Gary Vee in seconds 0–2, silence in seconds 2–4, and music in seconds 4–5. A recovered silent window should not automatically inherit both `Gary_Vee = 1` and `music_present = 1`.

This explains the experimental pattern:

- silence F1 improved strongly;
- several speaker and event labels declined under full recovery;
- removing the 747 non-silence rows restored global performance but reduced silence performance.

The correct conclusion is not to discard low-energy data. The correct conclusion is to avoid forcing uncertain inherited labels onto segment-level examples.

## AC. Tri-state nine-label manual review

The recovered clips now use the following annotation protocol for the remaining nine labels:

```text
 1 = confidently present
 0 = confidently absent
-1 = reviewed but uncertain or unknown
blank = not reviewed yet
```

`review_silence_present` remains the trusted human decision and must not be replaced with `-1`.

Overlapping labels are allowed. A segment may contain a target speaker, another speaker, music, and audience reaction simultaneously.

When the whole one-second clip is unclear, all nine new review fields may be set to `-1`, while the existing silence decision remains unchanged.

## AD. Planned masked supervision

For final training, unknown labels should not be treated as negatives. A label mask should be generated as:

```text
label is 0 or 1 -> mask = 1
label is -1     -> mask = 0
```

The loss should be calculated only over known labels:

```text
masked BCE = sum(BCE × mask) / sum(mask)
```

This preserves all 1,018 reviewed low-energy examples, allows the 747 hard negatives to supervise `silence_present = 0`, and prevents uncertain labels from contaminating the remaining classes.

## AE. Evaluation policy after partial annotation

For rigorous reporting:

1. Evaluate the original and recovered models on the same original 1,961-row test set.
2. Evaluate recovered low-energy clips separately for `silence_present`.
3. Use only fully known labels for per-label evaluation.
4. Do not report ordinary Exact Match for rows containing any `-1` labels.
5. Use masked per-label metrics for partially labelled recovered rows.
6. Use the original trusted validation rows for full-label checkpoint selection until recovered validation labels are fully reviewed.

The v0.8 parent-level main-model metrics and v0.9 segment-level TATA metrics must remain separate in all tables and thesis claims.
