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
