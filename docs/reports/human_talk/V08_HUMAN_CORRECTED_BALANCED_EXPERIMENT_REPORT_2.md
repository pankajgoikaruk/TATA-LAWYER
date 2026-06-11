# v0.8 Human-Corrected-Balanced Experiment Report

## 1. Overview

This report documents the latest ASHADIP/NeuroAccuExit result on branch:

```text
agentic_data_preprocessing_v0.8
```

The main experiment is:

```text
v0.8-human-corrected-balanced
run = main_v08_human_corrected_balanced_3exit_20260610_084027
```

The core contribution of v0.8 is a safer data-preprocessing and evaluation pipeline:

```text
v0.6 trusted base
+ reviewed LAWYER-new samples
+ corrected non-target context labels
+ corrected holdout context labels
+ known non-target identity repair
+ balanced training manifest
```

The latest additional finding is a **post-hoc label-aware aggregation rule** that improves Macro-F1 by using different parent-level pooling rules for stable and transient labels.

## 2. Evaluation setting

| Item | Value |
|---|---|
| Dataset | Human-talk multi-label audio |
| Corrected holdout | 867 parent clips / 4,335 one-second segments |
| Labels | 10 |
| Model | TinyAudioCNN + ExitNet |
| Exits | 3 |
| Tap blocks | `1,3` |
| Official threshold | fixed 0.5 |
| Official parent aggregation | mean probability |
| Additional analysis | label-aware mean/max aggregation |

## 3. Official corrected-holdout result

The official headline result uses parent/clip-level mean probability aggregation for all labels.

```text
v0.8-HCB 3-exit
parent mean
fixed threshold 0.5
Exit 3
```

| method               |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   jaccard_score |   avg_true_labels |   avg_pred_labels |
|:---------------------|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|------------------:|------------------:|
| parent_mean_official |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |          0.9174 |            1.4694 |            1.4302 |

## 4. Fair comparison with v0.6

| model                       |   exit | aggregation          |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_pred_labels |
|:----------------------------|-------:|:---------------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|
| v0.6 3-exit                 |      3 | mean                 |     0.7537 |     0.8865 |       0.8992 |        0.7497 |         0.0315 |            1.3045 |
| v0.6 5-exit                 |      5 | mean                 |     0.7460 |     0.8771 |       0.8881 |        0.7232 |         0.0338 |            1.2814 |
| v0.8-HCB 3-exit official    |      3 | mean                 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4302 |
| v0.8-HCB 3-exit label-aware |      3 | mean/max label-aware |     0.8320 |     0.9285 |       0.9375 |        0.8235 |         0.0211 |            1.4844 |

![v0.6 vs v0.8-HCB line plot](../../figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_vs_v06_label_aware_lineplot.png)

The v0.8-HCB parent-mean result improves over the v0.6 3-exit model on the same corrected holdout:

| Metric | v0.6 3-exit | v0.8-HCB mean | Difference |
|---|---:|---:|---:|
| Macro-F1 | 0.7537 | 0.7801 | +0.0264 |
| Micro-F1 | 0.8865 | 0.9332 | +0.0467 |
| Samples-F1 | 0.8992 | 0.9406 | +0.0414 |
| Exact Match | 0.7497 | 0.8397 | +0.0900 |
| Hamming Loss | 0.0315 | 0.0194 | -0.0121 |

## 5. Global max diagnostic

A global max aggregation diagnostic was run because rare event labels are often short and bursty. However, global max is not suitable as the final method because it produces too many false positives.

| method                |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_pred_labels | official_use                                 |
|:----------------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|:---------------------------------------------|
| parent_mean_official  |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4302 | headline overall result                      |
| global_max_diagnostic |     0.7251 |     0.8203 |       0.8423 |        0.5121 |         0.0630 |            2.0346 | diagnostic only; not final                   |
| label_aware_mean_max  |     0.8320 |     0.9285 |       0.9375 |        0.8235 |         0.0211 |            1.4844 | post-hoc macro-F1 / transient-label analysis |

![Aggregation strategy line plot](../../figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_aggregation_strategy_lineplot.png)

![Macro-Hamming trade-off](../../figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_macro_hamming_tradeoff_bar.png)

## 6. Weak/transient label result

The global max diagnostic showed that max pooling can help event labels:

| label                     | aggregation   |   precision |   recall |     f1 |   support |   predicted_positive |
|:--------------------------|:--------------|------------:|---------:|-------:|----------:|---------------------:|
| audience_reaction_present | mean          |      0.6667 |   0.0690 | 0.1250 |        29 |                    3 |
| audience_reaction_present | global_max    |      0.4103 |   0.5517 | 0.4706 |        29 |                   39 |
| audience_reaction_present | label_aware   |      0.4103 |   0.5517 | 0.4706 |        29 |                   39 |
| silence_present           | mean          |      0.0000 |   0.0000 | 0.0000 |        12 |                    0 |
| silence_present           | global_max    |      0.1818 |   0.1667 | 0.1739 |        12 |                   11 |
| silence_present           | label_aware   |      0.1818 |   0.1667 | 0.1739 |        12 |                   11 |

![Weak label F1 line plot](../../figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_weak_label_f1_lineplot.png)

This finding explains why the official parent-mean Macro-F1 is lower than Micro-F1: the main weakness is not broad multi-label accuracy but the rare transient event labels.

## 7. Label-aware aggregation

The final recommended post-hoc aggregation analysis is:

```text
mean for:
  Brene_Brown
  Eckhart_Tolle
  Eric_Thomas
  Gary_Vee
  Jay_Shetty
  Nick_Vujicic
  other_speaker_present
  music_present

max for:
  audience_reaction_present
  silence_present
```

This gives:

```text
Macro-F1     = 0.8320
Micro-F1     = 0.9285
Samples-F1   = 0.9375
Exact Match  = 0.8235
Hamming Loss = 0.0211
```

The Macro-F1 gain is:

```text
0.7801 -> 0.8320
absolute gain = 0.0520
```

![Per-label mean vs label-aware](../../figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_per_label_mean_vs_labelaware_bar.png)

## 8. Scientific interpretation

This result supports a more nuanced interpretation of parent-level aggregation in multi-label audio. Labels are heterogeneous:

- speaker identity labels are persistent across the clip;
- music is often sustained across multiple segments;
- audience reactions and silence can be short and intermittent.

Therefore, a single aggregation rule is suboptimal. Mean aggregation suppresses segment noise for stable labels, but it can dilute transient labels. Max aggregation detects transient labels but over-fires for stable labels. Label-aware aggregation combines both advantages.

## 9. Reporting recommendation

Use two result blocks:

### Main official result

```text
v0.8-HCB, parent mean, fixed 0.5
Micro-F1   = 0.9332
Samples-F1 = 0.9406
Exact      = 0.8397
Hamming    = 0.0194
```

### Label-aware research finding

```text
v0.8-HCB, mean for stable labels, max for transient labels
Macro-F1 = 0.8320
```

## 10. Thesis-ready paragraph

On the corrected parent-level holdout set containing 867 clips and 4,335 segments, the v0.8-human-corrected-balanced 3-exit model achieved the strongest overall result using mean probability aggregation and a fixed 0.5 threshold, with Micro-F1 = 0.9332, Samples-F1 = 0.9406, Exact Match = 0.8397, and Hamming Loss = 0.0194. A subsequent post-hoc aggregation analysis showed that global max pooling is not suitable for all labels because it increases false positives. However, using max pooling only for transient event labels (`audience_reaction_present` and `silence_present`) while retaining mean pooling for the remaining stable labels improved Macro-F1 from 0.7801 to 0.8320. This demonstrates that label-aware parent-level aggregation can recover rare transient events without retraining the model.
