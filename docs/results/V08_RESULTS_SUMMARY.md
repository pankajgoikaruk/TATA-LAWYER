# v0.8 Results Summary

## Official result

```text
Experiment: v0.8-human-corrected-balanced
Model: main_v08_human_corrected_balanced_3exit_20260610_084027
Evaluation: corrected holdout, parent/clip-level mean aggregation
Threshold: fixed 0.5
Parent clips: 867
Segments: 4,335
Official exit: 3
```

| model           | threshold_mode   | aggregation   |   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   jaccard_score |   avg_true_labels |   avg_pred_labels |
|:----------------|:-----------------|:--------------|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|------------------:|------------------:|
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |          0.9174 |            1.4694 |            1.4302 |

## Fair comparison with previous models

| model           |   final_exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:----------------|-------------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| v0.6 3-exit     |            3 |     0.7537 |     0.8865 |       0.8992 |        0.7497 |         0.0315 |            1.4694 |            1.3045 |
| v0.6 5-exit     |            5 |     0.746  |     0.8771 |       0.8881 |        0.7232 |         0.0338 |            1.4694 |            1.2814 |
| v0.8-HCB 3-exit |            3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4694 |            1.4302 |

## Improvement over v0.6 3-exit

| Metric       |   v0.8 minus v0.6 3-exit |
|:-------------|-------------------------:|
| Macro-F1     |                   0.0264 |
| Micro-F1     |                   0.0467 |
| Samples-F1   |                   0.0414 |
| Exact Match  |                   0.09   |
| Hamming Loss |                  -0.0121 |

The hamming-loss change is negative, which is desirable. The label error rate falls from 0.0315 to 0.0194.

## Fixed vs tuned threshold on corrected holdout

| setting        |   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_pred_labels |
|:---------------|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|
| fixed_0p5      |      1 |     0.113  |     0.3166 |       0.204  |        0.0288 |         0.1275 |            0.3956 |
| fixed_0p5      |      2 |     0.6315 |     0.7739 |       0.7197 |        0.5467 |         0.0591 |            1.1419 |
| fixed_0p5      |      3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4302 |
| tuned_per_exit |      1 |     0.3756 |     0.5239 |       0.547  |        0.1546 |         0.2146 |            3.0392 |
| tuned_per_exit |      2 |     0.7134 |     0.8107 |       0.8328 |        0.5409 |         0.0597 |            1.6863 |
| tuned_per_exit |      3 |     0.7487 |     0.9139 |       0.921  |        0.8143 |         0.0243 |            1.3576 |

Decision: use fixed 0.5 for official reporting.

## Corrected holdout per-label final-exit F1

| label                     |   precision |   recall |     f1 |   support |   predicted_positive |
|:--------------------------|------------:|---------:|-------:|----------:|---------------------:|
| Brene_Brown               |      1      |   0.9315 | 0.9645 |        73 |                   68 |
| Eckhart_Tolle             |      1      |   0.9643 | 0.9818 |        84 |                   81 |
| Eric_Thomas               |      0.9028 |   0.9559 | 0.9286 |        68 |                   72 |
| Gary_Vee                  |      1      |   0.9559 | 0.9774 |        68 |                   65 |
| Jay_Shetty                |      0.9278 |   1      | 0.9626 |        90 |                   97 |
| Nick_Vujicic              |      1      |   0.9592 | 0.9792 |        49 |                   47 |
| other_speaker_present     |      0.9156 |   0.9435 | 0.9293 |       460 |                  474 |
| music_present             |      0.964  |   0.9413 | 0.9525 |       341 |                  333 |
| audience_reaction_present |      0.6667 |   0.069  | 0.125  |        29 |                    3 |
| silence_present           |      0      |   0      | 0      |        12 |                    0 |
