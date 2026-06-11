# Human-talk confusion and final evaluation update

## What was analysed

This report uses the new confusion-matrix package:

- segment-level selected-policy confusion matrices
- segment-level final-exit confusion matrices
- clip-level dynamic-policy confusion matrices
- clip-level full-final confusion matrices

The provided clean2 screenshot is consistent with the exported `clean2 / 3-exit / segment final-exit` matrix: 634 correct Les_Brown segments, 631 correct Simon_Sinek segments, and only 13 total errors.

## Important interpretation note

The confusion matrices are **single-label views** of a model trained/evaluated through a multi-label pipeline.  
For these clean speaker stages this is valid because each true sample has exactly one speaker label. However, prediction conversion may use argmax when the multi-label output is empty or contains more than one active label. Therefore, confusion accuracy can look slightly different from multi-label Exact Match / Macro-F1.

For example, in `clean5 / 5-exit / segment selected-policy`, only 95.89% of predictions were already single-positive; 2.53% were multi-positive and 1.58% were empty before argmax conversion. This is why confusion-matrix results should be reported as **speaker-identification confusion analysis**, while Macro-F1/Hamming/Jaccard remain the main multi-label metrics.

## Confusion summary

| stage           | model_type   | level   | policy          |   n_samples |   accuracy |   errors | worst_class   |   worst_class_f1 |
|:----------------|:-------------|:--------|:----------------|------------:|-----------:|---------:|:--------------|-----------------:|
| clean2_balanced | 3-exit       | clip    | dynamic_policy  |         142 |     0.993  |        1 | Simon_Sinek   |           0.9929 |
| clean2_balanced | 3-exit       | clip    | full_final      |         142 |     1      |        0 | Les_Brown     |           1      |
| clean2_balanced | 5-exit       | clip    | dynamic_policy  |         142 |     1      |        0 | Les_Brown     |           1      |
| clean2_balanced | 5-exit       | clip    | full_final      |         142 |     1      |        0 | Les_Brown     |           1      |
| clean2_balanced | 3-exit       | segment | final_exit      |        1278 |     0.9898 |       13 | Simon_Sinek   |           0.9898 |
| clean2_balanced | 3-exit       | segment | selected_policy |        1278 |     0.9898 |       13 | Simon_Sinek   |           0.9898 |
| clean2_balanced | 5-exit       | segment | final_exit      |        1278 |     0.993  |        9 | Les_Brown     |           0.993  |
| clean2_balanced | 5-exit       | segment | selected_policy |        1278 |     0.9898 |       13 | Les_Brown     |           0.9898 |
| clean3_balanced | 3-exit       | clip    | dynamic_policy  |         213 |     0.9906 |        2 | Simon_Sinek   |           0.9859 |
| clean3_balanced | 3-exit       | clip    | full_final      |         213 |     1      |        0 | Les_Brown     |           1      |
| clean3_balanced | 5-exit       | clip    | dynamic_policy  |         213 |     0.9906 |        2 | Simon_Sinek   |           0.9859 |
| clean3_balanced | 5-exit       | clip    | full_final      |         213 |     1      |        0 | Les_Brown     |           1      |
| clean3_balanced | 3-exit       | segment | final_exit      |        1917 |     0.9812 |       36 | Simon_Sinek   |           0.9739 |
| clean3_balanced | 3-exit       | segment | selected_policy |        1917 |     0.9812 |       36 | Simon_Sinek   |           0.9739 |
| clean3_balanced | 5-exit       | segment | final_exit      |        1917 |     0.9812 |       36 | Simon_Sinek   |           0.9746 |
| clean3_balanced | 5-exit       | segment | selected_policy |        1917 |     0.9802 |       38 | Simon_Sinek   |           0.9721 |
| clean4_balanced | 3-exit       | clip    | dynamic_policy  |         284 |     0.993  |        2 | Simon_Sinek   |           0.9859 |
| clean4_balanced | 3-exit       | clip    | full_final      |         284 |     0.9965 |        1 | Oprah_Winfrey |           0.9929 |
| clean4_balanced | 5-exit       | clip    | dynamic_policy  |         284 |     1      |        0 | Les_Brown     |           1      |
| clean4_balanced | 5-exit       | clip    | full_final      |         284 |     1      |        0 | Les_Brown     |           1      |
| clean4_balanced | 3-exit       | segment | final_exit      |        2556 |     0.9812 |       48 | Simon_Sinek   |           0.9734 |
| clean4_balanced | 3-exit       | segment | selected_policy |        2556 |     0.9812 |       48 | Simon_Sinek   |           0.9734 |
| clean4_balanced | 5-exit       | segment | final_exit      |        2556 |     0.9828 |       44 | Simon_Sinek   |           0.9774 |
| clean4_balanced | 5-exit       | segment | selected_policy |        2556 |     0.98   |       51 | Simon_Sinek   |           0.9719 |
| clean5_balanced | 3-exit       | clip    | dynamic_policy  |         330 |     1      |        0 | Les_Brown     |           1      |
| clean5_balanced | 3-exit       | clip    | full_final      |         330 |     1      |        0 | Les_Brown     |           1      |
| clean5_balanced | 5-exit       | clip    | dynamic_policy  |         330 |     0.9879 |        4 | Oprah_Winfrey |           0.9688 |
| clean5_balanced | 5-exit       | clip    | full_final      |         330 |     0.9939 |        2 | Oprah_Winfrey |           0.9846 |
| clean5_balanced | 3-exit       | segment | final_exit      |        2970 |     0.9848 |       45 | Simon_Sinek   |           0.9702 |
| clean5_balanced | 3-exit       | segment | selected_policy |        2970 |     0.9848 |       45 | Simon_Sinek   |           0.9702 |
| clean5_balanced | 5-exit       | segment | final_exit      |        2970 |     0.9704 |       88 | Simon_Sinek   |           0.9385 |
| clean5_balanced | 5-exit       | segment | selected_policy |        2970 |     0.9667 |       99 | Simon_Sinek   |           0.9319 |

## 3-exit vs 5-exit confusion accuracy deltas

| stage_label   | comparison              |   accuracy_3exit |   accuracy_5exit |   accuracy_delta_5_minus_3 |   errors_3exit |   errors_5exit | worst_class_3exit   | worst_class_5exit   |
|:--------------|:------------------------|-----------------:|-----------------:|---------------------------:|---------------:|---------------:|:--------------------|:--------------------|
| 2 speakers    | Segment selected policy |           0.9898 |           0.9898 |                     0      |             13 |             13 | Simon_Sinek         | Les_Brown           |
| 2 speakers    | Segment final exit      |           0.9898 |           0.993  |                     0.0031 |             13 |              9 | Simon_Sinek         | Les_Brown           |
| 2 speakers    | Clip dynamic policy     |           0.993  |           1      |                     0.007  |              1 |              0 | Simon_Sinek         | Les_Brown           |
| 2 speakers    | Clip full final         |           1      |           1      |                     0      |              0 |              0 | Les_Brown           | Les_Brown           |
| 3 speakers    | Segment selected policy |           0.9812 |           0.9802 |                    -0.001  |             36 |             38 | Simon_Sinek         | Simon_Sinek         |
| 3 speakers    | Segment final exit      |           0.9812 |           0.9812 |                     0      |             36 |             36 | Simon_Sinek         | Simon_Sinek         |
| 3 speakers    | Clip dynamic policy     |           0.9906 |           0.9906 |                     0      |              2 |              2 | Simon_Sinek         | Simon_Sinek         |
| 3 speakers    | Clip full final         |           1      |           1      |                     0      |              0 |              0 | Les_Brown           | Les_Brown           |
| 4 speakers    | Segment selected policy |           0.9812 |           0.98   |                    -0.0012 |             48 |             51 | Simon_Sinek         | Simon_Sinek         |
| 4 speakers    | Segment final exit      |           0.9812 |           0.9828 |                     0.0016 |             48 |             44 | Simon_Sinek         | Simon_Sinek         |
| 4 speakers    | Clip dynamic policy     |           0.993  |           1      |                     0.007  |              2 |              0 | Simon_Sinek         | Les_Brown           |
| 4 speakers    | Clip full final         |           0.9965 |           1      |                     0.0035 |              1 |              0 | Oprah_Winfrey       | Les_Brown           |
| 5 speakers    | Segment selected policy |           0.9848 |           0.9667 |                    -0.0182 |             45 |             99 | Simon_Sinek         | Simon_Sinek         |
| 5 speakers    | Segment final exit      |           0.9848 |           0.9704 |                    -0.0145 |             45 |             88 | Simon_Sinek         | Simon_Sinek         |
| 5 speakers    | Clip dynamic policy     |           1      |           0.9879 |                    -0.0121 |              0 |              4 | Les_Brown           | Oprah_Winfrey       |
| 5 speakers    | Clip full final         |           1      |           0.9939 |                    -0.0061 |              0 |              2 | Les_Brown           | Oprah_Winfrey       |

## Clean5 segment selected-policy per-class confusion metrics

| model_type   | label         |   tp |   fp |   fn |   precision |   recall |     f1 |   support |   predicted |
|:-------------|:--------------|-----:|-----:|-----:|------------:|---------:|-------:|----------:|------------:|
| 3-exit       | Les_Brown     |  591 |    0 |    3 |      1      |   0.9949 | 0.9975 |       594 |         591 |
| 3-exit       | Mel_Robbins   |  589 |    2 |    5 |      0.9966 |   0.9916 | 0.9941 |       594 |         591 |
| 3-exit       | Oprah_Winfrey |  584 |   10 |   10 |      0.9832 |   0.9832 | 0.9832 |       594 |         594 |
| 3-exit       | Rabin_Sharma  |  592 |   23 |    2 |      0.9626 |   0.9966 | 0.9793 |       594 |         615 |
| 3-exit       | Simon_Sinek   |  569 |   10 |   25 |      0.9827 |   0.9579 | 0.9702 |       594 |         579 |
| 5-exit       | Les_Brown     |  587 |    6 |    7 |      0.9899 |   0.9882 | 0.989  |       594 |         593 |
| 5-exit       | Mel_Robbins   |  583 |    3 |   11 |      0.9949 |   0.9815 | 0.9881 |       594 |         586 |
| 5-exit       | Oprah_Winfrey |  547 |    4 |   47 |      0.9927 |   0.9209 | 0.9555 |       594 |         551 |
| 5-exit       | Rabin_Sharma  |  566 |    6 |   28 |      0.9895 |   0.9529 | 0.9708 |       594 |         572 |
| 5-exit       | Simon_Sinek   |  588 |   80 |    6 |      0.8802 |   0.9899 | 0.9319 |       594 |         668 |

## Main findings

1. **The confusion export is working correctly.** The ZIP contains 32 confusion matrices: segment selected, segment final, clip dynamic, and clip full-final for 3-exit and 5-exit across clean2–clean5.
2. **Clean2 is extremely strong.** The shown 2-class final-exit segment matrix has only 13 errors out of 1,278 segments, giving 98.98% single-label confusion accuracy.
3. **Clip-level performance is much stronger than segment-level performance.** This is expected because clip-level aggregation reduces isolated 1-second segment mistakes. For example, clean5 3-exit reaches 100% clip dynamic accuracy, while segment selected accuracy is 98.48%.
4. **The 3-exit model remains the stronger accuracy baseline at clean5.** Clean5 segment selected accuracy is 98.48% for 3-exit vs 96.67% for 5-exit.
5. **The 5-exit model still provides the efficiency story, but not the best confusion accuracy.** Its clean5 clip dynamic accuracy is 98.79%, and clip full-final accuracy is 99.39%, but the 3-exit clip model reaches 100%.
6. **Simon_Sinek is still the most frequent weak/confused class at segment level.** In clean5 segment selected-policy, Simon_Sinek is the worst class for both 3-exit and 5-exit.
7. **Oprah_Winfrey becomes the weakest class in clean5 clip-level 5-exit evaluation.** This suggests a small number of clip-level aggregation mistakes involving Oprah_Winfrey.

## Recommended paper wording

The confusion-matrix analysis confirms that the clean human-talk benchmark is separable at both segment and clip levels. Segment-level errors increase as the number of speakers grows, with Simon_Sinek emerging as the most frequently confused class. Clip-level aggregation substantially improves robustness by smoothing isolated segment mistakes. The 3-exit model provides the strongest speaker-identification accuracy, while the 5-exit model should be positioned as an efficiency-oriented dynamic model rather than an accuracy-improving model.

## Plots

![Segment selected accuracy](plots/segment_selected_confusion_accuracy.png)

![Segment final accuracy](plots/segment_final_confusion_accuracy.png)

![Clip dynamic accuracy](plots/clip_dynamic_confusion_accuracy.png)

![Clip full accuracy](plots/clip_full_confusion_accuracy.png)

![Clean5 errors](plots/clean5_confusion_errors.png)

![Clean5 per-class F1](plots/clean5_segment_selected_perclass_f1.png)

## Representative confusion matrices

![Clean2 3-exit segment final](confusion_images/c2__3e__segment_confusion__segment_final_exit_confusion_matrix.png)

![Clean5 3-exit segment selected](confusion_images/c5__3e__segment_confusion__segment_selected_policy_confusion_matrix.png)

![Clean5 5-exit segment selected](confusion_images/c5__5e__segment_confusion__segment_selected_policy_confusion_matrix.png)

![Clean5 5-exit clip dynamic](confusion_images/c5__5e__clip_confusion__clip_dynamic_policy_confusion_matrix.png)

![Clean5 5-exit clip full final](confusion_images/c5__5e__clip_confusion__clip_full_final_confusion_matrix.png)
