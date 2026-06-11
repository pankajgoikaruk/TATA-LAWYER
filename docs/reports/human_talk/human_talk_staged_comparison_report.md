# Human-talk staged benchmark comparison

## Scope

Analysed four clean staged speaker-identification experiments:

- `clean2_balanced`
- `clean3_balanced`
- `clean4_balanced`
- `clean5_balanced`

Each stage contains 3-exit and 5-exit no-hint models evaluated with the updated multi-label metric script.

**Important limitation:** the uploaded ZIPs contain segment-level/static-exit/dynamic-policy outputs. I did not find clip/window-level files such as `avg_windows_used`, `windows_saved_pct`, or detection-latency tables. Therefore, this report covers segment-level early-exit behaviour and depth-compute efficiency only. Clip-level window efficiency should be added as a next evaluation script.

## Dataset stage summary

| stage           |   n_labels | labels                                                           |   n_parent_clips |   n_segments |   train_segments |   val_segments |   test_segments |
|:----------------|-----------:|:-----------------------------------------------------------------|-----------------:|-------------:|-----------------:|---------------:|----------------:|
| clean2_balanced |          2 | Les_Brown, Simon_Sinek                                           |              944 |         8496 |             5940 |           1278 |            1278 |
| clean3_balanced |          3 | Les_Brown, Simon_Sinek, Rabin_Sharma                             |             1416 |        12744 |             8910 |           1917 |            1917 |
| clean4_balanced |          4 | Les_Brown, Simon_Sinek, Rabin_Sharma, Oprah_Winfrey              |             1888 |        16992 |            11880 |           2556 |            2556 |
| clean5_balanced |          5 | Les_Brown, Mel_Robbins, Oprah_Winfrey, Rabin_Sharma, Simon_Sinek |             2205 |        19845 |            13905 |           2970 |            2970 |

## Selected dynamic policy summary

| stage           | model_type   |   n_samples |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   hamming_accuracy |   jaccard_score |   macro_auprc |   avg_exit_depth |   depth_compute_saved_pct |   exit_consistency |   label_set_flip_any_rate |   avg_label_set_flip_count |   avg_label_bit_flip_count |
|:----------------|:-------------|------------:|-----------:|-----------:|-------------:|--------------:|---------------:|-------------------:|----------------:|--------------:|-----------------:|--------------------------:|-------------------:|--------------------------:|---------------------------:|---------------------------:|
| clean2_balanced | 3-exit       |        1278 |     0.9898 |     0.9898 |       0.9898 |        0.9898 |         0.0102 |             0.9898 |          0.9898 |        0.9996 |           3      |                    0      |             1      |                    0.1369 |                     0.1502 |                     0.2887 |
| clean2_balanced | 5-exit       |        1278 |     0.9898 |     0.9898 |       0.9898 |        0.9898 |         0.0102 |             0.9898 |          0.9898 |        0.9989 |           4.025  |                   19.4992 |             0.9953 |                    0.1385 |                     0.1612 |                     0.3036 |
| clean3_balanced | 3-exit       |        1917 |     0.9808 |     0.9809 |       0.9767 |        0.975  |         0.0127 |             0.9873 |          0.9763 |        0.9976 |           3      |                    0      |             1      |                    0.4956 |                     0.5712 |                     0.7173 |
| clean3_balanced | 5-exit       |        1917 |     0.9792 |     0.9794 |       0.9764 |        0.9729 |         0.0137 |             0.9863 |          0.9755 |        0.9955 |           4.121  |                   17.5796 |             0.987  |                    0.4971 |                     0.6802 |                     0.8023 |
| clean4_balanced | 3-exit       |        2556 |     0.9789 |     0.979  |       0.9717 |        0.9667 |         0.0105 |             0.9895 |          0.9705 |        0.9976 |           3      |                    0      |             1      |                    0.7433 |                     0.8345 |                     0.9769 |
| clean4_balanced | 5-exit       |        2556 |     0.9696 |     0.9695 |       0.9674 |        0.9515 |         0.0154 |             0.9846 |          0.9634 |        0.995  |           4.1451 |                   17.097  |             0.9855 |                    0.8075 |                     1.0278 |                     1.1373 |
| clean5_balanced | 3-exit       |        2970 |     0.9758 |     0.9758 |       0.9677 |        0.9589 |         0.0096 |             0.9904 |          0.9655 |        0.9976 |           3      |                    0      |             1      |                    0.7428 |                     0.8236 |                     0.9327 |
| clean5_balanced | 5-exit       |        2970 |     0.9629 |     0.9621 |       0.9582 |        0.9414 |         0.0152 |             0.9848 |          0.954  |        0.994  |           4.1535 |                   16.9293 |             0.9778 |                    0.731  |                     0.9886 |                     1.1114 |

## 3-exit vs 5-exit selected-policy deltas

Positive Macro-F1 delta means 5-exit is better than 3-exit. Negative means 3-exit is better.

| stage_label   |   macro_f1_3exit |   macro_f1_5exit |   macro_f1_delta_5_minus_3 |   exact_match_3exit |   exact_match_5exit |   compute_saved_5exit_pct |   exit_consistency_5exit |   flip_rate_5exit |
|:--------------|-----------------:|-----------------:|---------------------------:|--------------------:|--------------------:|--------------------------:|-------------------------:|------------------:|
| 2 speakers    |           0.9898 |           0.9898 |                     0      |              0.9898 |              0.9898 |                   19.4992 |                   0.9953 |            0.1385 |
| 3 speakers    |           0.9808 |           0.9792 |                    -0.0016 |              0.975  |              0.9729 |                   17.5796 |                   0.987  |            0.4971 |
| 4 speakers    |           0.9789 |           0.9696 |                    -0.0093 |              0.9667 |              0.9515 |                   17.097  |                   0.9855 |            0.8075 |
| 5 speakers    |           0.9758 |           0.9629 |                    -0.0129 |              0.9589 |              0.9414 |                   16.9293 |                   0.9778 |            0.731  |

## Final-exit static quality

This table ignores early-exit stopping and evaluates each model at its final exit.

| stage           | model_type   |   exit |   macro_f1 |   exact_match |   hamming_loss |   macro_auprc |
|:----------------|:-------------|-------:|-----------:|--------------:|---------------:|--------------:|
| clean2_balanced | 3-exit       |      3 |     0.9898 |        0.9898 |         0.0102 |        0.9996 |
| clean2_balanced | 5-exit       |      5 |     0.993  |        0.993  |         0.007  |        0.9998 |
| clean3_balanced | 3-exit       |      3 |     0.9808 |        0.975  |         0.0127 |        0.9976 |
| clean3_balanced | 5-exit       |      5 |     0.9795 |        0.9697 |         0.0136 |        0.9972 |
| clean4_balanced | 3-exit       |      3 |     0.9789 |        0.9667 |         0.0105 |        0.9976 |
| clean4_balanced | 5-exit       |      5 |     0.9724 |        0.9566 |         0.0139 |        0.9967 |
| clean5_balanced | 3-exit       |      3 |     0.9758 |        0.9589 |         0.0096 |        0.9976 |
| clean5_balanced | 5-exit       |      5 |     0.9654 |        0.9451 |         0.0141 |        0.996  |

## Policy vs final-exit quality

This shows how much the selected dynamic policy loses or preserves compared with always using the final exit.

| stage           | model_type   |   macro_f1_policy |   macro_f1_final |   macro_f1_policy_minus_final |   exact_match_policy |   exact_match_final |   exact_match_policy_minus_final |   depth_compute_saved_pct |
|:----------------|:-------------|------------------:|-----------------:|------------------------------:|---------------------:|--------------------:|---------------------------------:|--------------------------:|
| clean2_balanced | 3-exit       |            0.9898 |           0.9898 |                        0      |               0.9898 |              0.9898 |                           0      |                    0      |
| clean2_balanced | 5-exit       |            0.9898 |           0.993  |                       -0.0032 |               0.9898 |              0.993  |                          -0.0032 |                   19.4992 |
| clean3_balanced | 3-exit       |            0.9808 |           0.9808 |                        0      |               0.975  |              0.975  |                           0      |                    0      |
| clean3_balanced | 5-exit       |            0.9792 |           0.9795 |                       -0.0003 |               0.9729 |              0.9697 |                           0.0032 |                   17.5796 |
| clean4_balanced | 3-exit       |            0.9789 |           0.9789 |                        0      |               0.9667 |              0.9667 |                           0      |                    0      |
| clean4_balanced | 5-exit       |            0.9696 |           0.9724 |                       -0.0028 |               0.9515 |              0.9566 |                          -0.0051 |                   17.097  |
| clean5_balanced | 3-exit       |            0.9758 |           0.9758 |                        0      |               0.9589 |              0.9589 |                           0      |                    0      |
| clean5_balanced | 5-exit       |            0.9629 |           0.9654 |                       -0.0025 |               0.9414 |              0.9451 |                          -0.0037 |                   16.9293 |

## Clean5 selected-policy per-label quality

| stage           | model_type   | label         |   precision |   recall |     f1 |   support |   predicted_positive |
|:----------------|:-------------|:--------------|------------:|---------:|-------:|----------:|---------------------:|
| clean5_balanced | 3-exit       | Les_Brown     |      1      |   0.9899 | 0.9949 |       594 |                  588 |
| clean5_balanced | 3-exit       | Mel_Robbins   |      0.9966 |   0.9865 | 0.9915 |       594 |                  588 |
| clean5_balanced | 3-exit       | Oprah_Winfrey |      0.9862 |   0.9613 | 0.9736 |       594 |                  579 |
| clean5_balanced | 3-exit       | Rabin_Sharma  |      0.9379 |   0.9916 | 0.964  |       594 |                  628 |
| clean5_balanced | 3-exit       | Simon_Sinek   |      0.9805 |   0.931  | 0.9551 |       594 |                  564 |
| clean5_balanced | 5-exit       | Les_Brown     |      0.985  |   0.9933 | 0.9891 |       594 |                  599 |
| clean5_balanced | 5-exit       | Mel_Robbins   |      0.9983 |   0.9815 | 0.9898 |       594 |                  584 |
| clean5_balanced | 5-exit       | Oprah_Winfrey |      0.991  |   0.9276 | 0.9583 |       594 |                  556 |
| clean5_balanced | 5-exit       | Rabin_Sharma  |      0.9843 |   0.9495 | 0.9666 |       594 |                  573 |
| clean5_balanced | 5-exit       | Simon_Sinek   |      0.8499 |   0.9815 | 0.9109 |       594 |                  686 |

## Research findings

1. The clean speaker benchmark scales smoothly rather than sharply: selected dynamic-policy Macro-F1 decreases from 0.9898 at 2 speakers to 0.9758 for the 3-exit model and 0.9629 for the 5-exit model at 5 speakers.
2. The 3-exit model is the stronger accuracy baseline across clean3–clean5. It keeps final/selected Macro-F1 above 0.975 up to 5 speakers, but the selected policy uses the final exit and therefore saves no depth compute under the current stable_k=2 setting.
3. The 5-exit model provides useful early-exit efficiency: selected policy saves about 16.9–19.5% depth-compute across clean2–clean5, with a modest-to-moderate Macro-F1 cost that grows as the number of speakers increases.
4. AUPRC remains very high for all stages, suggesting that the probability ranking is strong even when thresholded exact-match/F1 declines under more classes.
5. Exit instability increases with class count. Flip-any rate rises from about 0.14 at clean2 to roughly 0.73–0.81 at clean4/clean5, showing that intermediate exits change label sets more often as the task becomes harder.
6. The main per-label weakness in the 5-exit clean5 selected policy is Simon_Sinek, where precision drops to 0.8499 while recall is high at 0.9815; this suggests over-prediction/confusion toward Simon_Sinek in the dynamic 5-exit policy.

## Interpretation

### RQ1: Does performance drop smoothly or sharply as clean classes increase?

Performance drops smoothly. The selected 3-exit Macro-F1 moves from 0.9898 → 0.9808 → 0.9789 → 0.9758 from clean2 to clean5. The selected 5-exit Macro-F1 moves from 0.9898 → 0.9792 → 0.9696 → 0.9629. The 5-exit model shows a larger drop, especially after 3 classes.

### RQ2: Does the 5-exit model create an accuracy-efficiency tradeoff?

Yes. The selected 5-exit policy saves about 17–19.5% estimated depth-compute across all clean stages. However, this comes with increasing Macro-F1 loss compared with the 3-exit baseline as classes increase.

### RQ3: Is 5-exit better than 3-exit?

Not as a pure accuracy model on this clean speaker benchmark. The 3-exit model is more accurate from clean3 to clean5. The 5-exit model is useful when the research objective is dynamic efficiency, because it offers measurable compute savings while keeping Macro-F1 reasonably high.

### RQ4: What does exit stability tell us?

Exit stability gets worse as classes increase. This is expected because early exits are less confident and label-set predictions change more often. The 5-exit model shows useful later-exit stopping, but label-set flip rate remains high in clean4/clean5, so the policy should be tuned carefully before making strong efficiency claims.

## Plots

![Selected Macro-F1](plots/selected_macro_f1_by_stage.png)

![Selected exact match](plots/selected_exact_match_by_stage.png)

![Compute saved](plots/selected_compute_saved_by_stage.png)

![Average exit depth](plots/selected_avg_exit_depth_by_stage.png)

![Flip rate](plots/selected_flip_rate_by_stage.png)

![Final exit Macro-F1](plots/final_exit_macro_f1_by_stage.png)

![5-exit policy sweep](plots/five_exit_policy_sweep_tradeoff.png)

![Clean5 per-label F1](plots/clean5_per_label_f1.png)

![Clean5 static per-exit Macro-F1](plots/clean5_static_per_exit_macro_f1.png)

## Recommended next steps

1. Keep the 3-exit model as the clean-stage accuracy baseline.
2. Report the 5-exit model as the efficiency-focused dynamic model.
3. Add a clip-level evaluation script before claiming window-level efficiency.
4. For clean5, inspect confusion involving `Simon_Sinek`, especially in the 5-exit dynamic policy.
5. Test tuned thresholds (`tuned_per_exit`) after the fixed-0.5 baseline, because high AUPRC suggests threshold tuning may recover some F1/exact-match loss.
