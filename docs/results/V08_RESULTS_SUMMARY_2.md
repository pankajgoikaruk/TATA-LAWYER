# v0.8 Results Summary — Human-Corrected-Balanced + Label-Aware Aggregation

## Official result

| method               |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:---------------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| parent_mean_official |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4694 |            1.4302 |

## Label-aware result

| method               |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:---------------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| label_aware_mean_max |     0.8320 |     0.9285 |       0.9375 |        0.8235 |         0.0211 |            1.4694 |            1.4844 |

## Aggregation comparison

| method                |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_pred_labels | official_use                                 |
|:----------------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|:---------------------------------------------|
| parent_mean_official  |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4302 | headline overall result                      |
| global_max_diagnostic |     0.7251 |     0.8203 |       0.8423 |        0.5121 |         0.0630 |            2.0346 | diagnostic only; not final                   |
| label_aware_mean_max  |     0.8320 |     0.9285 |       0.9375 |        0.8235 |         0.0211 |            1.4844 | post-hoc macro-F1 / transient-label analysis |

## Fair v0.6/v0.8 corrected-holdout comparison

| model                       |   exit | aggregation          |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_pred_labels |
|:----------------------------|-------:|:---------------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|
| v0.6 3-exit                 |      3 | mean                 |     0.7537 |     0.8865 |       0.8992 |        0.7497 |         0.0315 |            1.3045 |
| v0.6 5-exit                 |      5 | mean                 |     0.7460 |     0.8771 |       0.8881 |        0.7232 |         0.0338 |            1.2814 |
| v0.8-HCB 3-exit official    |      3 | mean                 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4302 |
| v0.8-HCB 3-exit label-aware |      3 | mean/max label-aware |     0.8320 |     0.9285 |       0.9375 |        0.8235 |         0.0211 |            1.4844 |

## Weak label comparison

| label                     | aggregation   |   precision |   recall |     f1 |   support |   predicted_positive |
|:--------------------------|:--------------|------------:|---------:|-------:|----------:|---------------------:|
| audience_reaction_present | mean          |      0.6667 |   0.0690 | 0.1250 |        29 |                    3 |
| audience_reaction_present | global_max    |      0.4103 |   0.5517 | 0.4706 |        29 |                   39 |
| audience_reaction_present | label_aware   |      0.4103 |   0.5517 | 0.4706 |        29 |                   39 |
| silence_present           | mean          |      0.0000 |   0.0000 | 0.0000 |        12 |                    0 |
| silence_present           | global_max    |      0.1818 |   0.1667 | 0.1739 |        12 |                   11 |
| silence_present           | label_aware   |      0.1818 |   0.1667 | 0.1739 |        12 |                   11 |

## Per-label F1 comparison

| label                     |   mean_f1 |   global_max_f1 |   label_aware_f1 |   label_aware_support |   label_aware_predicted_positive |
|:--------------------------|----------:|----------------:|-----------------:|----------------------:|---------------------------------:|
| Brene_Brown               |    0.9645 |          0.8834 |           0.9645 |                    73 |                               68 |
| Eckhart_Tolle             |    0.9818 |          0.9941 |           0.9818 |                    84 |                               81 |
| Eric_Thomas               |    0.9286 |          0.6175 |           0.9286 |                    68 |                               72 |
| Gary_Vee                  |    0.9774 |          0.9007 |           0.9774 |                    68 |                               65 |
| Jay_Shetty                |    0.9626 |          0.8333 |           0.9626 |                    90 |                               97 |
| Nick_Vujicic              |    0.9792 |          0.6853 |           0.9792 |                    49 |                               47 |
| other_speaker_present     |    0.9293 |          0.8422 |           0.9293 |                   460 |                              474 |
| music_present             |    0.9525 |          0.8496 |           0.9525 |                   341 |                              333 |
| audience_reaction_present |    0.1250 |          0.4706 |           0.4706 |                    29 |                               39 |
| silence_present           |    0.0000 |          0.1739 |           0.1739 |                    12 |                               11 |

## Figures

| Figure | Path |
|---|---|
| Aggregation strategy line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_aggregation_strategy_lineplot.png` |
| Hamming loss line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_aggregation_hamming_loss_lineplot.png` |
| Weak label F1 line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_weak_label_f1_lineplot.png` |
| v0.6 vs v0.8 line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_vs_v06_label_aware_lineplot.png` |
| Per-label mean vs label-aware | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_per_label_mean_vs_labelaware_bar.png` |
