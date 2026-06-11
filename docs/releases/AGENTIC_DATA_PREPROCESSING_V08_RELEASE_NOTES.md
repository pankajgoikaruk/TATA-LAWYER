# Release Notes — agentic_data_preprocessing_v0.8

## Summary

`agentic_data_preprocessing_v0.8` introduces the v0.8-human-corrected-balanced experiment. It consolidates the v0.6 trusted base, reviewed v0.8 LAWYER-new samples, corrected non-target labels, corrected holdout labels, known non-target identity repair, and controlled balancing.

## New/updated documentation

- Root `README.md` updated from v0.7 to v0.8.
- `DOC_STRUCTURE.md` updated to describe the v0.8 documentation layout.
- `docs/MULTILABEL_EXPERIMENT_LOG.md` expanded with chronological v0.8 experiment notes.
- `docs/APPENDIX.md` expanded with methods, settings, command log summary, and thesis-ready findings.
- `docs/reports/V08_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md` added.
- `docs/results/V08_RESULTS_SUMMARY.md` added.
- `docs/COMMANDS_V08.md` added.
- CSV tables added under `docs/tables/`.
- Plot images added under `docs/figures/`.

## Final result

| model           |   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:----------------|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| v0.8-HCB 3-exit |      3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4694 |            1.4302 |

## Comparison

| model           |   final_exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:----------------|-------------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| v0.6 3-exit     |            3 |     0.7537 |     0.8865 |       0.8992 |        0.7497 |         0.0315 |            1.4694 |            1.3045 |
| v0.6 5-exit     |            5 |     0.746  |     0.8771 |       0.8881 |        0.7232 |         0.0338 |            1.4694 |            1.2814 |
| v0.8-HCB 3-exit |            3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4694 |            1.4302 |

## Recommendation

Use the v0.8-HCB fixed-threshold parent-mean result as the current official ASHADIP/TATA result. Treat tuned thresholds as an ablation because they did not improve corrected holdout reliability.
