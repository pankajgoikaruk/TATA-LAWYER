---

## v0.8 human-talk documentation layout update

The v0.8 documentation artifacts are organised under human-talk and version-specific folders so that v0.7 and v0.8 results do not overwrite each other.

### Reports

```text
docs/reports/human_talk/V08_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md
```

Purpose: thesis-ready narrative report for the v0.8-human-corrected-balanced experiment, including methodology, training configuration, corrected-holdout results, fair v0.6 comparison, and label-aware aggregation analysis.

### Results summaries

```text
docs/results/human_talk/V08_RESULTS_SUMMARY.md
```

Purpose: compact results summary for GitHub readers, including headline metrics, corrected-holdout comparison, and final reporting decision.

### Tables

```text
docs/tables/agentic_data_preprocessing_v0.8/
```

Recommended v0.8 table files:

```text
v08_fair_comparison_corrected_holdout_parent_mean_fixed.csv
v08_corrected_holdout_parent_mean_fixed_by_exit.csv
v08_corrected_holdout_parent_mean_tuned_by_exit.csv
v08_corrected_holdout_parent_mean_fixed_per_label_exit3.csv
v08_internal_test_by_exit.csv
v08_internal_test_per_label_exit3.csv
v08_label_counts_before_after_balance.csv
v08_threshold_tuning_internal_val_test.csv
v08_final_exit_tuned_thresholds.csv
v08_experiment_commands_index.csv
v08_hcb_parent_aggregation_strategy_comparison.csv
v08_hcb_label_aware_fair_comparison_corrected_holdout.csv
v08_hcb_weak_label_f1_by_aggregation.csv
v08_hcb_per_label_mean_max_labelaware_exit3.csv
v08_hcb_label_aware_commands.csv
```

v0.7-related tables should stay separate:

```text
docs/tables/agentic_data_preprocessing_v0.7/
```

### Figures

All v0.8 human-talk figures should be under:

```text
docs/figures/human_talk/agentic_data_preprocessing_v0.8/
```

Recommended v0.8 figure files:

```text
v08_training_validation_curve.png
v08_training_loss_hamming_curve.png
v08_label_counts_before_after_balance.png
v08_internal_test_by_exit_lineplot.png
v08_corrected_holdout_fixed_by_exit_lineplot.png
v08_vs_v06_corrected_holdout_bar.png
v08_vs_v06_hamming_loss_bar.png
v08_corrected_holdout_per_label_f1_bar.png
v08_avg_true_vs_pred_labels_bar.png
v08_hcb_aggregation_strategy_lineplot.png
v08_hcb_aggregation_hamming_loss_lineplot.png
v08_hcb_weak_label_f1_lineplot.png
v08_hcb_vs_v06_label_aware_lineplot.png
v08_hcb_macro_hamming_tradeoff_bar.png
v08_hcb_per_label_mean_vs_labelaware_bar.png
```

### Command and methodology docs

```text
docs/COMMANDS_V08.md
docs/APPENDIX.md
docs/MULTILABEL_EXPERIMENT_LOG.md
```

Purpose:

| File | Purpose |
|---|---|
| `docs/COMMANDS_V08.md` | Full reproducible PowerShell command history, including delta review, manifest build, training, corrected holdout evaluation, v0.6 re-evaluation, global max diagnostic, and label-aware aggregation. |
| `docs/APPENDIX.md` | Thesis-style methodology appendix. |
| `docs/MULTILABEL_EXPERIMENT_LOG.md` | Chronological experiment log and decisions. |

### Final reporting policy

| Result type | Method | Use |
|---|---|---|
| Main official v0.8-HCB result | Parent mean aggregation, fixed threshold 0.5, Exit 3 | Overall corrected-holdout headline result. |
| Label-aware research finding | Mean for 8 stable labels, max for `audience_reaction_present` and `silence_present` | Macro-F1 and transient-label analysis. |
| Global max diagnostic | Max for all labels | Diagnostic only; not final because it over-predicts labels. |
