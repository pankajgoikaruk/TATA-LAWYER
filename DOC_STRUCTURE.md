# Human-Talk Documentation Structure

The documentation is versioned so that v0.8 main-model findings and v0.9 TATA-triage experiments remain independently reproducible.

---

## 1. Core documentation

```text
docs/README.md
docs/DOC_STRUCTURE.md
docs/APPENDIX.md
docs/MULTILABEL_EXPERIMENT_LOG.md
docs/COMMANDS_V08.md
docs/COMMANDS_V09.md
```

| File | Purpose |
|---|---|
| `README.md` | Current project status, headline results, and decisions. |
| `DOC_STRUCTURE.md` | Canonical documentation, table, figure, and report locations. |
| `APPENDIX.md` | Thesis-style methodology and experimental interpretation. |
| `MULTILABEL_EXPERIMENT_LOG.md` | Chronological record of experiments and decisions. |
| `COMMANDS_V08.md` | Reproducible v0.8 command history. |
| `COMMANDS_V09.md` | Reproducible v0.9 workspace, review, recovery, training, and ablation commands. |

---

## 2. v0.8 main-model documentation

### Reports

```text
docs/reports/human_talk/V08_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md
```

### Results summaries

```text
docs/results/human_talk/V08_RESULTS_SUMMARY.md
```

### Tables

```text
docs/tables/agentic_data_preprocessing_v0.8/
```

Recommended files:

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
v08_hcb_parent_aggregation_strategy_comparison.csv
v08_hcb_label_aware_fair_comparison_corrected_holdout.csv
v08_hcb_weak_label_f1_by_aggregation.csv
v08_hcb_per_label_mean_max_labelaware_exit3.csv
v08_hcb_label_aware_commands.csv
```

### Figures

```text
docs/figures/human_talk/agentic_data_preprocessing_v0.8/
```

Recommended files:

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

### Final v0.8 reporting policy

| Result type | Method | Use |
|---|---|---|
| Main official v0.8-HCB result | Parent mean, fixed 0.5, Exit 3 | Overall corrected-holdout headline result. |
| Label-aware research finding | Mean for eight stable labels and max for audience/silence | Macro-F1 and transient-label analysis. |
| Global max | Max for all labels | Diagnostic only because it over-predicts labels. |

---

## 3. v0.9 TATA-triage documentation

### Reports

Recommended location:

```text
docs/reports/human_talk/V09_TATA_TRIAGE_LOW_ENERGY_RECOVERY_REPORT.md
```

### Results summary

Recommended location:

```text
docs/results/human_talk/V09_RESULTS_SUMMARY.md
```

### Tables

```text
docs/tables/agentic_data_preprocessing_v0.9/
```

Recommended files:

```text
v09_seed_lineage_summary.csv
v09_initial_manifest_split_counts.csv
v09_low_energy_audit_summary.csv
v09_low_energy_review_counts.csv
v09_recovery_manifest_counts.csv
v09_original_vs_full_vs_positive_only_metrics.csv
v09_per_label_original_vs_recovery.csv
v09_exit_comparison.csv
v09_manual_review_tristate_schema.csv
v09_experiment_commands_index.csv
```

### Figures

```text
docs/figures/human_talk/agentic_data_preprocessing_v0.9/
```

Recommended files:

```text
v09_original_vs_recovery_overall_metrics.png
v09_original_vs_recovery_per_label_f1.png
v09_silence_precision_recall_f1.png
v09_exit_macro_f1_comparison.png
v09_low_energy_audit_priority_counts.png
v09_review_silence_positive_negative_counts.png
v09_split_size_comparison.png
v09_macro_exact_hamming_tradeoff.png
```

### v0.9 workspace layout

```text
human_talk_workspace/
└── tata_v0.9_pipeline/
    ├── tata_triage_model/
    │   ├── metadata/
    │   ├── feature_cache/
    │   ├── runs/
    │   ├── manual_review/
    │   │   └── low_energy_recovery_v09/
    │   └── silence_recovered_v09/
    │       ├── metadata/
    │       ├── feature_cache/
    │       ├── segment_wavs/
    │       ├── reports/
    │       └── runs/
    ├── neuroaccuexit_main_model/
    └── shared/
        ├── human_talk_10label_schema.json
        ├── corrected_holdout/
        └── correction_ledgers/
```

### Canonical v0.9 manifests

```text
# Immutable baseline
tata_triage_model/metadata/tata2_parent_manifest_12label_2074_BASELINE.csv

# Audited parent manifest
tata_triage_model/metadata/tata_seed_parent_manifest_v09_FINAL_REVIEWED.csv

# Initial v0.9 feature manifest
tata_triage_model/feature_cache/metadata/multilabel_features_manifest_v09_FINAL.csv

# Full low-energy recovery manifest
tata_triage_model/silence_recovered_v09/feature_cache/metadata/
multilabel_features_manifest_v09_SILENCE_RECOVERED.csv

# Positive-only ablation manifest
tata_triage_model/silence_recovered_v09/feature_cache/metadata/
multilabel_features_manifest_v09_ONLY_RECOVERED_SILENCE_POSITIVE.csv

# Nine-label tri-state manual review
tata_triage_model/manual_review/low_energy_recovery_v09/
low_energy_9label_manual_review_v09.csv
```

---

## 4. v0.9 result roles

| Result | Status | Purpose |
|---|---|---|
| Original v0.9 | General ten-label baseline | Cleanest current segment-level TATA comparison. |
| Full recovery | Diagnostic | Strongest silence result; affected by inherited-label uncertainty. |
| Positive-only recovery | Ablation | Tests removal of 747 low-energy non-silence rows. |
| Future tri-state/masked model | Planned final v0.9 | Uses known labels and masks unknown labels. |

---

## 5. Naming and preservation rules

1. Never edit files ending in `_BASELINE.csv`.
2. Do not overwrite the original v0.9 manifest when running a recovery experiment.
3. Store each ablation under a distinct manifest name and run variant.
4. Preserve train/validation/test split assignment at parent level.
5. Keep parent-level main-model results separate from one-second TATA results.
6. Use `-1` only for manually reviewed but uncertain labels; blank means not reviewed.
7. Store all exact v0.9 PowerShell commands in `docs/COMMANDS_V09.md`.
