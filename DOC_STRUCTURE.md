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
docs/COMMANDS_V010.md
docs/COMMANDS_V010_01.md
```

| File | Purpose |
|---|---|
| `README.md` | Current project status, headline results, and decisions. |
| `DOC_STRUCTURE.md` | Canonical documentation, table, figure, and report locations. |
| `APPENDIX.md` | Thesis-style methodology and experimental interpretation. |
| `MULTILABEL_EXPERIMENT_LOG.md` | Chronological record of experiments and decisions. |
| `COMMANDS_V08.md` | Reproducible v0.8 command history. |
| `COMMANDS_V09.md` | Reproducible v0.9 workspace, review, recovery, training, and ablation commands. |
| `COMMANDS_V010.md` | Reproducible v0.10 masked-training and v0.10.1 hybrid-finalisation commands. |
| `COMMANDS_V010_01.md` | Extended v0.10.1 plus downstream NeuroAccuExit hybrid weak-label v0.1–v0.6 command history. |

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


---

## 6. v0.10 human-reviewed masked TATA documentation

### Purpose

v0.10 documents the final low-energy human-review integration after the v0.9 diagnostic experiments. It should be treated as a separate, non-destructive experimental track because it introduces tri-state labels, per-label masks, and masked BCE.

### Recommended reports

```text
docs/reports/human_talk/V10_HUMAN_REVIEWED_MASKED_LOW_ENERGY_REPORT.md
```

### Recommended results summary

```text
docs/results/human_talk/V10_RESULTS_SUMMARY.md
```

### Recommended tables

```text
docs/tables/agentic_data_preprocessing_v0.10/
```

Suggested files:

```text
v10_masked_manifest_summary.csv
v10_masked_reviewed_label_summary.csv
v10_masked_review_match_audit.csv
v10_original_vs_masked_strict_metrics.csv
v10_strict_per_label_original_vs_masked.csv
v10_test_strict_all_recovered_metrics.csv
v10_threshold_tuning_strict_validation.csv
v10_original_vs_masked_recovered_test.csv
v10_unknown_label_counts.csv
v10_silence_revision_summary.csv
```

### Recommended figures

```text
docs/figures/human_talk/agentic_data_preprocessing_v0.10/
```

Suggested files:

```text
v10_original_vs_masked_strict_metrics.png
v10_strict_per_label_f1_delta.png
v10_precision_recall_shift_by_label.png
v10_strict_vs_recovered_test_metrics.png
v10_masked_validation_curve.png
v10_unknown_label_counts_bar.png
v10_silence_revision_counts_bar.png
```

### Canonical v0.10 files

```text
# Manual tri-state review
tata_triage_model/manual_review/low_energy_recovery_v09/
low_energy_9label_manual_review_v09.csv

# Masked manifest
tata_triage_model/silence_recovered_v09/human_reviewed_masked_v09/
feature_cache/metadata/multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv

# Masked manifest reports
tata_triage_model/silence_recovered_v09/human_reviewed_masked_v09/reports/
v09_masked_manifest_summary.json
v09_masked_reviewed_label_summary.csv
v09_masked_review_match_audit.csv

# Masked training run root
tata_triage_model/silence_recovered_v09/human_reviewed_masked_v09/runs/
```

### v0.10 result role

| Result | Status | Purpose |
|---|---|---|
| v0.10 human-reviewed masked manifest | Canonical cleaned low-energy annotation artifact | Replaces parent-inherited labels for recovered windows and masks uncertain labels. |
| v0.10 masked fixed-threshold model | Diagnostic result | Scientifically cleaner but lower strict fixed-threshold performance than original v0.9. |
| v0.10 recovered-threshold model | Low-energy specialist component | Selected for the recovered low-energy branch in v0.10.1. |

### Updated preservation rules

8. Do not overwrite `multilabel_features_manifest_v09_SILENCE_RECOVERED.csv`; v0.10 must write a separate masked manifest.
9. Unknown labels must be represented with `mask_<label> = 0`, not by forcing the target to negative supervision.
10. Checkpoint selection for v0.10 must use the strict original validation subset unless explicitly running a separate ablation.
11. The strict original test subset remains the main fair comparison to original v0.9.
12. Recovered human-reviewed test rows should be reported as a separate low-energy-domain evaluation.

---
## 7. v0.10.1 final domain-aware hybrid documentation

### Purpose

v0.10.1 documents the final model-selection outcome after the v0.10 masked human-reviewed experiment, recovered-domain threshold calibration, and domain-aware hybrid evaluation.

The final system is:

```text
Normal/original audio:
    Original v0.9 model
    Fixed threshold = 0.50

Recovered low-energy audio:
    Human-reviewed masked v0.10 model
    Recovered-domain thresholds
```

Branch name:

```text
tata_lawyer_v0.10.1
```

### Recommended final report

```text
docs/results/human_talk/V10_1_DOMAIN_AWARE_HYBRID_FINAL.md
```

### Recommended final tables

```text
docs/tables/agentic_data_preprocessing_v0.10.1/
```

Suggested files:

```text
v10_1_final_policy_comparison.csv
v10_1_original_vs_masked_recovered_test.csv
v10_1_recovered_domain_thresholds.csv
v10_1_combined_test_metrics.csv
v10_1_per_label_original_vs_hybrid.csv
v10_1_final_artifact_freeze_list.csv
```

### Recommended final figures

```text
docs/figures/human_talk/agentic_data_preprocessing_v0.10.1/
```

Suggested files:

```text
v10_1_final_policy_comparison_bar.png
v10_1_recovered_test_original_vs_masked_bar.png
v10_1_hybrid_vs_original_metric_delta.png
v10_1_domain_routing_diagram.png
v10_1_per_label_f1_original_vs_hybrid.png
```

### Canonical v0.10.1 artifacts

```text
# Final branch
tata_lawyer_v0.10.1

# Normal/original-audio model
tata_triage_model/runs/
tata_v09_human_corrected_3exit_*

# Low-energy specialist model
tata_triage_model/silence_recovered_v09/human_reviewed_masked_v09/runs/
tata_v09_human_reviewed_masked_3exit_20260616_114500

# Masked manifest
tata_triage_model/silence_recovered_v09/human_reviewed_masked_v09/
feature_cache/metadata/multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv

# Threshold profiles
tata_triage_model/silence_recovered_v09/human_reviewed_masked_v09/runs/
tata_v09_human_reviewed_masked_3exit_20260616_114500/threshold_profiles/
```

### v0.10.1 result role

| Result | Status | Purpose |
|---|---|---|
| Original v0.9 fixed-threshold model | Final normal-audio branch | Best tested normal/original-audio policy. |
| Masked v0.10 recovered-threshold model | Final low-energy branch | Best tested recovered low-energy specialist. |
| v0.10.1 domain-aware hybrid | Final selected TATA-LAWYER policy | Combines the strongest normal-audio and low-energy policies. |
| Strict-threshold profiles | Diagnostic/ablation | Useful for analysis but not the selected final hybrid. |
| Additional training with class weights | Deferred | Only for future ablation, not part of v0.10.1. |

### Updated preservation rules

13. Treat `tata_lawyer_v0.10.1` as the final documentation branch for the selected hybrid result.
14. Keep v0.9 original model artifacts and v0.10 masked artifacts separate; the final system routes between them.
15. Do not route using labels or test-only metadata at deployment time. Routing must be derived from raw audio energy/preprocessing metadata.
16. The low-energy branch uses recovered-domain thresholds; the normal branch uses the original v0.9 fixed 0.50 threshold.
17. Do not claim the recovered thresholds are universally optimal; report them as the best selected thresholds for the recovered low-energy validation/test protocol.
18. Future confirmation should use a new untouched low-energy holdout if a stronger publication claim is needed.
---

## 8. Downstream NeuroAccuExit hybrid weak-label documentation

### Purpose

This section documents the downstream experiments that use the frozen TATA-LAWYER v0.10.1 domain-aware hybrid as a weak-label generator for a NeuroAccuExit-style three-exit model.

This is a separate phase from TATA-LAWYER itself:

| Phase | Role |
|---|---|
| TATA-LAWYER v0.10.1 | Final weak-label/refinement teacher policy |
| NeuroAccuExit hybrid weak-labels | Downstream student/main-model training and inference-stage aggregation study |

### Recommended reports

```text
docs/results/human_talk/NEUROACCUEXIT_HYBRID_WEAKLABELS_V06_CONFIRMED_FINDING.md
```

Optional future report:

```text
docs/results/human_talk/NEUROACCUEXIT_HYBRID_WEAKLABELS_V07_THRESHOLD_CALIBRATION.md
```

### Recommended command history

```text
docs/COMMANDS_V010_01.md
```

This file now records v0.10.1 finalisation plus the downstream v0.1–v0.6 NeuroAccuExit hybrid weak-label experiments.

### Canonical local workspace

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/
```

Important subdirectories:

```text
metadata/
reports/
runs/
human_context_checked_holdout_v03/
```

### Canonical downstream manifest

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/metadata/final_hybrid_weaklabel_manifest.csv
```

This manifest contains 13,606 rows:

| Source | Rows |
|---|---:|
| Normal/original audio from original v0.9 branch | 12,588 |
| Recovered low-energy audio from masked v0.10 branch | 1,018 |

### Downstream training runs

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/runs/
```

Canonical runs:

```text
neuroaccuexit_hybrid_weaklabels_v01_plain_bce_20260625_194859/
neuroaccuexit_hybrid_weaklabels_v02_posweight5_20260625_202246/
```

Selected downstream v0.3 model:

```text
neuroaccuexit_hybrid_weaklabels_v02_posweight5_20260625_202246/
threshold_tuning_strict_validation/per_label_thresholds.json
```

Selected v0.3 thresholds:

| Label | Threshold |
|---|---:|
| `Brene_Brown` | 0.60 |
| `Eckhart_Tolle` | 0.46 |
| `Eric_Thomas` | 0.68 |
| `Gary_Vee` | 0.95 |
| `Jay_Shetty` | 0.95 |
| `Nick_Vujicic` | 0.50 |
| `other_speaker_present` | 0.38 |
| `music_present` | 0.74 |
| `audience_reaction_present` | 0.69 |
| `silence_present` | 0.88 |

### Human context-checked holdout workspace

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/human_context_checked_holdout_v03/
```

Source CSV:

```text
human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/corrected_holdout/
01_raw_final_holdout_GROUND_TRUTH_FINAL_v08_context_checked.csv
```

Evaluation directory:

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/human_context_checked_holdout_v03/evaluation_v02_tuned/
```

Important files:

```text
holdout_v03_eval_summary.json
holdout_v03_segment_predictions.csv
holdout_v03_parent_predictions_mean.csv
holdout_v03_parent_predictions_max.csv
holdout_v03_per_label_segment.csv
holdout_v03_per_label_parent_mean.csv
holdout_v03_per_label_parent_max.csv
```

### v0.4/v0.4b label-specific aggregation diagnostic artifacts

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/human_context_checked_holdout_v03/
evaluation_v02_tuned/v04_label_specific_aggregation_diagnostic/
```

Important files:

```text
v04_aggregation_summary.csv
v04_aggregation_per_label.csv
v04_labelwise_method.json
parent_predictions_mean.csv
parent_predictions_max.csv
parent_predictions_top2mean.csv
parent_predictions_labelwise_diagnostic.csv
```

### v0.5 calibration-selected aggregation artifacts

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/human_context_checked_holdout_v03/
evaluation_v02_tuned/v05_calibration_selected_labelwise_aggregation/
```

Important files:

```text
v05_parent_cal_eval_split.csv
v05_selected_labelwise_aggregation.csv
v05_selected_labelwise_aggregation.json
v05_calibration_eval_summary.csv
v05_calibration_eval_per_label.csv
```

### v0.6 repeated stability artifacts

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/human_context_checked_holdout_v03/
evaluation_v02_tuned/v06_repeated_calibration_eval_stability/
```

Important files:

```text
v06_seed_level_eval_results.csv
v06_repeated_eval_summary.csv
v06_selected_aggregation_by_seed.csv
v06_selection_frequency.csv
```

### v0.6 confirmed finding

| Method | Macro-F1 mean | Micro-F1 mean | Samples-F1 mean | Exact Match mean | Hamming Loss mean |
|---|---:|---:|---:|---:|---:|
| mean | 0.5345 | 0.6776 | 0.6138 | 0.4471 | **0.0759** |
| max | 0.5605 | 0.6519 | 0.6709 | 0.2589 | 0.1328 |
| top2mean | 0.6108 | 0.7252 | 0.7283 | 0.4233 | 0.0879 |
| calibration-selected labelwise | **0.6377** | **0.7470** | **0.7508** | **0.4520** | 0.0785 |

### v0.6 result role

| Result | Status | Purpose |
|---|---|---|
| v0.3 downstream selected model | Selected student model | Best weak-label strict-test deployment-style model. |
| Human context-checked holdout evaluation | Generalisation diagnostic | Shows human holdout is harder than weak-label reproduction. |
| v0.4/v0.4b labelwise aggregation | Diagnostic | Shows label-specific aggregation improves without retraining. |
| v0.5 calibration-selected labelwise aggregation | Stronger diagnostic | Selects aggregation on calibration half and evaluates on held-out half. |
| v0.6 repeated stability | Confirmed finding | Shows labelwise aggregation benefit is stable across 20 calibration/evaluation splits. |
| v0.7 threshold calibration | Optional future diagnostic | Not required before freezing v0.6. |

### Updated preservation rules

19. Treat `human_talk_workspace/neuroaccuexit_hybrid_weaklabels/` as a downstream workspace, not part of the frozen TATA-LAWYER teacher.
20. Do not overwrite the v0.1/v0.2 run directories or v0.3 tuned thresholds.
21. Do not modify `holdout_v03_segment_predictions.csv`; all v0.4–v0.6 diagnostics must be derived from it.
22. Treat v0.4/v0.4b as diagnostic because the mapping was influenced by holdout inspection.
23. Treat v0.5 and v0.6 as stronger evidence because aggregation choices are selected on calibration splits and evaluated separately.
24. Freeze v0.6 as the confirmed finding before any optional v0.7 threshold-calibration diagnostic.
25. When writing the thesis/paper, distinguish weak-label policy learning from human context-checked holdout generalisation.
