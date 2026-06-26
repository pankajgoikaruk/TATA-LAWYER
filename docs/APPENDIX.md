# Methodology Appendix

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

---

## V. v0.9 TATA-triage workspace and preservation policy

The v0.9 experiment was created as a separate workspace so that the original TATA-2/v0.6 assets and the v0.8 main-model assets remained unchanged.

```text
human_talk_workspace/
└── tata_v0.9_pipeline/
    ├── tata_triage_model/
    ├── neuroaccuexit_main_model/
    └── shared/
```

The baseline parent and feature manifests were copied with `_BASELINE` names and treated as immutable:

```text
tata2_parent_manifest_12label_2074_BASELINE.csv
tata_seed_features_manifest_10label_12469_BASELINE.csv
human_talk_10label_schema.json
```

The TATA triage manifest remained separate from the downstream main-model manifest. This separation prevents pseudo-label generation, triage experiments, and low-energy ablations from silently modifying the main-model training data.

## W. Audited v0.9 seed reconstruction

The original reviewed seed contained 2,074 parent clips and 12,469 one-second feature rows. Existing silence and rare-event labels were reviewed, and 27 additional verified silence clips were renamed without colliding with existing filenames.

The final audited parent count was:

```text
2,074 original parents
+   27 new verified silence parents
=2,101 final parents
```

The initial v0.9 feature cache reused the 12,469 legacy features and extracted 120 additional segments from the 27 new silence parents:

```text
12,469 reused feature rows
+  120 new silence feature rows
=12,589 v0.9 feature rows
```

The split remained parent-safe:

| Split | Rows |
|---|---:|
| Train | 8,745 |
| Validation | 1,883 |
| Test | 1,961 |

The v0.9 builder also repaired missing metadata, recomputed `labels` and `num_active_labels`, cleared stale fine-audience metadata, normalised paths, and retained one-second segments with a 0.5-second hop for the new silence data.

## X. Original v0.9 TATA result

The original v0.9 3-exit model used the same architecture and training settings as the previous TATA comparison:

```text
tap blocks = 1,3
loss weights = 0.3,0.3,1.0
epochs = 40
batch size = 64
learning rate = 0.001
threshold = 0.5
seed = 42
device = CPU
```

Final-exit internal test performance was:

| Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---:|---:|---:|---:|---:|
| 0.8195 | 0.8226 | 0.8226 | 0.6527 | 0.0483 |

This was approximately level with the earlier v0.6 internal TATA result while using a cleaner and more auditable manifest.

## Y. Low-energy filtering diagnosis

The legacy preprocessing pipeline generated features only for windows that survived the energy filter. Therefore, missing low-energy windows could not be recovered from the 12,469-row feature manifest alone; they had to be rediscovered from the original parent audio.

A non-destructive audit used:

```text
window length = 1.0 second
window hop = 0.5 second
raw RMS measured before peak normalisation
```

Audit result:

| Item | Count |
|---|---:|
| Original parents scanned | 2,074 |
| Parent audio resolved | 2,074 |
| Grid windows scanned | 22,391 |
| Already represented windows | 12,164 |
| Missing low-energy windows before parent cap | 1,178 |
| Review candidates retained | 1,018 |
| High priority | 250 |
| Medium priority | 90 |
| Low priority | 678 |

Manual silence review produced:

| Human silence decision | Count |
|---|---:|
| `silence_present = 1` | 271 |
| `silence_present = 0` | 747 |

The 747 negative examples are scientifically important because they demonstrate that low energy is not equivalent to silence.

## Z. Recovered timeline and feature cache

All 1,018 reviewed candidates were incorporated into the recovered timeline. One candidate already matched an existing feature-manifest location and was updated in place; 1,017 missing candidates received new features.

| Recovery item | Count |
|---|---:|
| Existing feature rows | 12,589 |
| Existing reviewed rows updated in place | 1 |
| Reviewed rows appended | 1,017 |
| Full recovered rows | 13,606 |
| Affected parents | 522 |
| Parent silence labels changed 0 to 1 | 0 |

The parent-level consecutive-silence rule was:

```text
at least two reviewed silent one-second windows
with start times separated by one 0.5-second hop
```

With one-second windows and a 0.5-second hop, two consecutive positives represent approximately 1.5 seconds of continuous silence evidence.

The lack of parent-label changes does not invalidate the recovery. It means the newly recovered silence windows occurred inside parents already labelled silence-positive, while the major benefit was improved segment-level evidence.

## AA. Low-energy recovery experiments

### AA.1 Full recovery

The full recovery experiment included all 271 silence-positive and 747 silence-negative reviewed candidates.

| Split | Rows |
|---|---:|
| Train | 9,445 |
| Validation | 2,042 |
| Test | 2,119 |

| Metric | Score |
|---|---:|
| Macro-F1 | 0.8064 |
| Micro-F1 | 0.8064 |
| Samples-F1 | 0.7968 |
| Exact Match | 0.6022 |
| Hamming Loss | 0.0539 |
| Silence F1 | **0.7875** |

### AA.2 Silence-positive-only ablation

The positive-only ablation removed all 747 reviewed non-silence candidates from train, validation, and test while retaining the 271 confirmed silence candidates.

| Split | Rows |
|---|---:|
| Train | 8,938 |
| Validation | 1,926 |
| Test | 1,995 |

| Metric | Score |
|---|---:|
| Macro-F1 | **0.8199** |
| Micro-F1 | 0.8120 |
| Samples-F1 | 0.8075 |
| Exact Match | 0.6110 |
| Hamming Loss | 0.0537 |
| Silence F1 | 0.7355 |

### AA.3 Comparison with original v0.9

| Experiment | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Silence F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original v0.9 | 0.8195 | **0.8226** | **0.8226** | **0.6527** | **0.0483** | 0.6667 |
| Full recovery | 0.8064 | 0.8064 | 0.7968 | 0.6022 | 0.0539 | **0.7875** |
| Silence-positive only | **0.8199** | 0.8120 | 0.8075 | 0.6110 | 0.0537 | 0.7355 |

The full recovery model produced the strongest silence classification, while the positive-only ablation restored most of the global Macro-F1 lost under full recovery.

## AB. Interpretation of inherited-label uncertainty

Only `silence_present` was manually verified for the recovered one-second clips. The other nine labels were initially inherited from the parent clip. Parent labels may describe another temporal region and are not guaranteed to be true for the recovered second.

For example, a five-second parent may contain Gary Vee in seconds 0–2, silence in seconds 2–4, and music in seconds 4–5. A recovered silent window should not automatically inherit both `Gary_Vee = 1` and `music_present = 1`.

This explains the experimental pattern:

- silence F1 improved strongly;
- several speaker and event labels declined under full recovery;
- removing the 747 non-silence rows restored global performance but reduced silence performance.

The correct conclusion is not to discard low-energy data. The correct conclusion is to avoid forcing uncertain inherited labels onto segment-level examples.

## AC. Tri-state nine-label manual review

The recovered clips now use the following annotation protocol for the remaining nine labels:

```text
 1 = confidently present
 0 = confidently absent
-1 = reviewed but uncertain or unknown
blank = not reviewed yet
```

`review_silence_present` remains the trusted human decision and must not be replaced with `-1`.

Overlapping labels are allowed. A segment may contain a target speaker, another speaker, music, and audience reaction simultaneously.

When the whole one-second clip is unclear, all nine new review fields may be set to `-1`, while the existing silence decision remains unchanged.

## AD. Planned masked supervision

For final training, unknown labels should not be treated as negatives. A label mask should be generated as:

```text
label is 0 or 1 -> mask = 1
label is -1     -> mask = 0
```

The loss should be calculated only over known labels:

```text
masked BCE = sum(BCE × mask) / sum(mask)
```

This preserves all 1,018 reviewed low-energy examples, allows the 747 hard negatives to supervise `silence_present = 0`, and prevents uncertain labels from contaminating the remaining classes.

## AE. Evaluation policy after partial annotation

For rigorous reporting:

1. Evaluate the original and recovered models on the same original 1,961-row test set.
2. Evaluate recovered low-energy clips separately for `silence_present`.
3. Use only fully known labels for per-label evaluation.
4. Do not report ordinary Exact Match for rows containing any `-1` labels.
5. Use masked per-label metrics for partially labelled recovered rows.
6. Use the original trusted validation rows for full-label checkpoint selection until recovered validation labels are fully reviewed.

The v0.8 parent-level main-model metrics and v0.9 segment-level TATA metrics must remain separate in all tables and thesis claims.


---

## AA. v0.10 human-reviewed masked low-energy experiment

### AA.1 Rationale

The v0.9 low-energy recovery experiments showed that the legacy RMS filtering stage removed meaningful low-energy windows before feature extraction. Recovering these windows improved silence recognition, but parent-inherited labels introduced noisy supervision at one-second resolution. The v0.10 experiment therefore replaces inherited labels for recovered windows with human-reviewed segment-level labels and masks uncertain annotations.

### AA.2 Tri-state annotation

Recovered low-energy clips were reviewed using:

```text
1  = confidently present
0  = confidently absent
-1 = reviewed but uncertain / unknown
```

The silence label was handled separately because it had already been reviewed and later corrected after second-pass listening.

Final annotation counts:

```text
Reviewed low-energy rows:      1,018
Fully known rows:                966
Partially known rows:             52
Silence positives:               277
Silence negatives:               741
```

### AA.3 Masked loss

For each label, v0.10 converts review values into a binary target and binary supervision mask:

```text
review = 1  -> target = 1, mask = 1
review = 0  -> target = 0, mask = 1
review = -1 -> target = 0, mask = 0
```

The zero target for unknown labels is a placeholder only. It contributes nothing to the loss because its mask is zero.

For each exit head:

```python
loss_k = sum(BCEWithLogits(logits_k, y) * mask) / max(sum(mask), 1)
```

The total three-exit loss is:

```python
loss = 0.3 * loss_exit1 + 0.3 * loss_exit2 + 1.0 * loss_exit3
```

### AA.4 Strict and masked evaluation

v0.10 separates evaluation into three subsets:

```text
test_strict:
  original 1,961 test rows only

test_all_masked:
  original + recovered test rows with unknown labels masked

test_recovered_masked:
  recovered manually reviewed low-energy test rows only
```

This design protects direct comparability with original v0.9 while still allowing low-energy-domain analysis.

### AA.5 Result

The v0.10 fixed-threshold model was selected by strict validation Macro-F1:

```text
Best epoch: 38
Best strict validation Macro-F1: 0.7771
```

Strict original-test result:

| Metric | Original v0.9 | v0.10 masked | Delta |
|---|---:|---:|---:|
| Macro-F1 | 0.8195 | 0.7950 | -0.0245 |
| Micro-F1 | 0.8226 | 0.7952 | -0.0274 |
| Samples-F1 | 0.8226 | 0.7655 | -0.0571 |
| Exact Match | 0.6527 | 0.5926 | -0.0601 |
| Hamming Loss | 0.0483 | 0.0552 | +0.0069 worse |

The result indicates that cleaner partial-label supervision did not automatically improve the fixed-threshold classifier.

### AA.6 Label-specific behaviour

The v0.10 model improved:

```text
music_present
other_speaker_present
Nick_Vujicic
Eckhart_Tolle
```

It decreased substantially on:

```text
audience_reaction_present
Gary_Vee
Eric_Thomas
```

The dropped labels showed high precision but low recall. This indicates that the corrected negative labels made the model more conservative. The problem is likely not data corruption; it is a threshold/calibration issue introduced by cleaner but more imbalanced supervision.

### AA.7 Final v0.10.1 research interpretation

The threshold-calibration and model-selection experiments resolved the earlier open question. The masked v0.10 model should not replace the original v0.9 model globally, but it is the better specialist for the recovered low-energy domain.

The final selected system is therefore:

```text
TATA-LAWYER v0.10.1 Domain-Aware Hybrid

Normal/original audio:
    Original v0.9 model
    Fixed threshold = 0.50

Recovered low-energy audio:
    Human-reviewed masked v0.10 model
    Recovered-domain thresholds
```

This is a model-selection contribution rather than another training run. It shows that a single global classifier was not the best solution after low-energy recovery; instead, the corrected low-energy subset benefits from a dedicated specialist and domain-specific thresholds.

### AA.8 Recovered-domain model-selection result

On the same 158 recovered human-reviewed test clips, the masked model outperformed the original v0.9 model under both fixed and recovered-domain threshold settings.

| Model on recovered audio | Threshold source | Macro-F1 | Micro-F1 | Samples-F1 | Fully-known Exact | Hamming Loss ↓ |
|---|---|---:|---:|---:|---:|---:|
| Original v0.9 | Fixed 0.50 | 0.4756 | 0.7262 | 0.6646 | 0.5586 | 0.0569 |
| Masked v0.10 | Fixed 0.50 | 0.5095 | 0.8065 | 0.7236 | 0.6828 | **0.0384** |
| Original v0.9 | Recovered-domain thresholds | 0.5126 | 0.7143 | 0.6424 | 0.5448 | 0.0589 |
| **Masked v0.10** | **Recovered-domain thresholds** | **0.5438** | **0.8075** | **0.7532** | **0.6966** | 0.0397 |

The fixed-threshold comparison is especially important because it shows that the masked model learned a better representation for low-energy clips before threshold tuning was applied.

### AA.9 Final combined hybrid result

The final combined 2,119-row test evaluation compared several possible deployment policies.

| Policy | Normal/original audio | Recovered low-energy audio | Macro-F1 | Micro-F1 | Samples-F1 | Known-label Exact | Fully-known Exact | Hamming Loss ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Original fixed everywhere | Original v0.9, threshold 0.50 | Original v0.9, threshold 0.50 | 0.8155 | 0.8171 | 0.8108 | 0.6465 | 0.6462 | 0.0489 |
| Masked fixed everywhere | Masked v0.10, threshold 0.50 | Masked v0.10, threshold 0.50 | 0.7991 | 0.7958 | 0.7624 | 0.5998 | 0.5988 | 0.0539 |
| Original domain-aware | Original v0.9, normal threshold | Original v0.9, recovered thresholds | 0.8162 | 0.8165 | 0.8092 | 0.6446 | 0.6453 | 0.0491 |
| Masked domain-aware | Masked v0.10, strict thresholds | Masked v0.10, recovered thresholds | 0.8173 | 0.8081 | 0.7813 | 0.6031 | 0.6030 | 0.0526 |
| **Hybrid recommended** | **Original v0.9, threshold 0.50** | **Masked v0.10, recovered thresholds** | **0.8224** | **0.8218** | **0.8174** | **0.6555** | **0.6557** | **0.0477** |

The hybrid was the only tested policy to improve every aggregate metric over the original fixed-threshold system.

### AA.10 Final recovered-domain thresholds

```text
Brene_Brown                0.22
Eckhart_Tolle              0.50
Eric_Thomas                0.44
Gary_Vee                   0.50
Jay_Shetty                 0.57
Nick_Vujicic               0.50
other_speaker_present      0.45
music_present              0.57
audience_reaction_present  0.18
silence_present            0.46
```

These thresholds should be reported as recovered-domain thresholds, not global thresholds. Several recovered-validation labels had very low positive support, so the threshold profile is useful for the selected low-energy routing policy but should not be claimed as universally optimal.

### AA.11 Thesis-ready conclusion

The TATA-LAWYER v0.10.1 experiments show that low-energy recovery, human-reviewed partial labels, masked BCE, and domain-aware routing together provide the strongest tested TATA triage system. The original preprocessing pipeline removed low-energy windows before feature extraction, which censored potentially useful silence and quiet-event evidence. Reintroducing those windows without review improved silence detection but created supervision noise by inheriting parent-level labels onto one-second clips.

Manual tri-state review and masked BCE corrected this problem by using known labels while ignoring uncertain labels in the loss. The resulting masked model did not become the best general normal-audio model, but it became the best recovered low-energy specialist. Combining the original v0.9 model for normal audio with the masked v0.10 model for recovered low-energy audio produced the final v0.10.1 domain-aware hybrid, improving Macro-F1, Micro-F1, Samples-F1, exact match, and Hamming Loss over the original fixed-threshold baseline.

### AA.12 Final research questions and answers

| Research question | Final answer |
|---|---|
| Can low-energy segments be ignored safely? | No. They contain useful silence and hard-negative evidence. |
| Can low-energy segments simply inherit parent labels? | No. Parent labels are unreliable at the one-second recovered-window level. |
| Is masked BCE appropriate for partial human review? | Yes. It prevents uncertain labels from becoming negative supervision. |
| Does the masked model replace original v0.9 everywhere? | No. It is a low-energy specialist, not the best universal model. |
| What is the final selected TATA-LAWYER policy? | A v0.10.1 domain-aware hybrid: original v0.9 for normal audio and masked v0.10 for recovered low-energy audio. |
---

## AB. NeuroAccuExit hybrid weak-label downstream experiments

After the TATA-LAWYER v0.10.1 domain-aware hybrid was frozen as the selected weak-label generation policy, a downstream NeuroAccuExit experiment track was started. The goal was no longer to improve the TATA-LAWYER teacher itself, but to test whether a downstream three-exit NeuroAccuExit model could learn from the final hybrid weak-label manifest and then generalise to a human context-checked final holdout.

The hybrid weak-label manifest contained:

| Source branch | Rows | Label source |
|---|---:|---|
| Normal/original audio | 12,588 | Original v0.9 TATA model with fixed 0.50 threshold |
| Recovered low-energy audio | 1,018 | Human-reviewed masked v0.10 model with recovered-domain thresholds |
| **Total** | **13,606** | **TATA-LAWYER v0.10.1 hybrid policy** |

The downstream training split was:

| Split | Normal/original rows | Recovered low-energy rows | Total |
|---|---:|---:|---:|
| Train | 8,744 | 701 | 9,445 |
| Validation | 1,883 | 159 | 2,042 |
| Test | 1,961 | 158 | 2,119 |

This stage should be interpreted as **weak-label policy learning / distillation**, because the main training target is generated by the final TATA-LAWYER hybrid system rather than by fully independent human annotation.

### AB.1 v0.1 and v0.2 downstream training

Two downstream training runs were compared using the same architecture, splits, features and seed.

| Run | Loss setting | Threshold during fixed evaluation | Purpose |
|---|---|---:|---|
| v0.1 | Plain masked BCE | 0.50 | Clean downstream baseline |
| v0.2 | Masked BCE with capped positive weighting, `pos_weight_max = 5.0` | 0.50 | Improve rare-label recall |

On the strict original test subset, the fixed-threshold results were:

| Run | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| v0.1 fixed | **0.8912** | **0.8946** | 0.8537 | **0.7766** | **0.0260** |
| v0.2 fixed | 0.8800 | 0.8793 | **0.8691** | 0.7374 | 0.0328 |

The positive-weighted model improved recall but over-predicted several labels. Therefore, per-label threshold calibration was required.

### AB.2 v0.3 threshold-calibrated downstream model

Both v0.1 and v0.2 were threshold tuned using validation data only. The strict test comparison was:

| Candidate | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| v0.1 fixed | 0.8912 | 0.8946 | 0.8537 | 0.7766 | 0.0260 |
| v0.1 tuned | **0.9058** | 0.9020 | 0.8711 | 0.7879 | 0.0250 |
| v0.2 fixed | 0.8800 | 0.8793 | 0.8691 | 0.7374 | 0.0328 |
| v0.2 tuned | 0.9040 | **0.9073** | **0.8814** | **0.7930** | **0.0239** |

The selected downstream v0.3 model was:

```text
NeuroAccuExit Hybrid WeakLabels v0.3
= v0.2 pos_weight5 model
+ per-label tuned thresholds
```

The selected thresholds were:

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

The v0.3 result shows that the downstream model can learn the hybrid weak-label policy well. However, this is not yet a final human-ground-truth claim.

### AB.3 Human context-checked holdout evaluation

The selected v0.3 model was then evaluated on a separate human context-checked final holdout from the v0.8 corrected-holdout pipeline. This holdout produced 4,335 one-second segments from 867 parent clips.

The initial holdout evaluation using the v0.2 tuned model produced:

| Evaluation level | Rows | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|---:|
| Segment-level | 4,335 | 0.5522 | 0.6568 | 0.6082 | 0.4076 | 0.0917 |
| Parent mean | 867 | 0.5379 | **0.6816** | 0.6191 | **0.4498** | **0.0752** |
| Parent max | 867 | **0.5715** | 0.6553 | **0.6742** | 0.2641 | 0.1309 |

This result shows moderate generalisation to human context-checked labels. Parent mean was the safest global strategy, while parent max recovered more transient evidence but created many false positives.

### AB.4 v0.4 and v0.4b label-specific aggregation diagnostics

Because different labels have different temporal behaviour, parent-level aggregation was diagnosed label-wise. Four strategies were compared:

```text
mean
max
top2mean
labelwise_diagnostic
```

The improved v0.4b labelwise diagnostic used:

| Label | Aggregation |
|---|---|
| `Brene_Brown` | mean |
| `Eckhart_Tolle` | mean |
| `Eric_Thomas` | top2mean |
| `Gary_Vee` | mean |
| `Jay_Shetty` | top2mean |
| `Nick_Vujicic` | top2mean |
| `other_speaker_present` | max |
| `music_present` | top2mean |
| `audience_reaction_present` | top2mean |
| `silence_present` | max |

The diagnostic result was:

| Method | Rows | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|---:|
| mean | 867 | 0.5379 | 0.6816 | 0.6191 | 0.4498 | **0.0752** |
| max | 867 | 0.5715 | 0.6553 | 0.6742 | 0.2641 | 0.1309 |
| top2mean | 867 | 0.6125 | 0.7278 | 0.7310 | 0.4268 | 0.0870 |
| v0.4b labelwise | 867 | **0.6647** | **0.7527** | **0.7560** | **0.4591** | 0.0762 |

This showed that label-specific aggregation can improve Macro-F1, Micro-F1, Samples-F1 and Exact Match without retraining. However, because the mapping was chosen after inspecting holdout behaviour, v0.4b remains diagnostic rather than final unbiased evidence.

### AB.5 v0.5 calibration-selected label-wise aggregation

To make the analysis more defensible, the 867 parent clips were split into a calibration half and an evaluation half. Aggregation choices were selected on calibration data and then evaluated on the held-out evaluation split.

The selected v0.5 mapping was:

| Label | Selected aggregation |
|---|---|
| `Brene_Brown` | mean |
| `Eckhart_Tolle` | mean |
| `Eric_Thomas` | top2mean |
| `Gary_Vee` | mean |
| `Jay_Shetty` | top2mean |
| `Nick_Vujicic` | max |
| `other_speaker_present` | max |
| `music_present` | top2mean |
| `audience_reaction_present` | top2mean |
| `silence_present` | max |

On the evaluation split:

| Method | Rows | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|---:|
| mean | 434 | 0.5551 | 0.7084 | 0.6462 | **0.4862** | **0.0687** |
| max | 434 | 0.5427 | 0.6533 | 0.6793 | 0.2765 | 0.1313 |
| top2mean | 434 | 0.6129 | 0.7335 | 0.7368 | 0.4447 | 0.0839 |
| v0.5 calibration-selected labelwise | 434 | **0.6355** | **0.7491** | **0.7521** | 0.4447 | 0.0772 |

This confirmed that calibration-selected label-specific aggregation improves F1-style metrics on an unseen evaluation half, although global mean remained more conservative for exact match and Hamming Loss.

### AB.6 v0.6 repeated calibration/evaluation stability analysis

Finally, the calibration/evaluation split was repeated across 20 random seeds. This tested whether the label-wise aggregation benefit was stable or only a one-split artefact.

| Method | Macro-F1 mean | Macro-F1 std | Micro-F1 mean | Micro-F1 std | Samples-F1 mean | Samples-F1 std | Exact Match mean | Hamming Loss mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 0.5345 | 0.0130 | 0.6776 | 0.0073 | 0.6138 | 0.0127 | 0.4471 | **0.0759** |
| max | 0.5605 | 0.0177 | 0.6519 | 0.0087 | 0.6709 | 0.0106 | 0.2589 | 0.1328 |
| top2mean | 0.6108 | 0.0096 | 0.7252 | 0.0074 | 0.7283 | 0.0087 | 0.4233 | 0.0879 |
| calibration-selected labelwise | **0.6377** | 0.0140 | **0.7470** | 0.0161 | **0.7508** | 0.0151 | **0.4520** | 0.0785 |

The selected aggregation frequency across 20 calibration splits was:

| Label | Most frequent selected aggregation | Frequency |
|---|---|---:|
| `Brene_Brown` | mean | 20/20 |
| `Eckhart_Tolle` | mean | 20/20 |
| `Eric_Thomas` | top2mean | 18/20 |
| `Gary_Vee` | mean | 17/20 |
| `Jay_Shetty` | top2mean | 20/20 |
| `Nick_Vujicic` | top2mean | 18/20 |
| `other_speaker_present` | max | 20/20 |
| `music_present` | top2mean | 20/20 |
| `audience_reaction_present` | top2mean | 15/20 |
| `silence_present` | max | 16/20 |

The v0.6 analysis is the confirmed finding to freeze:

> Across repeated calibration/evaluation splits, calibration-selected label-specific aggregation consistently outperformed global aggregation rules on Macro-F1, Micro-F1 and Samples-F1, while maintaining near-mean exact-match behaviour and only a small Hamming Loss penalty. This supports the claim that one parent-level aggregation rule is suboptimal for multi-label audio, and that label-specific temporal evidence aggregation is an important NeuroAccuExit inference-stage contribution.

### AB.7 v0.7 repeated aggregation + threshold calibration

After freezing v0.6 as the confirmed aggregation-only finding, v0.7 tested whether the same repeated calibration/evaluation protocol could also select a per-label decision threshold. This analysis still used the saved human context-checked holdout predictions and did not retrain the model.

For each of 20 random 50/50 parent-level splits:

```text
1. Use the calibration half to select, for each label:
   - aggregation method from mean, max, top2mean
   - threshold from a coarse grid 0.10, 0.15, ..., 0.95
2. Apply the selected label-specific aggregation and threshold policy to the held-out evaluation half.
3. Compare against global mean, max and top2mean baselines using the original fixed thresholds.
```

The repeated v0.7 result was:

| Method | Macro-F1 mean | Macro-F1 std | Micro-F1 mean | Micro-F1 std | Samples-F1 mean | Samples-F1 std | Exact Match mean | Hamming Loss mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean fixed thresholds | 0.5345 | 0.0130 | 0.6776 | 0.0073 | 0.6138 | 0.0127 | 0.4471 | 0.0759 |
| max fixed thresholds | 0.5605 | 0.0177 | 0.6519 | 0.0087 | 0.6709 | 0.0106 | 0.2589 | 0.1328 |
| top2mean fixed thresholds | 0.6108 | 0.0096 | 0.7252 | 0.0074 | 0.7283 | 0.0087 | 0.4233 | 0.0879 |
| v0.7 aggregation + threshold calibrated | **0.6637** | 0.0204 | **0.7725** | 0.0095 | **0.7861** | 0.0099 | **0.4809** | **0.0702** |

This result improves over the frozen v0.6 finding on every reported aggregate metric:

| Metric | v0.6 aggregation only | v0.7 aggregation + threshold calibration | Change |
|---|---:|---:|---:|
| Macro-F1 | 0.6377 | **0.6637** | +0.0260 |
| Micro-F1 | 0.7470 | **0.7725** | +0.0255 |
| Samples-F1 | 0.7508 | **0.7861** | +0.0353 |
| Exact Match | 0.4520 | **0.4809** | +0.0289 |
| Hamming Loss | 0.0785 | **0.0702** | -0.0083 |

The aggregation choices changed after allowing thresholds to calibrate, confirming that aggregation and threshold selection interact.

| Label | Most frequent aggregation after threshold calibration | Frequency |
|---|---|---:|
| `Brene_Brown` | mean | 13/20 |
| `Eckhart_Tolle` | top2mean | 11/20 |
| `Eric_Thomas` | mean/top2mean tie | 10/20 each |
| `Gary_Vee` | mean | 20/20 |
| `Jay_Shetty` | mean | 17/20 |
| `Nick_Vujicic` | mean/top2mean tie | 10/20 each |
| `other_speaker_present` | max | 11/20 |
| `music_present` | mean | 12/20 |
| `audience_reaction_present` | top2mean | 15/20 |
| `silence_present` | max | 19/20 |

The calibrated thresholds also differed substantially from the weak-label validation thresholds used earlier:

| Label | Mean threshold | Std | Min | Max | Mean calibration support |
|---|---:|---:|---:|---:|---:|
| `Brene_Brown` | 0.7375 | 0.1685 | 0.55 | 0.95 | 37.45 |
| `Eckhart_Tolle` | 0.6625 | 0.1621 | 0.35 | 0.85 | 42.10 |
| `Eric_Thomas` | 0.5750 | 0.1594 | 0.30 | 0.80 | 33.85 |
| `Gary_Vee` | 0.8450 | 0.0536 | 0.75 | 0.95 | 33.35 |
| `Jay_Shetty` | 0.7350 | 0.0947 | 0.65 | 0.95 | 45.15 |
| `Nick_Vujicic` | 0.3725 | 0.1118 | 0.20 | 0.70 | 23.85 |
| `other_speaker_present` | 0.1725 | 0.0343 | 0.10 | 0.20 | 228.60 |
| `music_present` | 0.6750 | 0.1509 | 0.50 | 0.95 | 171.40 |
| `audience_reaction_present` | 0.9050 | 0.0776 | 0.65 | 0.95 | 13.15 |
| `silence_present` | 0.5075 | 0.2944 | 0.10 | 0.95 | 6.65 |

The most important threshold findings were:

| Label | Interpretation |
|---|---|
| `other_speaker_present` | Needed a much lower threshold, suggesting the weak-label threshold was too conservative for the human context-checked holdout. |
| `audience_reaction_present` | Needed a high threshold, consistent with a bursty label that can create many false positives under aggressive aggregation. |
| `silence_present` | Remained unstable because calibration support was low; this label should be interpreted cautiously. |
| `Nick_Vujicic` | Needed a lower threshold, suggesting the v0.3 threshold was too strict for this holdout. |

### AB.8 Updated final interpretation

The downstream experiments now support four separate claims:

| Claim | Evidence |
|---|---|
| The TATA-LAWYER v0.10.1 hybrid weak-label policy can train a downstream three-exit NeuroAccuExit model. | v0.3 strict weak-label test reached Micro-F1 = 0.9073 and Exact Match = 0.7930. |
| Human context-checked holdout generalisation is harder than weak-label policy reproduction. | Parent-mean holdout Micro-F1 = 0.6816 and Exact Match = 0.4498. |
| Label-specific parent aggregation improves holdout F1 metrics without retraining. | v0.6 repeated analysis: labelwise Macro-F1 = 0.6377 vs mean = 0.5345. |
| Joint label-specific aggregation and threshold calibration gives the strongest diagnostic result. | v0.7 repeated analysis: Macro-F1 = 0.6637, Micro-F1 = 0.7725, Samples-F1 = 0.7861, Exact Match = 0.4809 and Hamming Loss = 0.0702. |

The current best downstream diagnostic is therefore:

```text
NeuroAccuExit downstream model:
v0.2 pos_weight5 trained checkpoint
+ v0.3 weak-label threshold tuning
+ v0.7 repeated calibration-selected parent aggregation
+ v0.7 repeated per-label threshold calibration
```

This should be reported as the **best calibrated diagnostic**, not as an external unbiased final test, because the repeated calibration/evaluation analysis still operates within the same human context-checked holdout collection.

The final methodological conclusion is:

> Multi-label audio classification should not force one global parent-level aggregation rule or one inherited threshold profile across all labels. Label-specific aggregation and per-label threshold calibration substantially improve human context-checked holdout performance without retraining the underlying model.
