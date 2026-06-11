---

### 11. Global max aggregation diagnostic

After the official parent-level mean corrected-holdout evaluation, global max aggregation was tested as a diagnostic.

Command section:

```text
13_eval_v08_global_max_parent_fixed
```

Result:

| Aggregation | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| Parent mean | **0.7801** | **0.9332** | **0.9406** | **0.8397** | **0.0194** | 1.4302 |
| Global max | 0.7251 | 0.8203 | 0.8423 | 0.5121 | 0.0630 | 2.0346 |

Decision:

```text
Do not use global max as the final aggregation strategy.
```

Reason: global max over-predicts parent labels, increasing false positives and worsening Exact Match and Hamming Loss.

### 12. Weak-label improvement under max aggregation

The global max diagnostic showed that the two weak transient labels improved:

| Label | Parent mean F1 | Global max F1 |
|---|---:|---:|
| `audience_reaction_present` | 0.1250 | **0.4706** |
| `silence_present` | 0.0000 | **0.1739** |

Interpretation:

```text
The model has some segment-level evidence for weak transient labels,
but parent-level mean aggregation dilutes that evidence.
```

This motivated a label-aware parent aggregation rule.

### 13. Label-aware aggregation experiment

Command section:

```text
15_compute_v08_label_aware_parent_aggregation
```

Rule:

```text
mean aggregation for 8 stable labels:
  Brene_Brown
  Eckhart_Tolle
  Eric_Thomas
  Gary_Vee
  Jay_Shetty
  Nick_Vujicic
  other_speaker_present
  music_present

max aggregation for 2 transient labels:
  audience_reaction_present
  silence_present
```

Result:

| Strategy | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| Parent mean official | 0.7801 | **0.9332** | **0.9406** | **0.8397** | **0.0194** | 1.4302 |
| Label-aware mean/max | **0.8320** | 0.9285 | 0.9375 | 0.8235 | 0.0211 | 1.4844 |

Finding:

```text
Label-aware aggregation improved Macro-F1 from 0.7801 to 0.8320
without retraining the model.
```

### 14. Updated final decision

The final reporting strategy is:

| Reporting role | Method | Reason |
|---|---|---|
| Main official overall result | Parent mean, fixed threshold 0.5, Exit 3 | Best Micro-F1, Samples-F1, Exact Match, and Hamming Loss. |
| Research/ablation contribution | Label-aware mean/max, fixed threshold 0.5, Exit 3 | Best Macro-F1 and better handling of weak transient labels. |
| Diagnostic only | Global max, fixed threshold 0.5, Exit 3 | Helps transient labels but damages overall multi-label prediction. |

Official result remains:

```text
v0.8-HCB
parent-level mean
fixed threshold 0.5
Exit 3
Macro-F1=0.7801
Micro-F1=0.9332
Samples-F1=0.9406
Exact=0.8397
Hamming=0.0194
```

Additional label-aware research finding:

```text
v0.8-HCB
label-aware parent aggregation
mean for stable labels
max for transient labels
Exit 3
Macro-F1=0.8320
Micro-F1=0.9285
Samples-F1=0.9375
Exact=0.8235
Hamming=0.0211
```

### 15. Updated limitation and future work

The v0.8-HCB experiment shows that weak labels should not be treated only as a training-data problem. Parent-level aggregation also matters.

Remaining limitations:

- `audience_reaction_present` and `silence_present` remain difficult under parent mean.
- Global max is too aggressive for stable labels.
- Label-aware aggregation was post-hoc; future work should integrate this rule into a formal evaluation script or learn label-specific pooling automatically.
- Future early-exit policy should consider label type: stable labels may exit based on accumulated mean confidence, while transient labels may require max/event-detection evidence.

Updated future direction:

```text
Develop label-type-aware early-exit inference:
  stable labels -> evidence accumulation / mean confidence
  transient labels -> event-triggered max confidence
```
