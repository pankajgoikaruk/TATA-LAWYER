# NeuroAccuExit Hybrid Weak-Labels Plan

## Branch

```text
neuroaccuexit_hybrid_weaklabels
```

## Purpose

This branch starts the downstream NeuroAccuExit main-model stage using the final TATA-LAWYER v0.10.1 domain-aware hybrid weak-label system.

The goal is not to continue TATA-LAWYER model selection. That phase is now frozen. The goal here is to convert the selected TATA-LAWYER hybrid predictions into a clean weak-label training source for the main NeuroAccuExit model.

---

## Frozen upstream result

The final selected TATA-LAWYER system is:

```text
Normal/original audio:
    model:      Original v0.9 TATA triage checkpoint
    threshold:  fixed 0.50

Recovered low-energy audio:
    model:      Human-reviewed masked v0.10 checkpoint
    threshold:  recovered-domain per-label thresholds
```

The selected low-energy thresholds are:

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

The final hybrid policy should be treated as immutable input for this branch unless a separate ablation branch is created.

---

## Final TATA-LAWYER combined outcome

The hybrid was selected because it improved every aggregate metric over the original fixed-threshold baseline on the combined 2,119-row test set.

| Policy | Macro-F1 | Micro-F1 | Samples-F1 | Known-label Exact | Fully-known Exact | Hamming Loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Original fixed everywhere | 0.8155 | 0.8171 | 0.8108 | 0.6465 | 0.6462 | 0.0489 |
| Hybrid recommended | **0.8224** | **0.8218** | **0.8174** | **0.6555** | **0.6557** | **0.0477** |

This establishes the hybrid as the final weak-label generator/refiner to be used for downstream main-model work.

---

## What this branch must not change

Do not modify or overwrite:

```text
Original v0.9 checkpoint
Human-reviewed masked v0.10 checkpoint
v0.10.1 recovered-domain thresholds
Manual review CSVs
Human-reviewed masked manifest
Final v0.10.1 hybrid result report
Strict and recovered evaluation subsets
```

Any change to these artifacts belongs in a new TATA-LAWYER ablation branch, not here.

---

## Main objective

Build a reproducible downstream training path:

```text
TATA-LAWYER v0.10.1 hybrid policy
↓
final hybrid weak-label manifest
↓
NeuroAccuExit main-model training dataset
↓
main-model training
↓
comparison against previous main-model baseline
```

---

## Expected first deliverable

Create a final hybrid weak-label manifest that records, for every segment or parent item used downstream:

1. the final 10-label weak-label vector;
2. which branch produced the label decision;
3. the threshold profile used;
4. confidence/probability values where available;
5. whether the row came from normal preprocessing or low-energy recovery;
6. whether labels are fully known, weakly inferred, or masked;
7. enough provenance to reproduce the label decision later.

Suggested output path:

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/
  metadata/final_hybrid_weaklabel_manifest.csv
```

Suggested reports:

```text
human_talk_workspace/neuroaccuexit_hybrid_weaklabels/reports/
  hybrid_weaklabel_manifest_summary.json
  hybrid_weaklabel_label_distribution.csv
  hybrid_weaklabel_source_distribution.csv
  hybrid_weaklabel_split_distribution.csv
```

---

## Suggested manifest schema

Core identity columns:

```text
row_id
clip_id
parent_clip_id
segment_id
split
start_sec
end_sec
audio_path
feat_relpath
```

Routing/provenance columns:

```text
routing_domain
label_source_model
threshold_profile_name
threshold_profile_path
is_low_energy_recovered
is_original_strict_row
is_human_reviewed_recovered
weak_label_generation_version
```

Recommended routing values:

```text
routing_domain = normal_original | recovered_low_energy
label_source_model = original_v09_tata | masked_v10_human_reviewed
threshold_profile_name = fixed_0_50 | recovered_low_energy_profile
weak_label_generation_version = tata_lawyer_v0.10.1_domain_aware_hybrid
```

Target columns:

```text
Brene_Brown
Eckhart_Tolle
Eric_Thomas
Gary_Vee
Jay_Shetty
Nick_Vujicic
other_speaker_present
music_present
audience_reaction_present
silence_present
```

Optional probability columns:

```text
prob_Brene_Brown
prob_Eckhart_Tolle
prob_Eric_Thomas
prob_Gary_Vee
prob_Jay_Shetty
prob_Nick_Vujicic
prob_other_speaker_present
prob_music_present
prob_audience_reaction_present
prob_silence_present
```

Optional mask/reliability columns:

```text
mask_Brene_Brown
mask_Eckhart_Tolle
mask_Eric_Thomas
mask_Gary_Vee
mask_Jay_Shetty
mask_Nick_Vujicic
mask_other_speaker_present
mask_music_present
mask_audience_reaction_present
mask_silence_present

label_reliability_group
has_any_unknown_label
known_label_count
unknown_label_count
```

---

## Routing rule requirement

The final downstream router must not depend on test labels.

Acceptable experimental routing fields:

```text
v09_data_origin
v09_masked_review_applied
recovery_candidate_id
```

These are useful for reconstructing experiments, but a deployable method should route from reproducible signal-processing conditions, such as:

```text
raw waveform
↓
pre-normalisation RMS / dBFS / speech-activity checks
↓
low-energy decision
↓
normal branch or recovered-low-energy branch
```

Absolute energy must be measured before peak normalisation, because peak normalisation destroys the low-energy signal used for routing.

---

## Planned scripts

### 1. Build final weak-label manifest

```text
scripts/build_neuroaccuexit_hybrid_weaklabel_manifest.py
```

Responsibilities:

- load original v0.9 prediction outputs;
- load masked v0.10 prediction outputs;
- apply the v0.10.1 routing policy;
- write final hybrid labels;
- preserve branch/model/threshold provenance;
- write summary reports.

### 2. Validate final weak-label manifest

```text
scripts/validate_neuroaccuexit_hybrid_weaklabels.py
```

Responsibilities:

- confirm row counts;
- confirm no missing features;
- confirm binary target columns;
- confirm split consistency;
- confirm label-source counts;
- confirm no accidental overwrite of frozen TATA-LAWYER files.

### 3. Train downstream main model

```text
training/train_neuroaccuexit_from_hybrid_weaklabels.py
```

Responsibilities:

- consume the final hybrid weak-label manifest;
- keep training/evaluation splits fixed;
- save run configuration and commit SHA;
- report segment-level and parent-level metrics if applicable.

### 4. Evaluate downstream model

```text
scripts/evaluate_neuroaccuexit_hybrid_weaklabels.py
```

Responsibilities:

- compare against the previous main-model baseline;
- report per-label F1, Macro-F1, Micro-F1, Samples-F1, Exact Match, and Hamming Loss;
- separate normal-domain and low-energy-domain performance.

---

## First implementation phase

### Phase 1 — Inventory

Find and document the exact files needed for manifest construction:

```text
original v0.9 checkpoint
original v0.9 prediction/probability outputs
masked v0.10 checkpoint
masked v0.10 prediction/probability outputs
v0.10.1 recovered-domain threshold JSON/profile
normal fixed-threshold policy
human-reviewed masked manifest
feature root
label schema JSON
```

### Phase 2 — Hybrid weak-label manifest

Implement only the manifest builder and validation report first. Do not train yet.

Expected output:

```text
final_hybrid_weaklabel_manifest.csv
hybrid_weaklabel_manifest_summary.json
```

### Phase 3 — Main-model baseline selection

Before training, decide what the downstream main-model comparison baseline is:

```text
previous NeuroAccuExit main model
v0.8 parent-level model
v0.9/v0.10 segment-level TATA-only baselines
```

Keep TATA/Lawyer segment-level metrics separate from main-model parent-level metrics.

### Phase 4 — First main-model run

Train the first downstream model only after the final hybrid manifest passes validation.

---

## Evaluation policy

Report at least:

```text
Macro-F1
Micro-F1
Samples-F1
Exact Match
Hamming Loss
Per-label precision/recall/F1
```

If parent-level aggregation is used, report separate tables for:

```text
segment-level metrics
parent-level mean aggregation
parent-level max aggregation
label-aware aggregation if used
```

Do not compare parent-level main-model metrics directly against one-second TATA triage metrics without clearly naming the evaluation level.

---

## Research interpretation target

This branch should support the following thesis-level claim only if the downstream model improves:

> TATA-LAWYER v0.10.1 provides a domain-aware weak-label generation and refinement layer. When used to construct downstream NeuroAccuExit training labels, it improves the quality of the main model compared with training from unrefined weak labels or earlier baselines.

If the downstream model does not improve, the still-valid claim is:

> TATA-LAWYER improves weak-label quality and low-energy triage performance, but downstream main-model benefit depends on aggregation, label balance, and training strategy.

---

## Current status

```text
TATA-LAWYER model-selection phase: complete
Hybrid weak-label branch: started
Main-model training: not started
Next task: build final hybrid weak-label manifest
```
