# TATA-LAWYER v0.10.1 Domain-Aware Hybrid Final Result

## Final selected system

```text
Normal/original audio:
    model:      Original v0.9 TATA triage checkpoint
    threshold:  fixed 0.50

Recovered low-energy audio:
    model:      Human-reviewed masked v0.10 checkpoint
    threshold:  recovered-domain per-label thresholds
```

Final branch:

```text
tata_lawyer_v0.10.1
```

This version is a final documentation and model-selection release. It does not introduce a new feature cache, a new manual review round, or a new training run.

---

## Final combined 2,119-row test outcome

| Policy | Normal/original audio | Recovered low-energy audio | Macro-F1 | Micro-F1 | Samples-F1 | Known-label Exact | Fully-known Exact | Hamming Loss ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Original fixed everywhere | Original v0.9, threshold 0.50 | Original v0.9, threshold 0.50 | 0.8155 | 0.8171 | 0.8108 | 0.6465 | 0.6462 | 0.0489 |
| Masked fixed everywhere | Masked v0.10, threshold 0.50 | Masked v0.10, threshold 0.50 | 0.7991 | 0.7958 | 0.7624 | 0.5998 | 0.5988 | 0.0539 |
| Original domain-aware | Original v0.9, normal threshold | Original v0.9, recovered thresholds | 0.8162 | 0.8165 | 0.8092 | 0.6446 | 0.6453 | 0.0491 |
| Masked domain-aware | Masked v0.10, strict thresholds | Masked v0.10, recovered thresholds | 0.8173 | 0.8081 | 0.7813 | 0.6031 | 0.6030 | 0.0526 |
| **Hybrid recommended** | **Original v0.9, threshold 0.50** | **Masked v0.10, recovered thresholds** | **0.8224** | **0.8218** | **0.8174** | **0.6555** | **0.6557** | **0.0477** |

### Improvement over original fixed baseline

| Metric | Original fixed everywhere | v0.10.1 hybrid | Absolute change |
|---|---:|---:|---:|
| Macro-F1 | 0.8155 | **0.8224** | **+0.0068** |
| Micro-F1 | 0.8171 | **0.8218** | **+0.0047** |
| Samples-F1 | 0.8108 | **0.8174** | **+0.0066** |
| Known-label Exact | 0.6465 | **0.6555** | **+0.0090** |
| Fully-known Exact | 0.6462 | **0.6557** | **+0.0095** |
| Hamming Loss ↓ | 0.0489 | **0.0477** | **-0.0013** |

The hybrid is the selected final policy because it is the only tested policy that improves every aggregate metric over the original fixed-threshold baseline.

---

## Recovered low-energy model comparison

| Model on recovered audio | Threshold source | Macro-F1 | Micro-F1 | Samples-F1 | Fully-known Exact | Hamming Loss ↓ |
|---|---|---:|---:|---:|---:|---:|
| Original v0.9 | Fixed 0.50 | 0.4756 | 0.7262 | 0.6646 | 0.5586 | 0.0569 |
| Masked v0.10 | Fixed 0.50 | 0.5095 | 0.8065 | 0.7236 | 0.6828 | **0.0384** |
| Original v0.9 | Recovered-domain thresholds | 0.5126 | 0.7143 | 0.6424 | 0.5448 | 0.0589 |
| **Masked v0.10** | **Recovered-domain thresholds** | **0.5438** | **0.8075** | **0.7532** | **0.6966** | 0.0397 |

The masked v0.10 checkpoint is selected for recovered low-energy audio because it outperforms the original v0.9 checkpoint under both fixed-threshold and recovered-threshold evaluations.

---

## Recovered-domain thresholds

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

These thresholds are domain-specific to recovered low-energy audio. They should not be reported as universal thresholds for normal/original audio.

---

## Experiment settings

### Original v0.9 normal-audio branch

```text
architecture: TinyAudioCNN + three exits
tap_blocks: 1,3
loss_weights: 0.3,0.3,1.0
epochs: 40
batch_size: 64
learning_rate: 0.001
threshold: 0.50
device: CPU
```

### Masked v0.10 low-energy branch

```text
architecture: TinyAudioCNN + three exits
tap_blocks: 1,3
loss_weights: 0.3,0.3,1.0
epochs: 40
batch_size: 64
learning_rate: 0.001
masked loss: BCEWithLogitsLoss(reduction="none") multiplied by per-label masks
checkpoint selection: strict original validation Macro-F1
best epoch: 38
best strict validation Macro-F1: 0.7771
device: CPU
```

### Evaluation subsets

```text
strict validation rows:           1,883
strict original test rows:        1,961
all masked test rows:             2,119
recovered validation rows:          159
recovered human-reviewed test rows: 158
```

---

## Research findings

1. The original preprocessing pipeline censored low-energy one-second windows before feature extraction.
2. Low-energy recovery is useful, but low energy is not the same as silence.
3. Parent-level labels are unsafe as one-second supervision for recovered low-energy windows.
4. Tri-state human review and masked BCE corrected the inherited-label contamination problem.
5. The masked v0.10 model is not the best universal model for normal audio.
6. The masked v0.10 model is the best tested recovered low-energy specialist.
7. Domain-aware routing gives the strongest final system.

---

## Final conclusion

TATA-LAWYER v0.10.1 should be reported as a domain-aware hybrid system. The original v0.9 checkpoint remains the selected model for normal/original audio. The human-reviewed masked v0.10 checkpoint becomes the selected model for recovered low-energy audio. The hybrid achieves the strongest combined performance across all tested policies and improves every aggregate metric over the original fixed-threshold baseline.

---

## Limitations

The recovered-domain threshold profile was selected using a small recovered validation set. Some labels have very low positive support, so the thresholds should be treated as empirically selected domain-specific thresholds rather than globally optimal thresholds. For a stronger publication claim, a future confirmation run should use a new untouched low-energy holdout.
