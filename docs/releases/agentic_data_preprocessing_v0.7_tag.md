# Agentic Data Preprocessing v0.7

This release documents a filtered target-speaker ablation built from v0.6.

## Highlights

- Removed five non-target source-speaker folders from filtered CSV manifests: `Les_Brown`, `Mel_Robbins`, `Oprah_Winfrey`, `Rabin_Sharma`, and `Simon_Sinek`.
- Rebuilt the final expanded training manifest for a cleaner six-target-speaker setting.
- Trained and evaluated `main_v07_filtered_3exit` on the filtered raw holdout.
- Compared v0.7 against the v0.6 model on the same filtered holdout.

## Best v0.7 result

`main_v07_filtered_3exit + fixed threshold 0.5 + parent-level mean aggregation`

| Metric | Score |
|---|---:|
| Macro-F1 | 0.7446 |
| Micro-F1 | 0.8983 |
| Samples-F1 | 0.9041 |
| Exact Match | 0.7596 |
| Hamming Loss | 0.0317 |

## Main finding

v0.7 slightly improves target-focused holdout behaviour over the v0.6 model on the same filtered holdout, especially Macro-F1, Samples-F1, Exact Match, and Hamming Loss. The improvement is modest. Weak labels remain `other_speaker_present`, `audience_reaction_present`, and `silence_present`, so the next strategy should focus on targeted weak-label repair rather than additional source filtering alone.
