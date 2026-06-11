# Release Notes — agentic_data_preprocessing_v0.8 documentation update

## Added

- Updated documentation for the post-hoc label-aware aggregation finding.
- Added CSV tables under `docs/tables/agentic_data_preprocessing_v0.8/`.
- Added line plots and bar plots under `docs/figures/human_talk/agentic_data_preprocessing_v0.8/`.
- Added thesis-ready report under `docs/reports/human_talk/`.
- Added compact results summary under `docs/results/human_talk/`.

## Key finding

Global max aggregation is not suitable as the final result because it increases false positives:

```text
Global max Exit 3:
Macro-F1=0.7251
Micro-F1=0.8203
Exact=0.5121
Hamming=0.0630
Avg predicted labels=2.0346
```

Label-aware aggregation is the best Macro-F1 setting:

```text
Label-aware Exit 3:
Macro-F1=0.8320
Micro-F1=0.9285
Samples-F1=0.9375
Exact=0.8235
Hamming=0.0211
```

## Recommended release interpretation

- Keep parent mean fixed 0.5 as the official overall result.
- Add label-aware aggregation as a post-hoc analysis / research contribution.
- Do not replace the model or retrain.
