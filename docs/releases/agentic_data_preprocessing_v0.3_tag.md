# agentic_data_preprocessing_v0.3_tag

## Summary

This tag marks the v0.3 Agentic AI preprocessing milestone for NeuroAccuExit-ASHADIP.

## Main contribution

This version introduces a non-destructive, manifest-first Agentic AI preprocessing workflow for Raw5 human-talk speaker data.

## Key components

- DatasetAuditorAgent
- ManifestBuilderAgent
- DatasetBuilderAgent
- cleaned Raw5 dataset generation
- accepted / needs_review / rejected / blocked routing
- traceable audit reports and split manifests
- manual exclusion record for Eric_Thomas__0175.wav

## First Raw5 cleaned experiment

Run:

raw5_agentic_cleaned_3exit_greedy_final_001

Results:

- Segment greedy accuracy: 96.83%
- Full-clip accuracy: 99.57%
- Depth×Time clip accuracy: 98.93%
- Windows saved: 75.87%
- Compute saved: 75.82%
- Final cleaned files: 3,108

## Next required baseline

Run matched baseline:

raw5_uncleaned_3exit_greedy

This will quantify the effect of agentic preprocessing.
