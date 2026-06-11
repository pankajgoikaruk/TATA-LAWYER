# Label-aware parent aggregation command

Add this script to:

```text
scripts/evaluate_label_aware_parent_aggregation.py
```

Then run after the parent mean evaluation has produced:

```text
parent_eval_segment_probs_fixed_0p5_mean.csv
```

PowerShell command:

```powershell
$HCBBRoot = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline"

$SegmentProbCsv = "$HCBBRoot\evaluation\existing_script_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"

$OutDir = "$HCBBRoot\evaluation\v08_hcb_parent_label_aware_fixed"

python scripts\evaluate_label_aware_parent_aggregation.py `
  --segment_probs_csv "$SegmentProbCsv" `
  --out_dir "$OutDir" `
  --labels_json "configs\human_talk_10label_schema.json" `
  --exit_idx 3 `
  --threshold 0.5 `
  --max_labels "audience_reaction_present,silence_present" `
  --model_name "main_v08_human_corrected_balanced_3exit_20260610_084027"
```

Expected outputs:

```text
human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\evaluation\v08_hcb_parent_label_aware_fixed\parent_holdout_eval_fixed_0p5_label_aware.json
human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\evaluation\v08_hcb_parent_label_aware_fixed\parent_holdout_static_fixed_0p5_label_aware.csv
human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\evaluation\v08_hcb_parent_label_aware_fixed\parent_holdout_per_label_fixed_0p5_label_aware_exit3.csv
human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\evaluation\v08_hcb_parent_label_aware_fixed\parent_label_aware_exit3_probabilities.csv
```

Expected result:

```text
Macro-F1     = 0.8320
Micro-F1     = 0.9285
Samples-F1   = 0.9375
Exact Match  = 0.8235
Hamming Loss = 0.0211
```

Rule:

```text
mean aggregation:
  Brene_Brown
  Eckhart_Tolle
  Eric_Thomas
  Gary_Vee
  Jay_Shetty
  Nick_Vujicic
  other_speaker_present
  music_present

max aggregation:
  audience_reaction_present
  silence_present
```
