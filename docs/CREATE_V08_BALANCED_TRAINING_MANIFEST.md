# Create v0.8 Human-Corrected-Balanced Manifest

Run this after the final expanded manifest is built.

It does not change labels. It only down-samples over-represented background-heavy rows while keeping all target-speaker, audience, silence, and seed rows.

## Run

```powershell
$HCBBRoot = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline"

python scripts\create_v08_balanced_training_manifest.py `
  --input_manifest "$HCBBRoot\final_expanded_training_dataset\metadata\multilabel_features_manifest.csv" `
  --out_root "$HCBBRoot\final_expanded_training_dataset_balanced" `
  --labels_json "configs\human_talk_10label_schema.json" `
  --lawyer_config "configs\lawyer_v08_human_talk.json" `
  --other_cap_multiplier 3.0 `
  --seed 42
```

## Output

```text
human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\final_expanded_training_dataset_balanced\metadata\multilabel_features_manifest_balanced.csv
```

Use this balanced manifest for the next training run.
