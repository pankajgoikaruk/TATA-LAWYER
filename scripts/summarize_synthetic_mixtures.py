from pathlib import Path
import pandas as pd

manifest = Path("multilabel_data/metadata/synthetic_mixed_manifest.csv")

df = pd.read_csv(manifest)

df["combo"] = df["labels"].apply(
    lambda x: " + ".join(sorted(str(x).split(";")))
)

summary = (
    df.groupby(["split", "combo"])
    .size()
    .reset_index(name="count")
    .sort_values(["split", "count"], ascending=[True, False])
)

print("\nMixture combination counts:")
print(summary.to_string(index=False))

out = Path("multilabel_data/metadata/synthetic_mixture_combo_summary.csv")
summary.to_csv(out, index=False)

print(f"\nSaved summary to: {out}")

print("\nExample mixed files:")
cols = ["sample_id", "split", "labels", "source_names", "filepath"]
print(df[cols].head(20).to_string(index=False))