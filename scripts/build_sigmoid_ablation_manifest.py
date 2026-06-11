from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def parse_labels(text: str) -> list[str]:
    labels = [x.strip() for x in str(text).split(",") if x.strip()]
    if not labels:
        raise ValueError("No labels provided.")
    return labels


def make_sample_id(row: pd.Series, idx: int) -> str:
    label = safe_name(row["label"])
    split = safe_name(row["split"])
    rel = str(row.get("segment_wav_relpath", "") or row.get("wav_relpath", ""))
    h = hashlib.md5(f"{rel}|{idx}".encode("utf-8")).hexdigest()[:12]
    return f"{split}_{label}_{idx:07d}_{h}"


def main():
    parser = argparse.ArgumentParser(
        description="Build sigmoid/BCE one-hot speaker manifest from ASHADIP segments.csv."
    )

    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--dataset_name",
        default="agentic_cleaned_sigmoid_ablation",
    )

    args = parser.parse_args()

    segments_csv = Path(args.segments_csv)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)

    labels = parse_labels(args.labels)

    if not segments_csv.exists():
        raise FileNotFoundError(f"segments.csv not found: {segments_csv}")

    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir not found: {cache_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(segments_csv)

    required = {"label", "split", "segment_wav_relpath"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"segments.csv missing required columns: {sorted(missing)}")

    unknown = sorted(set(df["label"].astype(str)) - set(labels))
    if unknown:
        raise RuntimeError(
            "segments.csv contains labels not listed in --labels:\n"
            f"{unknown}\n"
            f"Allowed labels: {labels}"
        )

    rows = []

    for idx, row in df.reset_index(drop=True).iterrows():
        label = str(row["label"])
        split = str(row["split"])

        rel = str(row["segment_wav_relpath"]).replace("\\", "/")
        abs_path = (cache_dir / rel).resolve()

        if not abs_path.exists():
            raise FileNotFoundError(f"Segment WAV not found: {abs_path}")

        sample_id = make_sample_id(row, idx)

        out = {
            "sample_id": sample_id,
            "abs_path": str(abs_path),
            "split": split,
            "primary_label": label,
            "labels": label,
            "num_labels": 1,
            "is_clean_seed": 1,
            "is_synthetic": 0,
            "source_mode": "sigmoid_onehot_speaker_ablation",
            "source_segments_csv": str(segments_csv.resolve()),
            "segment_wav_relpath": rel,
            "orig_relpath": str(row.get("orig_relpath", "")),
            "wav_relpath": str(row.get("wav_relpath", "")),
            "source_file": str(row.get("source_file", "")),
            "source_path": str(row.get("source_path", "")),
            "start": row.get("start", ""),
            "parent_start": row.get("parent_start", ""),
            "duration": row.get("duration", ""),
        }

        for lab in labels:
            out[lab] = 1 if lab == label else 0

        rows.append(out)

    out_df = pd.DataFrame(rows)

    split_order = {"train": 0, "val": 1, "test": 2}
    out_df["_split_order"] = out_df["split"].map(split_order).fillna(99)
    out_df = out_df.sort_values(
        by=["_split_order", "primary_label", "sample_id"],
        ascending=[True, True, True],
    ).drop(columns=["_split_order"])

    manifest_path = out_dir / "sigmoid_onehot_manifest.csv"
    labels_json_path = out_dir / "labels.json"
    summary_path = out_dir / "sigmoid_onehot_manifest_summary.json"

    out_df.to_csv(manifest_path, index=False)

    labels_payload = {
        "dataset_name": args.dataset_name,
        "task": "sigmoid_onehot_speaker_ablation",
        "description": (
            "Single-label speaker folders converted into one-hot targets for "
            "BCEWithLogitsLoss + sigmoid ablation."
        ),
        "labels": labels,
    }

    with labels_json_path.open("w", encoding="utf-8") as f:
        json.dump(labels_payload, f, indent=2)

    summary = {
        "dataset_name": args.dataset_name,
        "task": "sigmoid_onehot_speaker_ablation",
        "segments_csv": str(segments_csv.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "labels_json": str(labels_json_path.resolve()),
        "rows": int(len(out_df)),
        "labels": labels,
        "split_counts": out_df["split"].value_counts().to_dict(),
        "class_counts": out_df["primary_label"].value_counts().to_dict(),
        "positive_label_counts": {
            lab: int(out_df[lab].sum()) for lab in labels
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSigmoid one-hot ablation manifest created")
    print("-" * 80)
    print(f"Manifest:    {manifest_path}")
    print(f"Labels JSON: {labels_json_path}")
    print(f"Summary:     {summary_path}")
    print(f"Rows:        {len(out_df)}")

    print("\nSplit counts:")
    print(out_df["split"].value_counts().to_string())

    print("\nClass counts:")
    print(out_df["primary_label"].value_counts().to_string())

    print("-" * 80)


if __name__ == "__main__":
    main()
