# scripts/build_tata_pseudo_dataset_manifests.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
    "other_speaker_present",
    "music_present",
    "audience_reaction_present",
    "silence_present",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_labels_text(row) -> str:
    active = [lab for lab in LABELS if int(row.get(lab, 0)) == 1]
    return "|".join(active)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training-ready pseudo-label feature manifest for TATA v0.6."
    )

    parser.add_argument("--pseudo_parent_csv", required=True)
    parser.add_argument("--base_feature_manifest", required=True)
    parser.add_argument("--labels_json", required=True)
    parser.add_argument("--out_root", required=True)

    args = parser.parse_args()

    pseudo_csv = Path(args.pseudo_parent_csv)
    base_feature_manifest = Path(args.base_feature_manifest)
    labels_json = Path(args.labels_json)
    out_root = Path(args.out_root)

    if not pseudo_csv.exists():
        raise FileNotFoundError(pseudo_csv)
    if not base_feature_manifest.exists():
        raise FileNotFoundError(base_feature_manifest)
    if not labels_json.exists():
        raise FileNotFoundError(labels_json)

    out_meta = out_root / "metadata"
    out_feature_meta = out_root / "feature_cache" / "metadata"
    out_meta.mkdir(parents=True, exist_ok=True)
    out_feature_meta.mkdir(parents=True, exist_ok=True)

    pseudo_df = pd.read_csv(pseudo_csv)
    feat_df = pd.read_csv(base_feature_manifest)

    if "parent_clip_id" not in pseudo_df.columns:
        raise RuntimeError("pseudo_parent_csv must contain parent_clip_id")

    if "parent_clip_id" not in feat_df.columns:
        raise RuntimeError("base_feature_manifest must contain parent_clip_id")

    # Convert parent_pred_<label> columns into actual training label columns.
    for lab in LABELS:
        pred_col = f"parent_pred_{lab}"
        if pred_col not in pseudo_df.columns:
            raise RuntimeError(f"Missing pseudo prediction column: {pred_col}")

        pseudo_df[lab] = pd.to_numeric(pseudo_df[pred_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    pseudo_df["labels"] = pseudo_df.apply(make_labels_text, axis=1)
    pseudo_df["num_active_labels"] = pseudo_df[LABELS].sum(axis=1).astype(int)

    zero_active = pseudo_df[pseudo_df["num_active_labels"] == 0].copy()
    pseudo_df = pseudo_df[pseudo_df["num_active_labels"] > 0].copy()

    # Map pseudo labels from parent level to all existing segment-feature rows.
    parent_to_labels = pseudo_df.set_index("parent_clip_id")[LABELS + ["labels", "num_active_labels"]].to_dict("index")
    keep_parent_ids = set(parent_to_labels.keys())

    out_rows = []
    missing_parent_segments = []

    grouped = feat_df.groupby("parent_clip_id", dropna=False)
    for parent_id in sorted(keep_parent_ids):
        if parent_id not in grouped.groups:
            missing_parent_segments.append(parent_id)
            continue

        parent_segments = feat_df.loc[grouped.groups[parent_id]].copy()
        lab_payload = parent_to_labels[parent_id]

        for lab in LABELS:
            parent_segments[lab] = int(lab_payload[lab])

        parent_segments["labels"] = lab_payload["labels"]
        parent_segments["num_active_labels"] = int(lab_payload["num_active_labels"])
        parent_segments["is_pseudo_labeled"] = 1
        parent_segments["pseudo_parent_source_csv"] = str(pseudo_csv)
        parent_segments["pseudo_parent_clip_id"] = parent_id

        if "pseudo_source_mode" in pseudo_df.columns:
            mode = str(pseudo_df[pseudo_df["parent_clip_id"] == parent_id]["pseudo_source_mode"].iloc[0])
            parent_segments["pseudo_source_mode"] = mode

        if "routing_decision" in pseudo_df.columns:
            decision = str(pseudo_df[pseudo_df["parent_clip_id"] == parent_id]["routing_decision"].iloc[0])
            parent_segments["routing_decision"] = decision

        out_rows.append(parent_segments)

    if out_rows:
        out_df = pd.concat(out_rows, ignore_index=True)
    else:
        out_df = pd.DataFrame()

    out_parent = out_meta / "pseudo_parent_manifest_training_ready.csv"
    out_feature = out_feature_meta / "multilabel_features_manifest.csv"
    out_labels = out_meta / "tata_v06_labels.json"
    out_summary = out_meta / "pseudo_dataset_summary.json"
    out_summary_md = out_meta / "pseudo_dataset_summary.md"
    out_zero = out_meta / "zero_active_pseudo_rows.csv"
    out_missing = out_meta / "missing_parent_segments.csv"

    pseudo_df.to_csv(out_parent, index=False)
    out_df.to_csv(out_feature, index=False)
    zero_active.to_csv(out_zero, index=False)
    pd.DataFrame({"parent_clip_id": missing_parent_segments}).to_csv(out_missing, index=False)

    # Copy labels json payload.
    with labels_json.open("r", encoding="utf-8") as f:
        labels_payload = json.load(f)
    labels_payload["pseudo_dataset"] = True
    labels_payload["labels"] = LABELS

    out_labels.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    summary = {
        "generated_at": now_iso(),
        "pseudo_parent_csv": str(pseudo_csv),
        "base_feature_manifest": str(base_feature_manifest),
        "out_root": str(out_root),
        "parent_rows_input": int(len(pd.read_csv(pseudo_csv))),
        "parent_rows_training_ready": int(len(pseudo_df)),
        "zero_active_parent_rows": int(len(zero_active)),
        "segment_feature_rows": int(len(out_df)),
        "missing_parent_segments": int(len(missing_parent_segments)),
        "routing_counts": pseudo_df["routing_decision"].value_counts().to_dict() if "routing_decision" in pseudo_df.columns else {},
        "parent_label_counts": {lab: int(pseudo_df[lab].sum()) for lab in LABELS},
        "segment_label_counts": {lab: int(out_df[lab].sum()) for lab in LABELS} if len(out_df) else {},
        "outputs": {
            "pseudo_parent_manifest": str(out_parent),
            "feature_manifest": str(out_feature),
            "labels_json": str(out_labels),
            "summary_json": str(out_summary),
            "summary_md": str(out_summary_md),
            "zero_active_rows": str(out_zero),
            "missing_parent_segments": str(out_missing),
        },
        "important_note": "This creates pseudo-label feature manifests only. It does not move/copy/delete feature files.",
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# TATA v0.6 Pseudo Dataset Summary")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_at']}`")
    lines.append("")
    lines.append(f"- Parent rows input: `{summary['parent_rows_input']}`")
    lines.append(f"- Parent rows training-ready: `{summary['parent_rows_training_ready']}`")
    lines.append(f"- Segment feature rows: `{summary['segment_feature_rows']}`")
    lines.append(f"- Zero-active parent rows: `{summary['zero_active_parent_rows']}`")
    lines.append(f"- Missing parent segment groups: `{summary['missing_parent_segments']}`")
    lines.append("")
    lines.append("## Routing Counts")
    lines.append("")
    lines.append("| Routing decision | Count |")
    lines.append("|---|---:|")
    for k, v in summary["routing_counts"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Parent Label Counts")
    lines.append("")
    lines.append("| Label | Count |")
    lines.append("|---|---:|")
    for lab in LABELS:
        lines.append(f"| `{lab}` | {summary['parent_label_counts'].get(lab, 0)} |")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for k, v in summary["outputs"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("Note: Feature files are reused from the existing scratch feature cache.")

    out_summary_md.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("Pseudo dataset manifest created")
    print("-" * 90)
    print(f"Output root:          {out_root}")
    print(f"Parent rows:          {len(pseudo_df)}")
    print(f"Segment feature rows: {len(out_df)}")
    print(f"Zero-active rows:     {len(zero_active)}")
    print(f"Missing parents:      {len(missing_parent_segments)}")
    print("")
    print("Parent label counts:")
    for lab in LABELS:
        print(f"  {lab:28s}: {int(pseudo_df[lab].sum())}")
    print("")
    print(f"Feature manifest: {out_feature}")
    print(f"Summary:          {out_summary_md}")


if __name__ == "__main__":
    main()