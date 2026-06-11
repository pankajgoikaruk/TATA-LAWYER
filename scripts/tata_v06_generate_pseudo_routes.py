# scripts/tata_v06_generate_pseudo_routes.py

"""
Generate conservative pseudo-label + routing CSVs for TATA v0.6.

Purpose:
- Use trained v0.6 sigmoid multi-label model.
- Predict labels from final exit only.
- Aggregate 1-second segment predictions to parent clip level.
- Produce pseudo-label and routing manifests.

Conservative routing:
- accepted: target speaker only, high confidence, no risky labels.
- accepted_with_warning: target speaker + event/background labels, high confidence.
- needs_review: target speaker + other speaker, uncertain target, low confidence, conflicting speakers.
- rejected: no target speaker and strong silence/event-only/non-target evidence.

This script does not move/delete audio.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_factory import build_audio_exit_net


TARGET_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
]

RISK_LABELS = [
    "other_speaker_present",
]

EVENT_LABELS = [
    "music_present",
    "audience_reaction_present",
    "silence_present",
]


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        return str(x)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def parse_tap_blocks(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def load_model(run_dir: Path, device: str):
    cfg = load_json(run_dir / "config_used.json")

    labels = [str(x) for x in cfg["labels"]]
    n_mels = int(cfg.get("n_mels", 64))
    tap_blocks = parse_tap_blocks(cfg["tap_blocks"])

    model_cfg = cfg.get("exit_hint", None)
    model_cfg = {
        "exit_hint": model_cfg if isinstance(model_cfg, dict) else {
            "enable": False,
            "dim": 8,
            "source": "probs",
            "detach": True,
            "use_stats": True,
        }
    }

    model = build_audio_exit_net(
        num_classes=len(labels),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    ckpt = run_dir / "ckpt" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)

    model.load_state_dict(state)
    model.eval()

    return model, cfg, labels


def load_thresholds(
    run_dir: Path,
    labels: list[str],
    mode: str,
    weak_labels_use_tuned: list[str],
) -> dict[str, float]:
    fixed = {lab: 0.5 for lab in labels}

    threshold_path = run_dir / "threshold_tuning" / "multilabel_thresholds.json"
    tuned = {}

    if threshold_path.exists():
        payload = load_json(threshold_path)
        tuned = {str(k): float(v) for k, v in payload.get("thresholds", {}).items()}

    if mode == "fixed":
        return fixed

    if mode == "tuned":
        if not tuned:
            raise FileNotFoundError(f"Tuned threshold file not found: {threshold_path}")
        return {lab: tuned.get(lab, 0.5) for lab in labels}

    if mode == "hybrid":
        out = fixed.copy()
        for lab in weak_labels_use_tuned:
            if lab in tuned:
                out[lab] = tuned[lab]
        return out

    raise ValueError(f"Unknown threshold mode: {mode}")


def load_feature(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected [n_mels, T], got {arr.shape}: {path}")
    return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_segments(
    model,
    df: pd.DataFrame,
    features_root: Path,
    labels: list[str],
    thresholds: dict[str, float],
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    rows = []
    label_thresholds = np.asarray([thresholds[lab] for lab in labels], dtype=np.float32)

    total = len(df)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]

        xs = []
        valid_rows = []

        for _, row in batch.iterrows():
            feat_path = features_root / Path(str(row["feat_relpath"]).replace("\\", "/"))
            if not feat_path.exists():
                continue

            xs.append(load_feature(feat_path))
            valid_rows.append(row)

        if not xs:
            continue

        x = torch.cat(xs, dim=0).to(device)
        logits_list = model(x)
        probs = torch.sigmoid(logits_list[-1]).detach().cpu().numpy()
        preds = (probs >= label_thresholds.reshape(1, -1)).astype(int)

        for i, row in enumerate(valid_rows):
            out = row.to_dict()

            pred_labels = []
            true_labels = []

            for j, lab in enumerate(labels):
                out[f"prob_{lab}"] = float(probs[i, j])
                out[f"pred_{lab}"] = int(preds[i, j])

                if int(preds[i, j]) == 1:
                    pred_labels.append(lab)

                if lab in row and int(row.get(lab, 0)) == 1:
                    true_labels.append(lab)

            out["predicted_labels"] = "|".join(pred_labels)
            out["true_labels"] = "|".join(true_labels)
            out["num_predicted_labels"] = len(pred_labels)

            rows.append(out)

        if (start + batch_size) % (batch_size * 20) == 0:
            print(f"[predict] processed {min(end, total)}/{total}")

    return pd.DataFrame(rows)


def aggregate_parent_predictions(
    seg_df: pd.DataFrame,
    labels: list[str],
    thresholds: dict[str, float],
    min_positive_ratio: float,
) -> pd.DataFrame:
    if "parent_clip_id" not in seg_df.columns:
        raise RuntimeError("segment manifest must contain parent_clip_id")

    parent_rows = []

    for parent_id, g in seg_df.groupby("parent_clip_id", sort=True):
        row = {
            "parent_clip_id": parent_id,
            "num_segments": int(len(g)),
        }

        first = g.iloc[0]
        for col in [
            "source_path",
            "source_file",
            "source_rel_path",
            "primary_label",
            "split",
        ]:
            if col in g.columns:
                row[col] = first.get(col, "")

        predicted_parent_labels = []
        true_parent_labels = []

        for lab in labels:
            prob_col = f"prob_{lab}"
            pred_col = f"pred_{lab}"

            max_prob = float(g[prob_col].max())
            mean_prob = float(g[prob_col].mean())
            positive_ratio = float(g[pred_col].mean())
            threshold = float(thresholds[lab])

            # Conservative parent-level rule:
            # label active if any segment is very strong OR enough segments are above threshold.
            active = int((max_prob >= max(threshold, 0.70)) or (positive_ratio >= min_positive_ratio))

            row[f"parent_prob_max_{lab}"] = max_prob
            row[f"parent_prob_mean_{lab}"] = mean_prob
            row[f"parent_positive_ratio_{lab}"] = positive_ratio
            row[f"parent_pred_{lab}"] = active

            if active:
                predicted_parent_labels.append(lab)

            if lab in g.columns and int(g[lab].max()) == 1:
                true_parent_labels.append(lab)

        row["predicted_labels"] = "|".join(predicted_parent_labels)
        row["true_labels_if_available"] = "|".join(true_parent_labels)
        row["num_predicted_labels"] = len(predicted_parent_labels)

        parent_rows.append(row)

    return pd.DataFrame(parent_rows)


def route_parent_row(
    row: pd.Series,
    labels: list[str],
    target_high: float,
    target_min: float,
    risk_high: float,
    event_high: float,
) -> tuple[str, str]:
    target_scores = {
        lab: float(row.get(f"parent_prob_max_{lab}", 0.0))
        for lab in TARGET_LABELS
        if lab in labels
    }

    event_scores = {
        lab: float(row.get(f"parent_prob_max_{lab}", 0.0))
        for lab in EVENT_LABELS
        if lab in labels
    }

    risk_scores = {
        lab: float(row.get(f"parent_prob_max_{lab}", 0.0))
        for lab in RISK_LABELS
        if lab in labels
    }

    active_targets = [
        lab for lab in TARGET_LABELS
        if lab in labels and int(row.get(f"parent_pred_{lab}", 0)) == 1
    ]

    active_events = [
        lab for lab in EVENT_LABELS
        if lab in labels and int(row.get(f"parent_pred_{lab}", 0)) == 1
    ]

    active_risks = [
        lab for lab in RISK_LABELS
        if lab in labels and int(row.get(f"parent_pred_{lab}", 0)) == 1
    ]

    best_target = max(target_scores, key=target_scores.get) if target_scores else ""
    best_target_score = target_scores.get(best_target, 0.0)

    max_risk = max(risk_scores.values()) if risk_scores else 0.0
    max_event = max(event_scores.values()) if event_scores else 0.0

    # Corrupt/empty/no prediction case.
    if not active_targets and not active_events and not active_risks:
        return "needs_review", "no active pseudo-label"

    # No target speaker.
    if not active_targets:
        if max_event >= event_high and max_risk < risk_high:
            return "rejected", "event-only/no target speaker"
        if max_risk >= risk_high:
            return "needs_review", "other speaker/no target speaker"
        return "needs_review", "no confident target speaker"

    # Multiple target speakers predicted means overlap/confusion.
    if len(active_targets) > 1:
        return "needs_review", "multiple target speakers predicted"

    # One target but too low confidence.
    if best_target_score < target_min:
        return "needs_review", "target speaker below minimum confidence"

    # Target + other speaker is risky.
    if active_risks or max_risk >= risk_high:
        return "needs_review", "target speaker plus other_speaker_present"

    # High-confidence clean target.
    if best_target_score >= target_high and not active_events:
        return "accepted", "high-confidence target speaker only"

    # High-confidence target with event/background.
    if best_target_score >= target_high and active_events:
        return "accepted_with_warning", "target speaker plus event/background"

    return "needs_review", "target speaker present but not high-confidence"


def apply_routing(
    parent_df: pd.DataFrame,
    labels: list[str],
    target_high: float,
    target_min: float,
    risk_high: float,
    event_high: float,
) -> pd.DataFrame:
    decisions = []
    reasons = []

    for _, row in parent_df.iterrows():
        d, r = route_parent_row(
            row=row,
            labels=labels,
            target_high=target_high,
            target_min=target_min,
            risk_high=risk_high,
            event_high=event_high,
        )
        decisions.append(d)
        reasons.append(r)

    out = parent_df.copy()
    out["routing_decision"] = decisions
    out["routing_reason"] = reasons
    return out


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = []
    lines.append("# TATA v0.6 Conservative Pseudo-Label Routing Summary")
    lines.append("")
    lines.append("## Decision counts")
    lines.append("")
    lines.append("| Decision | Count |")
    lines.append("|---|---:|")
    for k, v in summary["decision_counts"].items():
        lines.append(f"| `{k}` | {v} |")

    lines.append("")
    lines.append("## Label counts")
    lines.append("")
    lines.append("| Label | Parent positives |")
    lines.append("|---|---:|")
    for k, v in summary["parent_pred_label_counts"].items():
        lines.append(f"| `{k}` | {v} |")

    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for k, v in summary["outputs"].items():
        lines.append(f"- `{k}`: `{v}`")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate conservative TATA v0.6 pseudo-label routing CSVs."
    )

    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features_root", required=True)
    parser.add_argument("--labels_json", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--split", default="all", help="all/train/val/test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--threshold_mode",
        choices=["fixed", "tuned", "hybrid"],
        default="fixed",
    )

    parser.add_argument(
        "--weak_labels_use_tuned",
        default="Nick_Vujicic,other_speaker_present,silence_present",
        help="Comma-separated labels that use tuned threshold in hybrid mode.",
    )

    parser.add_argument("--min_positive_ratio", type=float, default=0.20)
    parser.add_argument("--target_high", type=float, default=0.80)
    parser.add_argument("--target_min", type=float, default=0.55)
    parser.add_argument("--risk_high", type=float, default=0.60)
    parser.add_argument("--event_high", type=float, default=0.70)

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest = Path(args.manifest).resolve()
    features_root = Path(args.features_root).resolve()
    labels_json = Path(args.labels_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    labels_payload = load_json(labels_json)
    labels = [str(x) for x in labels_payload["labels"]]

    model, cfg, cfg_labels = load_model(run_dir, device)
    if list(labels) != list(cfg_labels):
        raise RuntimeError(f"Label mismatch:\nlabels_json={labels}\nconfig={cfg_labels}")

    weak_labels = [
        x.strip() for x in args.weak_labels_use_tuned.split(",")
        if x.strip()
    ]

    thresholds = load_thresholds(
        run_dir=run_dir,
        labels=labels,
        mode=args.threshold_mode,
        weak_labels_use_tuned=weak_labels,
    )

    df = pd.read_csv(manifest)

    if args.split != "all":
        df = df[df["split"].astype(str) == str(args.split)].reset_index(drop=True)

    print("")
    print("TATA v0.6 conservative pseudo-label routing")
    print("-" * 90)
    print(f"Run dir:        {run_dir}")
    print(f"Manifest:       {manifest}")
    print(f"Features root:  {features_root}")
    print(f"Labels JSON:    {labels_json}")
    print(f"Output dir:     {out_dir}")
    print(f"Split:          {args.split}")
    print(f"Rows:           {len(df)}")
    print(f"Device:         {device}")
    print(f"Threshold mode: {args.threshold_mode}")
    print(f"Thresholds:     {thresholds}")
    print("-" * 90)

    seg_pred_df = predict_segments(
        model=model,
        df=df,
        features_root=features_root,
        labels=labels,
        thresholds=thresholds,
        device=device,
        batch_size=args.batch_size,
    )

    parent_df = aggregate_parent_predictions(
        seg_df=seg_pred_df,
        labels=labels,
        thresholds=thresholds,
        min_positive_ratio=float(args.min_positive_ratio),
    )

    routing_df = apply_routing(
        parent_df=parent_df,
        labels=labels,
        target_high=float(args.target_high),
        target_min=float(args.target_min),
        risk_high=float(args.risk_high),
        event_high=float(args.event_high),
    )

    seg_path = out_dir / "per_segment_predictions.csv"
    routing_path = out_dir / "routing_manifest.csv"
    pseudo_path = out_dir / "pseudo_label_manifest.csv"

    seg_pred_df.to_csv(seg_path, index=False)
    routing_df.to_csv(routing_path, index=False)

    # Pseudo-label manifest is the same routing table, but with simpler name for next-stage use.
    routing_df.to_csv(pseudo_path, index=False)

    decision_paths = {}
    for decision in ["accepted", "accepted_with_warning", "needs_review", "rejected"]:
        subset = routing_df[routing_df["routing_decision"] == decision].copy()
        path = out_dir / f"{decision}_manifest.csv"
        subset.to_csv(path, index=False)
        decision_paths[decision] = str(path)

    decision_counts = routing_df["routing_decision"].value_counts().to_dict()
    reason_counts = routing_df["routing_reason"].value_counts().to_dict()

    parent_label_counts = {}
    for lab in labels:
        parent_label_counts[lab] = int(routing_df[f"parent_pred_{lab}"].sum())

    summary = {
        "run_dir": str(run_dir),
        "manifest": str(manifest),
        "features_root": str(features_root),
        "labels_json": str(labels_json),
        "split": args.split,
        "threshold_mode": args.threshold_mode,
        "thresholds": thresholds,
        "routing_params": {
            "min_positive_ratio": float(args.min_positive_ratio),
            "target_high": float(args.target_high),
            "target_min": float(args.target_min),
            "risk_high": float(args.risk_high),
            "event_high": float(args.event_high),
        },
        "num_segments": int(len(seg_pred_df)),
        "num_parent_clips": int(len(routing_df)),
        "decision_counts": {str(k): int(v) for k, v in decision_counts.items()},
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "parent_pred_label_counts": parent_label_counts,
        "outputs": {
            "per_segment_predictions": str(seg_path),
            "routing_manifest": str(routing_path),
            "pseudo_label_manifest": str(pseudo_path),
            **decision_paths,
            "summary_json": str(out_dir / "routing_summary.json"),
            "summary_md": str(out_dir / "routing_summary.md"),
        },
    }

    save_json(summary, out_dir / "routing_summary.json")
    write_markdown_summary(out_dir / "routing_summary.md", summary)

    print("")
    print("Routing complete")
    print("-" * 90)
    print("Decision counts:")
    for k, v in summary["decision_counts"].items():
        print(f"  {k:24s}: {v}")

    print("")
    print(f"Routing manifest: {routing_path}")
    print(f"Summary:          {out_dir / 'routing_summary.md'}")


if __name__ == "__main__":
    main()
