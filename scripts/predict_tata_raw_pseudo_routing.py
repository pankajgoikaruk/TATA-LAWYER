# scripts/predict_tata_raw_pseudo_routing.py

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_factory import build_audio_exit_net


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

TARGET_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
]

EVENT_LABELS = [
    "music_present",
    "audience_reaction_present",
    "silence_present",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def parse_tap_blocks(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


def load_model(run_dir: Path, device: str):
    config = load_json(run_dir / "config_used.json")

    labels = config.get("labels", LABELS)
    if labels != LABELS:
        raise RuntimeError(
            "Label mismatch.\n"
            f"Expected: {LABELS}\n"
            f"Config:   {labels}"
        )

    tap_blocks = parse_tap_blocks(config.get("tap_blocks", [1, 3]))
    n_mels = int(config.get("n_mels", 64))

    model_cfg = {
        "exit_hint": {
            "enable": False,
            "dim": 8,
            "source": "probs",
            "detach": True,
            "use_stats": True,
        }
    }

    model = build_audio_exit_net(
        num_classes=len(LABELS),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    ckpt_path = run_dir / "ckpt" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state)
    model.eval()

    return model, config


def load_feature(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)

    if arr.ndim != 2:
        raise RuntimeError(f"Expected [n_mels, T], got {arr.shape}: {path}")

    return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_segments(
    *,
    model,
    manifest: pd.DataFrame,
    features_root: Path,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    rows = []

    n = len(manifest)

    for start in range(0, n, batch_size):
        batch_df = manifest.iloc[start:start + batch_size].copy()

        xs = []
        for _, row in batch_df.iterrows():
            feat_rel = str(row["feat_relpath"]).replace("\\", "/")
            feat_path = features_root / Path(feat_rel)

            if not feat_path.exists():
                raise FileNotFoundError(f"Feature file not found: {feat_path}")

            xs.append(load_feature(feat_path))

        x = torch.cat(xs, dim=0).to(device)

        logits_list = model(x)
        probs = torch.sigmoid(logits_list[-1]).detach().cpu().numpy()

        for i, (_, row) in enumerate(batch_df.iterrows()):
            out = row.to_dict()

            for j, lab in enumerate(LABELS):
                out[f"segment_prob_{lab}"] = float(probs[i, j])

            rows.append(out)

        if (start + batch_size) % 1000 < batch_size:
            print(f"[predict] processed {min(start + batch_size, n)}/{n}")

    return pd.DataFrame(rows)


def aggregate_parent_predictions(segment_pred: pd.DataFrame) -> pd.DataFrame:
    out_rows = []

    prob_cols = [f"segment_prob_{lab}" for lab in LABELS]

    for parent_id, group in segment_pred.groupby("parent_clip_id", dropna=False):
        row = {
            "parent_clip_id": str(parent_id),
            "num_segments": int(len(group)),
        }

        for col in [
            "source_file",
            "source_path",
            "source_rel_path",
            "source_class_dir",
        ]:
            if col in group.columns:
                row[col] = group[col].iloc[0]

        # Parent-level probability = max over 1-sec segments.
        # This answers: did this label appear anywhere in the clip?
        for lab in LABELS:
            p = group[f"segment_prob_{lab}"].astype(float).values
            row[f"parent_prob_{lab}"] = float(np.max(p))
            row[f"parent_mean_prob_{lab}"] = float(np.mean(p))

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def apply_thresholds(parent_df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    df = parent_df.copy()

    pred_labels = []
    pred_counts = []

    for lab in LABELS:
        prob_col = f"parent_prob_{lab}"
        pred_col = f"parent_pred_{lab}"

        th = float(thresholds[lab])
        df[pred_col] = (df[prob_col].astype(float) >= th).astype(int)

        # Simple editable label column too.
        df[lab] = df[pred_col].astype(int)

    for _, row in df.iterrows():
        active = [lab for lab in LABELS if int(row[f"parent_pred_{lab}"]) == 1]
        pred_labels.append("|".join(active))
        pred_counts.append(len(active))

    df["pred_labels"] = pred_labels
    df["labels"] = pred_labels
    df["num_active_labels"] = pred_counts

    return df


def route_parent_rows(df: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    out = df.copy()

    decisions = []
    reasons = []

    for _, row in out.iterrows():
        target_active = [lab for lab in TARGET_LABELS if int(row[f"parent_pred_{lab}"]) == 1]
        event_active = [lab for lab in EVENT_LABELS if int(row[f"parent_pred_{lab}"]) == 1]
        other_active = int(row["parent_pred_other_speaker_present"]) == 1

        max_target_prob = max(float(row[f"parent_prob_{lab}"]) for lab in TARGET_LABELS)
        max_event_prob = max(float(row[f"parent_prob_{lab}"]) for lab in EVENT_LABELS)
        other_prob = float(row["parent_prob_other_speaker_present"])

        if len(target_active) == 1 and not other_active and len(event_active) == 0 and max_target_prob >= 0.75:
            decisions.append("accepted")
            reasons.append("single_target_high_confidence_no_event_no_other")

        elif len(target_active) == 1 and not other_active and len(event_active) > 0 and max_target_prob >= 0.65:
            decisions.append("accepted_with_warning")
            reasons.append("single_target_with_event_or_background")

        elif len(target_active) == 0 and (max_event_prob >= 0.70 or other_prob >= 0.70):
            decisions.append("rejected")
            reasons.append("no_target_high_confidence_event_or_other")

        else:
            decisions.append("needs_review")
            reasons.append("ambiguous_or_low_confidence_or_multi_target_or_other_speaker")

    out["routing_mode"] = mode_name
    out["routing_decision"] = decisions
    out["routing_reason"] = reasons
    out["review_status"] = "pseudo_routed"
    out["notes"] = ""

    return out


def write_routing_bundle(df: pd.DataFrame, out_dir: Path, mode_name: str) -> dict:
    mode_dir = out_dir / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)

    full_path = mode_dir / f"{mode_name}_parent_predictions_all.csv"
    df.to_csv(full_path, index=False)

    outputs = {
        "all": str(full_path),
    }

    for decision in ["accepted", "accepted_with_warning", "needs_review", "rejected"]:
        part = df[df["routing_decision"] == decision].copy()
        path = mode_dir / f"{mode_name}_{decision}.csv"
        part.to_csv(path, index=False)
        outputs[decision] = str(path)

    summary = {
        "mode": mode_name,
        "rows": int(len(df)),
        "routing_counts": df["routing_decision"].value_counts().to_dict(),
        "label_counts": {lab: int(df[lab].sum()) for lab in LABELS},
        "outputs": outputs,
    }

    save_json(summary, mode_dir / f"{mode_name}_routing_summary.json")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict TATA v0.6 labels on raw pseudo-pool features and create routing CSVs."
    )

    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--features_manifest", required=True)
    parser.add_argument("--features_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    features_manifest = Path(args.features_manifest)
    features_root = Path(args.features_root)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("Predicting TATA v0.6 on raw pseudo-pool")
    print("-" * 90)
    print(f"Run dir:           {run_dir}")
    print(f"Features manifest: {features_manifest}")
    print(f"Features root:     {features_root}")
    print(f"Output dir:        {out_dir}")
    print(f"Device:            {args.device}")
    print("-" * 90)

    model, config = load_model(run_dir, args.device)

    manifest = pd.read_csv(features_manifest)

    segment_pred = predict_segments(
        model=model,
        manifest=manifest,
        features_root=features_root,
        device=args.device,
        batch_size=args.batch_size,
    )

    segment_pred_path = out_dir / "raw_segment_predictions.csv"
    segment_pred.to_csv(segment_pred_path, index=False)

    parent_probs = aggregate_parent_predictions(segment_pred)
    parent_probs_path = out_dir / "raw_parent_probabilities.csv"
    parent_probs.to_csv(parent_probs_path, index=False)

    fixed_thresholds = {lab: 0.50 for lab in LABELS}

    # Conservative hybrid thresholds.
    # Lower other_speaker/silence/Nick threshold catches risky clips and routes them to review.
    hybrid_thresholds = {
        "Brene_Brown": 0.50,
        "Eckhart_Tolle": 0.50,
        "Eric_Thomas": 0.50,
        "Gary_Vee": 0.50,
        "Jay_Shetty": 0.50,
        "Nick_Vujicic": 0.35,
        "other_speaker_present": 0.35,
        "music_present": 0.50,
        "audience_reaction_present": 0.50,
        "silence_present": 0.35,
    }

    fixed_df = route_parent_rows(
        apply_thresholds(parent_probs, fixed_thresholds),
        mode_name="fixed_0p5",
    )

    hybrid_df = route_parent_rows(
        apply_thresholds(parent_probs, hybrid_thresholds),
        mode_name="hybrid",
    )

    fixed_summary = write_routing_bundle(fixed_df, out_dir, "fixed_0p5")
    hybrid_summary = write_routing_bundle(hybrid_df, out_dir, "hybrid")

    final_summary = {
        "generated_at": now_iso(),
        "run_dir": str(run_dir),
        "features_manifest": str(features_manifest),
        "features_root": str(features_root),
        "out_dir": str(out_dir),
        "segment_rows": int(len(segment_pred)),
        "parent_rows": int(len(parent_probs)),
        "segment_predictions": str(segment_pred_path),
        "parent_probabilities": str(parent_probs_path),
        "fixed_0p5": fixed_summary,
        "hybrid": hybrid_summary,
        "important_rule": "These are pseudo labels for raw_pseudo_pool only. Do not include final holdout clips.",
    }

    save_json(final_summary, out_dir / "raw_tata_pseudo_routing_summary.json")

    print("")
    print("Raw TATA pseudo-routing complete")
    print("-" * 90)
    print(f"Segment rows: {len(segment_pred)}")
    print(f"Parent rows:  {len(parent_probs)}")
    print("")
    print("Fixed 0.5 routing counts:")
    print(pd.Series(fixed_summary["routing_counts"]).to_string())
    print("")
    print("Hybrid routing counts:")
    print(pd.Series(hybrid_summary["routing_counts"]).to_string())
    print("")
    print(f"Summary: {out_dir / 'raw_tata_pseudo_routing_summary.json'}")


if __name__ == "__main__":
    main()
