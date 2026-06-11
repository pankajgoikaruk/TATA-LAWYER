# scripts/analyse_tata_error_clips.py
"""
Analyse exact TATA error clips/segments from a trained multi-label run.

Outputs:
  <run_dir>/error_analysis/per_segment_predictions.csv
  <run_dir>/error_analysis/parent_clip_error_summary.csv
  <run_dir>/error_analysis/label_error_summary.csv
  <run_dir>/error_analysis/combo_error_summary.csv
  <run_dir>/error_analysis/synthetic_priority_candidates.csv
  <run_dir>/error_analysis/error_analysis_summary.json

Example:
  python scripts/analyse_tata_error_clips.py `
    --run_dir human_talk_workspace\tata_2\runs\tata_2_3exit_weakclip_20260530_121030 `
    --manifest human_talk_workspace\tata_2\feature_cache\metadata\multilabel_features_manifest.csv `
    --features_root human_talk_workspace\tata_2\feature_cache\features `
    --labels_json human_talk_workspace\tata_2\segment_cache\metadata\tata_labels.json `
    --split test `
    --threshold_mode tuned `
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_factory import build_audio_exit_net


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_labels(labels_json: Path) -> list[str]:
    payload = load_json(labels_json)
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise RuntimeError(f"Invalid labels json: {labels_json}")
    return [str(x) for x in labels]


def find_checkpoint(run_dir: Path) -> Path:
    for rel in ["ckpt/best.pt", "ckpt/last_loaded_best.pt"]:
        p = run_dir / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint found in {run_dir}/ckpt")


def load_thresholds(run_dir: Path, labels: list[str], threshold_mode: str, fixed_threshold: float) -> np.ndarray:
    if threshold_mode == "fixed":
        return np.array([fixed_threshold] * len(labels), dtype=np.float32)

    th_path = run_dir / "threshold_tuning" / "multilabel_thresholds.json"
    if not th_path.exists():
        raise FileNotFoundError(f"Tuned threshold file not found: {th_path}")

    payload = load_json(th_path)
    th = payload.get("thresholds", {})
    return np.array([float(th.get(label, fixed_threshold)) for label in labels], dtype=np.float32)


def make_model(config: dict, num_labels: int, device: str):
    tap_blocks = tuple(int(x) for x in config.get("tap_blocks", [1, 3]))
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
        num_classes=num_labels,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)
    return model


def active_label_names(row, labels: list[str]) -> list[str]:
    return [lab for lab in labels if int(row.get(lab, 0)) == 1]


def label_list_from_bits(bits: np.ndarray, labels: list[str]) -> list[str]:
    return [lab for lab, v in zip(labels, bits.astype(int).tolist()) if v == 1]


def ensure_path(path_text: str, features_root: Path) -> Path:
    p = Path(str(path_text))
    if p.is_absolute() and p.exists():
        return p
    p2 = features_root / Path(str(path_text).replace("\\", "/"))
    return p2


def main():
    parser = argparse.ArgumentParser(description="Analyse exact TATA multi-label prediction errors.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--features_root", default="")
    parser.add_argument("--labels_json", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold_mode", choices=["fixed", "tuned"], default="tuned")
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config = load_json(run_dir / "config_used.json")

    manifest = Path(args.manifest) if args.manifest else Path(config["manifest"])
    features_root = Path(args.features_root) if args.features_root else Path(config["features_root"])
    labels_json = Path(args.labels_json) if args.labels_json else Path(config["labels_json"])

    labels = load_labels(labels_json)
    thresholds = load_thresholds(run_dir, labels, args.threshold_mode, args.fixed_threshold)

    df = pd.read_csv(manifest)
    df = df[df["split"].astype(str) == str(args.split)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"No rows found for split={args.split}")

    device = args.device
    model = make_model(config, num_labels=len(labels), device=device)
    ckpt = find_checkpoint(run_dir)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    rows = []

    with torch.no_grad():
        for start in range(0, len(df), args.batch_size):
            part = df.iloc[start:start + args.batch_size]
            xs = []
            valid_indices = []

            for i, row in part.iterrows():
                feat_path = ensure_path(row["feat_relpath"], features_root)
                if not feat_path.exists():
                    feat_path = Path(str(row.get("feature_path", "")))
                if not feat_path.exists():
                    raise FileNotFoundError(f"Feature not found for row {i}: {row.get('feat_relpath')}")
                feat = np.load(feat_path).astype(np.float32)
                xs.append(torch.from_numpy(feat).float().unsqueeze(0))
                valid_indices.append(i)

            x = torch.stack(xs, dim=0).to(device)  # [B,1,n_mels,T]
            logits_list = model(x)
            probs = torch.sigmoid(logits_list[-1]).detach().cpu().numpy()

            for local_idx, i in enumerate(valid_indices):
                row = df.iloc[i]
                true_bits = row[labels].astype(int).values.astype(int)
                prob = probs[local_idx]
                pred_bits = (prob >= thresholds).astype(int)

                missed = [lab for lab, yt, yp in zip(labels, true_bits, pred_bits) if yt == 1 and yp == 0]
                extra = [lab for lab, yt, yp in zip(labels, true_bits, pred_bits) if yt == 0 and yp == 1]
                true_labels = label_list_from_bits(true_bits, labels)
                pred_labels = label_list_from_bits(pred_bits, labels)

                out = {
                    "sample_id": row.get("sample_id", ""),
                    "parent_clip_id": row.get("parent_clip_id", ""),
                    "split": row.get("split", ""),
                    "primary_label": row.get("primary_label", ""),
                    "source_file": row.get("source_file", ""),
                    "source_path": row.get("source_path", ""),
                    "source_rel_path": row.get("source_rel_path", ""),
                    "start_sec": row.get("start_sec", ""),
                    "end_sec": row.get("end_sec", ""),
                    "true_labels": "|".join(true_labels),
                    "pred_labels": "|".join(pred_labels),
                    "missed_labels": "|".join(missed),
                    "extra_labels": "|".join(extra),
                    "num_missed": len(missed),
                    "num_extra": len(extra),
                    "exact_match": int(len(missed) == 0 and len(extra) == 0),
                    "failure_type": "correct" if not missed and not extra else (
                        "false_negative_only" if missed and not extra else (
                            "false_positive_only" if extra and not missed else "mixed_error"
                        )
                    ),
                }
                for lab, yt, yp, pr, th in zip(labels, true_bits, pred_bits, prob, thresholds):
                    out[f"true_{lab}"] = int(yt)
                    out[f"pred_{lab}"] = int(yp)
                    out[f"prob_{lab}"] = float(pr)
                    out[f"threshold_{lab}"] = float(th)
                rows.append(out)

    pred_df = pd.DataFrame(rows)
    out_dir = run_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "per_segment_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Label error summary.
    label_rows = []
    for lab in labels:
        yt = pred_df[f"true_{lab}"].astype(int)
        yp = pred_df[f"pred_{lab}"].astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        support = tp + fn
        pred_pos = tp + fp
        precision = tp / max(pred_pos, 1)
        recall = tp / max(support, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        label_rows.append({
            "label": lab,
            "support": support,
            "predicted_positive": pred_pos,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fn_rate": fn / max(support, 1),
            "fp_per_true_negative": fp / max(fp + tn, 1),
            "threshold": float(thresholds[labels.index(lab)]),
        })
    label_df = pd.DataFrame(label_rows).sort_values(["f1", "fn", "fp"], ascending=[True, False, False])
    label_df.to_csv(out_dir / "label_error_summary.csv", index=False)

    # Parent clip summary.
    parent_rows = []
    for parent, g in pred_df.groupby("parent_clip_id", dropna=False):
        miss_counter = Counter()
        extra_counter = Counter()
        for items in g["missed_labels"].fillna(""):
            miss_counter.update([x for x in str(items).split("|") if x])
        for items in g["extra_labels"].fillna(""):
            extra_counter.update([x for x in str(items).split("|") if x])
        parent_rows.append({
            "parent_clip_id": parent,
            "segments": len(g),
            "exact_match_segments": int(g["exact_match"].sum()),
            "error_segments": int((g["exact_match"] == 0).sum()),
            "error_rate": float((g["exact_match"] == 0).mean()),
            "true_labels": g["true_labels"].iloc[0],
            "primary_label": g["primary_label"].iloc[0],
            "source_file": g["source_file"].iloc[0],
            "source_path": g["source_path"].iloc[0],
            "top_missed_labels": "|".join([f"{k}:{v}" for k, v in miss_counter.most_common(5)]),
            "top_extra_labels": "|".join([f"{k}:{v}" for k, v in extra_counter.most_common(5)]),
        })
    parent_df = pd.DataFrame(parent_rows).sort_values(["error_rate", "error_segments"], ascending=[False, False])
    parent_df.to_csv(out_dir / "parent_clip_error_summary.csv", index=False)

    # True-combination summary.
    combo_rows = []
    for combo, g in pred_df.groupby("true_labels", dropna=False):
        miss_counter = Counter()
        extra_counter = Counter()
        for items in g["missed_labels"].fillna(""):
            miss_counter.update([x for x in str(items).split("|") if x])
        for items in g["extra_labels"].fillna(""):
            extra_counter.update([x for x in str(items).split("|") if x])
        combo_rows.append({
            "true_combo": combo,
            "segments": len(g),
            "exact_match_rate": float(g["exact_match"].mean()),
            "error_segments": int((g["exact_match"] == 0).sum()),
            "top_missed_labels": "|".join([f"{k}:{v}" for k, v in miss_counter.most_common(5)]),
            "top_extra_labels": "|".join([f"{k}:{v}" for k, v in extra_counter.most_common(5)]),
        })
    combo_df = pd.DataFrame(combo_rows).sort_values(["exact_match_rate", "segments"], ascending=[True, False])
    combo_df.to_csv(out_dir / "combo_error_summary.csv", index=False)

    # Synthetic priority candidates: parent clips with highest errors.
    cand = parent_df[parent_df["error_segments"] > 0].copy()
    cand.to_csv(out_dir / "synthetic_priority_candidates.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "manifest": str(manifest),
        "features_root": str(features_root),
        "labels_json": str(labels_json),
        "split": args.split,
        "threshold_mode": args.threshold_mode,
        "rows": int(len(pred_df)),
        "exact_match_segments": int(pred_df["exact_match"].sum()),
        "error_segments": int((pred_df["exact_match"] == 0).sum()),
        "outputs": {
            "per_segment_predictions": str(pred_path),
            "label_error_summary": str(out_dir / "label_error_summary.csv"),
            "parent_clip_error_summary": str(out_dir / "parent_clip_error_summary.csv"),
            "combo_error_summary": str(out_dir / "combo_error_summary.csv"),
            "synthetic_priority_candidates": str(out_dir / "synthetic_priority_candidates.csv"),
        },
    }
    (out_dir / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTATA error analysis completed")
    print("-" * 90)
    for k, v in summary["outputs"].items():
        print(f"{k}: {v}")
    print(f"error_segments: {summary['error_segments']} / {summary['rows']}")


if __name__ == "__main__":
    main()
