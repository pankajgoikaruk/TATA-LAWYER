# scripts\evaluate_tata_final_holdout_parent_level.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, jaccard_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_factory import build_audio_exit_net


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
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


def parse_tap_blocks(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def load_feature(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected [n_mels, T], got {arr.shape}: {path}")
    return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    result = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_score": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
        "per_label": {},
    }

    for i, lab in enumerate(labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        result["per_label"][lab] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
        }

    return result


def load_thresholds(run_dir: Path, labels: list[str], num_exits: int, mode: str) -> list[np.ndarray]:
    if mode == "fixed_0p5":
        return [np.full(len(labels), 0.5, dtype=np.float32) for _ in range(num_exits)]

    path = run_dir / "threshold_tuning" / "threshold_comparison.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned thresholds: {path}")

    payload = load_json(path)
    exits = payload["exits"]

    thresholds = []
    for exit_idx in range(num_exits):
        mapping = exits[exit_idx]["tuned_thresholds"]
        thresholds.append(np.asarray([float(mapping[lab]) for lab in labels], dtype=np.float32))

    return thresholds


def label_set_stability_policy(preds_by_exit: list[np.ndarray], min_exit: int, stable_k: int):
    n = preds_by_exit[0].shape[0]
    num_exits = len(preds_by_exit)

    selected = np.zeros_like(preds_by_exit[-1])
    selected_exit_idx = np.full(n, num_exits - 1, dtype=int)

    for i in range(n):
        prev = None
        stable_count = 0

        for exit_idx in range(min_exit - 1, num_exits):
            cur = preds_by_exit[exit_idx][i]

            if prev is not None and np.array_equal(cur, prev):
                stable_count += 1
            else:
                stable_count = 1

            prev = cur

            if stable_count >= stable_k:
                selected[i] = cur
                selected_exit_idx[i] = exit_idx
                break
        else:
            selected[i] = preds_by_exit[-1][i]
            selected_exit_idx[i] = num_exits - 1

    return selected, selected_exit_idx


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Parent-level final raw holdout evaluation.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--holdout_manifest", required=True)
    parser.add_argument("--features_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--threshold_mode", default="fixed_0p5", choices=["fixed_0p5", "tuned_per_exit"])
    parser.add_argument("--aggregation", default="max", choices=["max", "mean"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--run_dynamic_policy", action="store_true")
    parser.add_argument("--min_exit", type=int, default=2)
    parser.add_argument("--stable_k", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = Path(args.holdout_manifest)
    features_root = Path(args.features_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(run_dir / "config_used.json")
    labels = config["labels"]
    tap_blocks = parse_tap_blocks(config["tap_blocks"])
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
        num_classes=len(labels),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(args.device)

    ckpt = run_dir / "ckpt" / "best.pt"
    try:
        state = torch.load(ckpt, map_location=args.device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=args.device)

    model.load_state_dict(state)
    model.eval()

    df = pd.read_csv(manifest_path, low_memory=False)

    if "parent_clip_id" not in df.columns:
        raise RuntimeError("Holdout feature manifest missing parent_clip_id")

    probs_by_exit = None

    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]

        xs = []
        for _, row in batch.iterrows():
            feat_path = features_root / Path(str(row["feat_relpath"]))
            xs.append(load_feature(feat_path))

        x = torch.cat(xs, dim=0).to(args.device)
        logits_list = model(x)
        probs_list = [torch.sigmoid(logits).detach().cpu().numpy() for logits in logits_list]

        if probs_by_exit is None:
            probs_by_exit = [[] for _ in probs_list]

        for k, probs in enumerate(probs_list):
            probs_by_exit[k].append(probs)

    probs_by_exit = [np.concatenate(parts, axis=0) for parts in probs_by_exit]

    # Segment-level prediction table.
    seg_prob_df = df[["parent_clip_id", "sample_id", "source_file", "source_path"] + labels].copy()
    for exit_idx, probs in enumerate(probs_by_exit):
        for j, lab in enumerate(labels):
            seg_prob_df[f"exit{exit_idx+1}_prob_{lab}"] = probs[:, j]

    seg_prob_path = out_dir / f"parent_eval_segment_probs_{args.threshold_mode}_{args.aggregation}.csv"
    seg_prob_df.to_csv(seg_prob_path, index=False)

    # Parent ground truth: first row per parent, since labels are inherited from clip to segments.
    parent_gt = (
        df.groupby("parent_clip_id", as_index=False)
        .first()[["parent_clip_id", "source_file", "source_path"] + labels]
        .copy()
    )
    y_true_parent = parent_gt[labels].astype(int).values

    # Parent aggregated probabilities per exit.
    parent_probs_by_exit = []
    for exit_idx in range(len(probs_by_exit)):
        prob_cols = [f"exit{exit_idx+1}_prob_{lab}" for lab in labels]
        temp = seg_prob_df[["parent_clip_id"] + prob_cols].copy()

        if args.aggregation == "max":
            agg = temp.groupby("parent_clip_id", as_index=False)[prob_cols].max()
        else:
            agg = temp.groupby("parent_clip_id", as_index=False)[prob_cols].mean()

        agg = parent_gt[["parent_clip_id"]].merge(agg, on="parent_clip_id", how="left")
        parent_prob = agg[prob_cols].values.astype(np.float32)
        parent_probs_by_exit.append(parent_prob)

    thresholds_by_exit = load_thresholds(run_dir, labels, len(parent_probs_by_exit), args.threshold_mode)
    parent_preds_by_exit = [
        (parent_probs_by_exit[i] >= thresholds_by_exit[i].reshape(1, -1)).astype(int)
        for i in range(len(parent_probs_by_exit))
    ]

    static_rows = []
    details = {}

    for i, pred in enumerate(parent_preds_by_exit):
        m = compute_metrics(y_true_parent, pred, labels)
        row = {
            "model": run_dir.name,
            "threshold_mode": args.threshold_mode,
            "aggregation": args.aggregation,
            "split": "final_raw_holdout_parent_level",
            "exit": i + 1,
            "parent_clips": int(len(parent_gt)),
            "macro_f1": m["macro_f1"],
            "micro_f1": m["micro_f1"],
            "samples_f1": m["samples_f1"],
            "exact_match": m["exact_match"],
            "hamming_loss": m["hamming_loss"],
            "jaccard_score": m["jaccard_score"],
            "avg_true_labels": m["avg_true_labels"],
            "avg_pred_labels": m["avg_pred_labels"],
        }
        static_rows.append(row)
        details[f"exit_{i+1}"] = m

    static_csv = out_dir / f"parent_holdout_static_{args.threshold_mode}_{args.aggregation}.csv"
    pd.DataFrame(static_rows).to_csv(static_csv, index=False)

    result = {
        "run_dir": str(run_dir),
        "holdout_manifest": str(manifest_path),
        "features_root": str(features_root),
        "threshold_mode": args.threshold_mode,
        "aggregation": args.aggregation,
        "parent_clips": int(len(parent_gt)),
        "segment_rows": int(len(df)),
        "labels": labels,
        "static": details,
        "static_summary_rows": static_rows,
        "outputs": {
            "static_csv": str(static_csv),
            "segment_probs_csv": str(seg_prob_path),
        },
    }

    if args.run_dynamic_policy:
        selected_pred, selected_exit_idx = label_set_stability_policy(
            preds_by_exit=parent_preds_by_exit,
            min_exit=int(args.min_exit),
            stable_k=int(args.stable_k),
        )

        dyn_metrics = compute_metrics(y_true_parent, selected_pred, labels)

        exit_counts = {}
        for e in range(len(parent_preds_by_exit)):
            exit_counts[f"exit_{e+1}"] = int(np.sum(selected_exit_idx == e))

        avg_depth = float(np.mean(selected_exit_idx + 1))
        num_exits = len(parent_preds_by_exit)
        compute_saved = float((1.0 - avg_depth / num_exits) * 100.0)

        dyn_row = {
            "model": run_dir.name,
            "threshold_mode": args.threshold_mode,
            "aggregation": args.aggregation,
            "split": "final_raw_holdout_parent_level",
            "policy": "label_set_stability",
            "min_exit": int(args.min_exit),
            "stable_k": int(args.stable_k),
            "parent_clips": int(len(parent_gt)),
            "avg_exit_depth": avg_depth,
            "depth_compute_saved_pct": compute_saved,
            "macro_f1": dyn_metrics["macro_f1"],
            "micro_f1": dyn_metrics["micro_f1"],
            "samples_f1": dyn_metrics["samples_f1"],
            "exact_match": dyn_metrics["exact_match"],
            "hamming_loss": dyn_metrics["hamming_loss"],
        }

        dyn_csv = out_dir / f"parent_holdout_dynamic_{args.threshold_mode}_{args.aggregation}.csv"
        pd.DataFrame([dyn_row]).to_csv(dyn_csv, index=False)

        result["dynamic_policy"] = {
            **dyn_row,
            "exit_counts": exit_counts,
            "metrics": dyn_metrics,
        }
        result["outputs"]["dynamic_csv"] = str(dyn_csv)

    out_json = out_dir / f"parent_holdout_eval_{args.threshold_mode}_{args.aggregation}.json"
    save_json(result, out_json)

    print("\nParent-level final holdout evaluation complete")
    print("-" * 90)
    print(f"Run:            {run_dir.name}")
    print(f"Parent clips:   {len(parent_gt)}")
    print(f"Segments:       {len(df)}")
    print(f"Threshold mode: {args.threshold_mode}")
    print(f"Aggregation:    {args.aggregation}")
    print("")
    print(pd.DataFrame(static_rows).to_string(index=False))

    if args.run_dynamic_policy:
        d = result["dynamic_policy"]
        print("\nDynamic parent-level policy:")
        print(f"  avg_exit_depth={d['avg_exit_depth']:.4f}")
        print(f"  compute_saved={d['depth_compute_saved_pct']:.2f}%")
        print(f"  macroF1={d['metrics']['macro_f1']:.4f}")
        print(f"  microF1={d['metrics']['micro_f1']:.4f}")
        print(f"  exact={d['metrics']['exact_match']:.4f}")
        print(f"  hamming={d['metrics']['hamming_loss']:.4f}")

    print(f"\nOutput JSON: {out_json}")


if __name__ == "__main__":
    main()