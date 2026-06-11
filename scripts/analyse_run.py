from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix, roc_curve, auc

from data.datasets import make_loaders
from utils.model_factory import build_audio_exit_net, load_run_model_cfg


def load_json_safepath(path, default=None):
    """
    Small helper: load JSON if it exists, otherwise return a default value.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception:
        return default


def _parse_tap_blocks(value) -> Optional[tuple]:
    """
    Accept:
      - None
      - "1,2,3,4"
      - [1,2,3,4]
      - (1,2,3,4)
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    value = str(value).strip()
    if value == "":
        return None

    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _infer_tap_blocks_from_run(run_dir: Path) -> Optional[tuple]:
    """
    Try to recover tap_blocks from saved artifacts.
    """
    candidates = [
        run_dir / "metrics.json",
        run_dir / "report.json",
        run_dir / "temperature.json",
        run_dir / "thresholds.json",
        run_dir / "summary.json",
        run_dir / "meta.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        obj = load_json_safepath(path, {})
        if not isinstance(obj, dict):
            continue

        tb = None

        if isinstance(obj.get("meta"), dict):
            tb = obj["meta"].get("tap_blocks")
        if tb is None and isinstance(obj.get("policy_summary"), dict):
            tb = obj["policy_summary"].get("tap_blocks")
        if tb is None:
            tb = obj.get("tap_blocks")

        tb = _parse_tap_blocks(tb)
        if tb is not None:
            return tb

    # Optional YAML fallback
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            tb = _parse_tap_blocks((cfg.get("model") or {}).get("tap_blocks"))
            if tb is not None:
                return tb
        except Exception:
            pass

    return None


def _infer_n_mels_from_run(run_dir: Path, default=64) -> int:
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return int((cfg.get("features") or {}).get("n_mels", default))
        except Exception:
            pass
    return int(default)


def plot_training_curves(metrics_path: Path, plots_dir: Path):
    """
    Plot training loss and validation accuracy per exit over epochs
    using the content of metrics.json produced by training/train.py.
    """
    metrics = load_json_safepath(metrics_path, {})
    if not metrics:
        print(f"[analyse_run] No metrics.json found at {metrics_path}, skipping training curves.")
        return

    train_hist = metrics.get("train", [])
    val_hist = metrics.get("val", [])

    if not train_hist or not val_hist:
        print("[analyse_run] metrics.json has no train/val history, skipping training curves.")
        return

    # --- Train loss ---
    epochs = [e["epoch"] for e in train_hist]
    losses = [e["loss"] for e in train_hist]

    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss vs epoch")
    plt.grid(True)
    out = plots_dir / "train_loss.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyse_run] Saved {out}")

    # --- Validation accuracy per exit (dynamic K) ---
    epochs_val = [e["epoch"] for e in val_hist]

    first_acc = val_hist[0].get("acc", [])
    if not isinstance(first_acc, list) or len(first_acc) == 0:
        print("[analyse_run] metrics.json val history has no per-exit acc list, skipping val_acc_exits.png.")
        return

    num_exits = len(first_acc)

    plt.figure()
    for k in range(num_exits):
        acc_k = []
        for entry in val_hist:
            acc_list = entry.get("acc", [])
            if k < len(acc_list):
                acc_k.append(acc_list[k])
            else:
                acc_k.append(np.nan)
        plt.plot(epochs_val, acc_k, marker="o", label=f"exit{k+1}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy per exit vs epoch")
    plt.legend()
    plt.grid(True)
    out = plots_dir / "val_acc_exits.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyse_run] Saved {out}")


@torch.no_grad()
def collect_test_predictions(
    run_dir: Path,
    segments_csv: Path,
    features_root: Path,
    device: str = None,
    n_mels: int = 64,
    tap_blocks=None,
):
    """
    Reload the trained ExitNet model from ckpt/best.pt and run it on the test set.

    Returns:
      y_true       : (N,) numpy array of ground-truth labels
      y_pred_exits : list of length K, each (N,) array of predicted labels for that exit
      y_prob_exits : list of length K, each (N, C) array of softmax probabilities for that exit
      label2id     : mapping label string -> int
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data loaders ---
    _, _, dl_te, label2id = make_loaders(
        str(segments_csv),
        str(features_root),
        batch_size=64,
        num_workers=4,
    )
    num_classes = len(label2id)

    # --- Model ---
    ckpt_path = run_dir / "ckpt" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint at {ckpt_path}")

    model_cfg = load_run_model_cfg(str(run_dir))

    if tap_blocks is None:
        tap_blocks = _parse_tap_blocks((model_cfg or {}).get("tap_blocks"))
    if tap_blocks is None:
        tap_blocks = _infer_tap_blocks_from_run(run_dir)
    if tap_blocks is None:
        tap_blocks = (1, 3)

    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Run through test set ---
    y_true_list = []
    y_pred_exits = None
    y_prob_exits = None

    for x, y in dl_te:
        x = x.to(device)
        y_true_list.extend(y.numpy().tolist())

        logits_list = model(x)
        probs_list = [softmax(lg, dim=1).cpu().numpy() for lg in logits_list]

        if y_pred_exits is None:
            num_exits = len(probs_list)
            y_pred_exits = [[] for _ in range(num_exits)]
            y_prob_exits = [[] for _ in range(num_exits)]

        for k in range(len(probs_list)):
            preds = np.argmax(probs_list[k], axis=1)
            y_pred_exits[k].extend(preds.tolist())
            y_prob_exits[k].append(probs_list[k])

    if y_pred_exits is None or y_prob_exits is None:
        raise RuntimeError("Test loader is empty. No predictions collected.")

    y_true = np.array(y_true_list, dtype=np.int64)
    y_prob_exits = [
        np.concatenate(chunks, axis=0) if len(chunks) > 0 else None
        for chunks in y_prob_exits
    ]
    y_pred_exits = [np.array(preds, dtype=np.int64) for preds in y_pred_exits]

    return y_true, y_pred_exits, y_prob_exits, label2id


def compute_and_plot_confusion_matrices(
    y_true,
    y_pred_exits,
    label2id,
    plots_dir: Path,
    out_json: Path,
):
    """
    Compute confusion matrices (counts + row-normalised) for each exit,
    save them as images and a JSON summary.
    """
    id2label = {v: k for k, v in label2id.items()}
    labels_sorted = [id2label[i] for i in range(len(id2label))]

    cm_info = {}

    for exit_idx, y_pred in enumerate(y_pred_exits, start=1):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(id2label))))

        with np.errstate(all="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = cm.astype(float) / np.maximum(row_sums, 1e-12)

        # Save plot
        plt.figure(figsize=(5, 4))
        plt.imshow(cm_norm, interpolation="nearest")
        plt.title(f"Confusion matrix - exit{exit_idx}")
        plt.colorbar()

        tick_marks = np.arange(len(labels_sorted))
        plt.xticks(tick_marks, labels_sorted, rotation=45, ha="right")
        plt.yticks(tick_marks, labels_sorted)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                val = cm_norm[i, j]
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        out_png = plots_dir / f"cm_exit{exit_idx}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[analyse_run] Saved {out_png}")

        cm_info[f"exit{exit_idx}"] = {
            "labels": labels_sorted,
            "counts": cm.tolist(),
            "row_normalised": cm_norm.tolist(),
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cm_info, f, indent=2)
    print(f"[analyse_run] Saved confusion matrices JSON -> {out_json}")

    return cm_info


def compute_and_plot_roc(
    y_true,
    y_prob_exits,
    plots_dir: Path,
    out_json: Path,
):
    """
    Compute ROC curves and AUC per exit for binary classification (num_classes == 2).
    If num_classes != 2, this function returns {} and does nothing.

    We treat class '1' as the positive class by convention.
    """
    if not y_prob_exits or y_prob_exits[0] is None:
        print("[analyse_run] No probabilities for exit1, skipping ROC/AUC.")
        return {}

    num_classes = y_prob_exits[0].shape[1]
    if num_classes != 2:
        print(
            f"[analyse_run] ROC/AUC currently implemented only for binary tasks, "
            f"but got num_classes={num_classes}. Skipping ROC/AUC."
        )
        return {}

    roc_info = {}
    y_true_bin = (y_true == 1).astype(int)

    for exit_idx, probs in enumerate(y_prob_exits, start=1):
        if probs is None:
            continue

        y_score = probs[:, 1]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"exit{exit_idx} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC curve - exit{exit_idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_png = plots_dir / f"roc_exit{exit_idx}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[analyse_run] Saved {out_png}")

        roc_info[f"exit{exit_idx}"] = {
            "auc": float(roc_auc),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(roc_info, f, indent=2)
    print(f"[analyse_run] Saved ROC/AUC JSON -> {out_json}")

    return roc_info


def build_analysis_summary(
    run_dir: Path,
    cm_info: dict,
    roc_info: dict,
    label_names,
):
    """
    Aggregate key metrics into one 'analysis_run.json' file.

    - Per-exit classification metrics from report.json
    - Policy-level metrics from summary.json
    - Confusion matrix & AUC info
    - Label names
    """
    report = load_json_safepath(run_dir / "report.json", {})
    summary = load_json_safepath(run_dir / "summary.json", {})

    out = {
        "run_id": summary.get("run_id", run_dir.name),
        "classification_per_exit": report,
        "policy_summary": summary.get("policy_summary", {}),
        "confusion_matrices": cm_info,
        "roc_auc": roc_info,
        "label_names": label_names,
    }

    out_path = run_dir / "analysis_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[analyse_run] Saved consolidated analysis JSON -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Path to a single run directory, e.g. runs/variant/variant_001",
    )
    ap.add_argument(
        "--segments_csv",
        default="data_cache/segments.csv",
        help="Path to segments.csv used for this run",
    )
    ap.add_argument(
        "--features_root",
        default="data_cache/features",
        help="Root directory for .npy features used for this run",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--tap_blocks", type=str, default=None)
    ap.add_argument("--n_mels", type=int, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    tap_blocks = _parse_tap_blocks(args.tap_blocks)
    if tap_blocks is None:
        tap_blocks = _infer_tap_blocks_from_run(run_dir)

    n_mels = int(args.n_mels) if args.n_mels is not None else _infer_n_mels_from_run(run_dir, default=64)

    # 1) Training curves from metrics.json
    metrics_path = run_dir / "metrics.json"
    plot_training_curves(metrics_path, plots_dir)

    # 2) Test-set predictions, confusion matrices, ROC
    y_true, y_pred_exits, y_prob_exits, label2id = collect_test_predictions(
        run_dir=run_dir,
        segments_csv=Path(args.segments_csv),
        features_root=Path(args.features_root),
        device=args.device,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
    )

    id2label = {v: k for k, v in label2id.items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    cm_json_path = run_dir / "confusion_matrices.json"
    cm_info = compute_and_plot_confusion_matrices(
        y_true=y_true,
        y_pred_exits=y_pred_exits,
        label2id=label2id,
        plots_dir=plots_dir,
        out_json=cm_json_path,
    )

    roc_json_path = run_dir / "roc_curves.json"
    roc_info = compute_and_plot_roc(
        y_true=y_true,
        y_prob_exits=y_prob_exits,
        plots_dir=plots_dir,
        out_json=roc_json_path,
    )

    # 3) Consolidated analysis JSON
    build_analysis_summary(run_dir, cm_info, roc_info, label_names)


if __name__ == "__main__":
    main()