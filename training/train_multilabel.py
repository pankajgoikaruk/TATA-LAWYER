# training/train_multilabel.py
#
# Multi-label training script for ASHADIP / NeuroAccuExit.
#
# Difference from single-label training:
#   Single-label:
#     CrossEntropyLoss
#     softmax
#     argmax
#
#   Multi-label:
#     BCEWithLogitsLoss
#     sigmoid
#     threshold per label
#
# Input:
#   multilabel_cache/metadata/multilabel_features_manifest.csv
#   multilabel_cache/features
#   multilabel_data/metadata/labels.json
#
# Example:
#   python -m training.train_multilabel `
#     --manifest "multilabel_cache\metadata\multilabel_features_manifest.csv" `
#     --features_root "multilabel_cache\features" `
#     --labels_json "multilabel_data\metadata\labels.json" `
#     --runs_root "runs_multilabel" `
#     --variant "multilabel_3exit_nohint" `
#     --tap_blocks "1,3" `
#     --epochs 40 `
#     --batch_size 64 `
#     --log_every 25 `
#     --lr 0.001 `
#     --device cpu

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from torch.optim import Adam

# Make project root importable when running:
# python training\train_multilabel.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel import (
    make_multilabel_loaders,
    make_pos_weight_from_train,
)
from utils.model_factory import build_audio_exit_net


def set_global_seed(seed: int):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def parse_tap_blocks(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    value = str(value).strip()
    if not value:
        raise ValueError("tap_blocks cannot be empty.")

    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def default_loss_weights_for_num_exits(num_exits: int) -> list[float]:
    """
    Same default style as your single-label K-exit setup.

    3 exits:
      [0.3, 0.3, 1.0]

    5 exits:
      [0.3, 0.3, 0.6, 0.8, 1.0]
    """
    if num_exits == 3:
        return [0.3, 0.3, 1.0]

    if num_exits == 5:
        return [0.3, 0.3, 0.6, 0.8, 1.0]

    if num_exits <= 1:
        return [1.0]

    return [0.3] * (num_exits - 1) + [1.0]


def parse_loss_weights(raw: str | None, num_exits: int) -> list[float]:
    if raw is None or str(raw).strip() == "":
        return default_loss_weights_for_num_exits(num_exits)

    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]

    if len(vals) == num_exits:
        return vals

    print(
        f"[WARN] loss_weights length={len(vals)} but model has {num_exits} exits. "
        "Auto-adjusting."
    )

    if len(vals) > num_exits:
        return vals[:num_exits]

    if not vals:
        return default_loss_weights_for_num_exits(num_exits)

    while len(vals) < num_exits:
        vals.append(vals[-1])

    return vals


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_run_dir(runs_root: str | Path, variant: str) -> Path:
    runs_root = Path(runs_root)
    ensure_dir(runs_root)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    variant = str(variant).strip() or "multilabel_run"

    run_dir = runs_root / f"{variant}_{timestamp}"

    # Very rare collision protection.
    suffix = 1
    original = run_dir
    while run_dir.exists():
        run_dir = Path(str(original) + f"_{suffix}")
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def multilabel_loss_for_exits(
    logits_list: list[torch.Tensor],
    y: torch.Tensor,
    loss_weights: list[float],
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []

    for logits in logits_list:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=pos_weight,
        )
        losses.append(loss)

    if len(loss_weights) != len(losses):
        raise RuntimeError(
            f"loss_weights length={len(loss_weights)} but exits={len(losses)}"
        )

    total = sum(float(w) * loss for w, loss in zip(loss_weights, losses))
    return total


@torch.no_grad()
def evaluate_multilabel(
    model,
    dl,
    device: str,
    labels: list[str],
    threshold: float = 0.5,
):
    model.eval()

    y_true_all = []
    y_prob_by_exit = None

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        logits_list = model(x)
        probs_list = [torch.sigmoid(logits) for logits in logits_list]

        if y_prob_by_exit is None:
            y_prob_by_exit = [[] for _ in probs_list]

        y_true_all.append(y.detach().cpu().numpy())

        for k, probs in enumerate(probs_list):
            y_prob_by_exit[k].append(probs.detach().cpu().numpy())

    if not y_true_all:
        return {
            "threshold": threshold,
            "exit_metrics": [],
            "per_label_final": {},
        }

    y_true = np.concatenate(y_true_all, axis=0).astype(int)

    exit_metrics = []

    for exit_idx, prob_parts in enumerate(y_prob_by_exit):
        y_prob = np.concatenate(prob_parts, axis=0)
        y_pred = (y_prob >= float(threshold)).astype(int)

        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        samples_f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)

        micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
        micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)

        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))
        ham = float(hamming_loss(y_true, y_pred))

        avg_pred_labels = float(y_pred.sum(axis=1).mean())
        avg_true_labels = float(y_true.sum(axis=1).mean())

        exit_metrics.append({
            "exit": exit_idx + 1,
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "samples_f1": float(samples_f1),
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "exact_match": exact_match,
            "hamming_loss": ham,
            "avg_pred_labels": avg_pred_labels,
            "avg_true_labels": avg_true_labels,
        })

    # Per-label metrics for final exit.
    final_probs = np.concatenate(y_prob_by_exit[-1], axis=0)
    final_pred = (final_probs >= float(threshold)).astype(int)

    per_label_final = {}
    for i, label in enumerate(labels):
        yt = y_true[:, i]
        yp = final_pred[:, i]

        per_label_final[label] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
        }

    return {
        "threshold": float(threshold),
        "exit_metrics": exit_metrics,
        "per_label_final": per_label_final,
    }


def train_one_epoch(
    model,
    dl,
    optimizer,
    device: str,
    loss_weights: list[float],
    pos_weight: torch.Tensor | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    log_every: int = 0,
):
    model.train()

    loss_sum = 0.0
    n = 0

    total_batches = len(dl)
    log_every = int(log_every or 0)

    for batch_idx, (x, y) in enumerate(dl, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits_list = model(x)
        loss = multilabel_loss_for_exits(
            logits_list=logits_list,
            y=y,
            loss_weights=loss_weights,
            pos_weight=pos_weight,
        )

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs

        if log_every > 0 and (
            batch_idx == 1
            or batch_idx % log_every == 0
            or batch_idx == total_batches
        ):
            avg_loss = loss_sum / max(n, 1)
            epoch_text = ""
            if epoch is not None and total_epochs is not None:
                epoch_text = f"Epoch {epoch:03d}/{total_epochs} | "

            print(
                f"{epoch_text}"
                f"batch {batch_idx:05d}/{total_batches:05d} | "
                f"samples={n} | "
                f"batch_loss={float(loss.item()):.4f} | "
                f"avg_loss={avg_loss:.4f}",
                flush=True,
            )

    return loss_sum / max(n, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train TinyAudioCNN + ExitNet for multi-label audio classification."
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to multilabel_features_manifest.csv",
    )
    parser.add_argument(
        "--features_root",
        required=True,
        help="Path to multilabel_cache/features",
    )
    parser.add_argument(
        "--labels_json",
        required=True,
        help="Path to labels.json",
    )

    parser.add_argument("--runs_root", default="runs_multilabel")
    parser.add_argument("--variant", default="multilabel_3exit_nohint")

    parser.add_argument(
        "--tap_blocks",
        default="1,3",
        help='Tap blocks. "1,3" gives 3 exits; "1,2,3,4" gives 5 exits.',
    )

    parser.add_argument("--n_mels", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--log_every",
        type=int,
        default=0,
        help="Print training batch progress every N batches. 0 disables batch progress logging.",
    )

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto")

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed sigmoid threshold for validation/test metrics.",
    )

    parser.add_argument(
        "--loss_weights",
        default=None,
        help='Optional comma-separated exit loss weights, e.g. "0.3,0.3,1.0"',
    )

    parser.add_argument(
        "--use_pos_weight",
        action="store_true",
        help="Use BCEWithLogitsLoss positive label weighting from train split.",
    )
    parser.add_argument(
        "--pos_weight_max",
        type=float,
        default=20.0,
        help="Maximum cap for positive class weights.",
    )

    parser.add_argument(
        "--label_balance_power",
        type=float,
        default=0.0,
        help="WeightedRandomSampler label balancing. 0 disables.",
    )
    parser.add_argument(
        "--synthetic_balance_power",
        type=float,
        default=0.0,
        help="WeightedRandomSampler clean/synthetic balancing. 0 disables.",
    )

    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.device is None or str(args.device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    manifest = Path(args.manifest).resolve()
    features_root = Path(args.features_root).resolve()
    labels_json = Path(args.labels_json).resolve()

    run_dir = make_run_dir(args.runs_root, args.variant)
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\nMulti-label training")
    print("-" * 90)
    print(f"Manifest:      {manifest}")
    print(f"Features root: {features_root}")
    print(f"Labels JSON:   {labels_json}")
    print(f"Run dir:       {run_dir}")
    print(f"Device:        {device}")
    print(f"Tap blocks:    {args.tap_blocks}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Log every:     {args.log_every}")
    print(f"LR:            {args.lr}")
    print(f"Threshold:     {args.threshold}")
    print("-" * 90)

    dl_tr, dl_va, dl_te, labels = make_multilabel_loaders(
        manifest_csv=manifest,
        features_root=features_root,
        labels_json=labels_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        label_balance_power=args.label_balance_power,
        synthetic_balance_power=args.synthetic_balance_power,
    )

    num_labels = len(labels)
    tap_blocks = parse_tap_blocks(args.tap_blocks)

    # Important:
    # For this first multi-label version, keep exit_hint disabled.
    # Existing hint-passing was designed around softmax-style class probabilities.
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
        n_mels=args.n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(device)

    num_exits = int(model.num_exits)
    loss_weights = parse_loss_weights(args.loss_weights, num_exits)

    print("\nModel")
    print("-" * 90)
    print(f"Number of labels: {num_labels}")
    print(f"Labels:           {labels}")
    print(f"Number of exits:  {num_exits}")
    print(f"Loss weights:     {loss_weights}")
    print(f"Exit hint:        disabled for first multi-label version")
    print("-" * 90)

    pos_weight = None
    if args.use_pos_weight:
        pos_weight = make_pos_weight_from_train(
            manifest_csv=manifest,
            features_root=features_root,
            labels_json=labels_json,
            max_value=args.pos_weight_max,
        ).to(device)

        print("\nPositive label weights:")
        for label, weight in zip(labels, pos_weight.detach().cpu().numpy().tolist()):
            print(f"  {label}: {weight:.4f}")

    optimizer = Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    config_payload = {
        "task": "multi_label_audio",
        "manifest": str(manifest),
        "features_root": str(features_root),
        "labels_json": str(labels_json),
        "labels": labels,
        "num_labels": num_labels,
        "runs_root": str(args.runs_root),
        "variant": str(args.variant),
        "run_dir": str(run_dir),
        "tap_blocks": list(tap_blocks),
        "num_exits": num_exits,
        "n_mels": int(args.n_mels),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "log_every": int(args.log_every),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "threshold": float(args.threshold),
        "loss_weights": loss_weights,
        "use_pos_weight": bool(args.use_pos_weight),
        "pos_weight_max": float(args.pos_weight_max),
        "label_balance_power": float(args.label_balance_power),
        "synthetic_balance_power": float(args.synthetic_balance_power),
        "exit_hint": model_cfg["exit_hint"],
    }

    save_json(config_payload, run_dir / "config_used.json")

    metrics = {
        "config": config_payload,
        "epochs": [],
        "best": {},
        "test": {},
    }

    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(
            model=model,
            dl=dl_tr,
            optimizer=optimizer,
            device=device,
            loss_weights=loss_weights,
            pos_weight=pos_weight,
            epoch=epoch,
            total_epochs=int(args.epochs),
            log_every=int(args.log_every),
        )

        val_metrics = evaluate_multilabel(
            model=model,
            dl=dl_va,
            device=device,
            labels=labels,
            threshold=float(args.threshold),
        )

        final_exit = val_metrics["exit_metrics"][-1]
        score = float(final_exit["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val": val_metrics,
        }
        metrics["epochs"].append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"final_exit_macroF1={final_exit['macro_f1']:.4f} | "
            f"final_exit_microF1={final_exit['micro_f1']:.4f} | "
            f"exact_match={final_exit['exact_match']:.4f} | "
            f"hamming={final_exit['hamming_loss']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save(model.state_dict(), ckpt_dir / "best.pt")

            metrics["best"] = {
                "epoch": int(best_epoch),
                "score_name": "final_exit_macro_f1",
                "score": float(best_score),
                "val": val_metrics,
            }

            save_json(metrics, run_dir / "metrics.json")

    # Final test evaluation with best checkpoint.
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = evaluate_multilabel(
        model=model,
        dl=dl_te,
        device=device,
        labels=labels,
        threshold=float(args.threshold),
    )

    metrics["test"] = test_metrics
    save_json(metrics, run_dir / "metrics.json")

    torch.save(model.state_dict(), ckpt_dir / "last_loaded_best.pt")

    print("\nTraining completed.")
    print("-" * 90)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation final-exit macro-F1: {best_score:.4f}")
    print(f"Run dir: {run_dir}")
    print(f"Best checkpoint: {best_path}")

    print("\nTest metrics by exit:")
    for item in test_metrics["exit_metrics"]:
        print(
            f"  Exit {item['exit']}: "
            f"macroF1={item['macro_f1']:.4f}, "
            f"microF1={item['micro_f1']:.4f}, "
            f"samplesF1={item['samples_f1']:.4f}, "
            f"exact_match={item['exact_match']:.4f}, "
            f"hamming={item['hamming_loss']:.4f}"
        )

    print("\nFinal-exit per-label test F1:")
    for label, vals in test_metrics["per_label_final"].items():
        print(
            f"  {label}: "
            f"P={vals['precision']:.4f}, "
            f"R={vals['recall']:.4f}, "
            f"F1={vals['f1']:.4f}, "
            f"support={vals['support']}, "
            f"pred_pos={vals['predicted_positive']}"
        )

    print("-" * 90)


if __name__ == "__main__":
    main()
