# training/train_multilabel_masked.py
#
# Three-exit multi-label training with per-label supervision masks.
# Unknown labels (mask=0) do not contribute to BCE loss or masked metrics.
#
# Checkpoint selection:
#   strict original validation rows only (v09_checkpoint_eligible=1)
#
# Main fair test:
#   strict original test rows only (v09_standard_test_eligible=1)
#
# Secondary reports:
#   all test rows with masked metrics
#   recovered human-reviewed test rows with masked metrics

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
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel_masked import make_masked_multilabel_loaders
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


def default_loss_weights(num_exits: int) -> list[float]:
    if num_exits == 3:
        return [0.3, 0.3, 1.0]
    if num_exits == 5:
        return [0.3, 0.3, 0.6, 0.8, 1.0]
    if num_exits <= 1:
        return [1.0]
    return [0.3] * (num_exits - 1) + [1.0]


def parse_loss_weights(raw: str | None, num_exits: int) -> list[float]:
    if raw is None or not str(raw).strip():
        return default_loss_weights(num_exits)
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values:
        return default_loss_weights(num_exits)
    if len(values) > num_exits:
        return values[:num_exits]
    while len(values) < num_exits:
        values.append(values[-1])
    return values


def save_json(payload, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        return str(value)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=convert)


def make_run_dir(runs_root: str | Path, variant: str) -> Path:
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = runs_root / f"{str(variant).strip()}_{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def masked_bce_for_exits(
    logits_list: list[torch.Tensor],
    targets: torch.Tensor,
    masks: torch.Tensor,
    loss_weights: list[float],
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if len(logits_list) != len(loss_weights):
        raise RuntimeError(
            f"loss_weights={len(loss_weights)} but exits={len(logits_list)}"
        )

    masks = masks.float()
    denominator = masks.sum().clamp_min(1.0)
    exit_losses = []

    for logits in logits_list:
        elementwise = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        exit_loss = (elementwise * masks).sum() / denominator
        exit_losses.append(exit_loss)

    return sum(
        float(weight) * loss
        for weight, loss in zip(loss_weights, exit_losses)
    )


def _safe_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.size == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    return {
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _sample_f1_over_known(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    scores = []
    for true_row, pred_row, mask_row in zip(y_true, y_pred, mask):
        known = mask_row.astype(bool)
        if not known.any():
            continue
        yt = true_row[known]
        yp = pred_row[known]
        scores.append(float(f1_score(yt, yp, zero_division=0)))
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def evaluate_masked(
    model,
    loader,
    device: str,
    labels: list[str],
    threshold: float = 0.5,
):
    model.eval()

    targets_parts = []
    masks_parts = []
    probabilities_by_exit = None

    for x, y, mask in loader:
        x = x.to(device)
        logits_list = model(x)
        probs_list = [torch.sigmoid(logits) for logits in logits_list]

        if probabilities_by_exit is None:
            probabilities_by_exit = [[] for _ in probs_list]

        targets_parts.append(y.numpy())
        masks_parts.append(mask.numpy())
        for exit_idx, probabilities in enumerate(probs_list):
            probabilities_by_exit[exit_idx].append(
                probabilities.detach().cpu().numpy()
            )

    if not targets_parts:
        return {
            "threshold": float(threshold),
            "rows": 0,
            "known_label_decisions": 0,
            "fully_known_rows": 0,
            "exit_metrics": [],
            "per_label_final": {},
        }

    y_true = np.concatenate(targets_parts, axis=0).astype(int)
    mask = np.concatenate(masks_parts, axis=0).astype(int)
    known = mask.astype(bool)

    fully_known_rows = known.all(axis=1)
    exit_metrics = []

    for exit_idx, parts in enumerate(probabilities_by_exit):
        probabilities = np.concatenate(parts, axis=0)
        y_pred = (probabilities >= float(threshold)).astype(int)

        flat_true = y_true[known]
        flat_pred = y_pred[known]

        if flat_true.size:
            micro_f1 = float(
                f1_score(flat_true, flat_pred, average="binary", zero_division=0)
            )
            micro_precision = float(
                precision_score(
                    flat_true, flat_pred, average="binary", zero_division=0
                )
            )
            micro_recall = float(
                recall_score(
                    flat_true, flat_pred, average="binary", zero_division=0
                )
            )
            hamming = float(np.mean(flat_true != flat_pred))
        else:
            micro_f1 = micro_precision = micro_recall = hamming = 0.0

        per_label_f1 = []
        per_label_precision = []
        per_label_recall = []
        for label_idx in range(len(labels)):
            label_known = known[:, label_idx]
            metrics = _safe_binary_metrics(
                y_true[label_known, label_idx],
                y_pred[label_known, label_idx],
            )
            per_label_precision.append(metrics["precision"])
            per_label_recall.append(metrics["recall"])
            per_label_f1.append(metrics["f1"])

        known_labels_exact_match = float(
            np.mean(
                np.all(
                    np.logical_or(~known, y_true == y_pred),
                    axis=1,
                )
            )
        )

        if fully_known_rows.any():
            exact_match_fully_known = float(
                np.mean(
                    np.all(
                        y_true[fully_known_rows] == y_pred[fully_known_rows],
                        axis=1,
                    )
                )
            )
        else:
            exact_match_fully_known = 0.0

        exit_metrics.append(
            {
                "exit": exit_idx + 1,
                "micro_f1_known_decisions": micro_f1,
                "macro_f1_known_labels": float(np.mean(per_label_f1)),
                "samples_f1_known_labels": _sample_f1_over_known(
                    y_true, y_pred, mask
                ),
                "micro_precision_known_decisions": micro_precision,
                "micro_recall_known_decisions": micro_recall,
                "macro_precision_known_labels": float(
                    np.mean(per_label_precision)
                ),
                "macro_recall_known_labels": float(np.mean(per_label_recall)),
                "known_labels_exact_match": known_labels_exact_match,
                "exact_match_fully_known_rows": exact_match_fully_known,
                "masked_hamming_loss": hamming,
                "known_label_decisions": int(known.sum()),
                "fully_known_rows": int(fully_known_rows.sum()),
                "rows": int(len(y_true)),
            }
        )

    final_probabilities = np.concatenate(
        probabilities_by_exit[-1], axis=0
    )
    final_pred = (final_probabilities >= float(threshold)).astype(int)

    per_label_final = {}
    for idx, label in enumerate(labels):
        label_known = known[:, idx]
        metrics = _safe_binary_metrics(
            y_true[label_known, idx],
            final_pred[label_known, idx],
        )
        per_label_final[label] = {
            **metrics,
            "known_count": int(label_known.sum()),
            "unknown_count": int((~label_known).sum()),
            "support": int(y_true[label_known, idx].sum()),
            "predicted_positive": int(final_pred[label_known, idx].sum()),
        }

    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "known_label_decisions": int(known.sum()),
        "fully_known_rows": int(fully_known_rows.sum()),
        "exit_metrics": exit_metrics,
        "per_label_final": per_label_final,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: str,
    loss_weights: list[float],
    pos_weight: torch.Tensor | None,
    epoch: int,
    total_epochs: int,
    log_every: int,
):
    model.train()
    running_loss = 0.0
    samples = 0
    total_batches = len(loader)

    for batch_idx, (x, y, mask) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits_list = model(x)
        loss = masked_bce_for_exits(
            logits_list,
            y,
            mask,
            loss_weights,
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += float(loss.item()) * batch_size
        samples += batch_size

        if int(log_every) > 0 and (
            batch_idx == 1
            or batch_idx % int(log_every) == 0
            or batch_idx == total_batches
        ):
            print(
                f"Epoch {epoch:03d}/{total_epochs} | "
                f"batch {batch_idx:05d}/{total_batches:05d} | "
                f"samples={samples} | "
                f"batch_loss={float(loss.item()):.4f} | "
                f"avg_loss={running_loss / max(samples, 1):.4f}",
                flush=True,
            )

    return running_loss / max(samples, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train a masked multi-label three-exit audio model."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features_root", required=True)
    parser.add_argument("--labels_json", required=True)
    parser.add_argument("--runs_root", default="runs_multilabel_masked")
    parser.add_argument(
        "--variant",
        default="tata_v09_human_reviewed_masked_3exit",
    )
    parser.add_argument("--tap_blocks", default="1,3")
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--loss_weights", default="0.3,0.3,1.0")
    parser.add_argument("--use_pos_weight", action="store_true")
    parser.add_argument("--pos_weight_max", type=float, default=20.0)
    parser.add_argument("--label_balance_power", type=float, default=0.0)
    parser.add_argument("--synthetic_balance_power", type=float, default=0.0)
    args = parser.parse_args()

    set_global_seed(args.seed)

    if str(args.device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)

    manifest = Path(args.manifest).resolve()
    features_root = Path(args.features_root).resolve()
    labels_json = Path(args.labels_json).resolve()
    run_dir = make_run_dir(args.runs_root, args.variant)
    checkpoint_dir = run_dir / "ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loaders, datasets, labels = make_masked_multilabel_loaders(
        manifest_csv=manifest,
        features_root=features_root,
        labels_json=labels_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        label_balance_power=args.label_balance_power,
        synthetic_balance_power=args.synthetic_balance_power,
    )

    tap_blocks = parse_tap_blocks(args.tap_blocks)
    model_config = {
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
        n_mels=args.n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_config,
    ).to(device)

    loss_weights = parse_loss_weights(args.loss_weights, int(model.num_exits))

    pos_weight = None
    if args.use_pos_weight:
        pos_weight = datasets["train"].pos_weight_tensor(
            max_value=args.pos_weight_max
        ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    config = {
        "task": "masked_multi_label_audio",
        "manifest": str(manifest),
        "features_root": str(features_root),
        "labels_json": str(labels_json),
        "labels": labels,
        "run_dir": str(run_dir),
        "device": device,
        "tap_blocks": list(tap_blocks),
        "num_exits": int(model.num_exits),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "threshold": float(args.threshold),
        "loss_weights": loss_weights,
        "use_pos_weight": bool(args.use_pos_weight),
        "pos_weight_max": float(args.pos_weight_max),
        "label_balance_power": float(args.label_balance_power),
        "synthetic_balance_power": float(args.synthetic_balance_power),
        "checkpoint_selection": (
            "final-exit macro-F1 on strict original validation rows"
        ),
        "dataset_rows": {
            name: int(len(dataset))
            for name, dataset in datasets.items()
        },
    }
    save_json(config, run_dir / "config_used.json")

    metrics = {
        "config": config,
        "epochs": [],
        "best": {},
        "test_strict": {},
        "test_all_masked": {},
        "test_recovered_masked": {},
    }

    print("\nMasked multi-label training")
    print("-" * 90)
    print(f"Manifest:       {manifest}")
    print(f"Features root:  {features_root}")
    print(f"Run directory:  {run_dir}")
    print(f"Device:         {device}")
    print(f"Labels:         {labels}")
    print(f"Loss weights:   {loss_weights}")
    print(
        "Checkpointing:  strict original validation "
        f"({len(datasets['val_strict']):,} rows)"
    )
    print(
        "Fair test:      strict original test "
        f"({len(datasets['test_strict']):,} rows)"
    )
    print("-" * 90)

    best_score = -1.0
    best_epoch = -1
    best_path = checkpoint_dir / "best.pt"

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            loss_weights=loss_weights,
            pos_weight=pos_weight,
            epoch=epoch,
            total_epochs=int(args.epochs),
            log_every=int(args.log_every),
        )

        val_strict = evaluate_masked(
            model,
            loaders["val_strict"],
            device,
            labels,
            threshold=args.threshold,
        )
        final_exit = val_strict["exit_metrics"][-1]
        score = float(final_exit["macro_f1_known_labels"])

        epoch_row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_strict": val_strict,
        }
        metrics["epochs"].append(epoch_row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"strict_val_macroF1={score:.4f} | "
            f"strict_val_microF1="
            f"{final_exit['micro_f1_known_decisions']:.4f} | "
            f"strict_val_exact="
            f"{final_exit['exact_match_fully_known_rows']:.4f} | "
            f"strict_val_hamming="
            f"{final_exit['masked_hamming_loss']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            metrics["best"] = {
                "epoch": int(best_epoch),
                "score_name": "strict_val_final_exit_macro_f1",
                "score": float(best_score),
                "val_strict": val_strict,
            }
            save_json(metrics, run_dir / "metrics.json")

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    evaluation_loaders = [
        ("test_strict", "test_strict"),
        ("test_all_masked", "test_all_masked"),
        ("test_recovered_masked", "test_recovered_masked"),
    ]
    for output_name, loader_name in evaluation_loaders:
        metrics[output_name] = evaluate_masked(
            model,
            loaders[loader_name],
            device,
            labels,
            threshold=args.threshold,
        )

    save_json(metrics, run_dir / "metrics.json")
    torch.save(model.state_dict(), checkpoint_dir / "last_loaded_best.pt")

    print("\nTraining completed")
    print("-" * 90)
    print(f"Best epoch: {best_epoch}")
    print(f"Best strict validation Macro-F1: {best_score:.4f}")
    print(f"Run directory: {run_dir}")
    print(f"Best checkpoint: {best_path}")

    for report_name in (
        "test_strict",
        "test_all_masked",
        "test_recovered_masked",
    ):
        report = metrics[report_name]
        print(f"\n{report_name} ({report['rows']} rows)")
        for item in report["exit_metrics"]:
            print(
                f"  Exit {item['exit']}: "
                f"macroF1={item['macro_f1_known_labels']:.4f}, "
                f"microF1={item['micro_f1_known_decisions']:.4f}, "
                f"samplesF1={item['samples_f1_known_labels']:.4f}, "
                f"exactFullyKnown="
                f"{item['exact_match_fully_known_rows']:.4f}, "
                f"maskedHamming={item['masked_hamming_loss']:.4f}"
            )

    print("\nStrict final-exit per-label test F1")
    for label, values in metrics["test_strict"]["per_label_final"].items():
        print(
            f"  {label}: "
            f"P={values['precision']:.4f}, "
            f"R={values['recall']:.4f}, "
            f"F1={values['f1']:.4f}, "
            f"support={values['support']}, "
            f"known={values['known_count']}"
        )


if __name__ == "__main__":
    main()
