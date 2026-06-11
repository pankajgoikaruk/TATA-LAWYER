# training/eval.py

from __future__ import annotations

import argparse
import json
import os

import torch
from sklearn.metrics import classification_report, confusion_matrix

from data.datasets import make_loaders
from utils.config import load_config
from utils.model_factory import build_audio_exit_net, load_run_model_cfg


def _parse_tap_blocks(value):
    """
    Parse tap blocks from:
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


def _load_run_cfg(run_dir: str):
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        return {}
    try:
        return load_config(cfg_path) or {}
    except Exception:
        return {}


@torch.no_grad()
def main(
    run_dir,
    segments_csv,
    features_root,
    num_classes=2,
    n_mels=None,
    tap_blocks=None,
    batch_size=64,
    num_workers=4,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dl_tr, dl_va, dl_te, label2id = make_loaders(
        segments_csv, features_root, batch_size, num_workers
    )

    run_cfg = _load_run_cfg(run_dir)
    run_model_cfg = load_run_model_cfg(run_dir)

    # Prefer CLI, else run config, else backward-compatible default
    tap_blocks = _parse_tap_blocks(tap_blocks)
    if tap_blocks is None:
        tap_blocks = _parse_tap_blocks((run_model_cfg or {}).get("tap_blocks"))
    if tap_blocks is None:
        tap_blocks = (1, 3)

    # Prefer CLI, else run config, else 64
    if n_mels is None:
        n_mels = int(((run_cfg.get("features") or {}).get("n_mels", 64)))
    else:
        n_mels = int(n_mels)

    # Keep C-class generic: prefer dataset class count
    num_classes_data = len(label2id)
    num_classes_cfg = int(num_classes)
    if num_classes_cfg != num_classes_data:
        print(
            f"[WARN] eval num_classes={num_classes_cfg} but dataset has "
            f"{num_classes_data}. Using dataset value."
        )
    num_classes = num_classes_data

    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=run_model_cfg,
    ).to(device)

    ckpt_path = os.path.join(run_dir, "ckpt", "best.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_true = []
    y_pred = None
    num_exits = None

    for x, y in dl_te:
        x = x.to(device)
        logits = model(x)

        if y_pred is None:
            num_exits = len(logits)
            y_pred = [[] for _ in range(num_exits)]

        for k, lg in enumerate(logits):
            preds = torch.argmax(lg, dim=1).cpu().numpy().tolist()
            y_pred[k].extend(preds)

        y_true.extend(y.numpy().tolist())

    if y_pred is None:
        raise RuntimeError("Test loader is empty. No predictions were generated.")

    reports = {}
    confusion_matrices = {}

    for k in range(num_exits):
        exit_name = f"exit{k + 1}"
        reports[exit_name] = classification_report(
            y_true,
            y_pred[k],
            output_dict=True,
            zero_division=0,
        )
        confusion_matrices[exit_name] = confusion_matrix(
            y_true,
            y_pred[k],
            labels=list(range(int(num_classes))),
        ).tolist()

    id2label = {int(v): str(k) for k, v in label2id.items()}

    out = {
        "num_exits": num_exits,
        "num_classes": int(num_classes),
        "tap_blocks": list(tap_blocks) if tap_blocks is not None else None,
        "n_mels": int(n_mels),
        "label2id": label2id,
        "id2label": id2label,
        "reports": reports,
        "confusion_matrices": confusion_matrices,
        "exit_hint": (run_model_cfg or {}).get("exit_hint", {}),
    }

    out_path = os.path.join(run_dir, "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved {out_path}")
    for k in range(num_exits):
        exit_name = f"exit{k + 1}"
        acc = reports[exit_name]["accuracy"]
        print(f"{exit_name} accuracy: {acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--n_mels", type=int, default=None)
    ap.add_argument(
        "--tap_blocks",
        type=str,
        default=None,
        help='Example: "1,2,3,4" for 5 exits total, or "1,3" for 3 exits total.',
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    main(
        run_dir=args.run_dir,
        segments_csv=args.segments_csv,
        features_root=args.features_root,
        num_classes=args.num_classes,
        n_mels=args.n_mels,
        tap_blocks=args.tap_blocks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )