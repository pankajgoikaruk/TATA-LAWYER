# training/train.py

from __future__ import annotations

import os
import sys
import argparse
import random
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score

from utils.config import parse_args_with_config, ensure_dirs, save_config
from utils.logging import make_run_dir, save_json
from data.datasets import make_loaders
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


def _parse_tap_blocks(value) -> Optional[tuple]:
    """Accept None, "1,2,3,4", [1,2,3,4], or (1,2,3,4)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    value = str(value).strip()
    if value == "":
        return None
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _pad_or_trim(
    seq: Sequence[float],
    target_len: int,
    pad_value: Optional[float] = None,
) -> List[float]:
    """Pad/trim numeric sequence to target_len."""
    vals = [float(x) for x in seq]
    if len(vals) >= target_len:
        return vals[:target_len]
    if pad_value is None:
        pad_value = vals[-1] if len(vals) > 0 else 1.0
    vals = vals + [float(pad_value)] * (target_len - len(vals))
    return vals


def _default_loss_weights_for_num_exits(num_exits: int) -> List[float]:
    """
    Dynamic default loss schedules.

    Recommended defaults:
      - 3 exits: [0.3, 0.3, 1.0]
      - 5 exits: [0.3, 0.3, 0.6, 0.8, 1.0]
    """
    if num_exits == 3:
        return [0.3, 0.3, 1.0]
    if num_exits == 5:
        return [0.3, 0.3, 0.6, 0.8, 1.0]
    if num_exits <= 1:
        return [1.0]
    return [0.3] * (num_exits - 1) + [1.0]


def _resolve_loss_weights(cfg: dict, num_exits: int) -> List[float]:
    """
    Resolve effective exit-loss weights.

    1. Missing/empty train.loss_weights -> dynamic defaults.
    2. Wrong length -> pad/trim and warn.
    3. Always returns length == num_exits.
    """
    tr = cfg.setdefault("train", {})
    raw = tr.get("loss_weights", None)

    if raw is None or (isinstance(raw, (list, tuple)) and len(raw) == 0):
        loss_w = _default_loss_weights_for_num_exits(num_exits)
        print(
            f"[INFO] train.loss_weights missing/empty -> using dynamic defaults "
            f"for {num_exits} exits: {loss_w}"
        )
        return loss_w

    loss_w = [float(x) for x in raw]
    if len(loss_w) != num_exits:
        print(
            f"[WARN] train.loss_weights has length {len(loss_w)} but model has "
            f"{num_exits} exits. Auto-adjusting."
        )
        loss_w = _pad_or_trim(loss_w, num_exits)
    return loss_w


def _parse_optional_bool(value):
    """Parse optional boolean from CLI."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}. Use true/false.")


def _class_weights_from_dataset(ds, num_classes: int, power: float, device: str):
    """
    Build mean-normalised inverse-frequency class weights.

    power=0.0 disables class weights.
    power=0.5 gives moderate sqrt balancing.
    power=1.0 gives full inverse-frequency balancing.
    """
    power = float(power or 0.0)
    if power <= 0:
        return None

    counts_dict = ds.class_counts_by_id()
    counts = np.array([max(float(counts_dict.get(i, 0)), 1.0) for i in range(num_classes)], dtype=np.float64)
    weights = np.power(counts, -power)
    weights = weights / max(float(weights.mean()), 1e-12)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    print(f"[INFO] class_weight_power={power} -> weights={weights_t.detach().cpu().numpy().round(4).tolist()}")
    return weights_t


def _single_exit_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
):
    label_smoothing = float(label_smoothing or 0.0)
    focal_gamma = float(focal_gamma or 0.0)

    if focal_gamma <= 0:
        return F.cross_entropy(
            logits,
            y,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    ce = F.cross_entropy(
        logits,
        y,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, y.view(-1, 1)).squeeze(1).clamp_min(1e-6)
    return (((1.0 - pt) ** focal_gamma) * ce).mean()


def train_one_epoch(
    model,
    dl,
    opt,
    device,
    loss_w,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
):
    model.train()
    loss_sum, n = 0.0, 0
    correct = None

    for x, y in dl:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)

        if correct is None:
            correct = [0 for _ in range(len(logits))]

        losses = [
            _single_exit_loss(
                lg,
                y,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
            )
            for lg in logits
        ]
        weights = _pad_or_trim(loss_w, len(losses))
        loss = sum(w * l for w, l in zip(weights, losses))

        loss.backward()
        opt.step()

        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs

        for k, lg in enumerate(logits):
            pred = lg.argmax(1)
            correct[k] += int((pred == y).sum())

    if correct is None:
        correct = []

    acc = [c / max(n, 1) for c in correct]
    return loss_sum / max(n, 1), acc


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct = None
    n = 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        if correct is None:
            correct = [0 for _ in range(len(logits))]

        n += x.size(0)
        for k, lg in enumerate(logits):
            pred = lg.argmax(1)
            correct[k] += int((pred == y).sum())

    if correct is None:
        correct = []

    return [c / max(n, 1) for c in correct]


@torch.no_grad()
def evaluate_metrics(model, dl, device, num_classes: int):
    """Return per-exit accuracy and macro-F1 for validation/test monitoring."""
    model.eval()
    y_true = []
    y_pred = None

    for x, y in dl:
        x = x.to(device)
        logits = model(x)

        if y_pred is None:
            y_pred = [[] for _ in range(len(logits))]

        y_list = y.cpu().numpy().tolist()
        y_true.extend(y_list)
        for k, lg in enumerate(logits):
            y_pred[k].extend(torch.argmax(lg, dim=1).cpu().numpy().tolist())

    if y_pred is None:
        return [], []

    labels = list(range(int(num_classes)))
    acc = []
    macro_f1 = []
    y_true_arr = np.asarray(y_true)
    for preds in y_pred:
        preds_arr = np.asarray(preds)
        acc.append(float(np.mean(preds_arr == y_true_arr)) if len(y_true_arr) else 0.0)
        macro_f1.append(
            float(f1_score(y_true, preds, labels=labels, average="macro", zero_division=0))
        )
    return acc, macro_f1


def _parse_extra_args():
    """Parse optional args without breaking parse_args_with_config()."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument("--run_dir", type=str, default=None, help="Explicit run directory to write outputs.")
    p.add_argument("--device", type=str, default=None, help="Force device: cpu | cuda (default: auto).")
    p.add_argument("--cache_dir", type=str, default=None, help="Override cache directory containing segments.csv + features/.")
    p.add_argument("--segment_sec", type=float, default=None, help="Optional: record segment_sec in effective config.")
    p.add_argument("--hop_sec", type=float, default=None, help="Optional: record hop_sec in effective config.")
    p.add_argument("--variant", type=str, default=None, help="Optional: record variant name in effective config.")
    p.add_argument("--tap_blocks", type=str, default=None, help='Comma-separated tap blocks. Example: "1,3" or "1,2,3,4".')
    p.add_argument("--exit_hint_enable", type=_parse_optional_bool, default=None, help="Override model.exit_hint.enable from CLI.")

    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    extra = _parse_extra_args()
    cfg = parse_args_with_config()

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    # ---------------- Device ----------------
    if extra.device is not None:
        device = extra.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Paths ----------------
    paths = (cfg.get("paths") or {})
    runs_root = paths.get("runs_root", "runs")
    cache_root = paths.get("cache_root", "data_cache")

    if extra.cache_dir:
        cache_root = extra.cache_dir

    ensure_dirs(runs_root)

    # ---------------- Run dir ----------------
    if extra.run_dir:
        run_dir = extra.run_dir
        ensure_dirs(run_dir)
    else:
        run_dir = make_run_dir(runs_root)

    ckpt_dir = os.path.join(run_dir, "ckpt")
    ensure_dirs(ckpt_dir)

    # ---------------- Runtime config ----------------
    cfg.setdefault("paths", {})
    cfg["paths"]["runs_root"] = runs_root
    cfg["paths"]["cache_root"] = cache_root

    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = device
    if extra.variant is not None:
        cfg["runtime"]["variant"] = extra.variant

    if extra.segment_sec is not None or extra.hop_sec is not None:
        cfg.setdefault("audio", {})
        if extra.segment_sec is not None:
            cfg["audio"]["segment_sec"] = float(extra.segment_sec)
        if extra.hop_sec is not None:
            cfg["audio"]["segment_hop"] = float(extra.hop_sec)

    # ---------------- Model config overrides ----------------
    cfg.setdefault("model", {})
    cfg["model"].setdefault("exit_hint", {})

    if extra.exit_hint_enable is not None:
        cfg["model"]["exit_hint"]["enable"] = bool(extra.exit_hint_enable)

    model_cfg = (cfg.get("model") or {})
    tap_blocks = (
        _parse_tap_blocks(extra.tap_blocks)
        or _parse_tap_blocks(model_cfg.get("tap_blocks"))
        or (1, 3)
    )
    cfg["model"]["tap_blocks"] = list(tap_blocks)

    # ---------------- Data ----------------
    seg_csv = os.path.join(cache_root, "segments.csv")
    feat_root = os.path.join(cache_root, "features")

    tr = (cfg.get("train") or {})
    bs = int(tr.get("batch_size", 64))
    nw = int(tr.get("num_workers", 4))
    lr = float(tr.get("lr", 1e-3))
    wd = float(tr.get("weight_decay", 0.0))
    epochs = int(tr.get("epochs", 40))

    class_balance_power = float(tr.get("class_balance_power", 0.0) or 0.0)
    source_balance_power = float(tr.get("source_balance_power", 0.0) or 0.0)
    class_weight_power = float(tr.get("class_weight_power", 0.0) or 0.0)
    label_smoothing = float(tr.get("label_smoothing", 0.0) or 0.0)
    focal_gamma = float(tr.get("focal_gamma", 0.0) or 0.0)
    best_metric = str(tr.get("best_metric", "final_acc") or "final_acc").strip().lower()

    dl_tr, dl_va, dl_te, label2id = make_loaders(
        seg_csv,
        feat_root,
        bs,
        nw,
        seed=seed,
        class_balance_power=class_balance_power,
        source_balance_power=source_balance_power,
    )

    # ---------------- Model ----------------
    n_mels = int((cfg.get("features") or {}).get("n_mels", 64))

    num_classes_data = len(label2id)
    num_classes_cfg = int((cfg.get("model") or {}).get("num_classes", num_classes_data))
    if num_classes_cfg != num_classes_data:
        print(
            f"[WARN] config num_classes={num_classes_cfg} but dataset has "
            f"{num_classes_data}. Using dataset value."
        )
    num_classes = num_classes_data
    cfg["model"]["num_classes"] = int(num_classes)

    model = build_audio_exit_net(
        num_classes=num_classes,
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=(cfg.get("model") or {}),
    ).to(device)

    num_exits = int(model.num_exits)
    cfg["model"]["exits"] = num_exits

    loss_w = _resolve_loss_weights(cfg, num_exits)
    cfg.setdefault("train", {})
    cfg["train"]["loss_weights"] = [float(x) for x in loss_w]
    cfg["train"]["class_balance_power"] = class_balance_power
    cfg["train"]["source_balance_power"] = source_balance_power
    cfg["train"]["class_weight_power"] = class_weight_power
    cfg["train"]["label_smoothing"] = label_smoothing
    cfg["train"]["focal_gamma"] = focal_gamma
    cfg["train"]["best_metric"] = best_metric

    class_weights = _class_weights_from_dataset(
        dl_tr.dataset,
        num_classes=num_classes,
        power=class_weight_power,
        device=device,
    )

    save_config(cfg, os.path.join(run_dir, "config_used.yaml"))

    print(
        f"[INFO] effective model config -> tap_blocks={list(tap_blocks)}, "
        f"num_exits={num_exits}, loss_weights={loss_w}, "
        f"exit_hint_enable={cfg.get('model', {}).get('exit_hint', {}).get('enable', False)}"
    )
    print(
        f"[INFO] training balance -> class_sampler_power={class_balance_power}, "
        f"source_sampler_power={source_balance_power}, class_weight_power={class_weight_power}, "
        f"label_smoothing={label_smoothing}, focal_gamma={focal_gamma}, "
        f"best_metric={best_metric}"
    )

    # ---------------- Optimizer ----------------
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)

    metrics = {
        "meta": {
            "num_exits": num_exits,
            "tap_blocks": list(tap_blocks),
            "tap_dims": list(model.tap_dims) if hasattr(model, "tap_dims") else [],
            "final_dim": int(model.final_dim) if hasattr(model, "final_dim") else -1,
            "num_classes": num_classes,
            "loss_weights_used": loss_w,
            "exit_hint": (cfg.get("model") or {}).get("exit_hint", {}),
            "label2id": label2id,
            "train_class_counts": dl_tr.dataset.class_counts_by_label(),
            "class_balance_power": class_balance_power,
            "source_balance_power": source_balance_power,
            "class_weight_power": class_weight_power,
            "label_smoothing": label_smoothing,
            "focal_gamma": focal_gamma,
            "best_metric": best_metric,
        },
        "train": [],
        "val": [],
    }

    best = -1.0

    # ---------------- Training loop ----------------
    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(
            model,
            dl_tr,
            opt,
            device,
            loss_w,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )
        va_acc, va_macro_f1 = evaluate_metrics(model, dl_va, device, num_classes=num_classes)

        metrics["train"].append({"epoch": ep + 1, "loss": tr_loss, "acc": tr_acc})
        metrics["val"].append({"epoch": ep + 1, "acc": va_acc, "macro_f1": va_macro_f1})

        final_acc = va_acc[-1] if va_acc else 0.0
        final_macro_f1 = va_macro_f1[-1] if va_macro_f1 else 0.0
        print(
            f"Epoch {ep + 1}: loss={tr_loss:.4f}, "
            f"acc@exits={va_acc}, macroF1@exits={va_macro_f1}"
        )

        if best_metric in {"macro_f1", "final_macro_f1", "f1"}:
            score = final_macro_f1
        else:
            score = final_acc

        if score > best:
            best = score
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

    metrics["meta"]["best_score"] = float(best)
    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
