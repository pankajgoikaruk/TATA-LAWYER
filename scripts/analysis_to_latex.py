# scripts/analysis_to_latex.py

from __future__ import annotations

import json
import argparse
import csv
import re
from pathlib import Path


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_classification_report_dict(d):
    """
    Heuristic: detect sklearn classification_report-style dict.
    """
    if not isinstance(d, dict):
        return False
    if "accuracy" in d:
        return True
    if "macro avg" in d:
        return True
    if "weighted avg" in d:
        return True
    return False


def _normalize_classification_per_exit(obj):
    """
    Accept either:
      1) direct per-exit dict:
           {"exit1": {...}, "exit2": {...}, ...}
      2) wrapped report.json structure:
           {"num_exits": ..., "reports": {"exit1": {...}, ...}, ...}

    Return:
      {"exit1": {...}, "exit2": {...}, ...}
    """
    if not isinstance(obj, dict):
        raise ValueError("classification_per_exit must be a dict.")

    # New wrapped structure from training.eval/report.json
    if "reports" in obj and isinstance(obj["reports"], dict):
        obj = obj["reports"]

    # Keep only exit-like keys with valid report dicts
    out = {}
    for k, v in obj.items():
        ks = str(k)
        if re.fullmatch(r"exit\d+", ks) and _is_classification_report_dict(v):
            out[ks] = v

    if len(out) == 0:
        raise ValueError(
            "Could not find any valid exit reports. Expected keys like exit1, exit2, ..."
        )

    return out


def _sorted_exit_names(classification_per_exit):
    return sorted(
        classification_per_exit.keys(),
        key=lambda s: int(str(s).replace("exit", "")),
    )


def make_latex_table(classification_per_exit, run_label="ASHADIP_V0 run", label_names=None):
    """
    Build a LaTeX table string from a classification_report-style dict per exit.
    """
    classification_per_exit = _normalize_classification_per_exit(classification_per_exit)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(rf"  \caption{{Classification metrics per exit for {run_label}.}}")
    lines.append(r"  \label{tab:" + run_label.replace(" ", "_").lower() + r"_cls}")
    lines.append(r"  \begin{tabular}{lrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Class / summary & Precision & Recall & F1-score & Support \\")
    lines.append(r"    \midrule")

    def add_exit_block(exit_name, exit_dict):
        lines.append(r"    \multicolumn{5}{c}{" + exit_name.capitalize() + r"} \\")
        lines.append(r"    \midrule")

        aggregate_keys = {"accuracy", "macro avg", "weighted avg"}
        class_keys = [k for k in exit_dict.keys() if k not in aggregate_keys]

        try:
            class_keys = sorted(class_keys, key=lambda x: int(x))
        except ValueError:
            class_keys = sorted(class_keys)

        for cls in class_keys:
            stats = exit_dict[cls]
            if not isinstance(stats, dict):
                continue

            prec = stats.get("precision", 0.0)
            rec = stats.get("recall", 0.0)
            f1 = stats.get("f1-score", 0.0)
            sup = stats.get("support", 0)

            if label_names is not None:
                try:
                    cls_idx = int(cls)
                    cls_name = label_names[cls_idx]
                except (ValueError, IndexError):
                    cls_name = str(cls)
            else:
                cls_name = str(cls)

            lines.append(
                rf"    {cls_name} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {int(sup)} \\"
            )

        acc = exit_dict.get("accuracy", None)
        if acc is not None:
            lines.append(r"    \midrule")
            lines.append(rf"    accuracy & \multicolumn{{3}}{{r}}{{{acc:.3f}}} & -- \\")

        for agg_key in ["macro avg", "weighted avg"]:
            if agg_key in exit_dict and isinstance(exit_dict[agg_key], dict):
                stats = exit_dict[agg_key]
                prec = stats.get("precision", 0.0)
                rec = stats.get("recall", 0.0)
                f1 = stats.get("f1-score", 0.0)
                sup = stats.get("support", 0)
                lines.append(
                    rf"    {agg_key} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {int(sup)} \\"
                )

        lines.append(r"    \midrule")

    for exit_name in _sorted_exit_names(classification_per_exit):
        add_exit_block(exit_name, classification_per_exit[exit_name])

    for i in range(len(lines) - 1, -1, -1):
        if r"\midrule" in lines[i]:
            lines[i] = lines[i].replace(r"\midrule", r"\bottomrule")
            break

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def write_csv_and_txt(classification_per_exit, out_tex_path: Path, label_names=None):
    classification_per_exit = _normalize_classification_per_exit(classification_per_exit)

    csv_path = out_tex_path.with_suffix(".csv")
    txt_path = out_tex_path.with_suffix(".txt")

    rows = []
    header = ["exit", "class", "type", "precision", "recall", "f1", "support"]

    for exit_name in _sorted_exit_names(classification_per_exit):
        exit_dict = classification_per_exit[exit_name]

        aggregate_keys = {"accuracy", "macro avg", "weighted avg"}
        class_keys = [k for k in exit_dict.keys() if k not in aggregate_keys]

        try:
            class_keys = sorted(class_keys, key=lambda x: int(x))
        except ValueError:
            class_keys = sorted(class_keys)

        for cls in class_keys:
            stats = exit_dict[cls]
            if not isinstance(stats, dict):
                continue

            prec = stats.get("precision", None)
            rec = stats.get("recall", None)
            f1 = stats.get("f1-score", None)
            sup = stats.get("support", None)

            if label_names is not None:
                try:
                    cls_idx = int(cls)
                    cls_name = label_names[cls_idx]
                except (ValueError, IndexError):
                    cls_name = str(cls)
            else:
                cls_name = str(cls)

            rows.append([
                exit_name,
                cls_name,
                "class",
                f"{prec:.6f}" if prec is not None else "",
                f"{rec:.6f}" if rec is not None else "",
                f"{f1:.6f}" if f1 is not None else "",
                int(sup) if sup is not None else "",
            ])

        if "accuracy" in exit_dict:
            acc = exit_dict["accuracy"]
            rows.append([
                exit_name,
                "",
                "accuracy",
                "",
                "",
                f"{acc:.6f}",
                "",
            ])

        for agg_key in ["macro avg", "weighted avg"]:
            if agg_key in exit_dict and isinstance(exit_dict[agg_key], dict):
                stats = exit_dict[agg_key]
                prec = stats.get("precision", None)
                rec = stats.get("recall", None)
                f1 = stats.get("f1-score", None)
                sup = stats.get("support", None)

                rows.append([
                    exit_name,
                    "",
                    agg_key,
                    f"{prec:.6f}" if prec is not None else "",
                    f"{rec:.6f}" if rec is not None else "",
                    f"{f1:.6f}" if f1 is not None else "",
                    int(sup) if sup is not None else "",
                ])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    print(f"[analysis_to_latex] Wrote CSV to {csv_path} with {len(rows)} rows")
    print(f"[analysis_to_latex] Wrote TXT to {txt_path} with {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis_json",
        required=True,
        help="Path to analysis_run.json for a single run.",
    )
    ap.add_argument(
        "--out_tex",
        required=True,
        help="Output .tex file to write the table into.",
    )
    ap.add_argument(
        "--run_label",
        default="ASHADIP_V0 run",
        help="Label used in the table caption and label.",
    )
    args = ap.parse_args()

    analysis_path = Path(args.analysis_json)
    out_tex_path = Path(args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    analysis = load_json(analysis_path, default={}) or {}
    cls = analysis.get("classification_per_exit")
    label_names = analysis.get("label_names")

    if not cls:
        run_dir = analysis_path.parent
        report_path = run_dir / "report.json"
        report = load_json(report_path, default=None)
        if report is None:
            raise SystemExit(
                f"No 'classification_per_exit' in {analysis_path} and "
                f"no report.json at {report_path}."
            )
        cls = report
        print(f"[analysis_to_latex] Using classification metrics from {report_path}")
    else:
        print(f"[analysis_to_latex] Using classification_per_exit from {analysis_path}")

    table_str = make_latex_table(cls, run_label=args.run_label, label_names=label_names)
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"[analysis_to_latex] Wrote LaTeX table to {out_tex_path}")

    write_csv_and_txt(cls, out_tex_path, label_names=label_names)


if __name__ == "__main__":
    main()