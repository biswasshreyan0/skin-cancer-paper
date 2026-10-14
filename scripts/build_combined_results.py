#!/usr/bin/env python
"""Combine saved final-holdout metrics and per-class metrics into CSV tables.

This script does not train models. It reads existing metrics and prediction
artifacts under outputs/ and writes combined, target-labelled report tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    metrics_path = reports / "metrics.json"
    leaderboard_path = reports / "leaderboard.csv"
    if not metrics_path.exists() or not leaderboard_path.exists():
        raise FileNotFoundError("Expected reports/metrics.json and reports/leaderboard.csv")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(leaderboard_path)

    rows: list[dict] = []
    per_class_rows: list[dict] = []
    for feature_set, models in metrics.items():
        for model, payload in models.items():
            test = payload.get("test", {})
            base = {
                "target": "cancer_vs_non_cancer",
                "feature_set": feature_set,
                "model": model,
                "accuracy": test.get("accuracy"),
                "balanced_accuracy": test.get("balanced_accuracy"),
                "ROC-AUC": test.get("ROC-AUC"),
                "average_precision": test.get("Average Precision"),
                "F1": test.get("F1"),
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "confusion_matrix": json.dumps(test.get("confusion_matrix")),
                "prediction_file": f"predictions/{feature_set}_{model}_predictions.csv",
            }
            row = dict(base)
            for class_name, values in test.get("per_class", {}).items():
                for metric in ("precision", "recall", "F1", "AUPRC"):
                    row[f"{class_name}_{metric}"] = values.get(metric)
                    per_class_rows.append({
                        "target": base["target"], "feature_set": feature_set,
                        "model": model, "class": class_name, **values,
                    })
            rows.append(row)

    combined = pd.DataFrame(rows)
    # Include legacy leaderboard fields that may not be present in metrics.json.
    combined = combined.merge(leaderboard, on=["feature_set", "model"], how="left", suffixes=("", "_leaderboard"))
    combined.to_csv(reports / "combined_results.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(reports / "combined_per_class_metrics.csv", index=False)

    # Confusion counts recovered from saved predictions for the binary cancer
    # task, with dataset diagnoses attached to support class-specific review.
    split_path = out_dir / "splits" / "test.csv"
    if split_path.exists():
        split = pd.read_csv(split_path)
        diagnosis = split[["image_id", "dx"]].drop_duplicates("image_id")
        confusion_rows = []
        for pred_path in sorted((out_dir / "predictions").glob("*_predictions.csv")):
            try:
                pred = pd.read_csv(pred_path)
            except (OSError, pd.errors.ParserError):
                continue
            if not {"image_id", "true_label", "predicted_label"}.issubset(pred.columns):
                continue
            joined = pred.merge(diagnosis, on="image_id", how="inner")
            if len(joined) != len(pred):
                continue
            for dx, group in joined.groupby("dx"):
                for true_label, pred_label in zip(group.true_label, group.predicted_label):
                    confusion_rows.append({
                        "target": "binary_task_used_by_saved_prediction",
                        "prediction_file": str(pred_path.relative_to(out_dir)),
                        "diagnosis": dx,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                    })
        pd.DataFrame(confusion_rows).to_csv(reports / "prediction_confusion_detail.csv", index=False)

    print(f"Wrote {reports / 'combined_results.csv'} ({len(combined)} model rows)")
    print(f"Wrote {reports / 'combined_per_class_metrics.csv'}")


if __name__ == "__main__":
    main()
