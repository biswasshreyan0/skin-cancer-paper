"""Generate the final Table 1 and its clinical-interpretation companion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.targets import TargetSpec
from src.utils.io import write_json


def write_final_table(
    leaderboard: pd.DataFrame,
    metrics: dict,
    out_dir: str | Path,
    target: TargetSpec,
    run_manifest: dict,
    cv_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write one fully traceable row per final held-out model result."""
    out_dir = Path(out_dir)
    reports_dir = out_dir / "reports"
    rows = []
    for item in leaderboard.itertuples(index=False):
        feature_set = item.feature_set
        model = item.model
        payload = metrics.get(feature_set, {}).get(model, {})
        test = payload.get("test", {})
        prediction_path = out_dir / "predictions" / f"{feature_set}_{model}_predictions.csv"
        row = {
            "target": target.name,
            "target_definition": target.description,
            "label_column": target.label_column,
            "negative_class": target.class_names[0] if target.is_binary else "",
            "positive_class": target.class_names[1] if target.is_binary else "",
            "positive_dx": "|".join(target.positive_diagnoses),
            "negative_dx": "|".join(target.negative_diagnoses),
            "excluded_dx": "|".join(target.excluded_diagnoses),
            "decision_threshold": 0.5 if target.is_binary else np.nan,
            "feature_set": feature_set,
            "model": model,
            "accuracy": test.get("accuracy"),
            "balanced_accuracy": test.get("balanced_accuracy"),
            "ROC-AUC": test.get("ROC-AUC", test.get("macro ROC-AUC")),
            "AUPRC": test.get("Average Precision", test.get("macro AUPRC")),
            "training_seconds": payload.get("training_seconds"),
            "imbalance_strategy": payload.get("imbalance_strategy"),
            "heldout_predictions": str(prediction_path),
            "run_command": run_manifest.get("command"),
            "code_commit": run_manifest.get("code_commit"),
            "git_dirty": run_manifest.get("git_dirty"),
        }
        for class_name, class_metrics in test.get("per_class", {}).items():
            for metric_name in ("precision", "recall", "F1", "AUPRC"):
                row[f"{class_name}_{metric_name}"] = class_metrics.get(metric_name)
        rows.append(row)
    table = pd.DataFrame(rows)
    if cv_summary is not None and not cv_summary.empty:
        cv_columns = [
            column
            for column in cv_summary.columns
            if column in {"feature_set", "model"}
            or column.startswith(("AUPRC_", "ROC-AUC_", "balanced_accuracy_", "F1_"))
        ]
        # Table 1 is limited to rows with repeated-CV uncertainty estimates.
        # One-off neural baselines remain in leaderboard.csv as supplemental
        # results rather than being mixed with mean±SD rows.
        table = table.merge(cv_summary[cv_columns], on=["feature_set", "model"], how="inner")
    sort_column = "AUPRC_mean" if "AUPRC_mean" in table else "AUPRC"
    table = table.sort_values(sort_column, ascending=False, na_position="last").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.to_csv(reports_dir / "table1_final_results.csv", index=False)
    write_json(
        {
            "target": target.to_dict(),
            "command": run_manifest.get("command"),
            "code_commit": run_manifest.get("code_commit"),
            "git_dirty": run_manifest.get("git_dirty"),
            "heldout_split": run_manifest.get("split_files", {}).get("test"),
            "table": str(reports_dir / "table1_final_results.csv"),
        },
        reports_dir / "table1_provenance.json",
    )
    _write_clinical_implications(table, reports_dir / "clinical_implications.md", target)
    return table


def _write_clinical_implications(table: pd.DataFrame, path: Path, target: TargetSpec) -> None:
    if table.empty:
        path.write_text("# Clinical implications\n\nNo completed models were available.\n", encoding="utf-8")
        return
    best = table.iloc[0]
    positive = target.class_names[1] if target.is_binary else None
    positive_recall = best.get(f"{positive}_recall") if positive else np.nan
    positive_precision = best.get(f"{positive}_precision") if positive else np.nan
    positive_auprc = best.get(f"{positive}_AUPRC") if positive else np.nan
    content = f"""# Clinical implications

The highest-ranked repeated-CV row is `{best['feature_set']}` / `{best['model']}` for the target **{target.name}**. Its final held-out positive-class precision is {_format(positive_precision)}, recall is {_format(positive_recall)}, and AUPRC is {_format(positive_auprc)}.

This experiment is a research benchmark, not a diagnostic device. A false negative in the positive class could delay biopsy or treatment; a false positive could lead to anxiety, specialist referral, or unnecessary biopsy. Threshold selection therefore requires a prespecified clinical use case and external validation. The default 0.5 threshold is reported for reproducibility, not asserted to be clinically optimal.

HAM10000 is enriched for selected dermatoscopic lesions and does not establish performance in population screening, primary care photographs, diverse skin tones, or out-of-distribution disease. The `akiec` category is excluded from the strict cancer target because it mixes actinic keratoses with intraepithelial carcinoma; the model therefore must not be interpreted as ruling out every premalignant or malignant lesion.

The ABCD variables are computational image proxies. Evolution is not used because longitudinal observations are unavailable. The clinical diameter warning sign is commonly stated as greater than 6 mm, but HAM10000 provides no physical pixel-to-millimeter scale, so `diameter_fraction` is not a 6 mm measurement and must not be presented as one.
"""
    path.write_text(content, encoding="utf-8")


def _format(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "not available"
