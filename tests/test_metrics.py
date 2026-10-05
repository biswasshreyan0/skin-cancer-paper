"""Regression tests for aggregate and per-class classification metrics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.utils.metrics import binary_metrics, multiclass_metrics


class PerClassMetricsTests(unittest.TestCase):
    def test_binary_reports_both_clinical_classes_and_auprc(self) -> None:
        metrics = binary_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            np.array([0.1, 0.8, 0.7, 0.9]),
            class_names=("non_mel", "mel"),
        )

        self.assertEqual(set(metrics["per_class"]), {"non_mel", "mel"})
        self.assertAlmostEqual(metrics["per_class"]["mel"]["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["per_class"]["mel"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["mel"]["F1"], 0.8)
        self.assertAlmostEqual(metrics["per_class"]["mel"]["AUPRC"], 5 / 6)

    def test_multiclass_uses_probability_column_class_order(self) -> None:
        metrics = multiclass_metrics(
            np.array(["akiec", "bcc", "mel"]),
            np.array(["akiec", "mel", "mel"]),
            np.array(
                [
                    [0.9, 0.05, 0.05, 0.0],
                    [0.1, 0.2, 0.7, 0.0],
                    [0.1, 0.1, 0.8, 0.0],
                ]
            ),
            class_labels=["akiec", "bcc", "mel", "vasc"],
        )

        self.assertEqual(set(metrics["per_class"]), {"akiec", "bcc", "mel", "vasc"})
        self.assertAlmostEqual(metrics["per_class"]["mel"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["mel"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["mel"]["AUPRC"], 1.0)
        self.assertTrue(math.isnan(metrics["per_class"]["vasc"]["AUPRC"]))


if __name__ == "__main__":
    unittest.main()
