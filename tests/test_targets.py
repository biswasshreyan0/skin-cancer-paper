"""Tests for explicit task semantics and imbalance handling."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.data.targets import get_target_spec, prepare_task_dataframe
from src.training.train_tabular import _training_data_for_imbalance


class TargetDefinitionTests(unittest.TestCase):
    def test_cancer_target_is_exact_and_excludes_mixed_akiec(self) -> None:
        diagnoses = ["mel", "bcc", "nv", "bkl", "df", "vasc", "akiec"]
        labels = [1, 1, 0, 0, 0, 0, -1]
        frame = pd.DataFrame({"dx": diagnoses, "cancer_label": labels})

        result = prepare_task_dataframe(frame, "cancer")
        spec = get_target_spec("cancer")

        self.assertEqual(spec.positive_diagnoses, ("mel", "bcc"))
        self.assertEqual(spec.excluded_diagnoses, ("akiec",))
        self.assertEqual(result["dx"].tolist(), diagnoses[:-1])
        self.assertEqual(result["cancer_label"].tolist(), labels[:-1])

    def test_hybrid_oversamples_knn_training_data_only(self) -> None:
        X = np.arange(10).reshape(5, 2)
        y = np.array([0, 0, 0, 0, 1])

        sampled_X, sampled_y = _training_data_for_imbalance(
            KNeighborsClassifier(), X, y, imbalance_strategy="hybrid", seed=42
        )

        self.assertEqual(sampled_X.shape[0], 8)
        np.testing.assert_array_equal(np.bincount(sampled_y), np.array([4, 4]))


if __name__ == "__main__":
    unittest.main()
