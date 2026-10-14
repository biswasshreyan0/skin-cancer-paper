"""Explicit, auditable prediction-target definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    name: str
    label_column: str
    description: str
    class_names: tuple[str, ...]
    positive_diagnoses: tuple[str, ...] = ()
    negative_diagnoses: tuple[str, ...] = ()
    excluded_diagnoses: tuple[str, ...] = ()

    @property
    def is_binary(self) -> bool:
        return len(self.class_names) == 2

    def to_dict(self) -> dict:
        return asdict(self)


TARGET_SPECS = {
    "cancer": TargetSpec(
        name="cancer_vs_non_cancer",
        label_column="cancer_label",
        description=(
            "Cancer (melanoma or basal cell carcinoma) versus non-cancer "
            "(nevus, benign keratosis, dermatofibroma, or vascular lesion). "
            "The mixed akiec category is excluded because it combines actinic "
            "keratoses with intraepithelial carcinoma."
        ),
        class_names=("non_cancer", "cancer"),
        positive_diagnoses=("mel", "bcc"),
        negative_diagnoses=("nv", "bkl", "df", "vasc"),
        excluded_diagnoses=("akiec",),
    ),
    "melanoma": TargetSpec(
        name="melanoma_vs_other",
        label_column="melanoma_label",
        description="Melanoma (mel) versus every other HAM10000 diagnosis.",
        class_names=("non_melanoma", "melanoma"),
        positive_diagnoses=("mel",),
        negative_diagnoses=("akiec", "bcc", "bkl", "df", "nv", "vasc"),
    ),
    "multiclass": TargetSpec(
        name="seven_class_diagnosis",
        label_column="multiclass_label",
        description="Seven-class HAM10000 diagnosis typing.",
        class_names=("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"),
    ),
}


def canonical_task(task: str) -> str:
    """Normalize the deprecated ``binary`` spelling to the melanoma task."""
    return "melanoma" if task == "binary" else task


def get_target_spec(task: str) -> TargetSpec:
    task = canonical_task(task)
    if task not in TARGET_SPECS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TARGET_SPECS)}.")
    return TARGET_SPECS[task]


def prepare_task_dataframe(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Return rows with an unambiguous label for the requested target."""
    spec = get_target_spec(task)
    prepared = df.copy()
    if spec.excluded_diagnoses:
        prepared = prepared[~prepared["dx"].isin(spec.excluded_diagnoses)].copy()
    if prepared[spec.label_column].isna().any() or (prepared[spec.label_column] == -1).any():
        raise ValueError(f"Unresolved labels remain for target {spec.name!r}.")
    return prepared

