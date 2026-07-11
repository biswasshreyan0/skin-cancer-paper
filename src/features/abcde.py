"""Approximate ABCDE-style feature extraction from lesion images."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except Exception:
    cv2 = None

ABCDE_FEATURE_COLUMNS = [
    "asymmetry_score",
    "border_irregularity",
    "compactness",
    "area_ratio",
    "color_variance",
    "color_entropy",
    "diameter_proxy",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
]


def extract_abcde_dataframe(
    image_paths: Iterable[str | Path],
    image_ids: Iterable[str] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract one ABCDE feature row per image path."""
    _require_cv2()
    image_paths = list(image_paths)
    image_ids = list(image_ids) if image_ids is not None else [Path(p).stem for p in image_paths]
    rows = []
    iterator = zip(image_ids, image_paths)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Extracting ABCDE features")
    for image_id, image_path in iterator:
        row = extract_abcde_features(image_path)
        row["image_id"] = image_id
        row["image_path"] = str(image_path)
        rows.append(row)
    return pd.DataFrame(rows)


def extract_abcde_features(image: str | Path | Image.Image) -> dict[str, float | bool]:
    """Extract approximate asymmetry, border, color, and diameter features."""
    _require_cv2()
    try:
        rgb = _load_rgb(image)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        mask, contour = _segment_lesion(rgb)
        if mask is None or contour is None:
            return _failed_features()

        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0 or perimeter <= 0:
            return _failed_features()

        mask_bool = mask.astype(bool)
        lesion_pixels = rgb[mask_bool]
        if lesion_pixels.size == 0:
            return _failed_features()

        x, y, w, h = cv2.boundingRect(contour)
        border_irregularity = (perimeter * perimeter) / max(area, 1.0)
        compactness = (4.0 * math.pi * area) / max(perimeter * perimeter, 1.0)

        features: dict[str, float | bool] = {
            "asymmetry_score": _asymmetry_score(mask),
            "border_irregularity": border_irregularity,
            "compactness": compactness,
            "area_ratio": float(mask_bool.mean()),
            "color_variance": float(np.var(lesion_pixels.astype(np.float32))),
            "color_entropy": _color_entropy(lesion_pixels),
            "diameter_proxy": float(max(w, h) / 224.0),
            "mean_r": float(np.mean(lesion_pixels[:, 0])),
            "mean_g": float(np.mean(lesion_pixels[:, 1])),
            "mean_b": float(np.mean(lesion_pixels[:, 2])),
            "std_r": float(np.std(lesion_pixels[:, 0])),
            "std_g": float(np.std(lesion_pixels[:, 1])),
            "std_b": float(np.std(lesion_pixels[:, 2])),
            "segmentation_failed": False,
        }
        return features
    except Exception:
        return _failed_features()


def _load_rgb(image: str | Path | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = Image.open(Path(image)).convert("RGB")
    return np.asarray(pil)


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "ABCDE feature extraction requires opencv-python. "
            "Install project dependencies with `pip install -r requirements.txt`."
        )


def _segment_lesion(rgb: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [binary, cv2.bitwise_not(binary)]

    best_mask = None
    best_contour = None
    best_score = -1.0
    image_area = float(rgb.shape[0] * rgb.shape[1])
    for candidate in masks:
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / image_area
            if area_ratio < 0.01 or area_ratio > 0.95:
                continue
            score = area * (1.0 - abs(area_ratio - 0.25))
            if score > best_score:
                mask = np.zeros(candidate.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
                best_mask = mask
                best_contour = contour
                best_score = score
    return best_mask, best_contour


def _asymmetry_score(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(bool)
    h, w = crop.shape
    left = crop[:, : w // 2]
    right = np.fliplr(crop[:, w - w // 2 :])
    top = crop[: h // 2, :]
    bottom = np.flipud(crop[h - h // 2 :, :])
    lr = _mismatch(left, right)
    tb = _mismatch(top, bottom)
    return float((lr + tb) / 2.0)


def _mismatch(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    common_shape = (min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1]))
    a = a[: common_shape[0], : common_shape[1]]
    b = b[: common_shape[0], : common_shape[1]]
    denom = max(float(np.logical_or(a, b).sum()), 1.0)
    return float(np.logical_xor(a, b).sum() / denom)


def _color_entropy(lesion_pixels: np.ndarray) -> float:
    gray = np.dot(lesion_pixels[:, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    hist = np.bincount(gray, minlength=256).astype(np.float64)
    probs = hist / max(hist.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _failed_features() -> dict[str, float | bool]:
    values: dict[str, float | bool] = {name: 0.0 for name in ABCDE_FEATURE_COLUMNS}
    values["segmentation_failed"] = True
    return values
