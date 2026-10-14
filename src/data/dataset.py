"""PyTorch image dataset for the optional CNN baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str = "cancer_label", transform=None):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row[self.label_col]), row["image_id"]
