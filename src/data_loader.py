from pathlib import Path

import pandas as pd


class DataLoader:
    FEATURES = (
        "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active"
    )

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self, require_target: bool = True) -> pd.DataFrame:
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")

        data = pd.read_csv(self.file_path, sep=None, engine="python")
        required = set(self.FEATURES)
        if require_target:
            required.add("cardio")
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if data.empty:
            raise ValueError("Dataset is empty")
        return data
