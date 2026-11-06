import pandas as pd

from src.data_loader import DataLoader


class DataPreprocessor:
    RANGES = {
        "age": (18 * 365.25, 100 * 365.25),
        "height": (120, 220),
        "weight": (30, 250),
        "ap_hi": (70, 250),
        "ap_lo": (40, 150),
    }
    CATEGORIES = {
        "gender": (1, 2),
        "cholesterol": (1, 2, 3),
        "gluc": (1, 2, 3),
        "smoke": (0, 1),
        "alco": (0, 1),
        "active": (0, 1),
    }

    def flags(self, data: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
        required = list(DataLoader.FEATURES) + (["cardio"] if require_target else [])
        missing = sorted(set(required) - set(data.columns))
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        numeric = data[required].apply(pd.to_numeric, errors="coerce")
        flags = pd.DataFrame(index=data.index)
        flags["missing_or_non_numeric"] = numeric.isna().any(axis=1)
        for column, (lower, upper) in self.RANGES.items():
            flags[f"invalid_{column}"] = ~numeric[column].between(lower, upper)
        flags["invalid_pressure_order"] = numeric["ap_hi"] <= numeric["ap_lo"]
        for column, allowed in self.CATEGORIES.items():
            flags[f"invalid_{column}"] = ~numeric[column].isin(allowed)
        if require_target:
            flags["invalid_cardio"] = ~numeric["cardio"].isin((0, 1))
        return flags

    def clean_training_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        flags = self.flags(data, require_target=True)
        clean = data.loc[~flags.any(axis=1)].copy().reset_index(drop=True)
        if clean.empty:
            raise ValueError("No valid rows after cleaning")
        return clean, flags

    def validate_prediction_data(self, data: pd.DataFrame) -> pd.DataFrame:
        flags = self.flags(data, require_target=False)
        invalid = flags.any(axis=1)
        if invalid.any():
            positions = list(data.index[invalid][:10])
            raise ValueError(f"Invalid patient records at row indices: {positions}")
        return data.copy()
