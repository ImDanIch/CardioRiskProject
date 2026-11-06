import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_loader import DataLoader


class FeatureEngineer(TransformerMixin, BaseEstimator):
    NUMERIC = (
        "age_years", "height", "weight", "ap_hi", "ap_lo",
        "bmi", "pulse_pressure", "mean_arterial_pressure"
    )
    CATEGORICAL = ("gender", "cholesterol", "gluc", "smoke", "alco", "active")

    def fit(self, X: pd.DataFrame, y=None):
        self._check_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        result = X.loc[:, DataLoader.FEATURES].copy()
        result["age_years"] = result["age"] / 365.25
        result["bmi"] = result["weight"] / (result["height"] / 100.0) ** 2
        result["pulse_pressure"] = result["ap_hi"] - result["ap_lo"]
        result["mean_arterial_pressure"] = (result["ap_hi"] + 2 * result["ap_lo"]) / 3.0
        return result.loc[:, list(self.NUMERIC) + list(self.CATEGORICAL)]

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")
        missing = sorted(set(DataLoader.FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Missing columns: {missing}")
