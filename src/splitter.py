from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import DataLoader


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    test_ids: pd.Series


class DatasetSplitter:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def split(self, data: pd.DataFrame) -> DatasetSplits:
        X = data.loc[:, list(DataLoader.FEATURES)].copy()
        y = data["cardio"].astype(int)
        ids = data["id"].copy() if "id" in data else pd.Series(data.index, index=data.index)

        X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
            X, y, ids, test_size=0.30, stratify=y, random_state=self.seed
        )
        X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
            X_temp, y_temp, ids_temp, test_size=0.50,
            stratify=y_temp, random_state=self.seed
        )
        return DatasetSplits(
            X_train.reset_index(drop=True), X_val.reset_index(drop=True),
            X_test.reset_index(drop=True), y_train.reset_index(drop=True),
            y_val.reset_index(drop=True), y_test.reset_index(drop=True),
            ids_test.reset_index(drop=True)
        )
