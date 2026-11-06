from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor


class RiskPredictor:
    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Trained model not found: {path}")
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.model_name = str(bundle["model_name"])
        self.preprocessor = DataPreprocessor()

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        valid = self.preprocessor.validate_prediction_data(data)
        probabilities = self.pipeline.predict_proba(valid)[:, 1]
        result = pd.DataFrame({
            "probability_cardio_1": probabilities,
            "predicted_cardio": (probabilities >= self.threshold).astype(int),
        })
        if "id" in valid.columns:
            result.insert(0, "id", valid["id"].to_numpy())
        return result

    def predict_file(self, input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
        data = DataLoader(input_path).load(require_target=False)
        result = self.predict(data)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        return result
