import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import ProjectConfig
from src.final_evaluation import FinalModelStudy
from src.predictor import RiskPredictor


def test_threshold_table_f1_and_precision_recall():
    y = [0, 0, 1, 1]
    prob = [0.10, 0.35, 0.45, 0.90]
    table = FinalModelStudy.threshold_table(y, prob)
    assert len(table) == 46
    assert {"threshold", "precision", "recall", "f1"}.issubset(table.columns)
    assert FinalModelStudy.select_threshold(table) == pytest.approx(0.36)


def test_threshold_selection_uses_only_provided_validation_table():
    table = pd.DataFrame({"threshold": [0.4, 0.3, 0.5], "f1": [0.9, 0.9, 0.7]})
    assert FinalModelStudy.select_threshold(table) == pytest.approx(0.3)


def test_final_model_path_is_separate_from_baseline(tmp_path):
    config = ProjectConfig(root=tmp_path)
    assert config.final_model_file != config.model_file
    assert config.final_model_file.name == "cardio_final_model.joblib"


def test_stage4_integration_and_prediction(tmp_path):
    pytest.importorskip("catboost")
    rng = np.random.default_rng(42)
    n = 240
    ages = rng.integers(30, 80, n)
    ap_hi = rng.integers(105, 170, n)
    df = pd.DataFrame({
        "id": np.arange(n), "age": ages * 365,
        "gender": rng.integers(1, 3, n),
        "height": rng.integers(150, 190, n),
        "weight": rng.integers(50, 110, n),
        "ap_hi": ap_hi, "ap_lo": rng.integers(65, 101, n),
        "cholesterol": rng.integers(1, 4, n),
        "gluc": rng.integers(1, 4, n),
        "smoke": rng.integers(0, 2, n),
        "alco": rng.integers(0, 2, n),
        "active": rng.integers(0, 2, n),
        "cardio": ((ages > 55) | (ap_hi > 143)).astype(int),
    })
    (tmp_path / "data").mkdir()
    df.to_csv(tmp_path / "data" / "cardio_train.csv", sep=";", index=False)
    config = ProjectConfig(root=tmp_path)
    report = FinalModelStudy(config).run()
    assert report["test_used_for_selection"] is False
    assert report["train_rows"] + report["validation_rows"] + report["test_rows"] == n
    assert config.final_model_file.is_file()
    assert not config.model_file.exists()  # Stage 1 bundle is never overwritten.
    assert len(list(config.figures_dir.glob("1[2-6]_*.png"))) == 5
    assert (config.tables_dir / "final_validation_thresholds.csv").is_file()
    assert (config.tables_dir / "final_test_predictions.csv").is_file()
    assert (config.metrics_dir / "final_model_report.json").is_file()
    assert json.loads((config.metrics_dir / "final_model_report.json").read_text())[
        "test_metrics"]["n"] == report["test_rows"]
    bundle = joblib.load(config.final_model_file)
    assert bundle["threshold"] == report["threshold"]
    predictions = RiskPredictor(config.final_model_file).predict(df.head(3).drop(columns="cardio"))
    assert len(predictions) == 3
    assert predictions["probability_cardio_1"].between(0, 1).all()
