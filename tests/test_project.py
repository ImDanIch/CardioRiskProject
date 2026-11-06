import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor


def patient():
    return {
        "age": 50 * 365, "gender": 1, "height": 170, "weight": 75,
        "ap_hi": 125, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1, "cardio": 0,
    }


def test_valid_record_survives_cleaning():
    clean, flags = DataPreprocessor().clean_training_data(pd.DataFrame([patient()]))
    assert len(clean) == 1
    assert not flags.any(axis=1).any()


def test_invalid_pressure_is_excluded():
    row = patient()
    row["ap_hi"] = 60
    with pytest.raises(ValueError, match="No valid rows"):
        DataPreprocessor().clean_training_data(pd.DataFrame([row]))


def test_feature_engineering():
    frame = FeatureEngineer().fit_transform(pd.DataFrame([patient()]))
    assert frame.loc[0, "bmi"] == pytest.approx(75 / 1.7**2)
    assert frame.loc[0, "pulse_pressure"] == 45


def test_prediction_rejects_invalid_patient():
    row = patient()
    row["height"] = 0
    with pytest.raises(ValueError, match="Invalid patient"):
        DataPreprocessor().validate_prediction_data(pd.DataFrame([row]))
