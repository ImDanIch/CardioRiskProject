import math
from pathlib import Path

import pandas as pd
import pytest

from src.interactive_prediction import InteractiveRiskSession, run_interactive


class DummyPredictor:
    model_name = "catboost_final"
    threshold = 0.40

    def predict(self, data):
        assert list(data.columns) == [
            "age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active",
        ]
        assert data.loc[0, "age"] == math.ceil(45 * 365.25)
        return pd.DataFrame({"probability_cardio_1": [0.72], "predicted_cardio": [1]})


VALID_ANSWERS = ["45", "2", "175", "82", "140", "90", "2", "1", "0", "0", "1"]


def session(answers, *, output_path=None):
    iterator = iter(answers)
    lines = []
    obj = InteractiveRiskSession(
        DummyPredictor(), input_fn=lambda prompt: next(iterator),
        output_fn=lines.append, output_path=output_path,
    )
    return obj, lines


def test_collect_valid_record_and_predict():
    obj, lines = session(VALID_ANSWERS)
    record, pred = obj.predict_one()
    assert record.shape == (1, 11)
    assert pred.loc[0, "predicted_cardio"] == 1
    assert any("72.0%" in line for line in lines)
    assert any("не медичний діагноз" in line for line in lines)


def test_reprompts_invalid_age_and_pressure():
    answers = ["17", "45", "2", "175", "82", "140", "150", "90", "2", "1", "0", "0", "1"]
    obj, lines = session(answers)
    record = obj.collect()
    assert record.loc[0, "ap_lo"] == 90
    assert any("Некоректне значення" in line for line in lines)
    assert any("повинен бути нижчим" in line for line in lines)


def test_reprompts_invalid_category_and_non_number():
    answers = ["abc", "45", "9", "2", "175", "82", "140", "90", "2", "1", "0", "0", "1"]
    obj, lines = session(answers)
    record = obj.collect()
    assert record.loc[0, "gender"] == 2
    assert any("Оберіть один" in line for line in lines)


def test_run_without_output_does_not_write_files(tmp_path):
    obj, lines = session(VALID_ANSWERS + ["н"])
    assert obj.run() == 1
    assert not list(tmp_path.iterdir())
    assert any("Виконано прогнозів: 1" in line for line in lines)


def test_explicit_output_saves_predictions(tmp_path):
    output = tmp_path / "manual.csv"
    obj, _ = session(VALID_ANSWERS + ["так"] + VALID_ANSWERS + ["ні"], output_path=output)
    assert obj.run() == 2
    saved = pd.read_csv(output)
    assert len(saved) == 2
    assert saved.loc[0, "probability_cardio_1"] == pytest.approx(0.72)
    assert saved.loc[0, "age"] == math.ceil(45 * 365.25)


def test_missing_model_explains_finalize(tmp_path):
    with pytest.raises(FileNotFoundError, match="python main.py finalize"):
        run_interactive(tmp_path / "missing.joblib")
