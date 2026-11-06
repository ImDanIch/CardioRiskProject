import numpy as np
import pandas as pd
import pytest

from src.ensemble_experiments import FeatureImportanceStudy
from src.model_factory import ModelFactory


def test_ensemble_names_include_stage3_models():
    assert ModelFactory.ENSEMBLE_MODEL_NAMES == (
        "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost"
    )


@pytest.mark.parametrize("name,package", [
    ("xgboost", "xgboost"), ("lightgbm", "lightgbm"), ("catboost", "catboost")
])
def test_optional_ensemble_pipeline(name, package):
    pytest.importorskip(package)
    pipeline = ModelFactory(seed=42).create(name)
    assert list(pipeline.named_steps) == ["features", "columns", "classifier"]
    assert hasattr(pipeline.named_steps["classifier"], "predict_proba")


def test_importance_bar_chart(tmp_path):
    table = pd.DataFrame({"raw_feature": ["age", "ap_hi"],
                          "importance_mean": [0.02, 0.03]})
    out = tmp_path / "plot.png"
    FeatureImportanceStudy._bar_chart(table, "raw_feature", "importance_mean", out)
    assert out.is_file() and out.stat().st_size > 1000
