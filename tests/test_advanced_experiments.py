import pandas as pd

from src.advanced_experiments import ExperimentDataProvider, HyperparameterOptimizer
from src.model_factory import ModelFactory


def test_benchmark_models_have_consistent_pipelines():
    factory = ModelFactory(seed=42)
    for name in factory.BENCHMARK_MODEL_NAMES:
        pipeline = factory.create(name)
        assert list(pipeline.named_steps) == ["features", "columns", "classifier"]


def test_sampling_is_stratified_and_reproducible():
    X = pd.DataFrame({"value": range(200)})
    y = pd.Series([0] * 100 + [1] * 100)
    first_X, first_y = ExperimentDataProvider._sample(X, y, 60, seed=42)
    second_X, second_y = ExperimentDataProvider._sample(X, y, 60, seed=42)
    assert first_X.equals(second_X)
    assert first_y.equals(second_y)
    assert first_y.value_counts().to_dict() == {0: 30, 1: 30}


def test_optimizer_parameter_names_are_pipeline_scoped():
    params = HyperparameterOptimizer.PARAMETER_DISTRIBUTIONS
    assert params
    assert all(key.startswith("classifier__") for key in params)
