import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.config import ProjectConfig
from src.data_loader import DataLoader
from src.evaluator import ModelEvaluator
from src.model_factory import ModelFactory
from src.preprocessing import DataPreprocessor
from src.splitter import DatasetSplitter
from src.visualizer import ResultVisualizer


@dataclass(frozen=True)
class ExperimentSample:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series


class ExperimentDataProvider:
    def __init__(self, config: ProjectConfig, train_limit: int = 12000,
                 validation_limit: int = 4000):
        self.config = config
        self.train_limit = train_limit
        self.validation_limit = validation_limit

    @staticmethod
    def _sample(X: pd.DataFrame, y: pd.Series, size: int, seed: int):
        if size < 2:
            raise ValueError("Sample size must be at least 2")
        if len(y) <= size:
            return X.reset_index(drop=True), y.reset_index(drop=True)
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=size, stratify=y, random_state=seed
        )
        return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)

    def load(self) -> ExperimentSample:
        raw = DataLoader(self.config.data_file).load()
        clean, _ = DataPreprocessor().clean_training_data(raw)
        splits = DatasetSplitter(self.config.seed).split(clean)
        X_train, y_train = self._sample(
            splits.X_train, splits.y_train, self.train_limit, self.config.seed
        )
        X_val, y_val = self._sample(
            splits.X_val, splits.y_val, self.validation_limit, self.config.seed
        )
        return ExperimentSample(X_train, y_train, X_val, y_val)


class ModelBenchmark:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.factory = ModelFactory(config.seed)
        self.evaluator = ModelEvaluator()

    def run(self) -> pd.DataFrame:
        self.config.ensure_output_dirs()
        sample = ExperimentDataProvider(self.config).load()
        records = []
        for name in self.factory.BENCHMARK_MODEL_NAMES:
            model = self.factory.create(name)
            started = perf_counter()
            model.fit(sample.X_train, sample.y_train)
            probabilities = model.predict_proba(sample.X_val)[:, 1]
            seconds = perf_counter() - started
            threshold = self.evaluator.best_threshold(sample.y_val, probabilities)
            metrics = self.evaluator.metrics(sample.y_val, probabilities, threshold)
            records.append({"model": name, **metrics, "fit_predict_seconds": round(seconds, 3)})

        table = pd.DataFrame(records).sort_values(
            ["roc_auc", "f1"], ascending=False
        ).reset_index(drop=True)
        table.to_csv(
            self.config.tables_dir / "extended_benchmark_validation.csv",
            index=False, encoding="utf-8-sig",
        )
        metadata = {
            "purpose": "validation-only algorithm comparison",
            "seed": self.config.seed,
            "train_rows": len(sample.y_train),
            "validation_rows": len(sample.y_val),
            "test_rows_used": 0,
            "selection_metric": "validation_roc_auc",
            "best_model": str(table.iloc[0]["model"]),
        }
        (self.config.metrics_dir / "extended_benchmark.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ResultVisualizer(self.config.figures_dir).extended_model_comparison(table)
        return table


class HyperparameterOptimizer:
    PARAMETER_DISTRIBUTIONS = {
        "classifier__max_iter": [80, 120, 160, 200],
        "classifier__learning_rate": [0.03, 0.05, 0.07, 0.10],
        "classifier__max_leaf_nodes": [15, 23, 31],
        "classifier__min_samples_leaf": [20, 40, 60],
        "classifier__l2_regularization": [0.0, 0.5, 1.0, 2.0],
    }

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.factory = ModelFactory(config.seed)
        self.evaluator = ModelEvaluator()

    def _score(self, model, sample: ExperimentSample) -> dict:
        probabilities = model.predict_proba(sample.X_val)[:, 1]
        threshold = self.evaluator.best_threshold(sample.y_val, probabilities)
        return self.evaluator.metrics(sample.y_val, probabilities, threshold)

    def run(self) -> dict:
        self.config.ensure_output_dirs()
        sample = ExperimentDataProvider(self.config).load()
        baseline = self.factory.create("gradient_boosting")
        baseline.fit(sample.X_train, sample.y_train)
        baseline_metrics = self._score(baseline, sample)

        search = RandomizedSearchCV(
            estimator=self.factory.create("gradient_boosting"),
            param_distributions=self.PARAMETER_DISTRIBUTIONS,
            n_iter=8,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.seed),
            n_jobs=2,
            pre_dispatch=4,
            refit=True,
            random_state=self.config.seed,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(sample.X_train, sample.y_train)
        tuned_metrics = self._score(search.best_estimator_, sample)

        trials = pd.DataFrame(search.cv_results_)[
            ["rank_test_score", "mean_test_score", "std_test_score", "params"]
        ].sort_values("rank_test_score")
        trials["params"] = trials["params"].map(lambda item: json.dumps(item, sort_keys=True))
        trials.to_csv(
            self.config.tables_dir / "hyperparameter_search_cv.csv",
            index=False, encoding="utf-8-sig",
        )

        report = {
            "purpose": "training-only cross-validation and validation comparison",
            "seed": self.config.seed,
            "train_rows": len(sample.y_train),
            "validation_rows": len(sample.y_val),
            "test_rows_used": 0,
            "cv_folds": 3,
            "search_iterations": 8,
            "best_cv_roc_auc": float(search.best_score_),
            "best_params": search.best_params_,
            "baseline_validation": baseline_metrics,
            "tuned_validation": tuned_metrics,
            "validation_roc_auc_delta": tuned_metrics["roc_auc"] - baseline_metrics["roc_auc"],
        }
        (self.config.metrics_dir / "hyperparameter_tuning.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        candidate = {
            "pipeline": search.best_estimator_,
            "model_name": "gradient_boosting_tuned_candidate",
            "threshold": tuned_metrics["threshold"],
            "training_rows": len(sample.y_train),
            "test_evaluated": False,
        }
        joblib.dump(candidate, self.config.model_file.parent / "cardio_tuned_candidate.joblib")
        ResultVisualizer(self.config.figures_dir).tuning_comparison(
            baseline_metrics, tuned_metrics
        )
        return report
