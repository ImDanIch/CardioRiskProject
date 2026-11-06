"""Stage 3: ensemble comparison and explainability (validation only)."""

import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.advanced_experiments import ExperimentDataProvider
from src.config import ProjectConfig
from src.evaluator import ModelEvaluator
from src.model_factory import ModelFactory


class EnsembleBenchmark:
    """Compare five ensembles on the same stratified train/validation subsets."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.factory = ModelFactory(config.seed)
        self.evaluator = ModelEvaluator()

    def run(self) -> pd.DataFrame:
        self.config.ensure_output_dirs()
        sample = ExperimentDataProvider(self.config).load()
        rows = []
        for name in self.factory.ENSEMBLE_MODEL_NAMES:
            model = self.factory.create(name)
            start = perf_counter()
            model.fit(sample.X_train, sample.y_train)
            probabilities = model.predict_proba(sample.X_val)[:, 1]
            elapsed = perf_counter() - start
            threshold = self.evaluator.best_threshold(sample.y_val, probabilities)
            metrics = self.evaluator.metrics(sample.y_val, probabilities, threshold)
            rows.append({"model": name, **metrics, "fit_predict_seconds": round(elapsed, 3)})

        table = pd.DataFrame(rows).sort_values(
            ["roc_auc", "f1"], ascending=False
        ).reset_index(drop=True)
        table.to_csv(self.config.tables_dir / "ensemble_benchmark_validation.csv",
                     index=False, encoding="utf-8-sig")
        report = {
            "purpose": "ensemble validation comparison; test data never used",
            "seed": self.config.seed,
            "train_rows": len(sample.y_train),
            "validation_rows": len(sample.y_val),
            "test_rows_used": 0,
            "selection_metric": "validation_roc_auc",
            "best_model": str(table.iloc[0]["model"]),
            "models": list(self.factory.ENSEMBLE_MODEL_NAMES),
        }
        (self.config.metrics_dir / "ensemble_benchmark.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(table["model"], table["roc_auc"])
        ax.set(xlabel="Model", ylabel="Validation ROC-AUC",
               title="Ensemble models comparison", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(self.config.figures_dir / "07_ensemble_benchmark.png", dpi=160)
        plt.close(fig)
        return table


class FeatureImportanceStudy:
    """Explain a preselected XGBoost model on a validation subset, not on test."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.factory = ModelFactory(config.seed)
        self.evaluator = ModelEvaluator()

    @staticmethod
    def _bar_chart(table: pd.DataFrame, name: str, value: str, path: Path) -> None:
        top = table.head(14).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(top[name].astype(str), top[value])
        ax.set(xlabel=value, title="XGBoost feature importance")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _save_shap(self, model, X_val: pd.DataFrame) -> dict:
        """Optional SHAP; operates on transformed features, not raw feature names."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError("SHAP requires: pip install -r requirements-stage3.txt") from exc

        X_small = X_val.iloc[: min(120, len(X_val))]
        X_transformed = model[:-1].transform(X_small)
        transformed_names = model.named_steps["columns"].get_feature_names_out()
        explainer = shap.TreeExplainer(model.named_steps["classifier"])
        values = explainer(X_transformed)
        values.feature_names = list(transformed_names)

        shap.plots.bar(values, max_display=14, show=False)
        plt.gcf().savefig(self.config.figures_dir / "10_xgb_shap_bar.png",
                          dpi=160, bbox_inches="tight")
        plt.close('all')
        shap.plots.beeswarm(values, max_display=14, show=False)
        plt.gcf().savefig(self.config.figures_dir / "11_xgb_shap_beeswarm.png",
                          dpi=160, bbox_inches="tight")
        plt.close('all')
        return {"shap_records": len(X_small), "shap_feature_count": len(transformed_names),
                "shap_scale": "raw model output (log-odds)",
                "shap_files": ["10_xgb_shap_bar.png", "11_xgb_shap_beeswarm.png"]}

    def run(self, include_shap: bool = False) -> dict:
        self.config.ensure_output_dirs()
        sample = ExperimentDataProvider(self.config).load()
        model = self.factory.create("xgboost")
        model.fit(sample.X_train, sample.y_train)

        # Validation ROC-AUC is for context; the independent test set is untouched.
        probabilities = model.predict_proba(sample.X_val)[:, 1]
        val_auc = self.evaluator.metrics(sample.y_val, probabilities, 0.5)["roc_auc"]

        names = model.named_steps["columns"].get_feature_names_out()
        weights = model.named_steps["classifier"].feature_importances_
        intrinsic = pd.DataFrame({"transformed_feature": names, "importance": weights})
        intrinsic = intrinsic.sort_values("importance", ascending=False).reset_index(drop=True)
        intrinsic.to_csv(self.config.tables_dir / "xgb_intrinsic_importance.csv",
                         index=False, encoding="utf-8-sig")
        self._bar_chart(intrinsic, "transformed_feature", "importance",
                        self.config.figures_dir / "09_xgb_intrinsic_importance.png")

        X_perm, y_perm = ExperimentDataProvider._sample(
            sample.X_val, sample.y_val, size=1000, seed=self.config.seed
        )
        importance = permutation_importance(
            model, X_perm, y_perm, scoring="roc_auc", n_repeats=5,
            random_state=self.config.seed, n_jobs=1,
        )
        perm = pd.DataFrame({
            "raw_feature": list(X_perm.columns),
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        perm.to_csv(self.config.tables_dir / "xgb_permutation_importance.csv",
                    index=False, encoding="utf-8-sig")
        self._bar_chart(perm, "raw_feature", "importance_mean",
                        self.config.figures_dir / "08_xgb_permutation_importance.png")

        report = {
            "purpose": "XGBoost global feature importance on validation subset",
            "train_rows": len(sample.y_train),
            "validation_rows": len(sample.y_val),
            "permutation_rows": len(y_perm),
            "permutation_repeats": 5,
            "validation_roc_auc": val_auc,
            "test_rows_used": 0,
            "note": "Permutation importance measures validation ROC-AUC decrease; it is not causal importance.",
            "top_raw_features": perm.head(5)["raw_feature"].tolist(),
        }
        if include_shap:
            report.update(self._save_shap(model, sample.X_val))
        (self.config.metrics_dir / "xgb_feature_importance.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
