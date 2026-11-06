import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.config import ProjectConfig
from src.data_loader import DataLoader
from src.evaluator import ModelEvaluator
from src.model_factory import ModelFactory
from src.preprocessing import DataPreprocessor
from src.splitter import DatasetSplitter
from src.visualizer import ResultVisualizer


class ModelTrainer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.factory = ModelFactory(seed=config.seed)
        self.evaluator = ModelEvaluator()

    def run(self) -> dict:
        self.config.ensure_output_dirs()
        raw = DataLoader(self.config.data_file).load()
        clean, _ = DataPreprocessor().clean_training_data(raw)
        splits = DatasetSplitter(self.config.seed).split(clean)
        candidates = []
        fitted = {}

        for name in self.factory.MODEL_NAMES:
            model = self.factory.create(name)
            model.fit(splits.X_train, splits.y_train)
            val_probabilities = model.predict_proba(splits.X_val)[:, 1]
            threshold = self.evaluator.best_threshold(splits.y_val, val_probabilities)
            metrics = self.evaluator.metrics(splits.y_val, val_probabilities, threshold)
            candidates.append({"model": name, **metrics})
            fitted[name] = (model, threshold)

        comparison = pd.DataFrame(candidates).sort_values(
            ["roc_auc", "f1"], ascending=False
        ).reset_index(drop=True)
        comparison.to_csv(self.config.tables_dir / "model_comparison_validation.csv",
                          index=False, encoding="utf-8-sig")
        ResultVisualizer(self.config.figures_dir).model_comparison(comparison)

        best_name = str(comparison.iloc[0]["model"])
        model, threshold = fitted[best_name]
        test_probabilities = model.predict_proba(splits.X_test)[:, 1]
        test_metrics = self.evaluator.metrics(splits.y_test, test_probabilities, threshold)
        report = {
            "selected_model": best_name,
            "selection_metric": "validation_roc_auc",
            "train_rows": int(len(splits.y_train)),
            "validation_rows": int(len(splits.y_val)),
            "test_rows": int(len(splits.y_test)),
            "test_metrics": test_metrics,
        }
        (self.config.metrics_dir / "training_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.evaluator.confusion_table(
            splits.y_test, test_probabilities, threshold
        ).to_csv(self.config.tables_dir / "test_confusion_matrix.csv", encoding="utf-8-sig")
        pd.DataFrame({
            "id": splits.test_ids,
            "actual_cardio": splits.y_test,
            "probability_cardio_1": test_probabilities,
            "predicted_cardio": (test_probabilities >= threshold).astype(int),
        }).to_csv(self.config.tables_dir / "test_predictions.csv",
                  index=False, encoding="utf-8-sig")

        bundle = {
            "pipeline": model,
            "model_name": best_name,
            "threshold": threshold,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": self.config.data_file.name,
        }
        joblib.dump(bundle, self.config.model_file)

        example = splits.X_test.head(5).copy()
        example.insert(0, "id", splits.test_ids.head(5).to_numpy())
        example.to_csv(self.config.root / "data" / "new_patients_example.csv",
                       index=False, encoding="utf-8-sig")
        return report
