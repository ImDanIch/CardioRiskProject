"""Stage 4: validation-selected threshold and final test evaluation.

The model family was selected from Stage 3's validation-only ensemble
benchmark. The test split is not used for model or threshold selection.
The same test split was already evaluated in Stage 1, so it is not a
previously unseen external test set.
"""

import json
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_curve,
)

from src.config import ProjectConfig
from src.data_loader import DataLoader
from src.evaluator import ModelEvaluator
from src.model_factory import ModelFactory
from src.preprocessing import DataPreprocessor
from src.splitter import DatasetSplitter


class FinalModelStudy:
    """Train CatBoost, choose threshold on validation, then refit and test."""

    SELECTED_MODEL = "catboost"
    THRESHOLDS = np.round(np.arange(0.20, 0.651, 0.01), 2)

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.factory = ModelFactory(config.seed)
        self.evaluator = ModelEvaluator()

    @classmethod
    def threshold_table(cls, y_true, probabilities) -> pd.DataFrame:
        """Threshold sensitivity on validation labels only (not test)."""
        y_true = np.asarray(y_true)
        probabilities = np.asarray(probabilities)
        records = []
        for threshold in cls.THRESHOLDS:
            predicted = (probabilities >= threshold).astype(int)
            records.append({
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, predicted, zero_division=0)),
                "recall": float(recall_score(y_true, predicted, zero_division=0)),
                "f1": float(f1_score(y_true, predicted, zero_division=0)),
            })
        return pd.DataFrame(records)

    @staticmethod
    def select_threshold(table: pd.DataFrame) -> float:
        """Maximize validation F1; smallest threshold wins ties."""
        best_row = table.sort_values(
            ["f1", "threshold"], ascending=[False, True]
        ).iloc[0]
        return float(best_row["threshold"])

    def _threshold_plot(self, table: pd.DataFrame, threshold: float) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, label in (("f1", "F1"), ("precision", "Precision"),
                           ("recall", "Recall")):
            ax.plot(table["threshold"], table[key], label=label)
        ax.axvline(threshold, color="black", linestyle="--",
                   label=f"Selected threshold: {threshold:.2f}")
        ax.set(xlabel="Decision threshold", ylabel="Score",
               title="Validation: decision threshold sensitivity", ylim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.config.figures_dir / "12_validation_thresholds.png", dpi=160)
        plt.close(fig)

    def _test_plots(self, y_true, probabilities, threshold: float) -> None:
        y_true = np.asarray(y_true)
        probabilities = np.asarray(probabilities)
        predictions = (probabilities >= threshold).astype(int)

        fpr, tpr, _ = roc_curve(y_true, probabilities)
        roc_auc = self.evaluator.metrics(y_true, probabilities, threshold)["roc_auc"]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(fpr, tpr, label=f"CatBoost (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title="Final model: test ROC curve", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.config.figures_dir / "13_final_test_roc.png", dpi=160)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        ap = average_precision_score(y_true, probabilities)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(recall, precision, label=f"CatBoost (AP = {ap:.3f})")
        ax.axhline(y_true.mean(), color="gray", linestyle="--", label="Class prevalence")
        ax.set(xlabel="Recall", ylabel="Precision",
               title="Final model: test precision-recall curve",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.config.figures_dir / "14_final_test_pr.png", dpi=160)
        plt.close(fig)

        cm = confusion_matrix(y_true, predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        for (i, j), count in np.ndenumerate(cm):
            ax.text(j, i, f"{count:,}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black")
        ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
        ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
        ax.set(title="Final model: test confusion matrix")
        fig.tight_layout()
        fig.savefig(self.config.figures_dir / "15_final_test_confusion.png", dpi=160)
        plt.close(fig)

        observed, predicted = calibration_curve(
            y_true, probabilities, n_bins=10, strategy="quantile"
        )
        calibration = pd.DataFrame({
            "mean_predicted_probability": predicted,
            "observed_positive_fraction": observed,
        })
        calibration.to_csv(
            self.config.tables_dir / "final_test_calibration.csv",
            index=False, encoding="utf-8-sig",
        )
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.plot(predicted, observed, "o-", label="CatBoost")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction",
               title="Final model: test calibration", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.config.figures_dir / "16_final_test_calibration.png", dpi=160)
        plt.close(fig)

    def run(self) -> dict:
        self.config.ensure_output_dirs()
        raw = DataLoader(self.config.data_file).load()
        clean, _ = DataPreprocessor().clean_training_data(raw)
        splits = DatasetSplitter(self.config.seed).split(clean)

        # The Stage 3 benchmark preselected CatBoost by validation ROC-AUC.
        # The test set is deliberately not consulted during threshold selection.
        validation_model = self.factory.create(self.SELECTED_MODEL)
        validation_model.fit(splits.X_train, splits.y_train)
        val_probabilities = validation_model.predict_proba(splits.X_val)[:, 1]
        sweep = self.threshold_table(splits.y_val, val_probabilities)
        threshold = self.select_threshold(sweep)
        validation_metrics = self.evaluator.metrics(
            splits.y_val, val_probabilities, threshold
        )
        sweep.to_csv(self.config.tables_dir / "final_validation_thresholds.csv",
                     index=False, encoding="utf-8-sig")
        self._threshold_plot(sweep, threshold)

        # Refit the same model family on train + validation. Keep the threshold
        # fixed from the validation phase; never optimize it using test labels.
        X_final_train = pd.concat([splits.X_train, splits.X_val], ignore_index=True)
        y_final_train = pd.concat([splits.y_train, splits.y_val], ignore_index=True)
        final_model = self.factory.create(self.SELECTED_MODEL)
        final_model.fit(X_final_train, y_final_train)

        test_probabilities = final_model.predict_proba(splits.X_test)[:, 1]
        test_metrics = self.evaluator.metrics(
            splits.y_test, test_probabilities, threshold
        )
        cm = confusion_matrix(splits.y_test, test_probabilities >= threshold,
                              labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        test_metrics.update({
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "brier_score": float(brier_score_loss(splits.y_test, test_probabilities)),
        })
        self.evaluator.confusion_table(
            splits.y_test, test_probabilities, threshold
        ).to_csv(self.config.tables_dir / "final_test_confusion_matrix.csv",
                 encoding="utf-8-sig")
        predicted = (test_probabilities >= threshold).astype(int)
        error_types = np.select(
            [(splits.y_test == 1) & (predicted == 0),
             (splits.y_test == 0) & (predicted == 1)],
            ["false_negative", "false_positive"], default="correct"
        )
        pd.DataFrame({
            "id": splits.test_ids.to_numpy(),
            "actual_cardio": splits.y_test.to_numpy(),
            "probability_cardio_1": test_probabilities,
            "predicted_cardio": predicted,
            "result_type": error_types,
        }).to_csv(self.config.tables_dir / "final_test_predictions.csv",
                  index=False, encoding="utf-8-sig")
        self._test_plots(splits.y_test, test_probabilities, threshold)

        bundle = {
            "pipeline": final_model,
            "model_name": "catboost_final",
            "threshold": threshold,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": self.config.data_file.name,
            "trained_rows": int(len(y_final_train)),
            "validation_selected_threshold": True,
        }
        joblib.dump(bundle, self.config.final_model_file)

        report = {
            "purpose": "final model evaluation and deployment preparation",
            "selected_model": self.SELECTED_MODEL,
            "model_selection_basis": "Stage 3 validation ROC-AUC (12k/4k subset)",
            "threshold_selection_metric": "validation_f1",
            "threshold": threshold,
            "train_rows": int(len(splits.y_train)),
            "validation_rows": int(len(splits.y_val)),
            "final_train_rows": int(len(y_final_train)),
            "test_rows": int(len(splits.y_test)),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_used_for_selection": False,
            "test_split_note": (
                "The same held-out test partition was evaluated in Stage 1. "
                "It is not a completely unseen external dataset."
            ),
            "saved_model": str(self.config.final_model_file.relative_to(self.config.root)),
            "figures": [
                "12_validation_thresholds.png", "13_final_test_roc.png",
                "14_final_test_pr.png", "15_final_test_confusion.png",
                "16_final_test_calibration.png",
            ],
        }
        (self.config.metrics_dir / "final_model_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
