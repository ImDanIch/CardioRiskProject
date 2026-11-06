import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


class ModelEvaluator:
    @staticmethod
    def best_threshold(y_true, probabilities) -> float:
        thresholds = np.round(np.arange(0.20, 0.651, 0.01), 2)
        scores = [f1_score(y_true, probabilities >= threshold, zero_division=0)
                  for threshold in thresholds]
        return float(thresholds[int(np.argmax(scores))])

    @staticmethod
    def metrics(y_true, probabilities, threshold: float) -> dict:
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "average_precision": float(average_precision_score(y_true, probabilities)),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "threshold": float(threshold),
            "n": int(len(y_true)),
        }

    @staticmethod
    def confusion_table(y_true, probabilities, threshold: float) -> pd.DataFrame:
        predicted = (np.asarray(probabilities) >= threshold).astype(int)
        return pd.DataFrame(
            confusion_matrix(y_true, predicted, labels=[0, 1]),
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        )
