"""Public sklearn-compatible adapter for CatBoost, without monkey-patching."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class CatBoostAdapter(ClassifierMixin, BaseEstimator):
    """Expose CatBoost through sklearn's estimator interface (incl. sklearn tags)."""

    def __init__(self, iterations=180, depth=5, learning_rate=0.05, seed=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.estimator_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, loss_function="Logloss",
            verbose=False, thread_count=2, random_seed=self.seed,
            allow_writing_files=False,
        )
        self.estimator_.fit(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X).reshape(-1)
