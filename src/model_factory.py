from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import FeatureEngineer


class ModelFactory:
    MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")
    ENSEMBLE_MODEL_NAMES = (
        "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost",
    )
    BENCHMARK_MODEL_NAMES = (
        "logistic_regression", "decision_tree", "knn",
        "svm_rbf", "random_forest", "gradient_boosting",
    )

    def __init__(self, seed: int = 42):
        self.seed = seed

    def create(self, name: str) -> Pipeline:
        # Optional libraries are loaded only when an ensemble model is requested.
        if name == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError("Install stage 3 packages: pip install -r requirements-stage3.txt") from exc
            classifier = XGBClassifier(
                n_estimators=180, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", n_jobs=2, random_state=self.seed,
            )
        elif name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError("Install stage 3 packages: pip install -r requirements-stage3.txt") from exc
            classifier = LGBMClassifier(
                n_estimators=180, learning_rate=0.05, num_leaves=23,
                min_child_samples=40, n_jobs=2, random_state=self.seed,
                verbosity=-1,
            )
        elif name == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:
                raise ImportError("Install stage 3 packages: pip install -r requirements-stage3.txt") from exc
            from src.ensemble_adapters import CatBoostAdapter
            classifier = CatBoostAdapter(
                iterations=180, depth=5, learning_rate=0.05, seed=self.seed,
            )
        else:
            classifier = None

        estimators = {
            "logistic_regression": lambda: LogisticRegression(max_iter=1500, random_state=self.seed),
            "decision_tree": lambda: DecisionTreeClassifier(
                max_depth=7, min_samples_leaf=40, random_state=self.seed
            ),
            "knn": lambda: KNeighborsClassifier(
                n_neighbors=21, weights="distance", n_jobs=-1
            ),
            "svm_rbf": lambda: SVC(
                kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=self.seed
            ),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=120, max_depth=12, min_samples_leaf=5,
                n_jobs=-1, random_state=self.seed
            ),
            "gradient_boosting": lambda: HistGradientBoostingClassifier(
                max_iter=130, learning_rate=0.07, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=self.seed
            ),
        }
        if classifier is None:
            if name not in estimators:
                raise ValueError(f"Unknown model: {name}")
            classifier = estimators[name]()

        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        columns = ColumnTransformer([
            ("numeric", numeric, list(FeatureEngineer.NUMERIC)),
            ("categorical", categorical, list(FeatureEngineer.CATEGORICAL)),
        ])
        return Pipeline([
            ("features", FeatureEngineer()),
            ("columns", columns),
            ("classifier", classifier),
        ])
