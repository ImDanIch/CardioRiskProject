import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import ProjectConfig
from src.data_loader import DataLoader
from src.model_factory import ModelFactory
from src.preprocessing import DataPreprocessor
from src.splitter import DatasetSplitter
from src.visualizer import ResultVisualizer


class TrainingSizeExperiment:
    FRACTIONS = (0.20, 0.50, 0.75, 1.00)

    def __init__(self, config: ProjectConfig):
        self.config = config

    def run(self, model_name: str) -> pd.DataFrame:
        self.config.ensure_output_dirs()
        raw = DataLoader(self.config.data_file).load()
        clean, _ = DataPreprocessor().clean_training_data(raw)
        splits = DatasetSplitter(self.config.seed).split(clean)
        rows = []
        for fraction in self.FRACTIONS:
            if fraction < 1.0:
                X_sub, _, y_sub, _ = train_test_split(
                    splits.X_train, splits.y_train, train_size=fraction,
                    stratify=splits.y_train, random_state=self.config.seed
                )
            else:
                X_sub, y_sub = splits.X_train, splits.y_train
            model = ModelFactory(self.config.seed).create(model_name)
            model.fit(X_sub, y_sub)
            probabilities = model.predict_proba(splits.X_val)[:, 1]
            rows.append({
                "training_fraction": fraction,
                "training_rows": len(y_sub),
                "validation_roc_auc": float(roc_auc_score(splits.y_val, probabilities)),
            })
        table = pd.DataFrame(rows)
        table.to_csv(self.config.tables_dir / "training_size_experiment.csv",
                     index=False, encoding="utf-8-sig")
        ResultVisualizer(self.config.figures_dir).learning_curve(table)
        return table
