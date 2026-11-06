import json

import pandas as pd

from src.config import ProjectConfig
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.visualizer import ResultVisualizer


class DataAnalyzer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.preprocessor = DataPreprocessor()

    def run(self) -> dict:
        self.config.ensure_output_dirs()
        raw = DataLoader(self.config.data_file).load()
        clean, flags = self.preprocessor.clean_training_data(raw)
        engineered = FeatureEngineer().fit_transform(clean)

        quality = flags.sum().rename("count").sort_values(ascending=False)
        quality.to_csv(self.config.tables_dir / "data_quality_flags.csv", encoding="utf-8-sig")
        clean.groupby("cardio").size().rename("count").to_csv(
            self.config.tables_dir / "class_distribution.csv", encoding="utf-8-sig"
        )
        engineered.assign(cardio=clean["cardio"]).groupby("cardio")[
            list(FeatureEngineer.NUMERIC)
        ].agg(["mean", "median", "std"]).to_csv(
            self.config.tables_dir / "numeric_statistics_by_class.csv", encoding="utf-8-sig"
        )

        visualizer = ResultVisualizer(self.config.figures_dir)
        visualizer.target_distribution(clean)
        visualizer.age_distribution(clean)
        summary = {
            "source_rows": int(len(raw)),
            "clean_rows": int(len(clean)),
            "excluded_rows": int(len(raw) - len(clean)),
            "positive_class_rate": float(clean["cardio"].mean()),
        }
        (self.config.metrics_dir / "data_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
