import joblib

from src.analyzer import DataAnalyzer
from src.config import ProjectConfig
from src.learning_curve_experiment import TrainingSizeExperiment
from src.predictor import RiskPredictor
from src.trainer import ModelTrainer


class ExperimentManager:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def analyze(self) -> dict:
        return DataAnalyzer(self.config).run()

    def train(self) -> dict:
        return ModelTrainer(self.config).run()

    def experiments(self):
        if not self.config.model_file.is_file():
            raise FileNotFoundError("Train a model before running experiments")
        name = joblib.load(self.config.model_file)["model_name"]
        return TrainingSizeExperiment(self.config).run(name)

    def predict(self, input_path, output_path):
        return RiskPredictor(self.config.model_file).predict_file(input_path, output_path)

    def run(self) -> dict:
        data_summary = self.analyze()
        training_report = self.train()
        experiment = self.experiments()
        predictions = self.predict(
            self.config.root / "data" / "new_patients_example.csv",
            self.config.tables_dir / "example_predictions.csv",
        )
        return {
            "data": data_summary,
            "training": training_report,
            "experiment_rows": len(experiment),
            "example_predictions": len(predictions),
        }
