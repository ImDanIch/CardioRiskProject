from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    seed: int = 42

    @property
    def data_file(self) -> Path:
        return self.root / "data" / "cardio_train.csv"

    @property
    def model_file(self) -> Path:
        return self.root / "models" / "cardio_model.joblib"

    @property
    def tables_dir(self) -> Path:
        return self.root / "results" / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "results" / "metrics"

    def ensure_output_dirs(self) -> None:
        for directory in (self.model_file.parent, self.tables_dir,
                          self.figures_dir, self.metrics_dir):
            directory.mkdir(parents=True, exist_ok=True)
