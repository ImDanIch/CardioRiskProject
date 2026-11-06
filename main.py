import argparse
import json
from pathlib import Path

from src.config import ProjectConfig
from src.experiment_manager import ExperimentManager


def main() -> None:
    parser = argparse.ArgumentParser(prog="CardioRiskProject")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "train", "experiments", "run", "benchmark", "tune"):
        subparsers.add_parser(name)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input", required=True, type=Path)
    predict_parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = ProjectConfig(root=Path(__file__).resolve().parent)
    manager = ExperimentManager(config)
    if args.command == "predict":
        output = args.output or config.tables_dir / "new_predictions.csv"
        result = manager.predict(args.input, output)
        print(f"Predictions saved: {output} ({len(result)} records)")
    else:
        result = getattr(manager, args.command)()
        if hasattr(result, "to_string"):
            print(result.to_string(index=False))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
