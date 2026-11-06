from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class ResultVisualizer:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def target_distribution(self, data: pd.DataFrame) -> None:
        counts = data["cardio"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set(xlabel="cardio", ylabel="Number of patients", title="Target distribution")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(self.directory / "01_target_distribution.png", dpi=160)
        plt.close(fig)

    def age_distribution(self, data: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        for target in (0, 1):
            subset = data.loc[data["cardio"] == target, "age"] / 365.25
            ax.hist(subset, bins=35, density=True, alpha=0.5, label=f"cardio={target}")
        ax.set(xlabel="Age (years)", ylabel="Density", title="Age by target class")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.directory / "02_age_by_class.png", dpi=160)
        plt.close(fig)

    def model_comparison(self, table: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(table["model"], table["roc_auc"])
        ax.set(ylabel="Validation ROC-AUC", title="Model comparison", ylim=(0.0, 1.0))
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(self.directory / "03_model_comparison.png", dpi=160)
        plt.close(fig)

    def learning_curve(self, table: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(table["training_fraction"], table["validation_roc_auc"], marker="o")
        ax.set(xlabel="Fraction of training data", ylabel="Validation ROC-AUC",
               title="Effect of training set size", ylim=(0.5, 1.0))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.directory / "04_learning_curve.png", dpi=160)
        plt.close(fig)

    def extended_model_comparison(self, table: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(table["model"], table["roc_auc"])
        ax.set(ylabel="Validation ROC-AUC", title="Algorithm benchmark", ylim=(0.0, 1.0))
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(self.directory / "05_extended_benchmark.png", dpi=160)
        plt.close(fig)

    def tuning_comparison(self, baseline: dict, tuned: dict) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(["Baseline", "Tuned"], [baseline["roc_auc"], tuned["roc_auc"]])
        ax.set(ylabel="Validation ROC-AUC", title="Hyperparameter optimization", ylim=(0.0, 1.0))
        fig.tight_layout()
        fig.savefig(self.directory / "06_tuning_comparison.png", dpi=160)
        plt.close(fig)
