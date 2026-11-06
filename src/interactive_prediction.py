"""Stage 5: validated manual patient input for the finalized model.

This is a coursework classifier of the dataset's cardio label, not a
clinical diagnosis or a prospective cardiovascular event risk calculator.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import pandas as pd

from src.predictor import RiskPredictor
from src.preprocessing import DataPreprocessor


class InteractiveRiskSession:
    """Collect one or more patient records and run the saved ML pipeline."""

    def __init__(
        self,
        predictor: RiskPredictor,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        output_path: Path | None = None,
    ) -> None:
        self.predictor = predictor
        self.input = input_fn
        self.output = output_fn
        self.output_path = Path(output_path) if output_path is not None else None
        self.preprocessor = DataPreprocessor()

    def _number(self, label: str, lower: float, upper: float, *, integer: bool = True):
        """Request a number until its type and allowed range are valid."""
        while True:
            raw = self.input(f"{label} [{lower:g}–{upper:g}]: ").strip().replace(",", ".")
            try:
                number = float(raw)
                if not math.isfinite(number):
                    raise ValueError
                if integer and not number.is_integer():
                    raise ValueError
                if not lower <= number <= upper:
                    raise ValueError
                return int(number) if integer else number
            except ValueError:
                self.output(f"Некоректне значення. Введіть число від {lower:g} до {upper:g}.")

    def _category(self, label: str, options: dict[int, str]) -> int:
        choices = ", ".join(f"{code} — {name}" for code, name in options.items())
        while True:
            raw = self.input(f"{label} ({choices}): ").strip()
            if raw.isdigit() and int(raw) in options:
                return int(raw)
            self.output(f"Оберіть один із кодів: {', '.join(map(str, options))}.")

    def collect(self) -> pd.DataFrame:
        """Collect fields in dataset units, including age in *days*."""
        age_years = self._number("Вік, повних років", 18, 100)
        gender = self._category("Стать", {1: "жіноча", 2: "чоловіча"})
        height = self._number("Зріст, см", 120, 220)
        weight = self._number("Маса тіла, кг", 30, 250, integer=False)
        ap_hi = self._number("Систолічний тиск, мм рт. ст.", 70, 250)
        while True:
            ap_lo = self._number("Діастолічний тиск, мм рт. ст.", 40, 150)
            if ap_lo < ap_hi:
                break
            self.output("Діастолічний тиск повинен бути нижчим за систолічний.")
        cholesterol = self._category("Холестерин", {
            1: "норма", 2: "підвищений", 3: "значно підвищений",
        })
        gluc = self._category("Глюкоза", {
            1: "норма", 2: "підвищена", 3: "значно підвищена",
        })
        smoke = self._category("Куріння", {0: "ні", 1: "так"})
        alco = self._category("Вживання алкоголю", {0: "ні", 1: "так"})
        active = self._category("Фізична активність", {0: "ні", 1: "так"})
        # Training data stores age in days. ceil keeps exactly 18 years valid.
        age_days = math.ceil(age_years * 365.25)
        record = pd.DataFrame([{
            "age": age_days, "gender": gender, "height": height,
            "weight": weight, "ap_hi": ap_hi, "ap_lo": ap_lo,
            "cholesterol": cholesterol, "gluc": gluc,
            "smoke": smoke, "alco": alco, "active": active,
        }])
        # Reuse the same validation rules as file-based prediction.
        return self.preprocessor.validate_prediction_data(record)

    def predict_one(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        record = self.collect()
        prediction = self.predictor.predict(record)
        probability = float(prediction.loc[0, "probability_cardio_1"])
        predicted_class = int(prediction.loc[0, "predicted_cardio"])
        self.output("\n--- РЕЗУЛЬТАТ КЛАСИФІКАЦІЇ ---")
        self.output(f"Модель: {self.predictor.model_name}")
        self.output(f"Оцінка ймовірності класу cardio=1: {probability:.1%}")
        self.output(f"Поріг класифікації: {self.predictor.threshold:.2f}")
        self.output(f"Прогнозований клас: {predicted_class} "
                    f"({'наявність ознак ССЗ у датасеті' if predicted_class else 'відсутність ознак ССЗ у датасеті'})")
        self.output("Увага: це навчальна ML-модель, а не медичний діагноз "
                    "чи прогноз захворювання на певний період.")
        return record, prediction

    def _save(self, record: pd.DataFrame, prediction: pd.DataFrame) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result = pd.concat([record.reset_index(drop=True),
                            prediction.reset_index(drop=True)], axis=1)
        exists = self.output_path.exists() and self.output_path.stat().st_size > 0
        result.to_csv(self.output_path, mode="a" if exists else "w",
                      header=not exists, index=False, encoding="utf-8-sig" if not exists else "utf-8")
        self.output(f"Результат збережено локально: {self.output_path}")

    def run(self) -> int:
        """Run interactive loop; Ctrl+C or EOF exits without a traceback."""
        self.output("Інтерактивне прогнозування CardioRiskProject")
        self.output("Для виходу натисніть Ctrl+C. Дані не зберігаються без --output.\n")
        count = 0
        try:
            while True:
                record, prediction = self.predict_one()
                self._save(record, prediction)
                count += 1
                while True:
                    again = self.input("\nВвести ще одну людину? [т/н]: ").strip().lower()
                    if again in ("т", "так", "y", "yes"):
                        break
                    if again in ("н", "ні", "n", "no", ""):
                        self.output(f"Завершено. Виконано прогнозів: {count}.")
                        return count
                    self.output("Введіть 'т' або 'н'.")
        except (EOFError, KeyboardInterrupt):
            self.output(f"\nЗавершено. Виконано прогнозів: {count}.")
            return count


def run_interactive(model_path: Path, output_path: Path | None = None) -> int:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Не знайдено фінальну модель: {model_path}. "
            "Спочатку виконайте: python main.py finalize"
        )
    predictor = RiskPredictor(model_path)
    return InteractiveRiskSession(predictor, output_path=output_path).run()
