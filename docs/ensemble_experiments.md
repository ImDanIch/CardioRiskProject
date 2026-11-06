# Етап 3 — ансамблеві моделі та пояснюваність

## Встановлення

У кореневій папці проєкту з активованим `.venv`:

```powershell
python -m pip install -r requirements-stage3.txt
python -m pytest -q
```

## Порівняння п'яти ансамблевих моделей

```powershell
python main.py ensemble
```

Порівнюються Random Forest, HistGradientBoosting, XGBoost, LightGBM та CatBoost на **однакових** стратифікованих підвибірках (12 000 train, 4 000 validation). Критерій ранжування — validation ROC-AUC. Поріг для F1 підбирається на validation, тому F1 є оптимістичною валідаційною оцінкою. Тестова вибірка не використовується.

Вихідні файли:
- `results/tables/ensemble_benchmark_validation.csv`
- `results/metrics/ensemble_benchmark.json`
- `results/figures/07_ensemble_benchmark.png`

## Важливість ознак XGBoost

```powershell
python main.py explain
```

Модель XGBoost **заздалегідь обрана** для інтерпретації, незалежно від того, хто переміг у benchmark. Виводяться два різні рейтинги:

- `xgb_intrinsic_importance.csv` — вбудована важливість **перетворених** ознак (включно з one-hot та інженерними ознаками);
- `xgb_permutation_importance.csv` — зміна ROC-AUC після перемішування **первинних** вхідних ознак на 1 000 валідаційних записах (5 повторів).

Графіки: `results/figures/08_xgb_permutation_importance.png` і `09_xgb_intrinsic_importance.png`. Зведення: `results/metrics/xgb_feature_importance.json`.

Для необов'язкового аналізу SHAP:

```powershell
python main.py explain --shap
```

SHAP-рисунки: `10_xgb_shap_bar.png`, `11_xgb_shap_beeswarm.png`. Пояснення побудовано на 120 валідаційних записах у шкалі **log-odds**, а не ймовірності. Це асоціативні пояснення моделі, не причинно-наслідкові твердження.

**Важливо:** не перезаписується `models/cardio_model.joblib`; тестова вибірка залишається недоторканою. Дані зберігаються лише локально. Висновки стосуються класифікації наявного в датасеті статусу `cardio`, а не прогнозу захворювання в майбутньому.
