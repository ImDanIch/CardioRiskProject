# Етап 4. Фінальне оцінювання та підготовка моделі до прогнозування

## Команди

```powershell
python -m pytest -q
python main.py finalize
python main.py predict --final --input data/new_patients_example.csv
```

У цьому етапі не потрібні нові пакети, але залежності етапу 3 (`requirements-stage3.txt`) мають бути встановлені.

## Методика

1. Відтворюємо стратифікований поділ даних із seed=42: 70% train, 15% validation, 15% test.
2. **CatBoost обрано за ROC-AUC на валідаційній підвибірці етапу 3.** Вибір моделі не проводиться за тестовою вибіркою.
3. Навчаємо CatBoost на train і визначаємо поріг класифікації за максимумом F1 на validation у діапазоні 0.20–0.65 з кроком 0.01.
4. Навчаємо новий CatBoost на об'єднаних train+validation, **не змінюючи** знайдений поріг.
5. Оцінюємо модель на test: ROC-AUC, Average Precision, Accuracy, Precision, Recall, F1, Specificity, Brier score; TN/FP/FN/TP.
6. Зберігаємо модель, результати, ROC/PR-криві, матрицю помилок і калібрувальний графік.

**Обмеження оцінювання:** тестовий поділ уже використовувався на етапі 1 для базової моделі, тому його не можна називати абсолютно новим незалежним набором даних. Він не використовується для підбору порогу або вибору моделі на етапі 4. Для суворої зовнішньої валідації потрібні нові дані.

## Файли

- `models/cardio_final_model.joblib` — фінальна модель із порогом, використовується на етапі 5.
- `results/metrics/final_model_report.json` — параметри процедури та всі метрики.
- `results/tables/final_validation_thresholds.csv` — залежність метрик від порогу на validation.
- `results/tables/final_test_confusion_matrix.csv` — матриця помилок на test.
- `results/tables/final_test_predictions.csv` — прогноз, справжній клас і тип помилки для кожного тестового запису.
- `results/tables/final_test_calibration.csv` — дані для калібрувального графіка.
- `results/figures/12_validation_thresholds.png` — вибір порогу за validation.
- `results/figures/13_final_test_roc.png` — ROC-крива.
- `results/figures/14_final_test_pr.png` — PR-крива.
- `results/figures/15_final_test_confusion.png` — матриця помилок.
- `results/figures/16_final_test_calibration.png` — калібрування ймовірностей.

## Прогнозування на етапі 5

Команда `predict --final` уже вміє прогнозувати з CSV за фінальною моделлю. Інтерактивне введення характеристик людини з термінала VS Code реалізуємо окремим п'ятим етапом.

Результат моделі — ймовірність класу `cardio=1` у навчальному датасеті. Це не клінічний діагноз і не оцінка десятирічного ризику захворювання.
