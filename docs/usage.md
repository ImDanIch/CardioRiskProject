# Запуск у VS Code (Windows / PowerShell)

1. Розпакуйте вміст ZIP **всередину** клонованої папки `CardioRiskProject`. Існуючі `README.md` та `.gitignore` залиште без змін.
2. Переконайтесь, що `data/cardio_train.csv` уже лежить у папці проєкту.
3. У терміналі VS Code виконайте:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py run
```

Перший запуск може тривати кілька хвилин через навчання кількох моделей і додаткові експерименти.

Окремі команди:

```powershell
.\.venv\Scripts\python.exe main.py analyze
.\.venv\Scripts\python.exe main.py train
.\.venv\Scripts\python.exe main.py experiments
.\.venv\Scripts\python.exe main.py predict --input data/new_patients_example.csv
.\.venv\Scripts\python.exe -m pytest -q
```

`main.py train` створює `models/cardio_model.joblib` і `data/new_patients_example.csv`. Для `predict` спочатку треба виконати `train` або `run`.

Результати: `results/tables/`, `results/figures/`, `results/metrics/`.

Не завантажуйте реальні медичні дані пацієнтів у публічний репозиторій.

## Розширені експерименти (етап 2)

```powershell
python main.py benchmark
python main.py tune
```

Перша команда порівнює шість алгоритмів на однакових підвибірках, друга підбирає гіперпараметри Gradient Boosting за крос-валідацією. Деталі та шляхи результатів: `docs/advanced_experiments.md`.
