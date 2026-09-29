\# Анализ данных и предсказание цены бриллиантов


![Python](https://img.shields.io/badge/Python-3.14-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Учебный Data Science проект: полный цикл от загрузки данных в PostgreSQL

до линейной регрессии для предсказания цены бриллианта.



\## 🎯 Цель проекта



Пройти полный пайплайн анализа данных и машинного обучения:



1\. Загрузить сырые данные в реляционную базу данных (PostgreSQL).

2\. Выгрузить данные в Pandas.

3\. Провести разведочный анализ (EDA) и визуализацию.

4\. Подготовить признаки и обучить модель линейной регрессии.

5\. Оценить качество, проверить остатки и интерпретировать коэффициенты.



\## 📊 Датасет



Классический датасет \*\*Diamonds\*\* — 53 940 бриллиантов, 10 признаков:



| Признак | Тип | Описание |

| :--- | :--- | :--- |

| `carat` | float | Вес в каратах |

| `cut` | str | Качество огранки (Fair → Ideal) |

| `color` | str | Цвет (J → D, D — лучший) |

| `clarity` | str | Чистота (I1 → IF) |

| `depth` | float | Глубина в % от среднего диаметра |

| `table` | float | Ширина верхней площадки |

| `price` | int | Цена в USD (целевая переменная) |

| `x`, `y`, `z` | float | Физические размеры (мм) |



\## 🏗 Архитектура проекта



```

Data Science/

├── data/

│   └── raw/

│       └── diamonds.csv              # Исходные данные

├── src/

│   ├── \_\_init\_\_.py

│   ├── db.py                          # Подключение к PostgreSQL

│   ├── load\_diamonds.py               # Загрузка CSV в БД

│   ├── eda\_diamonds.py                # EDA и визуализация

│   ├── eda\_price\_log.py               # Логарифмирование цены

│   └── regression\_diamonds.py         # ML-пайплайн

├── reports/

│   └── figures/                       # Все графики

├── notebooks/                         # Jupyter-ноутбуки (опционально)

├── models/                            # Сохранённые модели

├── .env                               # Параметры БД (НЕ коммитится)

├── .gitignore

├── requirements.txt

└── README.md

```



\## 🛠 Стек



\- \*\*Python\*\* 3.14

\- \*\*PostgreSQL\*\* 17 + \*\*SQLAlchemy\*\* + \*\*psycopg2-binary\*\*

\- \*\*Pandas\*\*, \*\*NumPy\*\* — обработка данных

\- \*\*Matplotlib\*\*, \*\*Seaborn\*\* — визуализация

\- \*\*Scikit-learn\*\*, \*\*SciPy\*\* — ML и статистика

\- \*\*VS Code\*\* + \*\*SQLTools\*\* — среда разработки



\## 🚀 Быстрый старт



\### 1. Клонировать репозиторий



```bash

git clone <ваш-репозиторий>

cd "Data Science"

```



\### 2. Создать виртуальное окружение



```powershell

python -m venv .venv

.venv\\Scripts\\Activate

```



\### 3. Установить зависимости



```powershell

pip install -r requirements.txt

```



\### 4. Настроить `.env`



Создайте файл `.env` в корне проекта:



```

DB\_HOST=127.0.0.1

DB\_PORT=5432

DB\_NAME=postgres

DB\_USER=postgres

DB\_PASSWORD=ваш\_пароль

```



\### 5. Запустить пайплайн



```powershell

\# Загрузка CSV в PostgreSQL

python -m src.load\_diamonds



\# EDA и визуализация

python -m src.eda\_diamonds

python -m src.eda\_price\_log



\# ML-пайплайн

python -m src.regression\_diamonds

```



Все графики появятся в `reports/figures/`.



\## 📈 Результаты



\### EDA — ключевые находки



\- Цена сильно скошена вправо (skew = 1.62). После `log(1+price)` — skew = 0.12.

\- `carat` — доминирующий признак: корреляция с ценой 0.92.

\- `x`, `y`, `z` сильно коррелируют с `carat` (0.95–0.98) — \*\*мультиколлинеарность\*\*,

&#x20; поэтому они исключены из модели.

\- `depth` практически не влияет на цену (корреляция -0.01).



\### Модель — LinearRegression



\*\*Признаки:\*\* `carat`, `carat²`, `depth`, `table`, `cut\_ord`, `color\_ord`, `clarity\_ord`.



\*\*Целевая переменная:\*\* `log(1 + price)`.



| Метрика | В лог-шкале | В USD |

| :--- | :--- | :--- |

| \*\*R²\*\* | 0.9708 | 0.9387 |

| \*\*MAE\*\* | 0.1292 | 509 USD |

| \*\*RMSE\*\* | 0.1734 | 987 USD |



\### Важность признаков (по коэффициентам)



1\. `clarity` — +11.9% к цене за ступень

2\. `color` — +8.4% за ступень

3\. `cut` — +2.9% за ступень

4\. `depth`, `table` — практически не влияют



\### Ограничения модели



\- \*\*Линейная модель\*\*: не улавливает сложные взаимодействия признаков.

\- \*\*Клиппинг предсказаний\*\* по диапазону `y\_train` защищает от экстраполяции,

&#x20; но модель всё ещё плохо работает на очень дорогих камнях (>15 000 USD).

\- \*\*Мультиколлинеарность\*\* `x`, `y`, `z`, `carat` — решена удалением размеров.



\## 📊 Графики



Все визуализации — в `reports/figures/`:



| Файл | Содержание |

| :--- | :--- |

| `diamonds\_eda.png` | EDA: распределения цены и карата, scatter, boxplot |

| `price\_log\_transform.png` | Сравнение цены и log(1+цена) |

| `correlation\_matrix.png` | Корреляционная матрица признаков |

| `residuals\_diagnostics.png` | Диагностика остатков (гистограмма, Q-Q, scatter) |

| `actual\_vs\_predicted.png` | Факт vs прогноз |



\## 🔮 Возможные улучшения



\- \*\*Кросс-валидация\*\* (KFold) вместо одного train/test split.

\- \*\*Сравнение моделей\*\*: Ridge, Lasso, RandomForest, GradientBoosting.

\- \*\*Feature engineering\*\*: отношения `x/y`, `depth/table`, бинарные признаки.

\- \*\*Гиперпараметрический поиск\*\*: GridSearchCV, Optuna.

\- \*\*SHAP\*\* для интерпретации вклада признаков.



\## 📚 Источники



\- \[Diamonds dataset](https://ggplot2.tidyverse.org/reference/diamonds.html)

\- \[Scikit-learn documentation](https://scikit-learn.org/)

\- \[Pandas user guide](https://pandas.pydata.org/docs/user\_guide/)

\- Wes McKinney, \*Python for Data Analysis\* (3rd ed.)



\## 👤 Автор



\*\*Хороший человек\*\* — начинающий Data Scientist.



\- GitHub: \[@username](https://github.com/username)

\- Telegram: \[@username](https://t.me/username)



\## 📄 Лицензия



Учебный проект. Данные — публичный датасет `diamonds`.

