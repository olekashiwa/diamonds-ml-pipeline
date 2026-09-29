"""Шаги 5–10: линейная регрессия для предсказания цены бриллианта.

Пайплайн:
1. Загрузка данных из PostgreSQL.
2. Изучение мультиколлинеарности.
3. Подготовка признаков (ordinal-кодирование категорий).
4. Train/test split.
5. Обучение LinearRegression на log(1 + price).
6. Метрики в лог- и исходной шкале.
7. Диагностика остатков.
8. График «факт vs прогноз».
9. Интерпретация коэффициентов.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.db import get_engine


# --- Настройка шрифтов ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- Пути проекта ---
BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Шаг 1: Загрузка данных
# ============================================================
engine = get_engine()
df = pd.read_sql("SELECT * FROM diamonds;", engine)

print(f"Размер данных: {df.shape}")
print(f"Пропуски:\n{df.isnull().sum()}")
print(f"Типы:\n{df.dtypes}")

# ============================================================
# Шаг 2: Мультиколлинеарность
# ============================================================
numeric_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Корреляционная матрица числовых признаков')
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=100, bbox_inches='tight')
print(f"\n✅ correlation_matrix.png сохранён в {FIG_DIR}")
print("\nКорреляции с ценой:")
print(corr_matrix['price'].sort_values(ascending=False).round(3))


# ============================================================
# Шаг 3: Подготовка признаков
# ============================================================
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']  # D — лучший
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

df_model = df.copy()
df_model['cut_ord'] = df_model['cut'].map({v: i for i, v in enumerate(cut_order)})
df_model['color_ord'] = df_model['color'].map({v: i for i, v in enumerate(color_order)})
df_model['clarity_ord'] = df_model['clarity'].map({v: i for i, v in enumerate(clarity_order)})   
df_model['log_price'] = np.log1p(df_model['price'])
df_model['carat_sq'] = df_model['carat'] ** 2   # ← добавили

feature_cols = ['carat', 'carat_sq', 'depth', 'table', 'cut_ord', 'color_ord', 'clarity_ord']
X = df_model[feature_cols]
y = df_model['log_price']

print(f"\nМатрица признаков: {X.shape}")
print(f"Целевая переменная: {y.shape}")

print("Колонки X:", X.columns.tolist())
print("y (log_price) — первые 5:", y.head().tolist())
print("y (log_price) — describe:")
print(y.describe())


# ============================================================
# Шаг 4: Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} строк")
print(f"Test:  {X_test.shape[0]} строк")


# ============================================================
# Шаг 5: Обучение модели
# ============================================================
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nIntercept: {model.intercept_:.4f}")
print("\nКоэффициенты (в лог-шкале):")
for name, coef in zip(feature_cols, model.coef_):
    print(f"  {name:15s}: {coef:+.4f}")


# ============================================================
# Шаг 6: Метрики
# ============================================================
y_pred_log = model.predict(X_test)
y_pred_log = np.clip(
    y_pred_log,
    float(y_train.min()),
    float(y_train.max())
)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

print("Проверка y_true:")
print(f"  NaN: {y_true.isna().sum()}")
print(f"  inf: {np.isinf(y_true).sum()}")
print(f"  min: {y_true.min()}, max: {y_true.max()}")
print(f"  отрицательных: {(y_true < 0).sum()}")
print(f"  нулей: {(y_true == 0).sum()}")

print("\nПроверка y_pred:")
print(f"  NaN: {np.isnan(y_pred).sum()}")
print(f"  inf: {np.isinf(y_pred).sum()}")
print(f"  min: {y_pred.min()}, max: {y_pred.max()}")

r2_log = r2_score(y_test, y_pred_log)
mae_log = mean_absolute_error(y_test, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))

r2_usd = r2_score(y_true, y_pred)
mae_usd = mean_absolute_error(y_true, y_pred)
rmse_usd = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"\nDEBUG: y_true (USD) — первые 5: {y_true.head().tolist()}")
print(f"DEBUG: y_pred (USD) — первые 5: {y_pred[:5].tolist()}")
print(f"DEBUG: y_test (log) — первые 5: {y_test.head().tolist()}")
print(f"DEBUG: y_pred_log — первые 5: {y_pred_log[:5].tolist()}")

print("\n" + "=" * 60)
print("МЕТРИКИ В ЛОГ-ШКАЛЕ")
print("=" * 60)
print(f"R²:   {r2_log:.4f}")
print(f"MAE:  {mae_log:.4f}")
print(f"RMSE: {rmse_log:.4f}")

print("\n" + "=" * 60)
print("МЕТРИКИ В ИСХОДНОЙ ШКАЛЕ (USD)")
print("=" * 60)
print(f"R²:   {r2_usd:.4f}")
print(f"MAE:  {mae_usd:.2f} USD")
print(f"RMSE: {rmse_usd:.2f} USD")


# ============================================================
# Шаг 7: Диагностика остатков
# ============================================================
residuals = y_test - y_pred_log

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Диагностика остатков', fontsize=16, fontweight='bold')

sns.histplot(residuals, bins=60, kde=True, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title(f'Распределение остатков (skew = {residuals.skew():.2f})')
axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)

stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q plot остатков')

axes[1, 0].scatter(y_pred_log, residuals, alpha=0.3, s=5)
axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Предсказанные log(цена)')
axes[1, 0].set_ylabel('Остатки')
axes[1, 0].set_title('Остатки vs Предсказания')

axes[1, 1].scatter(X_test['carat'], residuals, alpha=0.3, s=5)
axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Карат')
axes[1, 1].set_ylabel('Остатки')
axes[1, 1].set_title('Остатки vs Карат')

plt.tight_layout()
plt.savefig(FIG_DIR / "residuals_diagnostics.png", dpi=100, bbox_inches='tight')
print(f"\n✅ residuals_diagnostics.png сохранён в {FIG_DIR}")
print(f"Среднее остатков: {residuals.mean():.4f} (должно быть ≈ 0)")
print(f"Std остатков:     {residuals.std():.4f}")


# ============================================================
# Шаг 8: Факт vs Прогноз
# ============================================================
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.3, s=5)
lims = [0, max(y_true.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', label='Идеальное предсказание')
plt.xlabel('Фактическая цена, USD')
plt.ylabel('Предсказанная цена, USD')
plt.title(f'Факт vs Прогноз (R² = {r2_usd:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=100, bbox_inches='tight')
print(f"\n✅ actual_vs_predicted.png сохранён в {FIG_DIR}")


# ============================================================
# Шаг 9: Интерпретация коэффициентов
# ============================================================
print("\n" + "=" * 60)
print("ИНТЕРПРЕТАЦИЯ КОЭФФИЦИЕНТОВ")
print("=" * 60)
for name, coef in zip(feature_cols, model.coef_):
    pct = (np.exp(coef) - 1) * 100
    print(f"{name:15s}: {coef:+.4f}  →  {pct:+.2f}% к цене")

print("\n🎉 Пайплайн завершён. Все графики в папке reports/figures/")