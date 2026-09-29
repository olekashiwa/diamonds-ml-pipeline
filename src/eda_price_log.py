"""Шаг 4: Логарифмическое преобразование цены — сравнение распределений."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # без окна, только сохранение в файл
import matplotlib.pyplot as plt
import seaborn as sns

from src.db import get_engine


# --- Настройка шрифтов ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- Пути проекта ---
BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Подключение к БД ---
engine = get_engine()
df = pd.read_sql("SELECT * FROM diamonds;", engine)

# --- Логарифмическое преобразование цены ---
df['log_price'] = np.log1p(df['price'])

print("Сравнение статистик:")
print(f"  Цена:        mean={df['price'].mean():.0f}, median={df['price'].median():.0f}, "
      f"skew={df['price'].skew():.2f}")
print(f"  log(1+Цена): mean={df['log_price'].mean():.2f}, median={df['log_price'].median():.2f}, "
      f"skew={df['log_price'].skew():.2f}")

# --- Сравнение распределений ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Сравнение: цена vs log(1 + цена)', fontsize=16, fontweight='bold')

# Верхний ряд — исходная цена
sns.histplot(df['price'], bins=60, kde=True, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title(f'Цена (skew = {df["price"].skew():.2f})')
axes[0, 0].set_xlabel('Цена, USD')

sns.boxplot(x=df['price'], ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title('Цена — boxplot')
axes[0, 1].set_xlabel('Цена, USD')

# Нижний ряд — логарифм цены
sns.histplot(df['log_price'], bins=60, kde=True, ax=axes[1, 0], color='coral')
axes[1, 0].set_title(f'log(1 + Цена) (skew = {df["log_price"].skew():.2f})')
axes[1, 0].set_xlabel('log(1 + Цена)')

sns.boxplot(x=df['log_price'], ax=axes[1, 1], color='coral')
axes[1, 1].set_title('log(1 + Цена) — boxplot')
axes[1, 1].set_xlabel('log(1 + Цена)')

plt.tight_layout()
plt.savefig(FIG_DIR / "price_log_transform.png", dpi=100, bbox_inches='tight')
print(f"\n✅ График сохранён: {FIG_DIR / 'price_log_transform.png'}")