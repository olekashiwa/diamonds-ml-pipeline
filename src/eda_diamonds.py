"""Шаг 3: EDA датасета diamonds — статистика и визуализация."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.db import get_engine


# --- Настройка шрифтов для кириллицы ---
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

# --- EDA ---
print("=" * 60)
print("ОБЩАЯ ИНФОРМАЦИЯ")
print("=" * 60)
print(df.info())

print("\n" + "=" * 60)
print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("КОРРЕЛЯЦИИ")
print("=" * 60)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print(df[numeric_cols].corr())

# --- Визуализация ---
sns.set_theme(style="whitegrid", font='DejaVu Sans')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Разведочный анализ данных: Бриллианты', fontsize=16, fontweight='bold')

# 1. Распределение цены
sns.histplot(data=df, x='price', bins=50, kde=True, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Распределение цены')
axes[0, 0].set_xlabel('Цена, USD')

# 2. Распределение веса (карат)
sns.histplot(data=df, x='carat', bins=50, kde=True, ax=axes[0, 1], color='coral')
axes[0, 1].set_title('Распределение веса (карат)')
axes[0, 1].set_xlabel('Карат')

# 3. Зависимость цены от карата
sns.scatterplot(data=df, x='carat', y='price', alpha=0.3, ax=axes[1, 0], s=10)
axes[1, 0].set_title('Цена vs Вес')
axes[1, 0].set_xlabel('Карат')
axes[1, 0].set_ylabel('Цена, USD')

# 4. Boxplot цены по качеству огранки
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
sns.boxplot(data=df, x='cut', y='price', order=cut_order, ax=axes[1, 1])
axes[1, 1].set_title('Цена по качеству огранки')
axes[1, 1].set_xlabel('Огранка')
axes[1, 1].set_ylabel('Цена, USD')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(FIG_DIR / "diamonds_eda.png", dpi=100, bbox_inches='tight')
print(f"\n✅ График сохранён: {FIG_DIR / 'diamonds_eda.png'}")