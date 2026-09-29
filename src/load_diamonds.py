"""Шаг 1: Загрузка diamonds.csv в PostgreSQL."""
import os
from pathlib import Path

import pandas as pd

from src.db import get_engine


# Корень проекта: поднимаемся на два уровня вверх от src/load_diamonds.py
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "raw" / "diamonds.csv"


def main():
    """Читает CSV и загружает его в PostgreSQL."""
    engine = get_engine()

    # 1. Читаем CSV
    print(f"📂 Читаю файл: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    print("\n📊 Исходные данные:")
    print(df.head())
    print(f"\nРазмер: {df.shape}")
    print(f"Колонки: {list(df.columns)}")

    # 2. Загружаем в PostgreSQL
    # if_exists='replace' — если таблица уже есть, она будет пересоздана
    df.to_sql(
        name="diamonds",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",   # пакетная вставка — быстрее построчной
        chunksize=1000,   # по 1000 строк за раз
    )
    print("\n✅ Данные успешно загружены в таблицу 'diamonds'")

    # 3. Проверяем, что данные на месте
    check = pd.read_sql("SELECT COUNT(*) AS cnt FROM diamonds;", engine)
    print(f"📈 Строк в таблице: {check['cnt'].iloc[0]}")


if __name__ == "__main__":
    main()