import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

# Читаем таблицу через SQL-запрос
df = pd.read_sql("SELECT * FROM diamonds;", engine)

print(f"Размер DataFrame: {df.shape}")
print(f"\nТипы данных:\n{df.dtypes}")
print(f"\nПервые 5 строк:\n{df.head()}")