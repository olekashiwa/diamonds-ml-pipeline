import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Собираем строку подключения
DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# Создаём движок SQLAlchemy
engine = create_engine(DB_URL)

try:
    df = pd.read_sql("SELECT version();", engine)
    print("✅ Подключение успешно!")
    print(df)
except Exception as e:
    print(f"❌ Ошибка подключения: {e}")