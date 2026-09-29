"""Общий модуль подключения к PostgreSQL."""
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv


def get_engine():
    """Создаёт движок SQLAlchemy для подключения к PostgreSQL."""
    load_dotenv()
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(db_url)