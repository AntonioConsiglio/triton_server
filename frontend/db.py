import os
from io import BytesIO
from PIL import Image

from sqlalchemy import create_engine
from sqlalchemy.sql import text
import pandas as pd

import sqlalchemy
from sqlalchemy import Column, Integer, String, TIMESTAMP,inspect
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserResults(Base):
    __tablename__ = 'user_results'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False)
    result_image = Column(sqlalchemy.LargeBinary, nullable=False)
    timestamp = Column(TIMESTAMP, server_default=sqlalchemy.func.now())

    def __repr__(self):
        return f"<UserResults(id={self.id}, user_id={self.user_id}, timestamp={self.timestamp})>"
    
class DBClass():
    def __init__(self,session_id):
        self.engine = create_engine(os.getenv("DATABASE_URL"))
        self.userid = session_id
        self._create_table()
    
    def _create_table(self,):
        inspector = inspect(self.engine)
        if 'user_results' not in inspector.get_table_names():
            Base.metadata.create_all(self.engine)

    def _fetch_data(self,query):
        with self.engine.connect() as connection:
            result = pd.read_sql(query, connection)
        return result
    
    def save_result(self,image:Image):
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        with self.engine.connect() as connection:
            connection.execute(text("""
                INSERT INTO user_results (user_id, result_image)
                VALUES (:user_id, :result_image)
            """), {"user_id": self.userid, "result_image": img_byte_arr})
            connection.commit()

    def fetch_latest_results(self,n_results=5):
        with self.engine.connect() as connection:
            result = connection.execute(text(f"""
                SELECT result_image FROM user_results
                WHERE user_id = CAST(:user_id AS VARCHAR)
                ORDER BY timestamp DESC
                LIMIT :limit
            """), {"user_id": self.userid,"limit": n_results})
            rows = result.fetchall()
        return [row[0] for row in rows]