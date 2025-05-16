from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
import pandas as pd
import json
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def extract_data():
    """Извлечение данных из PostgreSQL"""
    logging.info("Начало извлечения данных из PostgreSQL")
    hook = PostgresHook(postgres_conn_id='hotels_db')
    query = "SELECT * FROM hotels;"
    df = hook.get_pandas_df(query)
    data_path = os.path.join(os.getenv('AIRFLOW_HOME'), 'data', 'hotels_data.csv')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    logging.info(f"Извлечено записей: {len(df)}")

def transform_data():
    """Преобразование данных"""
    logging.info("Начало преобразования данных")
    data_path = os.path.join(os.getenv('AIRFLOW_HOME'), 'data', 'hotels_data.csv')
    df = pd.read_csv(data_path)
    
    df['rubrics'] = df['rubrics'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df['text'])
    vectorizer_path = os.path.join(os.getenv('AIRFLOW_HOME'), 'data', 'test_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    df['vector'] = [json.dumps(vectors[i].toarray().tolist()[0]) for i in range(vectors.shape[0])]
    
    df['rating'] = df['rating'].astype(float)
    
    transformed_path = os.path.join(os.getenv('AIRFLOW_HOME'), 'data', 'transformed_data.csv')
    df.to_csv(transformed_path, index=False)
    logging.info("Данные преобразованы")

def load_data():
    """Загрузка данных в SQLite"""
    logging.info("Начало загрузки данных в SQLite")
    db_path = os.path.join(os.getenv('AIRFLOW_HOME'), 'data', 'hotels_vectors.db')
    transformed_path = os.path.join(os.getenv('AIRFLOW_HOME'), 'data', 'transformed_data.csv')
    df = pd.read_csv(transformed_path)
    
    with sqlite3.connect(db_path) as conn:
        df.to_sql('hotels_vectors', conn, if_exists='replace', index=False)
    logging.info("Данные загружены в SQLite")

with DAG(
    dag_id='hotel_etl_pipeline',
    start_date=datetime(2025, 2, 1),
    schedule_interval='@weekly',  
    catchup=False
) as dag:
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data
    )
    
    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data
    )
    
    load_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )
    
    extract_task >> transform_task >> load_task