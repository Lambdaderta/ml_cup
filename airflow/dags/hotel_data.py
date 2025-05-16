import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk


def data_work():
    data = []

    with open('/home/lambda/projects/mm_2/airflow/data/geo-reviews-dataset-2023.tskv', mode='r', encoding='UTF-8') as lines:
        for line in lines:
            row = {}
            line = line.split('\t')
            for i in line:
                lol = i.split('=', 1)
                row[lol[0]] = lol[1]
            data.append(row)

    df = pd.DataFrame(data=data)
    df['rating'] = df['rating'].apply(lambda x: int(x[0]))
    df['rubrics'] = df['rubrics'].apply(lambda x: set(x.split(';')))
    df = df[df['rubrics'].apply(lambda x: "Гостиница" in x)]
    df = df.dropna()
    df['city'] = df['address'].str.split(',').apply(lambda x: x[0])
    df = df.reset_index(drop=True)
    df['hotel_id'] = df.index 
    df.to_csv('/home/lambda/projects/mm_2/airflow/data/dataset.csv', index=False)

    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('punkt')

    ru_stp = stopwords.words('russian')

    def text_obr(line):
        line = re.sub(r'[^а-яА-ЯёЁ\s]+', ' ', line)
        line = line.lower()
        line = line.split()
        line = [SnowballStemmer(language="russian").stem(i) for i in line if line not in ru_stp]
        return " ".join(line)

    X = df['text'].apply(lambda x: text_obr(x))
    tfidf = TfidfVectorizer(ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(X)


    joblib.dump(tfidf, '/home/lambda/projects/mm_2/airflow/data/tfidf_model.pkl') 
    joblib.dump({'ids': df['hotel_id'], 'vectors': X_tfidf}, '/home/lambda/projects/mm_2/airflow/data/tfidf_vectors.pkl') 

data_work()

with DAG(
    dag_id='hotels_dag',
    start_date=datetime(2025, 2, 1),
    schedule_interval='@weekly',  
    catchup=False
) as dag:
    task = PythonOperator(
        task_id='extract_data',
        python_callable=data_work
    )
    task

#echo $AIRFLOW_HOME
#export AIRFLOW_HOME=~/projects/mm_2/airflow/
#source ~/.bashrc