from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import os


base_path = '/home/lambda/projects/ML Championship/airflow/data'


tfidf_path = os.path.join(base_path, 'tfidf_model.pkl')
tfidf_vec_path = os.path.join(base_path, 'tfidf_vectors.pkl')
dataset_path = os.path.join(base_path, 'dataset.csv')


tfidf = joblib.load(tfidf_path)
df = pd.read_csv(dataset_path)


cities = df['city'].unique().tolist()


app = Flask(__name__)



def text_obr(line):
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('punkt')
        ru_stp = stopwords.words('russian')

        line = re.sub(r'[^а-яА-ЯёЁ\s]+', ' ', line)
        line = line.lower()
        line = line.split()
        line = [SnowballStemmer(language="russian").stem(i) for i in line if line not in ru_stp]
        return " ".join(line)

@app.route("/", methods=['POST', 'GET'])
def index():
    results = None  
    if request.method == "POST":
        city = request.form.get('city')
        rating = list(map(int, request.form.getlist('rating')))
        user_text = request.form.get('description', '').strip()
        
        filtered_df = df[
            (df['city'] == city) & 
            (df['rating'].isin(rating))
        ] 
        
        if filtered_df.empty:
            results = pd.DataFrame()  
        else:
            data = joblib.load(tfidf_vec_path)
            X_tfidf = data['vectors']
            
            processed_text = text_obr(user_text)
            user_vector = tfidf.transform([processed_text])
            
            similarities = cosine_similarity(
                user_vector,
                X_tfidf[filtered_df.index]  
            ).flatten()
            
            filtered_df = filtered_df.copy()
            filtered_df['similarity'] = similarities
            results = filtered_df.sort_values(by='similarity', ascending=False).head(20)
    
    return render_template('index.html', cities=cities, results=results.to_dict('records') if results is not None else [])
if __name__=="__main__":
    app.run(debug=True)