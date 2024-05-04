# from flask import Flask, request, render_template
# import openai
# import joblib
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import re

# app = Flask(__name__)

# # Assuming you have set your OpenAI API key in your environment variables
# # openai.api_key = 'sk-proj-s4wIpQmM6waSOCFn5k5ZT3BlbkFJG7XHtm2iVSZhkhBmXaYo'

# # Load the recommender model and other necessary objects
# knn_model = joblib.load("Recommender.pkl")
# data = pd.read_csv("./aitools(2).csv")
# tfidf_vectorizer = TfidfVectorizer()
# stop_words = set(stopwords.words("english"))


# # Preprocess text data
# def preprocess_text(text):
#     if isinstance(text, str):  # Check if text is not NaN
#         text = text.lower()
#         text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
#         words = word_tokenize(text)  # Tokenize
#         words = [word for word in words if word not in stop_words]  # Remove stopwords
#         return " ".join(words)
#     else:
#         return ""  # Return empty string for NaN
    
# # Fit the TfidfVectorizer on your text data
# tfidf_vectorizer.fit(data["Description"].apply(preprocess_text))


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/submit', methods=['POST'])
# def submit():
#     problem = request.form['problem']
#     # response = openai.Completion.create(
#     #   engine="gpt-3.5-turbo",
#     #   prompt=problem,
#     #   max_tokens=150
#     # )
#     # answer = response.choices[0].text.strip()

#     user_input = preprocess_text(problem)
#     user_input_vector = tfidf_vectorizer.transform([user_input])
#     distances, indices = knn_model.kneighbors(user_input_vector)
    
#     recommended_tool_names = []

#     for i in indices[0]:
#         tool_name = data.loc[i, "AI Tool Name"]
#         recommended_tool_names.append(tool_name)

#     # return render_template('output.html', response=answer, tool_names=recommended_tool_names)
#     return render_template('output.html', tool_names=recommended_tool_names)

# if __name__ == '__main__':
#     app.run(debug=True)

import requests
from flask import Flask, request, render_template
import openai
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2-large"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Assuming you have set your OpenAI API key in your environment variables
# openai.api_key = 'sk-proj-s4wIpQmM6waSOCFn5k5ZT3BlbkFJG7XHtm2iVSZhkhBmXaYo'

# Load the recommender model and other necessary objects
knn_model = joblib.load("Recommender.pkl")
data = pd.read_csv("./aitools(2).csv")
tfidf_vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words("english"))


# Preprocess text data
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is not NaN
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        words = word_tokenize(text)  # Tokenize
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return " ".join(words)
    else:
        return ""  # Return empty string for NaN
    
# Fit the TfidfVectorizer on your text data
tfidf_vectorizer.fit(data["Description"].apply(preprocess_text))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    problem = request.form['problem']
    gpt2_output = query({
        "inputs": problem,
    })
    answer = gpt2_output[0]['generated_text']

    user_input = preprocess_text(problem)
    user_input_vector = tfidf_vectorizer.transform([user_input])
    distances, indices = knn_model.kneighbors(user_input_vector)
    
    recommended_tool_names = []

    for i in indices[0]:
        tool_name = data.loc[i, "AI Tool Name"]
        recommended_tool_names.append(tool_name)

    return render_template('output.html', response=answer, tool_names=recommended_tool_names)

if __name__ == '__main__':
    app.run(debug=True)
