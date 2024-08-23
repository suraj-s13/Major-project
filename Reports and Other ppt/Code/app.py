from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

app = Flask(__name__)

# Load the model and other necessary objects
knn_model = joblib.load("Recommender.pkl")
data = pd.read_csv("./aitools(2).csv")
tfidf_vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words("english"))
selected_columns = [
    "AI Tool Name",
    "Free/Paid/Other",
    "Charges",
    "Languages",
    "Major Category",
    "Tool Link",
    "Description",
]


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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    user_description = request.form["description"]
    pay = request.form["payment_option"]
    budget = float(request.form["budget"])
    user_input = preprocess_text(user_description)
    user_input_vector = tfidf_vectorizer.transform([user_input])
    distances, indices = knn_model.kneighbors(user_input_vector)
    print(budget)
    recommended_tools_with_info = pd.DataFrame(columns=selected_columns)
    answers_dict = {}

    # Iterate through the Series and add each element to the answers_dict
    # Initialize an empty list to store dictionaries
    tools_list = []

    for dist, i in zip(distances[0], indices[0]):
        tool_data = data.loc[i, selected_columns].to_dict()

        if pay == "All":
            tools_list.append(tool_data)
        elif pay == "Paid" and tool_data["Free/Paid/Other"] == "Paid":
            if budget == 300.0:
                tools_list.append(tool_data)
            elif budget >= float(tool_data["Charges"]):
                tools_list.append(tool_data)
        elif pay == "Free" and tool_data["Free/Paid/Other"] == "Free":
            tools_list.append(tool_data)

    # Now, tools_list contains a list of dictionaries, where each dictionary represents a row of data

    # print(tools_list)
    return render_template("recommend.html", tools=tools_list)


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(debug=False, host="0.0.0.0")
