import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Add some background decoration using HTML and CSS
html_style = """
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton {
            background-color: #4CAF50;
            color: white;
        }
    </style>
"""

st.markdown(html_style, unsafe_allow_html=True)

# Streamlit App
st.title("Spam Classifier App")

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = pd.read_csv(url, compression="zip", sep="\t", names=["label", "message"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# Build a pipeline that combines a text feature extractor with a Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# User input
user_input = st.text_area("Enter a message to classify:", "Hello there!")

# Make predictions
prediction = model.predict([user_input])

# Display the prediction
st.subheader("Prediction:")
st.write(prediction[0])

# Evaluate the model
y_pred = model.predict(X_test)

# Display evaluation metrics
st.subheader("Model Evaluation:")
st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))
st.write("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
