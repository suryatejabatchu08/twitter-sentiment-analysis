import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load the saved model and vectorizer
log_model = joblib.load("log_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit App
st.title("Sentiment Analysis")
st.write("Enter text to classify it as Negative or Positive sentiment.")

user_input = st.text_area("Enter Text:", "")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned_text = clean_text(user_input)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = log_model.predict(transformed_text)[0]

        result = "Negative 🔴" if prediction == 1 else "Positive 🟢"
        st.title(f"Prediction: **{result}**")
    else:
        st.title("Please enter text.")