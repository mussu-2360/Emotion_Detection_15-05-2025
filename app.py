import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# UI
st.title("Emotion Detection App")
st.write("Enter a sentence to detect the emotion.")

user_input = st.text_area("Type your sentence:")

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        st.success(f"Predicted Emotion: **{prediction.capitalize()}**")
