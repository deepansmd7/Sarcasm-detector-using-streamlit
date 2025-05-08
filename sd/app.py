# app.py
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("😎 Sarcasm Detection App")

user_input = st.text_area("Enter a headline or sentence to detect sarcasm:")

if st.button("Analyze"):
    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)[0]
    
    if prediction == 1:
        st.error("🟥 This sentence is SARCASM.")
    else:
        st.success("🟩 This sentence is NOT sarcastic.")
