# train_model.py
import pandas as pd
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

# Download the stopwords corpus from NLTK if it's not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Load the dataset (make sure 'sarcasm.json' is in the same directory)
data = pd.read_json("sarcasm.json", lines=True)

# Clean and preprocess the text (removing punctuation, stopwords, and converting to lowercase)
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()  # Split text into words
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Apply the cleaning function to the 'headline' column
data["cleaned_text"] = data["headline"].apply(clean_text)

# Features (X) and target variable (y)
X = data["cleaned_text"]
y = data["is_sarcastic"]

# TF-IDF Vectorization (transform the cleaned text into a numeric format)
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model (optional)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {accuracy:.2f}")

# Save the trained model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ model.pkl and vectorizer.pkl saved!")
