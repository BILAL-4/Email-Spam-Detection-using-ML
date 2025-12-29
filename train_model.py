import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')

# 2. Load Dataset
df = pd.read_csv("emails.csv")
df.columns = ['Text', 'Label']
df.drop_duplicates(inplace=True)

# 3. Text Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['Text'] = df['Text'].apply(preprocess_text)

# 4. Train-Test Split
X = df['Text']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# 5. Feature Extraction
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 6. Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# 7. Evaluation
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 8. Save Model & Vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and Vectorizer saved successfully.")
