import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/text_data.csv")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(report)

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(model_pipeline, "model/text_classifier.joblib")

with open("results/metrics.txt", "w") as f:
    f.write(f"Model: Text Classification (TF-IDF + Logistic Regression)\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(f"Classification Report:\n{report}\n")

print("Text Classification model trained and saved successfully!")
