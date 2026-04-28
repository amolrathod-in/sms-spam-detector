import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from processing import clean_text
from features import get_vectorizer


# Load Dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']


# Preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].apply(clean_text)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 4. Vectorization
vectorizer = get_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Logistic Regression Model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)


# Evaluation
y_pred = model.predict(X_test_vec)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Save Model
os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("\nTraining complete")
print("Model and vectorizer saved in /models")
