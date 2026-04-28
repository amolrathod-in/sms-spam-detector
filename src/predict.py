import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, '../models/model.pkl')
vectorizer_path = os.path.join(BASE_DIR, '../models/vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_message(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Spam" if prediction[0] == 1 else "Ham"


if __name__ == "__main__":
    print(predict_message("Congratulations! You won a free ticket"))
    print(predict_message("Hey, are we meeting today?"))
    print(predict_message("Free entry in a contest now!!!"))
    print(predict_message("Reminder: Your subscription is about to expire. Renew now to continue services."))
    print(predict_message("Your package couldn’t be delivered today. Please confirm your address here: http://bit.ly/update"))
