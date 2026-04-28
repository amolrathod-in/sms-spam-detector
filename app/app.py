from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, '../models/model.pkl')
vectorizer_path = os.path.join(BASE_DIR, '../models/vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_message(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Spam" if prediction[0] == 1 else "Ham"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message')

    if not message:
        return render_template('index.html', prediction="Please enter a message")

    result = predict_message(message)

    return render_template(
        'index.html',
        prediction=result,
        message=message
    )


if __name__ == '__main__':
    app.run(debug=True)