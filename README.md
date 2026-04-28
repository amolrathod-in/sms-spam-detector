# 📩 SMS Spam Detection using Python & Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-red)

---

## 🚀 Project Overview

This project is a **Machine Learning-based SMS Spam Detection system** that classifies messages as:

* ✅ **Ham (Not Spam)**
* 🚫 **Spam**

It leverages **Natural Language Processing (NLP)** and a trained classification model to detect unwanted, misleading, or phishing messages in real-time.

---

## 🎯 Problem Statement

With the rise of spam, phishing, and fraudulent SMS messages, users are increasingly vulnerable to scams.

This project aims to:

* Automatically detect spam messages
* Improve user safety
* Demonstrate real-world application of NLP + ML

---

## 🧠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK
* Flask (Web Application)

### Core Concepts

* TF-IDF
* Logistic Regression

---

## 📂 Project Structure

```
sms-spam-detector/
│── data/
│   ├── raw/
│   │   └── spam.csv
│   └── processed/
│
│── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
│── notebooks/
│   └── sms_spam_analysis.ipynb
│
│── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
│── app/
│   ├── app.py
│   ├── templates/
│   │    └── index.html   
│   └── static/
│        └── styles.css
│
│── tests/
│   └── test_predict.py
│
│── requirements.txt
│── README.md
```

---

## ⚙️ How It Works

1. **Data Preprocessing**

   * Lowercasing
   * Removing punctuation
   * Cleaning text

2. **Feature Extraction**

   * Convert text into numerical vectors using TF-IDF
   * Add custom features (caps, digits, keywords, etc.)

3. **Model Training**

   * Train a Logistic Regression model on labeled data
   * Handle class imbalance

4. **Prediction**

   * Input message → Vectorization → Model → Output

5. **Web Interface**

   * Built using Flask
   * Real-time prediction with clean UI

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone [<your-repo-link>](https://github.com/amolrathod-in/sms-spam-detector)
cd sms-spam-detector
```

### 2. Create environment

```bash
conda create -n spam_project python=3.12 -y
conda activate spam_project
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

### 5. Run the Flask app

```bash
cd app
python app.py
```

### 6. Open in browser

```
http://127.0.0.1:5000/
```

---

## 🧪 Example Inputs

### ✅ Ham

* "Hey, are we meeting today?"
* "I reached home safely."

### 🚫 Spam

* "Congratulations! You won a free prize!"
* "Your account requires verification. Please update details."

---

## 📊 Model Performance

| Metric         | Score |
| -------------- | ----- |
| Accuracy       | 97%   |
| Spam Recall    | 91%   |
| Spam Precision | 89%   |

### Key Insights:

* Strong performance on standard spam
* Improved detection of tricky spam using feature engineering
* Balanced model with reduced bias toward ham

---

## ⚠️ Limitations

* May struggle with highly **contextual or semantic spam**
* Relies on statistical patterns, not full language understanding
* Performance depends on dataset quality

---

## 🔥 Future Improvements

* Use advanced NLP models like BERT
* Add prediction probability (confidence score)
* Improve UI/UX (animations, history, dashboard)
* Deploy as a live web app
* Build REST API for integration

---

## 🌐 Deployment (Optional)

You can deploy this project using:

* Render
* Railway
* AWS

---

## 👨‍💻 Author

**Your Name**

---

## ⭐ Acknowledgements

* Dataset from Kaggle
* Scikit-learn & NLP community

---

## 📌 Conclusion

This project demonstrates how machine learning and NLP can be applied to solve real-world problems like spam detection.

It highlights:

* End-to-end ML pipeline
* Model optimization
* Deployment with Flask

---

