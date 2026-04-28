import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

SUSPICIOUS_WORDS = [
    "verify", "update", "account", "urgent", "reward",
    "eligible", "subscription", "renew", "confirm", "login",
    "process", "request", "action", "required", "attention"
]

def get_vectorizer():
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2
    )

def extra_features(texts):
    features = []
    
    for text in texts:
        text_lower = text.lower()

        num_digits = sum(c.isdigit() for c in text)
        num_caps = sum(1 for c in text if c.isupper())
        length = len(text)

        keyword_hits = sum(1 for word in SUSPICIOUS_WORDS if word in text_lower)

        has_urgent_tone = int(any(word in text_lower for word in ["urgent", "immediately", "now"]))

        features.append([
            num_digits,
            num_caps,
            length,
            keyword_hits,
            has_urgent_tone
        ])

    return csr_matrix(np.array(features))

def build_features(vectorizer, texts, fit=False):
    if fit:
        X_tfidf = vectorizer.fit_transform(texts)
    else:
        X_tfidf = vectorizer.transform(texts)

    X_extra = extra_features(texts)
    return hstack([X_tfidf, X_extra])