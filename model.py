import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ================= STEP 1: Load dataset =================
data = pd.read_csv("medical_advice_dataset_clean.csv", encoding='utf-8')
data.columns = data.columns.str.strip()

print("Columns:", data.columns)


# ================= STEP 2: Clean text =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

data['text'] = data['text'].apply(clean_text)


# ================= STEP 3: Split =================
X = data['text']
y = data['label']


# ================= STEP 4: Train-test split =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ================= STEP 5: Vectorization =================
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ================= STEP 6: Train model =================
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)


# ================= STEP 7: Accuracy =================
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))


# ================= STEP 8: Save model =================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))


# ================= STEP 9: Rule-based layer =================

SUSPICIOUS_PHRASES = [
    "cure", "guaranteed", "no doctor", "stop medication",
    "instant relief", "permanent cure", "no side effects",
    "miracle", "natural remedy", "100%"
]

MEDICAL_KEYWORDS = [
    "medicine", "doctor", "disease", "treatment",
    "cure", "health", "symptom", "therapy",
    "drug", "tablet", "diabetes", "cancer",
    "fever", "infection", "pain", "insulin",
    "hospital", "diagnosis", "surgery"
]


def is_medical_related(text):
    text = text.lower()
    return any(word in text for word in MEDICAL_KEYWORDS)


def find_suspicious_phrases(text):
    text = text.lower()
    return [phrase for phrase in SUSPICIOUS_PHRASES if phrase in text]


def get_risk_level(prob):
    if prob < 0.4:
        return "Safe"
    elif prob < 0.7:
        return "Potentially Misleading"
    else:
        return "Harmful"


def get_action(risk):
    if risk == "Safe":
        return "Allow"
    elif risk == "Potentially Misleading":
        return "Flag for Review"
    else:
        return "Block"


# ================= STEP 10: Final function =================
def medical_safety_assessment(text):
    cleaned = clean_text(text)

    # A: Check relevance
    if not is_medical_related(text):
        return {
            "Category": "Not Related",
            "Explanation": "Input is not medical-related",
            "Action": "Ignore"
        }

    # B: ML prediction
    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0][1]

    # C: Uncertainty
    if 0.45 <= prob <= 0.55:
        return {
            "Category": "Uncertain",
            "Confidence": round(prob * 100, 2),
            "Action": "Needs Review"
        }

    # D: Final decision
    risk = get_risk_level(prob)
    suspicious = find_suspicious_phrases(text)
    action = get_action(risk)

    return {
        "Category": risk,
        "Confidence": round(prob * 100, 2),
        "Suspicious": suspicious,
        "Action": action
    }