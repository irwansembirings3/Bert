import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

### 🔹 Step 1: Load Dataset (Example Data)
data = {
    "text": [
        "Dear user, your account has been compromised! Click here to reset your password: http://fake-url.com",
        "Your PayPal account needs verification. Please login here: http://phishingsite.com",
        "Hello John, your meeting is scheduled for tomorrow at 10 AM.",
        "Win a FREE iPhone now! Click this link: http://scam.com",
        "Congratulations! You won $1000! Claim now: http://fraud.com",
        "Reminder: Your subscription expires soon. Renew at http://legit-site.com"
    ],
    "label": [1, 1, 0, 1, 1, 0]  # 1 = Phishing, 0 = Legitimate
}

df = pd.DataFrame(data)

### 🔹 Step 2: Text Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

df["clean_text"] = df["text"].apply(preprocess_text)

### 🔹 Step 3: Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

### 🔹 Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 🔹 Step 5: Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

### 🔹 Step 6: Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

### 🔹 Step 7: Predict New Emails
new_emails = [
    "Your account is locked! Click here to unlock: http://phishing.com",
    "Team meeting is scheduled for today at 3 PM."
]
new_cleaned = [preprocess_text(email) for email in new_emails]
new_features = vectorizer.transform(new_cleaned)
predictions = model.predict(new_features)

# Print Results
for email, pred in zip(new_emails, predictions):
    print(f"🔍 Email: {email}\n📌 Prediction: {'Phishing' if pred == 1 else 'Legitimate'}\n")