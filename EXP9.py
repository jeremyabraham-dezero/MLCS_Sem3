
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create email dataset
data = {
    "email": [
        "Win a free prize now",
        "Claim your cash reward",
        "You won a lottery",
        "Free offer click now",
        "Get a free gift today",
        "Meeting scheduled for tomorrow",
        "Please send the project report",
        "Let us meet this afternoon",
        "Your assignment is ready",
        "The class starts at ten"
    ],
    "label": [
        "Spam", "Spam", "Spam", "Spam", "Spam",
        "Not Spam", "Not Spam", "Not Spam",
        "Not Spam", "Not Spam"
    ]
}

df = pd.DataFrame(data)

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Predict a new email
new_email = ["Congratulations! You won a free prize"]

new_features = vectorizer.transform(new_email)
prediction = model.predict(new_features)

print("Email:", new_email[0])
print("Prediction:", prediction[0])
Accuracy: 33.33333333333333 %
Email: Congratulations! You won a free prize
Prediction: Spam
 
