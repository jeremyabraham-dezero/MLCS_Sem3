
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create a sample email dataset
data = {
    "sender_match": [1, 1, 1, 0, 0, 0, 1, 0, 1, 0,
                     1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "spf_pass":     [1, 1, 1, 0, 0, 0, 1, 0, 1, 0,
                     1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "dkim_pass":    [1, 1, 1, 0, 0, 0, 1, 0, 1, 0,
                     1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "label":        [0, 0, 0, 1, 1, 1, 0, 1, 0, 1,
                     0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[["sender_match", "spf_pass", "dkim_pass"]]
y = df["label"]  # 0 = Legitimate, 1 = Potentially Spoofed

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Legitimate", "Potentially Spoofed"],
    zero_division=0
))

# Test a new email
new_email = pd.DataFrame([{
    "sender_match": 0,
    "spf_pass": 0,
    "dkim_pass": 0
}])

prediction = model.predict(new_email)[0]

if prediction == 1:
    print("Email Status: Potentially Spoofed")
else:
    print("Email Status: Legitimate")
Accuracy: 100.0 %

Classification Report:
                     precision    recall  f1-score   support

         Legitimate       1.00      1.00      1.00         3
Potentially Spoofed       1.00      1.00      1.00         3

           accuracy                           1.00         6
          macro avg       1.00      1.00      1.00         6
       weighted avg       1.00      1.00      1.00         6

Email Status: Potentially Spoofed
 
