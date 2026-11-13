
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create sample WSN dataset
data = {
    "hour": [9, 10, 11, 12, 14, 2, 3, 1, 15, 16,
             4, 5, 10, 2, 13, 3, 11, 1, 14, 4],

    "source_ip": [
        "192.168.1.2", "192.168.1.3", "192.168.1.4",
        "192.168.1.5", "192.168.1.2", "10.0.0.5",
        "10.0.0.6", "10.0.0.7", "192.168.1.3",
        "192.168.1.4", "10.0.0.8", "10.0.0.9",
        "192.168.1.5", "10.0.0.5", "192.168.1.2",
        "10.0.0.6", "192.168.1.3", "10.0.0.7",
        "192.168.1.4", "10.0.0.8"
    ],

    "event_type": [
        "Normal", "Normal", "Normal", "Normal", "Normal",
        "Flooding", "Flooding", "Unauthorized", "Normal",
        "Normal", "Unauthorized", "Flooding", "Normal",
        "Flooding", "Normal", "Unauthorized", "Normal",
        "Flooding", "Normal", "Unauthorized"
    ],

    "label": [
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
        1, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

# Convert event type into numerical values
df["event_type"] = df["event_type"].map({
    "Normal": 0,
    "Flooding": 1,
    "Unauthorized": 2
})

# Convert source IP into numerical values
df["source_ip"] = df["source_ip"].astype("category").cat.codes

# Select features and target
X = df[["hour", "source_ip", "event_type"]]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Test a new network event
new_event = pd.DataFrame([{
    "hour": 2,
    "source_ip": df["source_ip"].max() + 1,
    "event_type": 1
}])

prediction = model.predict(new_event)

if prediction[0] == 1:
    print("\nResult: Potentially Malicious Event")
else:
    print("\nResult: Normal Event")
Accuracy: 100.0 %

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       1.00      1.00      1.00         3

    accuracy                           1.00         6
   macro avg       1.00      1.00      1.00         6
weighted avg       1.00      1.00      1.00         6


Result: Normal Event
 
