
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create sample database dataset
data = {
    "special_characters": [0, 1, 0, 2, 8, 10, 7, 9, 1, 0,
                           6, 8, 0, 9, 1, 7, 0, 10, 2, 8],
    "input_length": [10, 15, 12, 18, 90, 120, 85, 110, 20, 14,
                     95, 100, 12, 115, 16, 88, 10, 130, 18, 105],
    "sql_keywords": [0, 0, 0, 1, 5, 6, 4, 5, 0, 0,
                     4, 5, 0, 6, 0, 4, 0, 7, 1, 5],
    "label": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
              1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Select input features and target
X = df[["special_characters", "input_length", "sql_keywords"]]
y = df["label"]  # 0 = Normal, 1 = Potentially Malicious

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

# Test a new database input
new_data = pd.DataFrame([{
    "special_characters": 8,
    "input_length": 100,
    "sql_keywords": 5
}])

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("\nResult: Potentially Malicious Data Injection")
else:
    print("\nResult: Normal Database Input")
    
Accuracy: 100.0 %

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       1.00      1.00      1.00         3

    accuracy                           1.00         6
   macro avg       1.00      1.00      1.00         6
weighted avg       1.00      1.00      1.00         6


Result: Potentially Malicious Data Injection
 
