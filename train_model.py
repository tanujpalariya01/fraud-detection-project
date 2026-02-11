import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Load dataset
data = pd.read_csv("creditcard.csv")

# Split features and label
X = data.drop("Class", axis=1)
y = data["Class"]

# Train-test split (important for imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("✅ model.pkl created successfully")