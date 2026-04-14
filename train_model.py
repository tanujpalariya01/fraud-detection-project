# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# import pickle

# # Load dataset
# data = pd.read_csv("creditcard.csv")

# # Split features and label
# X = data.drop("Class", axis=1)
# y = data["Class"]

# # Train-test split (important for imbalance)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # # Train Logistic Regression model
# model = LogisticRegression(max_iter=1000, class_weight="balanced")
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# # Precision, Recall, F1-score
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix:")
# print(cm)


# # Evaluate model
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Save model
# pickle.dump(model, open("model.pkl", "wb"))
# print("✅ model.pkl created successfully")

##################################     last code ############################### 

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import pickle

# # Load dataset
# data = pd.read_csv("creditcard.csv")

# # Split features and label
# X = data.drop("Class", axis=1)
# y = data["Class"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train Logistic Regression model
# model = LogisticRegression(max_iter=1000, class_weight="balanced")
# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)

# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Precision, Recall, F1-score
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix:")
# print(cm)

# # Save model
# pickle.dump(model, open("model.pkl", "wb"))
# print("✅ model.pkl created successfully")

######################### new code ###################

import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_FOLDER = "./fraud_data"        # Folder where BAF CSVs are saved
USE_VARIANT  = "Base"               # Options: "Base", "Variant I", "Variant II",
                                    #          "Variant III", "Variant IV", "Variant V"
MODEL_TYPE   = "random_forest"      # Options: "logistic_regression", "random_forest"
MODEL_OUTPUT = "model.pkl"

# ─────────────────────────────────────────────
# STEP 1: LOAD DATASET
# ─────────────────────────────────────────────

# csv_path = os.path.join(DATA_FOLDER, f"{USE_VARIANT}.csv")

# if not os.path.exists(csv_path):
#     available = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
#     print(f"❌ File not found: {csv_path}")
#     print(f"   Available files: {[os.path.basename(f) for f in available]}")
#     raise FileNotFoundError(f"Could not find {csv_path}")

# print(f"📂 Loading dataset: {USE_VARIANT}")
# data = pd.read_csv(csv_path)
# print(f"✅ Loaded — Shape: {data.shape}")
# print(f"   Columns: {list(data.columns)}\n")
# STEP 1: LOAD ALL DATASETS

import glob

print("📂 Loading ALL dataset variants...\n")

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

df_list = []

for file in csv_files:
    print(f"👉 Loading: {os.path.basename(file)}")
    df = pd.read_csv(file)
    df_list.append(df)

# Combine all datasets
data = pd.concat(df_list, ignore_index=True)

print(f"\n✅ Combined dataset shape: {data.shape}")
print(f"📊 Total files used: {len(csv_files)}")
print(f"   Columns: {list(data.columns)}\n")

# ─────────────────────────────────────────────
# STEP 2: EXPLORE CLASS DISTRIBUTION
# ─────────────────────────────────────────────

print("📊 Class Distribution:")
print(data["fraud_bool"].value_counts())
fraud_rate = data["fraud_bool"].mean() * 100
print(f"   Fraud rate: {fraud_rate:.2f}%\n")

# ─────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ─────────────────────────────────────────────

# Separate features and label
X = data.drop("fraud_bool", axis=1)
y = data["fraud_bool"]

# Encode categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
label_encoders = {}

if categorical_cols:
    print(f"🔤 Encoding categorical columns: {categorical_cols}")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Handle any remaining NaN values
X = X.fillna(X.median(numeric_only=True))

print(f"✅ Feature shape after preprocessing: {X.shape}\n")

# ─────────────────────────────────────────────
# STEP 4: TRAIN-TEST SPLIT
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"📦 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

# ─────────────────────────────────────────────
# STEP 5: TRAIN MODEL
# ─────────────────────────────────────────────

if MODEL_TYPE == "logistic_regression":
    print("🤖 Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)

elif MODEL_TYPE == "random_forest":
    print("🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

model.fit(X_train, y_train)
print("✅ Training complete!\n")

# ─────────────────────────────────────────────
# STEP 6: EVALUATE MODEL
# ─────────────────────────────────────────────

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_prob)

print(f"📈 Accuracy : {accuracy:.4f}")
print(f"📈 ROC-AUC  : {roc_auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

cm = confusion_matrix(y_test, y_pred)
print("📊 Confusion Matrix:")
print(f"   True Negatives  (Legit correctly identified): {cm[0][0]}")
print(f"   False Positives (Legit flagged as Fraud)    : {cm[0][1]}")
print(f"   False Negatives (Fraud missed)              : {cm[1][0]}")
print(f"   True Positives  (Fraud correctly caught)    : {cm[1][1]}\n")

# ─────────────────────────────────────────────
# STEP 7: SAVE MODEL + METADATA
# ─────────────────────────────────────────────

# Save feature names and encoders alongside model
model_bundle = {
    "model": model,
    "feature_names": list(X.columns),
    "label_encoders": label_encoders,
    "variant_used": USE_VARIANT,
    "model_type": MODEL_TYPE
}

pickle.dump(model_bundle, open(MODEL_OUTPUT, "wb"))
print(f"✅ Model saved to: {MODEL_OUTPUT}")
print(f"   Features used  : {len(X.columns)}")
print(f"   Variant trained: {USE_VARIANT}")
print(f"   Model type     : {MODEL_TYPE}")