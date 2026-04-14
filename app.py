from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model bundle
bundle = pickle.load(open("model.pkl", "rb"))
model = bundle["model"]
feature_names = bundle["feature_names"]
label_encoders = bundle["label_encoders"]

print(f"✅ Model loaded | Features: {len(feature_names)} | Variant: {bundle['variant_used']}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({"features": feature_names})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        row = {}
        for feat in feature_names:
            row[feat] = data.get(feat, 0)

        df = pd.DataFrame([row])

        # Encode categorical columns
        for col, le in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except:
                    df[col] = 0

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        result = "Fraud" if prediction == 1 else "Not Fraud"

        return jsonify({
            "prediction": result,
            "fraud_probability": round(float(probability) * 100, 2),
            "is_fraud": bool(prediction == 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)