# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load trained ML model
# model = pickle.load(open("model.pkl", "rb"))

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     amount = float(data["amount"])

#     # Create feature vector (30 features: Time + V1–V28 + Amount)
#     features = np.zeros(30)
#     features[-1] = amount   # Amount feature

#     prediction = model.predict([features])[0]

#     result = "Fraud" if prediction == 1 else "Not Fraud"
#     return jsonify({"prediction": result})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load trained model bundle
# ─────────────────────────────────────────────

bundle          = pickle.load(open("model.pkl", "rb"))
model           = bundle["model"]
feature_names   = bundle["feature_names"]
label_encoders  = bundle["label_encoders"]

print(f"✅ Model loaded | Features: {len(feature_names)} | Variant: {bundle['variant_used']}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/features", methods=["GET"])
def get_features():
    """Return required feature names so frontend can build the form dynamically."""
    return jsonify({"features": feature_names})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Build a single-row dataframe with the correct feature order
        row = {}
        missing_features = []

        for feat in feature_names:
            if feat in data:
                row[feat] = data[feat]
            else:
                missing_features.append(feat)
                row[feat] = 0  # Default to 0 for missing features

        if missing_features:
            print(f"⚠️  Missing features (defaulted to 0): {missing_features}")

        df = pd.DataFrame([row])

        # Apply label encoding for categorical columns
        for col, le in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0  # Unknown category → default

        # Predict
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


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)