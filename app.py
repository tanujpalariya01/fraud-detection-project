from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    amount = float(data["amount"])

    # Create feature vector (30 features: Time + V1–V28 + Amount)
    features = np.zeros(30)
    features[-1] = amount   # Amount feature

    prediction = model.predict([features])[0]

    result = "Fraud" if prediction == 1 else "Not Fraud"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)