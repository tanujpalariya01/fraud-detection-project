# 🚀 Fraud Detection System

An online fraud detection system using Machine Learning to identify fraudulent transactions.
This project integrates a trained ML model with a Flask-based web application for real-time prediction.

---

## 📂 Dataset

This project uses the **Bank Account Fraud (BAF) Dataset Suite (NeurIPS 2022)**.

🔗 Dataset Link:
https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?select=Variant+I.csv

### 📊 Dataset Description

The BAF dataset is a **large-scale synthetic dataset** designed for fraud detection and ML evaluation. It consists of multiple dataset variants with different types of bias and realistic financial behavior.

### 🔑 Key Characteristics

* Contains **6 dataset variants** (Base + Variant I–V)
* Each dataset has **~1 million records** ([Kaggle][1])
* Includes **30+ features** related to user behavior, transactions, and risk signals ([Kaggle][1])
* Highly **imbalanced dataset** (very low fraud rate) ([Kaggle][1])
* Designed to simulate **real-world fraud scenarios**
* Privacy-preserving (uses synthetic data generation techniques)

### 📌 Features include:

* Customer demographics (age, income, employment)
* Transaction details (amount, velocity, history)
* Device & behavioral data
* Risk indicators (credit score, fraud signals)

---

## 🧠 What I Did

* 📁 Organized all dataset variants into a single directory
* 🔗 Combined **6 CSV files (~6 million records)** into one dataset
* 🧹 Performed data preprocessing:

  * Handled missing values
  * Encoded categorical features using Label Encoding
* ⚖️ Worked with **imbalanced data (~1% fraud cases)**
* 🤖 Trained a **Random Forest Classifier**
* 📊 Evaluated model using:

  * Accuracy
  * ROC-AUC Score
  * Confusion Matrix
  * Classification Report
* 🌐 Built a **Flask web app** for real-time fraud prediction

---

## 📊 Model Performance

* Accuracy: **~99.46%**
* ROC-AUC Score: **~0.96**
* Fraud Recall: **~51%**

📌 Note: High accuracy is due to class imbalance. Fraud detection recall is a more important metric.

---

## ⚙️ Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Flask
* HTML / CSS / JavaScript

---

## ▶️ How to Run

```bash
git clone <your-repo-link>
cd fraud-detection-project
pip install -r requirements.txt
python train_model.py
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📌 Features

* Dynamic feature input (auto-generated from model)
* Real-time fraud prediction
* Probability-based output
* Handles missing inputs gracefully

---

## 📚 Reference & Authors

This project uses the **Bank Account Fraud (BAF) Dataset (NeurIPS 2022)**.

### 🧑‍🔬 Dataset Authors

* Sérgio Jesus
* José Pombal
* Duarte Alves
* André Cruz
* Pedro Saleiro
* Rita Ribeiro
* João Gama
* Pedro Bizarro

### 📖 Citation

```text
@article{jesusTurningTablesBiased2022,
  title={Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation},
  author={Jesus, Sérgio and Pombal, José and Alves, Duarte and Cruz, André and Saleiro, Pedro and Ribeiro, Rita and Gama, João and Bizarro, Pedro},
  journal={NeurIPS},
  year={2022}
}
```

---



## 🚀 Future Improvements

* Improve fraud detection recall using SMOTE / XGBoost
* Build analytics dashboard
* Deploy on cloud (AWS / Render / Railway)

---

## 👨‍💻 Author

Sumit Tiwari
Tanuj Palariya
Tanay Ranjan
Vaibhav Jaiswal
Nitin Kargeti
GitHub: https://github.com/tanujpalariya01


