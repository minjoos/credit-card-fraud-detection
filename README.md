# Credit Card Transaction Fraud Detection using LightGBM

This repository contains a real-world fraud detection project completed as part of the MSBA program at USC. The goal was to build an effective machine learning model to identify fraudulent credit card transactions, minimizing financial loss while reducing false positives.

## 🧠 Problem Statement
Detect fraudulent transactions from a large dataset of credit card activity from 2010. The key challenge was to catch as many frauds as possible while keeping false positives low.

## 📊 Dataset
- ~100,000 real-world credit card transactions from 2010
- Includes timestamps, transaction amounts, anonymized account IDs, and fraud labels

## ⚙️ Methodology

### 1. Data Cleaning & Exploration
- Removed duplicates and missing values
- Identified and handled outliers
- Performed EDA to understand fraud vs. non-fraud distributions

### 2. Feature Engineering
- Time-based features (e.g., transaction time difference)
- Frequency-based features (e.g., transactions per day/hour)
- Aggregated behavior per account

### 3. Model Building
- Model: LightGBM Classifier
- Objective: Binary classification (fraud vs. non-fraud)
- Metrics: AUC, precision, recall, detection rate within top flagged %

### 4. Feature Selection
- Used top 200 features based on LightGBM importance ranking
- Prevented overfitting by removing highly correlated features

## ✅ Results
- **Detection rate:** 57.24% of fraudulent transactions caught in the top 3% of flagged cases (OOT validation)
- **Estimated savings:** Over $40M/year in potential fraud loss
- **High interpretability:** Feature importance and behavior-based variables improved explainability

## 🛠️ Tech Stack
- Python (Pandas, NumPy, LightGBM, Scikit-learn)
- Jupyter Notebook
- Matplotlib / Seaborn (EDA)
- Git / GitHub

## 📁 Files
- `01_data_exploration.ipynb` – EDA & cleaning
- `02_model_baseline.ipynb` – Baseline model setup
- `03_feature_engineering.ipynb` – Variable creation
- `04_model_training_lightgbm.ipynb` – Final model training & evaluation

## 🧾 Report
- 📄 [EDA Summary Notebook](./report/eda_summary.ipynb)

## 💡 Key Lessons Learned
- Importance of clean, engineered features for fraud patterns
- Real-world modeling requires balancing business goals and technical accuracy
- How to optimize detection thresholds based on cost impact

---

## 📬 Contact
If you're interested in learning more about this project or seeing additional notebooks (redacted), feel free to reach out!
