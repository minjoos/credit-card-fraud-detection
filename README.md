# 💳 Credit Card Transaction Fraud Detection using LightGBM

This repository contains a real-world fraud detection project completed as part of the MSBA program at USC. The goal was to build an effective machine learning model to identify fraudulent credit card transactions, minimizing financial loss while reducing false positives.

---

## 🧠 Problem Statement

Detect fraudulent transactions from a large credit card transaction dataset. The key challenge was to maximize fraud identification while minimizing false positives — achieving operational efficiency and business value.

---

## 📊 Dataset

- Real-world anonymized credit card transaction data (2010)
- ~100,000 transactions with binary fraud labels
- Highly imbalanced (fraud rate < 1%)
- OOT = Out-of-Time validation set  
- FDR = Fraud Detection Rate @ top 3% of model-scored transactions

---

## ⚙️ Methodology

### 1. Data Cleaning & Exploration
- Removed duplicates and missing values
- Detected and handled outliers
- Conducted EDA to uncover fraud-related patterns

### 2. Feature Engineering
- Time-based features (e.g., time since last transaction)
- Frequency-based metrics (e.g., transactions per hour/day)
- Aggregated behavior per account and merchant

### 3. Model Development
- Models tested: Logistic Regression, Decision Tree, Random Forest, Neural Network (MLP), LightGBM
- ✅ Final model: **LightGBM Classifier**
  - Best FDR@3% on OOT set: **57.24%**
  - Robust to imbalanced data
  - Fast training and high interpretability
- Evaluation metrics: AUC, Precision, Recall, FDR@3%

### 4. Feature Selection
- Selected top 200 variables based on LightGBM importance scores
- Removed multicollinearity and low-signal features to prevent overfitting

---

## 📈 Performance Comparison

| Model               | FDR@3% (OOT) | Notes                         |
|---------------------|--------------|-------------------------------|
| Logistic Regression | 0.469        | Simple baseline               |
| Decision Tree       | 0.491        | Overfit-prone                 |
| Random Forest       | 0.504        | Decent, slower                |
| **LightGBM**        | **0.5724**   | ✅ Final choice: best overall |
| Neural Network      | 0.490        | Less stable, lower precision |

---

## ✅ Results & Impact

- **Detection rate:** 57.24% of fraudulent transactions captured in top 3% of flagged cases (OOT)
- **Estimated financial savings:** $40M+ annually
- **Operational benefit:** Reduced manual review volume and enhanced fraud analyst efficiency
- **Business value:** Balanced fraud mitigation and customer experience

---

## 🛠️ Tech Stack

- Python: `pandas`, `numpy`, `lightgbm`, `scikit-learn`
- Jupyter Notebook
- Matplotlib / Seaborn (visualizations)
- Git / GitHub

---

## 📁 Project Structure
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training_lightgbm.ipynb
├── report/
│   ├── eda_summary.ipynb
│   └── final_report.pdf
├── requirements.txt
└── README.md

---

## 📄 Reports

- 📘 [EDA Summary Notebook](./report/eda_summary.ipynb)
- 📄 [Final Report PDF](./report/final_report.pdf)

---

## 💡 Key Lessons Learned

- Carefully engineered features significantly enhance model performance
- Out-of-time validation provides realistic business impact estimates
- Business-aligned fraud detection requires balance between recall and operational cost

---

## 📬 Contact

Interested in this project or want to explore additional notebooks?  
Feel free to reach out or visit my portfolio (Notion/GitHub link coming soon).
