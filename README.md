# Card Transaction Fraud Detection (LightGBM)

Scoring ~100K real card transactions so a limited review team can prioritize the riskiest ones.
Academic project · USC Marshall, DSO 562 Fraud Analytics · Fall 2024 · individual submission

**Result:** the final LightGBM model captured **57.2% of all fraud within the top 3% highest-risk transactions** on out-of-time data (FDR@3%).

---

## Data

- 97,852 card transactions from 2010 (1,645 cards, 13,091 merchants), course-provided and **not redistributed here**
- 2,047 fraud labels (≈ 2.1%)
- Kept purchase transactions only and removed one extreme non-fraud outlier → 97,496 rows
- Out-of-time set: transactions from November 1 onward

## Approach

**1. Cleaning.** Filled missing merchant number, state, and ZIP through merchant-description and ZIP mappings, marking the remainder as unknown/foreign instead of dropping rows.

**2. Feature engineering: 2,600+ unique candidates.** Starting from ~3,300 engineered fields, I removed duplicates to get 2,634 unique variables:
- Frequency and amount: count / average / max / median / total over 0, 1, 3, 7, 14, 30, and 60-day windows, per card, merchant, and location combination
- Velocity: short-window vs long-window activity ratios
- Recency: days since a card–merchant or card–location pair was last seen; new-entity flags
- Behavioral consistency: unique counts and variability
- Context: day-of-week risk and merchant state/ZIP risk (target-encoded using pre-OOT data only), foreign-ZIP flag, Benford's-law deviation

**3. Feature selection.** KS filter (top 500) → LightGBM forward wrapper → 20 final variables.
The strongest themes were card velocity, short-vs-long window ratios, unique card–merchant/location counts, and recency.

**4. Models.** Compared Logistic Regression, Decision Tree, Random Forest, Neural Network, and LightGBM.
In the first comparison LightGBM had the best OOT FDR@3% (0.539 vs. 0.504 RF and 0.469 LR).
The tuned final model scored train / test / OOT = **0.785 / 0.746 / 0.572**.

**5. Cutoff.** A fraud-caught vs false-positive cost curve under course assumptions recommended a 5% review cutoff.

## Limitations

- The ~21-point gap between train and OOT suggests some overfitting. The final run was also selected with OOT performance in view.
- Dollar savings from the course cost assumptions are simulations, not realized results.
- 2010 academic data; no production deployment.

## Repository

```
notebooks/
  01_data_exploration.ipynb        # data quality review
  02_model_baseline.ipynb          # baseline models
  03_feature_engineering.ipynb     # variable creation
  04_model_training_lightgbm.ipynb # selection, tuning, OOT evaluation, cutoff
report/final_report.pdf
report/eda_summary.ipynb
```

Notebooks build on course-provided starter code; cleaning, feature design, selection, tuning, and the report are my own work. The dataset is not included.

**Stack:** Python · pandas · NumPy · scikit-learn · LightGBM · Matplotlib
