# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer will churn. Includes a complete Jupyter Notebook for data analysis and model training, plus a **Flask web application** with a modern dark-themed UI for real-time predictions.

---

## Screenshots

<img width="1911" height="910" alt="image" src="https://github.com/user-attachments/assets/b7fe09d5-f204-4418-abae-b353776baf24" />
<img width="1908" height="909" alt="image" src="https://github.com/user-attachments/assets/fac9e978-b5bf-4b44-beb3-344bad308a75" />

---

## Key Features

| Feature | Description |
|---|---|
| **Ensemble Model** | VotingClassifier (XGBoost + Gradient Boosting) — **80.9% accuracy**, **0.847 AUC** |
| **Smart UI** | Dark-themed, responsive form with conditional field logic and auto-calculated Total Charges |
| **Feature Engineering** | 10 new features derived from raw data (see below) |
| **Dual Training Strategy** | Compares SMOTE resampling vs. class-weight balancing, picks the best automatically |
| **Threshold Optimization** | Decision threshold tuned on validation set for maximum accuracy |
| **Production Pipeline** | Single `ColumnTransformer` (StandardScaler + OneHotEncoder) for consistent inference |

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Backend** | Flask |
| **Frontend** | HTML, Tailwind CSS, JavaScript |
| **ML** | Scikit-learn, XGBoost, imbalanced-learn |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

---

## Project Structure

```
customer-churn-prediction/
├── app.py                     # Flask web application
├── train_model.py             # Full training pipeline (dual strategy + grid search)
├── best_model.pkl             # Trained VotingClassifier
├── preprocessor.pkl           # ColumnTransformer (scaler + encoder)
├── feature_config.pkl         # Feature lists, threshold (0.48), model metadata
├── customer_churn_pred.ipynb  # Jupyter Notebook (EDA + training)
├── dataset_telco.csv          # Telco Customer Churn dataset (7,043 rows × 21 cols)
├── requirements.txt           # Python dependencies
├── README.md
└── templates/
    └── index.html             # Web UI
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/onyxfuzzx/customer-chrun-prediction.git
cd customer-chrun-prediction
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Web App

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### 3. Retrain the Model (optional)

```bash
python train_model.py
```

This will re-run the full pipeline and overwrite the `.pkl` artifacts.

### 4. Explore the Notebook

```bash
jupyter notebook customer_churn_pred.ipynb
```

---

## UI Logic

The web form has smart conditional logic built in:

| Condition | Behavior |
|---|---|
| **Phone Service = No** | "Multiple Lines" auto-sets to *No phone service* and is disabled |
| **Internet Service = No** | All 6 internet-dependent fields auto-set to *No internet service* and are disabled |
| **Tenure** | Slider input, range 0–72 months (matches dataset) |
| **Monthly Charges** | Validated range $18–$120 (matches dataset) |
| **Total Charges** | **[Auto-fill]** button calculates `Monthly Charges × Tenure` as suggestion |

---

## Model Details

### Performance

| Metric | Score |
|---|---|
| **Accuracy** | 80.9% |
| **F1 Score** | 0.608 |
| **AUC-ROC** | 0.847 |
| **Threshold** | 0.480 |

### Architecture

- **VotingClassifier** (soft voting) combining:
  - XGBoost with class weights
  - Gradient Boosting on original data
- Threshold optimized on validation set for best accuracy

### Preprocessing

- `TotalCharges` converted from string to float; missing values filled with 0
- `OneHotEncoder(drop="first", handle_unknown="ignore")` for 16 categorical features
- `StandardScaler` for 13 numerical features (including engineered features)
- Everything wrapped in a single `ColumnTransformer`

### Feature Engineering (10 new features)

| Feature | Formula / Logic |
|---|---|
| `AvgChargesPerMonth` | `TotalCharges / (tenure + 1)` |
| `TenureGroup` | Bucketed: 0–12, 13–24, 25–48, 49–60, 61–72 |
| `ChargesRatio` | `TotalCharges / (MonthlyCharges × (tenure + 1) + 1)` |
| `NumServices` | Count of all subscribed services |
| `HasInternet` | Binary — has any internet service |
| `HasBundle` | Binary — has both phone and internet |
| `ChargesPerService` | `MonthlyCharges / (NumServices + 1)` |
| `IsNewCustomer` | Binary — tenure ≤ 6 months |
| `NumSecurityFeatures` | Count of security/backup/protection/support services |
| `NumStreamingFeatures` | Count of streaming services |

### Training Strategy

Two parallel strategies are evaluated; the best model is saved automatically:

1. **SMOTE Strategy** — Synthetic oversampling of minority class, then train RF + GB + XGB ensemble
2. **Class Weight Strategy** — Use `scale_pos_weight` / `class_weight="balanced"`, then optimize decision threshold

---

## Dataset

**Telco Customer Churn** — 7,043 customers, 21 columns.

- **Target**: `Churn` (Yes/No) — ~26.5% churn rate
- **Features**: Demographics (gender, senior citizen, partner, dependents), Account info (tenure, contract, billing, charges), Services (phone, internet, security, streaming)

---

## Contributing

Issues, pull requests, and forks are welcome.

---

## License

[MIT License](LICENSE)
