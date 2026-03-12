# Customer Churn Prediction & Web App

An end-to-end machine learning project that predicts customer churn for a telecommunications company. This repository includes a detailed Jupyter Notebook for data analysis and model training, as well as a **Flask web application** that serves the trained model through an interactive user interface.

---

## Live Demo & Screenshot

The project is deployed as a web application where users can input customer details and receive **real-time churn predictions**.

<img width="1911" height="910" alt="image" src="https://github.com/user-attachments/assets/b7fe09d5-f204-4418-abae-b353776baf24" />
<img width="1908" height="909" alt="image" src="https://github.com/user-attachments/assets/fac9e978-b5bf-4b44-beb3-344bad308a75" />

---

## Key Features

* **Interactive Web Interface**
  A clean and user-friendly form built using **Flask** and styled with **Tailwind CSS**, with a visual probability progress bar and model info display.

* **Detailed Data Analysis**
  The `customer_churn_pred.ipynb` notebook offers rich exploratory data analysis (EDA) with business insights, correlation heatmaps, and feature importance analysis.

* **High-Accuracy Ensemble Model**
  A **Voting Classifier** (XGBoost + Gradient Boosting) with optimized decision threshold achieves **80.9% accuracy** and **0.847 AUC**.

* **Advanced Feature Engineering**
  10 new features engineered from raw data including `AvgChargesPerMonth`, `TenureGroup`, `ChargesRatio`, `NumServices`, `HasBundle`, `NumSecurityFeatures`, and more.

* **Dual Training Strategy**
  Two approaches compared — SMOTE resampling vs. class-weight balancing with threshold optimization — and the best is automatically selected.

* **Production-Ready Pipeline**
  `ColumnTransformer` with `StandardScaler` + `OneHotEncoder` ensures consistent preprocessing between training and inference.

---

## Tech Stack

| Layer        | Tools/Technologies                                     |
| ------------ | ------------------------------------------------------ |
| **Backend**  | Flask                                                  |
| **Frontend** | HTML, Tailwind CSS                                     |
| **ML/DS**    | Scikit-learn, XGBoost, Pandas, NumPy, imbalanced-learn |
| **Viz**      | Matplotlib, Seaborn                                    |
| **Dev Env**  | Jupyter Notebook                                       |

---

## Repository Structure

```
customer-churn-prediction/
├── app.py                     # Flask application (loads model + serves predictions)
├── train_model.py             # Full training pipeline (dual strategy, grid search)
├── best_model.pkl             # Trained VotingClassifier (XGBoost + GradientBoosting)
├── preprocessor.pkl           # ColumnTransformer (StandardScaler + OneHotEncoder)
├── feature_config.pkl         # Feature lists, threshold, model metadata
├── customer_churn_pred.ipynb  # Jupyter Notebook (EDA + Model Training)
├── dataset_telco.csv          # Telco Customer Churn dataset (7043 rows)
├── requirements.txt           # Required Python packages
└── templates/
    └── index.html             # Web UI (HTML + Tailwind CSS)
```

---

## Setup & Installation

1. **Clone the Repository**

```bash
git clone https://github.com/onyxfuzzx/customer-chrun-prediction.git
cd customer-chrun-prediction
```

2. **Create Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## How to Run

### Run the Flask Web App

```bash
python app.py
```

Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Retrain the Model

```bash
python train_model.py
```

This will:
- Load and clean the dataset
- Engineer 10 new features
- Train models with two strategies (SMOTE and class weights)
- Optimize decision thresholds
- Save the best model and artifacts as `.pkl` files

### Run the Jupyter Notebook

```bash
jupyter notebook customer_churn_pred.ipynb
```

---

## Model Details

| Metric       | Value                                             |
| ------------ | ------------------------------------------------- |
| **Model**    | VotingClassifier (XGBoost-CW + GradientBoosting)  |
| **Accuracy** | 80.9%                                             |
| **F1 Score** | 0.608                                             |
| **AUC-ROC**  | 0.847                                             |
| **Threshold**| 0.480 (optimized for accuracy)                    |

### Preprocessing
- Missing value treatment for `TotalCharges` (converted from string, filled with 0)
- `OneHotEncoder` (drop="first") for 16 categorical features
- `StandardScaler` for 13 numerical features (including engineered features)
- All preprocessing packaged in a single `ColumnTransformer`

### Feature Engineering (10 new features)
| Feature              | Description                                           |
| -------------------- | ----------------------------------------------------- |
| AvgChargesPerMonth   | TotalCharges / (tenure + 1)                           |
| TenureGroup          | Bucketed tenure (0-12, 13-24, 25-48, 49-60, 61+)     |
| ChargesRatio         | MonthlyCharges / (TotalCharges + 1)                   |
| NumServices          | Count of subscribed services                          |
| HasInternet          | Binary: has any internet service                      |
| HasBundle            | Binary: phone + internet                              |
| ChargesPerService    | MonthlyCharges / (NumServices + 1)                    |
| IsNewCustomer        | Binary: tenure <= 6 months                            |
| NumSecurityFeatures  | Count of security services                            |
| NumStreamingFeatures | Count of streaming services                           |

### Training Strategy
Two strategies are compared and the best is selected automatically:

- **Strategy A (SMOTE)**: Apply SMOTE to balance classes, train RF/GB/XGB, combine into VotingClassifier
- **Strategy B (Class Weights)**: Use `scale_pos_weight` / `class_weight="balanced"` in models, optimize decision threshold on validation set

---

## Contributing

Got ideas to improve? Issues or bugs?
You're welcome to **open issues**, create pull requests, or fork this project.

---

## License

This project is licensed under the [MIT License](LICENSE).
