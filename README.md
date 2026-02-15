
#  Customer Churn Prediction & Web App

An end-to-end machine learning project that predicts customer churn for a telecommunications company. This repository includes a detailed Jupyter Notebook for data analysis and model training, as well as a **Flask web application** that serves the trained model through an interactive user interface.

---

##  Live Demo & Screenshot

The project is deployed as a web application where users can input customer details and receive **real-time churn predictions**.

<img width="1911" height="910" alt="image" src="https://github.com/user-attachments/assets/b7fe09d5-f204-4418-abae-b353776baf24" />
<img width="1908" height="909" alt="image" src="https://github.com/user-attachments/assets/fac9e978-b5bf-4b44-beb3-344bad308a75" />



---

##  Key Features

* **Interactive Web Interface**
  A clean and user-friendly form built using **Flask** and styled with **Tailwind CSS**.

* **Detailed Data Analysis**
  The `customer_churn_pred.ipynb` notebook offers rich exploratory data analysis (EDA) to uncover business insights.

* **Robust Model Training**
  Multiple models were evaluated, and a **Random Forest Classifier** was selected as the final model.

* **⚖️ Class Imbalance Handling**
  Implements **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset effectively.

* **🔄 Complete Preprocessing Pipeline**
  Label encoding and scaling with saved encoders (`.pkl`) for consistent inference in production.

---

## 🛠️ Tech Stack

| Layer        | Tools/Technologies                                     |
| ------------ | ------------------------------------------------------ |
| **Backend**  | Flask                                                  |
| **Frontend** | HTML, Tailwind CSS                                     |
| **ML/DS**    | Scikit-learn, Pandas, NumPy, imbalanced-learn, XGBoost |
| **Viz**      | Matplotlib, Seaborn                                    |
| **Dev Env**  | Jupyter Notebook                                       |

---

## 📁 Repository Structure

```
customer-churn-prediction/
├── app.py                     # Flask application
├── best_model.pkl             # Trained RandomForest model
├── encoder.pkl                # Label encoders for categorical features
├── scaler.pkl                 # Standard scaler for numerical features
├── customer_churn_pred.ipynb  # Jupyter Notebook (EDA + Model Training)
├── dataset_telco.csv          # Telco Customer Churn dataset
├── requirements.txt           # Required Python packages
└── templates/
    └── index.html             # Web UI (HTML + Tailwind)
```

---

## ⚙️ Setup & Installation

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

Make sure you have `requirements.txt` with the following:

```txt
flask
pandas
scikit-learn==1.3.2
numpy
matplotlib
seaborn
jupyter
imbalanced-learn
xgboost
```

Then install:

```bash
pip install -r requirements.txt
```

---

## 🚦 How to Run the Project

### 1️⃣ Run Jupyter Notebook (for EDA & Training)

```bash
jupyter notebook customer_churn_pred.ipynb
```

### 2️⃣ Run the Flask Web App

```bash
python app.py
```

Then, visit 👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Model Details

* **Model Used**: Random Forest Classifier
* **Preprocessing**:

  * Missing value treatment for `TotalCharges`
  * Label Encoding for categorical columns
  * Standard Scaling for numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`)
* **Handling Imbalance**: SMOTE applied on training data
* **Hyperparameter Tuning**: GridSearchCV
* **Serialization**: Final model + encoders saved as `.pkl` files

---

## 🤝 Contributing

Got ideas to improve? Issues or bugs?
You're welcome to **open issues**, create pull requests, or fork this project.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
