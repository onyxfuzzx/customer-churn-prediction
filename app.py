"""
Customer Churn Prediction - Flask Web Application
===================================================
Serves the trained ML model via a web interface.
Uses the preprocessor pipeline and feature engineering
from the training script.
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# ─── Load Model Artifacts ─────────────────────────────────────────────────────

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("feature_config.pkl", "rb") as f:
    config = pickle.load(f)

THRESHOLD = config.get("threshold", 0.5)
MODEL_NAME = config.get("model_name", "Unknown")
MODEL_ACC = config.get("accuracy", 0)
MODEL_AUC = config.get("auc", 0)

print(f"Loaded model: {MODEL_NAME}")
print(f"Threshold: {THRESHOLD:.3f} | Accuracy: {MODEL_ACC:.4f} | AUC: {MODEL_AUC:.4f}")

app = Flask(__name__)


def engineer_features(input_df):
    """Apply the same feature engineering as training."""
    df = input_df.copy()

    # Simplify redundant labels
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[col] = df[col].replace("No internet service", "No")

    # Engineered features
    df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    )

    df["ChargesRatio"] = df["TotalCharges"] / (df["MonthlyCharges"] * (df["tenure"] + 1) + 1)

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumServices"] = sum(
        (df[c].isin(["Yes", "DSL", "Fiber optic"])).astype(int) for c in service_cols
    )

    df["HasInternet"] = (df["InternetService"] != "No").astype(int)
    df["HasBundle"] = ((df["PhoneService"] == "Yes") & (df["InternetService"] != "No")).astype(int)
    df["ChargesPerService"] = df["MonthlyCharges"] / (df["NumServices"] + 1)
    df["IsNewCustomer"] = (df["tenure"] <= 12).astype(int)

    df["NumSecurityFeatures"] = sum(
        (df[c] == "Yes").astype(int) for c in
        ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    )
    df["NumStreamingFeatures"] = sum(
        (df[c] == "Yes").astype(int) for c in ["StreamingTV", "StreamingMovies"]
    )

    return df


def make_prediction(input_data):
    """Make a churn prediction from raw input data."""
    input_df = pd.DataFrame([input_data])
    input_df = engineer_features(input_df)

    X = preprocessor.transform(input_df)
    probability = model.predict_proba(X)[0, 1]
    prediction = "Churn" if probability >= THRESHOLD else "No Churn"

    return prediction, probability


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error_message = None

    if request.method == "POST":
        try:
            tenure = int(request.form["tenure"])
            monthly_charges = float(request.form["MonthlyCharges"])
            total_charges = float(request.form["TotalCharges"])

            # Validation
            if tenure < 0 or tenure > 120:
                error_message = "Tenure must be between 0 and 120 months."
            elif monthly_charges <= 0 or monthly_charges > 500:
                error_message = "Monthly charges must be between $0.01 and $500."
            elif total_charges < 0:
                error_message = "Total charges cannot be negative."
            else:
                input_data = {
                    "gender": request.form["gender"],
                    "SeniorCitizen": int(request.form["SeniorCitizen"]),
                    "Partner": request.form["Partner"],
                    "Dependents": request.form["Dependents"],
                    "tenure": tenure,
                    "PhoneService": request.form["PhoneService"],
                    "MultipleLines": request.form["MultipleLines"],
                    "InternetService": request.form["InternetService"],
                    "OnlineSecurity": request.form["OnlineSecurity"],
                    "OnlineBackup": request.form["OnlineBackup"],
                    "DeviceProtection": request.form["DeviceProtection"],
                    "TechSupport": request.form["TechSupport"],
                    "StreamingTV": request.form["StreamingTV"],
                    "StreamingMovies": request.form["StreamingMovies"],
                    "Contract": request.form["Contract"],
                    "PaperlessBilling": request.form["PaperlessBilling"],
                    "PaymentMethod": request.form["PaymentMethod"],
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }

                prediction, probability = make_prediction(input_data)

        except ValueError:
            error_message = "Please enter valid numeric values for tenure and charges."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error_message=error_message,
        model_name=MODEL_NAME,
        model_accuracy=MODEL_ACC,
        threshold=THRESHOLD,
    )


if __name__ == "__main__":
    app.run(debug=True)
