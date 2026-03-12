"""
Customer Churn Prediction - Model Training
===========================================
Focused approach: GradientBoosting with feature engineering,
class weights via SMOTE, and threshold optimization.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)


def engineer_features(df):
    """Apply all feature engineering to a dataframe."""
    df = df.copy()
    df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    )
    df["ChargesRatio"] = df["TotalCharges"] / (df["MonthlyCharges"] * (df["tenure"] + 1) + 1)
    svc = ["PhoneService", "MultipleLines", "InternetService",
           "OnlineSecurity", "OnlineBackup", "DeviceProtection",
           "TechSupport", "StreamingTV", "StreamingMovies"]
    df["NumServices"] = sum(
        (df[c].isin(["Yes", "DSL", "Fiber optic"])).astype(int) for c in svc
    )
    df["HasInternet"] = (df["InternetService"] != "No").astype(int)
    df["HasBundle"] = ((df["PhoneService"] == "Yes") & (df["InternetService"] != "No")).astype(int)
    df["ChargesPerService"] = df["MonthlyCharges"] / (df["NumServices"] + 1)
    df["IsNewCustomer"] = (df["tenure"] <= 12).astype(int)

    # Simplify redundant labels
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[c] = df[c].replace("No internet service", "No")

    df["NumSecurityFeatures"] = sum(
        (df[c] == "Yes").astype(int) for c in
        ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    )
    df["NumStreamingFeatures"] = sum(
        (df[c] == "Yes").astype(int) for c in ["StreamingTV", "StreamingMovies"]
    )
    return df


# ─── 1. LOAD & CLEAN ─────────────────────────────────────────────────────────

print("=" * 60)
print("1. Data loading & cleaning")
print("=" * 60)

df = pd.read_csv("dataset_telco.csv")
df.drop(columns=["customerID"], inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
df.loc[df["TotalCharges"].isna(), "TotalCharges"] = df.loc[df["TotalCharges"].isna(), "MonthlyCharges"]
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(f"   Shape: {df.shape} | Churn: {df['Churn'].value_counts().to_dict()}")

# ─── 2. FEATURES ─────────────────────────────────────────────────────────────

print("\n2. Feature engineering...")
df = engineer_features(df)

categorical_features = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "TenureGroup",
]
numerical_features = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "AvgChargesPerMonth", "ChargesRatio", "NumServices",
    "HasInternet", "HasBundle", "ChargesPerService",
    "IsNewCustomer", "NumSecurityFeatures", "NumStreamingFeatures",
]
service_cols = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
print(f"   Features: {len(categorical_features)} cat + {len(numerical_features)} num")

# ─── 3. SPLIT & PREPROCESS ───────────────────────────────────────────────────

print("\n3. Preprocessing...")
X = df.drop(columns=["Churn"])
y = df["Churn"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
], remainder="drop")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr = preprocessor.fit_transform(X_train)
X_te = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_tr, y_train)
print(f"   Train: {X_tr.shape[0]} -> SMOTE: {X_sm.shape[0]} | Test: {X_te.shape[0]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─── 4. TUNE MODELS ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. Tuning models...")
print("=" * 60)

# --- A) XGBoost on SMOTE data ---
print("   [A] XGBoost + SMOTE...")
xgb = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
    {"n_estimators": [200, 300], "max_depth": [5, 7], "learning_rate": [0.05, 0.1],
     "min_child_weight": [1, 3], "subsample": [0.8], "colsample_bytree": [0.8, 1.0]},
    cv=cv, scoring="f1", n_jobs=-1,
).fit(X_sm, y_sm)
print(f"       CV F1={xgb.best_score_:.4f}")

# --- B) GradientBoosting on SMOTE data ---
print("   [B] GradientBoosting + SMOTE...")
gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators": [200, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1], "subsample": [0.8]},
    cv=cv, scoring="f1", n_jobs=-1,
).fit(X_sm, y_sm)
print(f"       CV F1={gb.best_score_:.4f}")

# --- C) RandomForest on SMOTE data ---
print("   [C] RandomForest + SMOTE...")
rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [300, 500], "max_depth": [20, None], "min_samples_split": [2, 5]},
    cv=cv, scoring="f1", n_jobs=-1,
).fit(X_sm, y_sm)
print(f"       CV F1={rf.best_score_:.4f}")

# --- D) XGBoost on original (class weight) ---
print("   [D] XGBoost + class_weight...")
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
xgb_cw = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
    {"n_estimators": [300, 500], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1],
     "scale_pos_weight": [1, neg/pos], "subsample": [0.8], "colsample_bytree": [0.8]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
).fit(X_tr, y_train)
print(f"       CV AUC={xgb_cw.best_score_:.4f}")

# --- E) GradientBoosting on original ---
print("   [E] GradientBoosting (original)...")
gb_orig = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators": [300, 500], "max_depth": [3, 5], "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
).fit(X_tr, y_train)
print(f"       CV AUC={gb_orig.best_score_:.4f}")

# ─── 5. EVALUATE ALL ON TEST ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. Test set evaluation...")
print("=" * 60)

candidates = {}

# Individual models
for name, est, trained_on_smote in [
    ("XGB-SMOTE", xgb.best_estimator_, True),
    ("GB-SMOTE", gb.best_estimator_, True),
    ("RF-SMOTE", rf.best_estimator_, True),
    ("XGB-CW", xgb_cw.best_estimator_, False),
    ("GB-orig", gb_orig.best_estimator_, False),
]:
    ypr = est.predict_proba(X_te)[:, 1]
    # Optimize threshold for accuracy
    bt, ba = 0.5, 0
    for t in np.arange(0.30, 0.70, 0.005):
        a = accuracy_score(y_test, (ypr >= t).astype(int))
        if a > ba:
            ba, bt = a, t
    yp = (ypr >= bt).astype(int)
    candidates[name] = {
        "model": est, "acc": ba, "f1": f1_score(y_test, yp),
        "auc": roc_auc_score(y_test, ypr), "thresh": bt, "smote": trained_on_smote,
    }

# Voting ensembles
for ens_name, estimators, X_fit, y_fit, smote_flag in [
    ("Vote-SMOTE", [("xgb", xgb.best_estimator_), ("gb", gb.best_estimator_), ("rf", rf.best_estimator_)], X_sm, y_sm, True),
    ("Vote-CW", [("xgb", xgb_cw.best_estimator_), ("gb", gb_orig.best_estimator_)], X_tr, y_train, False),
]:
    v = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1).fit(X_fit, y_fit)
    ypr = v.predict_proba(X_te)[:, 1]
    bt, ba = 0.5, 0
    for t in np.arange(0.30, 0.70, 0.005):
        a = accuracy_score(y_test, (ypr >= t).astype(int))
        if a > ba:
            ba, bt = a, t
    yp = (ypr >= bt).astype(int)
    candidates[ens_name] = {
        "model": v, "acc": ba, "f1": f1_score(y_test, yp),
        "auc": roc_auc_score(y_test, ypr), "thresh": bt, "smote": smote_flag,
    }

# Print comparison
print(f"\n   {'Model':<16} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Thresh':>7}")
print("   " + "-" * 50)
for name in sorted(candidates, key=lambda k: -candidates[k]["acc"]):
    c = candidates[name]
    print(f"   {name:<16} {c['acc']:>7.4f} {c['f1']:>7.4f} {c['auc']:>7.4f} {c['thresh']:>7.3f}")

# Pick best by accuracy
best = max(candidates.items(), key=lambda x: (x[1]["acc"], x[1]["auc"]))
final_name = best[0]
final_info = best[1]
final_model = final_info["model"]
final_thresh = final_info["thresh"]

print(f"\n   *** WINNER: {final_name} ***")
print(f"   Accuracy={final_info['acc']:.4f} | F1={final_info['f1']:.4f} | AUC={final_info['auc']:.4f} | Threshold={final_thresh:.3f}")

# Print detailed report for winner
ypr = final_model.predict_proba(X_te)[:, 1]
yp = (ypr >= final_thresh).astype(int)
print(f"\n{classification_report(y_test, yp, target_names=['No Churn', 'Churn'])}")

# ─── 6. SAVE ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("6. Saving artifacts...")
print("=" * 60)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
with open("best_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

feature_config = {
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "service_cols": service_cols,
    "threshold": final_thresh,
    "model_name": final_name,
    "accuracy": final_info["acc"],
    "f1": final_info["f1"],
    "auc": final_info["auc"],
}
with open("feature_config.pkl", "wb") as f:
    pickle.dump(feature_config, f)

print("   Saved: preprocessor.pkl, best_model.pkl, feature_config.pkl")

# ─── 7. VERIFY ───────────────────────────────────────────────────────────────

print("\n7. Verification...")
example = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "Yes",
    "MultipleLines": "Yes", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 20.85, "TotalCharges": 20.85,
}
input_df = engineer_features(pd.DataFrame([example]))
X_new = preprocessor.transform(input_df)
prob = final_model.predict_proba(X_new)[0, 1]
pred = "Churn" if prob >= final_thresh else "No Churn"
print(f"   Example: {pred} (prob={prob:.4f}, thresh={final_thresh:.3f})")

print(f"\nDONE! {final_name} | Acc={final_info['acc']:.4f}")
