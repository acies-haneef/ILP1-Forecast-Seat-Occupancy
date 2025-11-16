# train_polynomial_model.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

# ----------------------------------------------
# CONFIG
# ----------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"

PREDICTOR_COLS = [
    "Headcount",
    "DayOfWeek",
    "Is_Mandatory_Holiday",
    "Is_Restricted_Holiday",
    "Event_Flag",
    "Hiring_Flag",
]

TARGET_COL = "Total_WFO"

OUTPUT_FILES = {
    "scaler":  f"{MODEL_DIR}/scaler.pkl",
    "poly":    f"{MODEL_DIR}/poly.pkl",
    "model":   f"{MODEL_DIR}/model_poly.pkl"
}

# ----------------------------------------------
# LOAD DATA
# ----------------------------------------------
def read_master(loc):
    path = f"{DATA_DIR}/master_{loc}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

print("📥 Loading Bangalore & Chennai master data...")
blr = read_master("bangalore")
che = read_master("chennai")

print("➡ Combining datasets")
df = pd.concat([blr, che], ignore_index=True)

# ----------------------------------------------
# CLEANING
# ----------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

bad_rows = df[df["Date"].isna()]
if len(bad_rows) > 0:
    print("⚠️ Warning: Some rows had invalid dates and will be dropped:")
    print(bad_rows)

df = df.dropna(subset=["Date"])

df = df.sort_values("Date").reset_index(drop=True)
df["DayOfWeek"] = df["Date"].dt.weekday

# Ensure all required columns exist
missing = [c for c in PREDICTOR_COLS+[TARGET_COL] if c not in df]
if missing:
    raise ValueError(f"Missing columns in master data: {missing}")

# ----------------------------------------------
# TRAINING DATA
# ----------------------------------------------
X = df[PREDICTOR_COLS].astype(float)
y = df[TARGET_COL].astype(float)

print("📊 Dataset size:", X.shape)

# ----------------------------------------------
# SCALER
# ----------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------
# POLYNOMIAL FEATURES
# ----------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# ----------------------------------------------
# MODEL TRAINING
# ----------------------------------------------
print("🤖 Training Polynomial Regression...")
model = LinearRegression()
model.fit(X_poly, y)

# ----------------------------------------------
# SAVE ARTIFACTS
# ----------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

pickle.dump(scaler, open(OUTPUT_FILES["scaler"], "wb"))
pickle.dump(poly, open(OUTPUT_FILES["poly"], "wb"))
pickle.dump(model, open(OUTPUT_FILES["model"], "wb"))

print("\n✅ Polynomial Model Training COMPLETE!")
print(f"Saved scaler → {OUTPUT_FILES['scaler']}")
print(f"Saved poly    → {OUTPUT_FILES['poly']}")
print(f"Saved model   → {OUTPUT_FILES['model']}")
