# train_prophet_model.py
import pandas as pd
import os
import pickle
from prophet import Prophet

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_FILE = f"{MODEL_DIR}/model_prophet.pkl"

# ----------------------------------------------
# LOAD & MERGE DATA
# ----------------------------------------------
def read_master(loc):
    path = f"{DATA_DIR}/master_{loc}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path, parse_dates=["Date"])

print("📥 Loading master data...")
blr = read_master("bangalore")
che = read_master("chennai")

df = pd.concat([blr, che], ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df = df.sort_values("Date")

# Prophet expects columns: ds (Date), y (Target)
df_prophet = df.rename(columns={"Date": "ds", "Total_WFO": "y"})
df_prophet = df_prophet[["ds", "y"]].dropna()

print("📊 Training rows:", len(df_prophet))

# ----------------------------------------------
# TRAIN PROPHET MODEL
# ----------------------------------------------
print("🤖 Training Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model.fit(df_prophet)

# ----------------------------------------------
# SAVE MODEL
# ----------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
pickle.dump(model, open(OUTPUT_FILE, "wb"))

print("\n✅ Prophet Model Training COMPLETE!")
print(f"Model saved → {OUTPUT_FILE}")
