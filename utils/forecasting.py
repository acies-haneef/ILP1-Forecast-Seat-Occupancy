# utils/forecasting.py
import os
import pandas as pd
import numpy as np

from models.poly_forecast import predict_poly
from models.prophet_forecast import predict_prophet

PREDICTOR_COLS = [
    "Headcount",
    "DayOfWeek",
    "Is_Mandatory_Holiday",
    "Is_Restricted_Holiday",
    "Event_Flag",
    "Hiring_Flag",
]


def load_historical_for_location(loc, data_dir="data"):
    path = os.path.join(data_dir, f"master_{loc.lower()}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing historical file: {path}")
    df = pd.read_csv(path)
    # ensure Date parsed and drop bad rows
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Normalize flags to 0/1 ints if they exist
    for col in ["Is_Mandatory_Holiday", "Is_Restricted_Holiday", "Event_Flag", "Hiring_Flag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0
    return df


def apply_future_flags(future_df, flags, start_headcount, seating_capacity):
    """
    Apply future flags (filtered for location) to a future_df of Dates.
    Produces columns:
      Headcount, Hiring_Flag, Is_Mandatory_Holiday, Is_Restricted_Holiday,
      Event_Flag, Seating_Capacity, DayOfWeek
    """
    df = future_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # initialize columns
    df["Headcount"] = np.nan
    df["Hiring_Flag"] = 0
    df["Is_Mandatory_Holiday"] = 0
    df["Is_Restricted_Holiday"] = 0
    df["Event_Flag"] = 0
    df["Seating_Capacity"] = seating_capacity
    df["DayOfWeek"] = df["Date"].dt.weekday

    # normalize incoming flags' dates
    normalized = []
    for f in flags:
        ff = dict(f)
        ff["Date"] = pd.to_datetime(ff["Date"], dayfirst=True, errors="coerce").date() if pd.notna(ff.get("Date")) else None
        # normalize Type, Count, Event_Name presence
        ff["Type"] = str(ff.get("Type"))
        ff["Count"] = int(ff.get("Count", 0)) if ff.get("Count") not in (None, "") else 0
        ff.setdefault("Event_Name", "")
        normalized.append(ff)

    hc = int(start_headcount)

    for i, row in df.iterrows():
        d = row["Date"].date()
        todays = [f for f in normalized if f["Date"] == d]

        change = 0
        for f in todays:
            t = f["Type"]
            if t == "Hiring":
                hc += int(f.get("Count", 0))
                change += int(f.get("Count", 0))
            elif t == "Exit":
                hc -= int(f.get("Count", 0))
                change -= int(f.get("Count", 0))
            elif t == "Mandatory":
                df.at[i, "Is_Mandatory_Holiday"] = 1
            elif t == "Restricted":
                df.at[i, "Is_Restricted_Holiday"] = 1
            elif t == "Event":
                df.at[i, "Event_Flag"] = 1

        df.at[i, "Hiring_Flag"] = change
        df.at[i, "Headcount"] = hc

    # forward fill headcount if needed (safety)
    df["Headcount"] = df["Headcount"].ffill().astype(int)

    # ensure flags numeric ints
    for col in ["Is_Mandatory_Holiday", "Is_Restricted_Holiday", "Event_Flag", "Hiring_Flag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def run_forecast_models(hist_df, future_df):
    """
    hist_df: historical dataframe (with Date, Total_WFO, Headcount, and flags)
    future_df: future dataframe returned by apply_future_flags (Date, Headcount, DayOfWeek, flags)
    Returns: future_df (subset) with short_term_pred, long_term_pred, final_pred, plus flags
    """
    hist = hist_df.copy()
    fut = future_df.copy()

    # ensure dayofweek present
    fut["DayOfWeek"] = fut["Date"].dt.weekday

    # Ensure predictor columns exist in fut
    for c in PREDICTOR_COLS:
        if c not in fut.columns:
            fut[c] = 0

    # --- Short-term polynomial predictions ---
    try:
        fut["short_term_pred"] = predict_poly(fut, PREDICTOR_COLS)
    except Exception as e:
        fut["short_term_pred"] = np.nan
        fut["poly_error"] = str(e)

    # --- Long-term Prophet predictions ---
    try:
        fut["long_term_pred"] = predict_prophet(fut["Date"])
    except Exception as e:
        fut["long_term_pred"] = np.nan
        fut["prophet_error"] = str(e)

    # --- Blend ---
    def blend(row):
        a = row.get("short_term_pred", np.nan)
        b = row.get("long_term_pred", np.nan)
        if pd.isna(a) and pd.isna(b):
            return np.nan
        if pd.isna(a):
            return b
        if pd.isna(b):
            return a
        return 0.6 * a + 0.4 * b

    fut["final_pred"] = fut.apply(blend, axis=1)

    # --------- HARD OVERRIDE: Mandatory Holidays => ZERO WFO ----------
    # Normalize the flag column then apply
    if "Is_Mandatory_Holiday" in fut.columns:
        fut["Is_Mandatory_Holiday_flag"] = pd.to_numeric(fut["Is_Mandatory_Holiday"], errors="coerce").fillna(0).astype(int)
        mask = fut["Is_Mandatory_Holiday_flag"] == 1
        if mask.any():
            fut.loc[mask, ["short_term_pred", "long_term_pred", "final_pred"]] = 0
        fut.drop(columns=["Is_Mandatory_Holiday_flag"], inplace=True)

    # Ensure final_pred numeric
    fut["final_pred"] = pd.to_numeric(fut["final_pred"], errors="coerce")

    # Return selected columns (future rows with preds and flags)
    out_cols = ["Date", "Headcount", "Hiring_Flag", "Is_Mandatory_Holiday",
                "Is_Restricted_Holiday", "Event_Flag", "Seating_Capacity",
                "short_term_pred", "long_term_pred", "final_pred"]
    # keep other columns if present
    for c in out_cols:
        if c not in fut.columns:
            fut[c] = np.nan
    return fut[out_cols]

def find_best_blend_weight(val_df, val_preds):
    """
    Tests poly weights from 0.0 to 1.0 and finds weight that minimizes MAE.
    val_df: actual validation dataframe (with Total_WFO)
    val_preds: predictions dataframe (with short_term_pred, long_term_pred)
    """
    results = []
    weights = np.arange(0, 1.01, 0.05)

    y_true = val_df["Total_WFO"].values
    poly = val_preds["short_term_pred"].values
    prophet = val_preds["long_term_pred"].values

    for w in weights:
        y_hat = w * poly + (1 - w) * prophet
        score = np.mean(np.abs(y_true - y_hat))
        results.append((w, score))

    # get best
    best_w, best_score = min(results, key=lambda x: x[1])

    result_df = pd.DataFrame(results, columns=["weight", "mae"])
    return best_w, best_score, result_df
