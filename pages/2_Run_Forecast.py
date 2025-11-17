# pages/2_Run_Forecast.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go

from utils.forecasting import (
    load_future_flags,
    load_location_history,
    apply_future_flags,
    run_forecast_models
)


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
PREDICTOR_COLS = [
    "Headcount",
    "DayOfWeek",
    "Is_Mandatory_Holiday",
    "Is_Restricted_Holiday",
    "Event_Flag",
    "Hiring_Flag",
]
VAL_DAYS = 22
st.set_page_config(page_title="Run Forecast", layout="wide")
st.title("📈 WFO Forecast — Seating-first View (Fixed Buffer)")

# ----------------------------------------------------
# LOAD PERSISTENT FUTURE FLAGS ON PAGE LOAD
# ----------------------------------------------------
if "future_flags" not in st.session_state:
    st.session_state.future_flags = load_future_flags()

# Auto-load seats from historical data for both locations
hist_blr = load_location_history("bangalore")
hist_che = load_location_history("chennai")

default_blr_seats = int(hist_blr["Seating_Capacity"].iloc[-1]) if len(hist_blr) > 0 else 0
default_che_seats = int(hist_che["Seating_Capacity"].iloc[-1]) if len(hist_che) > 0 else 0

if "seats_blr" not in st.session_state:
    st.session_state.seats_blr = default_blr_seats

if "seats_che" not in st.session_state:
    st.session_state.seats_che = default_che_seats

# ----------------------------------------------------
# METRIC FUNCTIONS
# ----------------------------------------------------
def mae(y, yhat): return np.mean(np.abs(y - yhat))
def rmse(y, yhat): return np.sqrt(np.mean((y - yhat) ** 2))
def smape(y, yhat): return np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-6)) * 100
def mase(y, yhat, naive): return np.mean(np.abs(y - yhat)) / (np.mean(np.abs(y - naive)) + 1e-6)

# ----------------------------------------------------
# AUTO-BLEND FUNCTION
# ----------------------------------------------------
def find_best_blend_weight(val_df, val_preds):
    weights = np.arange(0, 1.01, 0.05)
    results = []
    y_true = val_df["Total_WFO"].values
    poly = val_preds["short_term_pred"].values
    prophet = val_preds["long_term_pred"].values

    for w in weights:
        blended = w * poly + (1 - w) * prophet
        results.append((w, mae(y_true, blended)))

    best_w, best_score = min(results, key=lambda x: x[1])
    table = pd.DataFrame(results, columns=["weight", "mae"])
    return best_w, best_score, table

# ----------------------------------------------------
# SIDEBAR INPUTS
# ----------------------------------------------------
st.sidebar.header("Settings")

horizon = st.sidebar.number_input(
    "Forecast Horizon (Business Days)",
    min_value=5, max_value=250, value=30
)

locs = st.sidebar.multiselect(
    "Locations",
    ["bangalore", "chennai"],
    default=["bangalore", "chennai"]
)

# ALWAYS enabled
show_contrib_lines = True
show_contrib_table = True

# ----------------------------------------------------
# SEATING BUFFER INPUT
# ----------------------------------------------------
st.markdown("### Seating Buffer")
buffer_seats = st.number_input(
    "Buffer seats to add on top of predicted WFO",
    min_value=0,
    value=5,
    step=1
)

# ----------------------------------------------------
# RUN FORECAST BUTTON
# ----------------------------------------------------
if st.button("▶️ Run Forecast"):

    # Reload persistent flags at button click (if updated in Page 1)
    st.session_state.future_flags = load_future_flags()

    if len(st.session_state.future_flags) == 0:
        st.info("⚠️ No future flags found. Forecast will run using default assumptions.")

    combined_outputs = {}

    # ------------------------------------------------
    # LOOP OVER SELECTED LOCATIONS
    # ------------------------------------------------
    for loc in locs:
        st.markdown(f"## 📍 {loc.title()}")

        # ------------------------------------------------
        # 1) Load historical data
        # ------------------------------------------------
        hist = load_location_history(loc)
        hist = hist.sort_values("Date").reset_index(drop=True)

        if len(hist) <= VAL_DAYS:
            st.error(f"Not enough data for {loc}. Need > {VAL_DAYS} rows.")
            continue

        train_df = hist.iloc[:-VAL_DAYS].reset_index(drop=True)
        val_df = hist.iloc[-VAL_DAYS:].reset_index(drop=True)

        # ------------------------------------------------
        # 2) Generate future business-day dates
        # ------------------------------------------------
        last_date = hist["Date"].max()
        last_hc = int(hist.loc[hist["Date"] == last_date, "Headcount"].iloc[0])
        seating_capacity = (
            st.session_state.seats_blr if loc == "bangalore" else st.session_state.seats_che
        )

        raw_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon * 2)
        business_dates = raw_dates[raw_dates.weekday < 5][:horizon]
        future_base = pd.DataFrame({"Date": business_dates})

        relevant_flags = [
            f for f in st.session_state.future_flags
            if str(f.get("Location", "")).strip().lower() == loc.strip().lower()
        ]

        future_ready = apply_future_flags(future_base, relevant_flags, last_hc, seating_capacity)

        # ------------------------------------------------
        # 3) Predict future using models
        # ------------------------------------------------
        preds = run_forecast_models(train_df, future_ready)
        preds["short_term_pred"] = preds["short_term_pred"].astype(float)
        preds["long_term_pred"] = preds["long_term_pred"].astype(float)

        # ------------------------------------------------
        # 4) Build validation set
        # ------------------------------------------------
        val_ready = val_df[["Date"]].copy()
        val_ready["Headcount"] = val_df["Headcount"].values
        val_ready["DayOfWeek"] = val_ready["Date"].dt.weekday
        val_ready["Seating_Capacity"] = seating_capacity
        val_ready["Hiring_Flag"] = 0

        for col in ["Is_Mandatory_Holiday", "Is_Restricted_Holiday", "Event_Flag"]:
            val_ready[col] = val_df[col].values

        val_preds = run_forecast_models(train_df, val_ready)
        val_preds["short_term_pred"] = val_preds["short_term_pred"].astype(float)
        val_preds["long_term_pred"] = val_preds["long_term_pred"].astype(float)

        # ------------------------------------------------
        # 5) Auto-blend
        # ------------------------------------------------
        best_w, best_score, blend_table = find_best_blend_weight(val_df, val_preds)

        val_df["pred"] = (
            best_w * val_preds["short_term_pred"]
            + (1 - best_w) * val_preds["long_term_pred"]
        ).round().astype(int)

        mand_mask_val = val_ready["Is_Mandatory_Holiday"] == 1
        val_df.loc[mand_mask_val, "pred"] = 0

        val_df["naive"] = val_df["Total_WFO"].shift(1).fillna(method="bfill").astype(int)

        # ------------------------------------------------
        # 6) Apply blend to future predictions
        # ------------------------------------------------
        preds["final_pred"] = (
            best_w * preds["short_term_pred"]
            + (1 - best_w) * preds["long_term_pred"]
        )

        if "Is_Mandatory_Holiday" in preds.columns:
            mand_mask = preds["Is_Mandatory_Holiday"] == 1
            preds.loc[mand_mask, ["final_pred", "short_term_pred", "long_term_pred"]] = 0

        preds["predicted_wfo"] = preds["final_pred"].round().astype(int)
        preds["required_seats"] = (preds["predicted_wfo"] + buffer_seats).astype(int)
        preds["seating_capacity"] = int(seating_capacity)

        # ------------------------------------------------
        # 7) Seating Summary
        # ------------------------------------------------
        exceed_days = preds[preds["required_seats"] > seating_capacity]
        exceed_count = len(exceed_days)

        st.markdown("### Seating Requirement Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Predicted WFO", int(preds["predicted_wfo"].max()))
        c2.metric("Max Required Seats", int(preds["required_seats"].max()))
        c3.metric("Seating Capacity", seating_capacity)
        c4.metric("Days Exceeding Capacity", exceed_count)

        # ------------------------------------------------
        # 8) Seating chart
        # ------------------------------------------------
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=preds["Date"],
            y=preds["predicted_wfo"],
            name="Predicted WFO",
            marker_color="indianred"
        ))

        fig.add_trace(go.Scatter(
            x=preds["Date"],
            y=preds["required_seats"],
            name="Required Seats (buffer)",
            mode="lines+markers",
            line=dict(color="orange", width=3)
        ))

        fig.add_trace(go.Scatter(
            x=preds["Date"],
            y=preds["seating_capacity"],
            name="Seating Capacity",
            mode="lines",
            line=dict(color="royalblue", width=3, dash="dash")
        ))

        fig.update_layout(
            title="Upcoming Seating vs Requirements",
            hovermode="x unified",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------
        # 9) Seating warnings
        # ------------------------------------------------
        with st.expander("⚠️ Seating Warnings"):
            if exceed_count == 0:
                st.success("No seating issues!")
            else:
                st.warning(f"{exceed_count} day(s) exceed capacity.")
                st.dataframe(exceed_days.reset_index(drop=True))

        # ------------------------------------------------
        # 10) Validation Metrics
        # ------------------------------------------------
        with st.expander("🔬 Blend Diagnostics & Validation Metrics"):
            st.markdown("### 📆 Date Ranges")
            st.write(f"Training: **{train_df['Date'].min().date()} → {train_df['Date'].max().date()}**")
            st.write(f"Validation: **{val_df['Date'].min().date()} → {val_df['Date'].max().date()}**")

            st.markdown("---")
            st.markdown("### 📊 Metrics")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", round(mae(val_df["Total_WFO"], val_df["pred"]), 2))
            m2.metric("RMSE", round(rmse(val_df["Total_WFO"], val_df["pred"]), 2))
            m3.metric("sMAPE (%)", round(smape(val_df["Total_WFO"], val_df["pred"]), 2))
            m4.metric("MASE", round(mase(val_df["Total_WFO"], val_df["pred"], val_df["naive"]), 3))

            st.markdown("---")
            st.markdown("### Validation Predictions")
            st.dataframe(val_df[["Date", "Total_WFO", "pred", "naive"]])

        with st.expander("🧠 Feature Importance (Short-Term Poly Model)"):

            try:
                # Refit poly model on training data to capture coefficients
                from models.poly_forecast import train_poly_model

                # train_poly_model must return (model, poly_transformed_X)
                poly_model, X_train_poly, feature_names = train_poly_model(
                    train_df, 
                    target_col="Total_WFO",
                    feature_cols=PREDICTOR_COLS
                )

                # Standard deviation of each feature
                stds = np.std(X_train_poly, axis=0)

                importances = np.abs(poly_model.coef_ * stds)

                imp_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False)

                st.dataframe(imp_df.reset_index(drop=True))

            except Exception as e:
                st.error(f"Could not compute feature importance: {e}")

        # ------------------------------------------------
        # 11) Forecast Plot
        # ------------------------------------------------
        with st.expander("📊 Forecast Plot & Model Contributions"):

            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=train_df["Date"], y=train_df["Total_WFO"],
                mode="lines", name="Training Actual", line=dict(color="royalblue")
            ))

            fig2.add_trace(go.Scatter(
                x=val_df["Date"], y=val_df["Total_WFO"],
                mode="lines+markers", name="Validation Actual",
                line=dict(color="green")
            ))

            fig2.add_trace(go.Scatter(
                x=val_df["Date"], y=val_df["pred"],
                mode="lines+markers", name="Validation Predicted",
                line=dict(color="orange")
            ))

            # Contribution Models
            fig2.add_trace(go.Scatter(
                x=val_preds["Date"], y=val_preds["short_term_pred"],
                mode="lines", name="Validation Poly",
                line=dict(color="orange", dash="dash")
            ))
            fig2.add_trace(go.Scatter(
                x=val_preds["Date"], y=val_preds["long_term_pred"],
                mode="lines", name="Validation Prophet",
                line=dict(color="purple", dash="dot")
            ))
            fig2.add_trace(go.Scatter(
                x=preds["Date"], y=preds["short_term_pred"],
                mode="lines", name="Poly (Future)",
                line=dict(color="orange")
            ))
            fig2.add_trace(go.Scatter(
                x=preds["Date"], y=preds["long_term_pred"],
                mode="lines", name="Prophet (Future)",
                line=dict(color="purple")
            ))

            fig2.add_trace(go.Scatter(
                x=preds["Date"], y=preds["final_pred"],
                mode="lines+markers", name="Final Forecast",
                line=dict(color="red", width=3)
            ))

            fig2.update_layout(
                title="Forecast & Model Contributions",
                height=600,
                hovermode="x unified"
            )

            st.plotly_chart(fig2, use_container_width=True)

        # ------------------------------------------------
        # 12) Contribution Table & Full Forecast
        # ------------------------------------------------
        with st.expander("📋 Contribution Table & Full Forecast"):

            contrib = preds[["Date", "short_term_pred", "long_term_pred", "final_pred"]].copy()
            contrib["poly_contrib"] = best_w * contrib["short_term_pred"]
            contrib["prophet_contrib"] = (1 - best_w) * contrib["long_term_pred"]

            st.markdown("### Contribution Table")
            st.dataframe(contrib)

            st.markdown("### Full Future Forecast")
            st.dataframe(preds.reset_index(drop=True))

        combined_outputs[loc] = preds