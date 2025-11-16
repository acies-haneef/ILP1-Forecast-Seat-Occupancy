# pages/2_Run_Forecast.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from utils.forecasting import load_future_flags


from utils.forecasting import (
    load_historical_for_location,
    apply_future_flags,
    run_forecast_models
)

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
VAL_DAYS = 22
st.set_page_config(page_title="Run Forecast", layout="wide")
st.title("📈 WFO Forecast — Seating-first View (Fixed Buffer)")


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
# SIDEBAR INPUTS (cleaned)
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

# ALWAYS ENABLE CONTRIBUTION VISUALS
show_contrib_lines = True
show_contrib_table = True


# ----------------------------------------------------
# TOP SEATING BUFFER INPUT
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

    if "future_flags" not in st.session_state:
        st.session_state.future_flags = load_future_flags()
        st.error("Please configure future flags on Page 1.")
        st.stop()

    combined_outputs = {}  # aggregated output for multi-location view

    # ------------------------------------------------
    # LOOP OVER LOCATIONS
    # ------------------------------------------------
    for loc in locs:

        st.markdown(f"## 📍 {loc.title()}")

        # ------------------------------------------------
        # 1) Load historical
        # ------------------------------------------------
        hist = load_historical_for_location(loc)
        hist = hist.sort_values("Date").reset_index(drop=True)

        if len(hist) <= VAL_DAYS:
            st.error(f"Not enough data for {loc}. Need > {VAL_DAYS} rows.")
            continue

        train_df = hist.iloc[:-VAL_DAYS].reset_index(drop=True)
        val_df = hist.iloc[-VAL_DAYS:].reset_index(drop=True)

        # ------------------------------------------------
        # 2) Build future business-day dates
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
            f for f in st.session_state.future_flags if f["Location"].lower() == loc
        ]

        future_ready = apply_future_flags(future_base, relevant_flags, last_hc, seating_capacity)

        # ------------------------------------------------
        # 3) Predict future using models
        # ------------------------------------------------
        preds = run_forecast_models(train_df, future_ready)
        preds["short_term_pred"] = preds["short_term_pred"].astype(float)
        preds["long_term_pred"] = preds["long_term_pred"].astype(float)

        # ------------------------------------------------
        # 4) Build validation set (existing flags from history)
        # ------------------------------------------------
        val_ready = val_df[["Date"]].copy()
        val_ready["Headcount"] = last_hc
        val_ready["DayOfWeek"] = val_ready["Date"].dt.weekday
        val_ready["Seating_Capacity"] = seating_capacity

        for col in ["Is_Mandatory_Holiday", "Is_Restricted_Holiday", "Event_Flag"]:
            val_ready[col] = hist.loc[hist["Date"].isin(val_df["Date"]), col].values

        val_ready["Hiring_Flag"] = 0

        val_preds = run_forecast_models(train_df, val_ready)
        val_preds["short_term_pred"] = val_preds["short_term_pred"].astype(float)
        val_preds["long_term_pred"] = val_preds["long_term_pred"].astype(float)

        # ------------------------------------------------
        # 5) Auto-blend
        # ------------------------------------------------
        best_w, best_score, blend_table = find_best_blend_weight(val_df, val_preds)

        # Apply blend to validation
        val_df["pred"] = (
            best_w * val_preds["short_term_pred"]
            + (1 - best_w) * val_preds["long_term_pred"]
        )

        # Mandatory override
        mask_val = val_ready["Is_Mandatory_Holiday"] == 1
        val_df.loc[mask_val, "pred"] = 0

        val_df["naive"] = val_df["Total_WFO"].shift(1).fillna(method="bfill")

        # ------------------------------------------------
        # 6) Apply blend to predictions
        # ------------------------------------------------
        preds["final_pred"] = (
            best_w * preds["short_term_pred"]
            + (1 - best_w) * preds["long_term_pred"]
        )

        if "Is_Mandatory_Holiday" in preds.columns:
            mand_mask = preds["Is_Mandatory_Holiday"] == 1
            preds.loc[mand_mask, ["final_pred", "short_term_pred", "long_term_pred"]] = 0

        preds["predicted_wfo"] = preds["final_pred"].round(2)
        preds["required_seats"] = preds["predicted_wfo"] + buffer_seats
        preds["seating_capacity"] = seating_capacity

        # ------------------------------------------------
        # 7) Seating summary
        # ------------------------------------------------
        max_pred = int(preds["predicted_wfo"].max())
        max_required = int(preds["required_seats"].max())
        exceed_days = preds[preds["required_seats"] > seating_capacity]
        exceed_count = len(exceed_days)

        st.markdown("### Seating Requirement Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Predicted WFO", max_pred)
        c2.metric("Max Required Seats", max_required)
        c3.metric("Seating Capacity", seating_capacity)
        c4.metric("Days Exceeding Capacity", exceed_count)

        # ------------------------------------------------
        # 8) Seating chart (Bar + Lines)
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
            name="Required Seats (buffer added)",
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
            title="Upcoming Seating vs Requirements (Business Days)",
            xaxis_title="Date",
            yaxis_title="Seats",
            hovermode="x unified",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------
        # 9) Seating warnings
        # ------------------------------------------------
        with st.expander("⚠️ Seating Warnings & Days exceeding capacity"):
            if exceed_count == 0:
                st.success("No capacity issues detected.")
            else:
                st.warning(f"{exceed_count} day(s) exceed capacity.")
                st.dataframe(
                    exceed_days[[
                        "Date", "predicted_wfo", "required_seats", "seating_capacity"
                    ]].reset_index(drop=True)
                )

        # ------------------------------------------------
        # 🔬 10) ALL TECHNICAL DETAILS IN EXPANDER
        # ------------------------------------------------
        with st.expander("🔬 Blend Diagnostics & Validation Metrics"):

            st.markdown("### 📆 Training & Validation Ranges")
            st.write(f"**Training:** {train_df['Date'].min().date()} → {train_df['Date'].max().date()}")
            st.write(f"**Validation:** {val_df['Date'].min().date()} → {val_df['Date'].max().date()}")

            st.markdown("---")
            st.markdown("### 📊 Validation Metrics")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", round(mae(val_df["Total_WFO"], val_df["pred"]), 2))
            m2.metric("RMSE", round(rmse(val_df["Total_WFO"], val_df["pred"]), 2))
            m3.metric("sMAPE (%)", round(smape(val_df["Total_WFO"], val_df["pred"]), 2))
            m4.metric("MASE", round(mase(val_df["Total_WFO"], val_df["pred"], val_df["naive"]), 3))

            st.markdown("---")
            st.markdown("### 📄 Validation Predictions")
            st.dataframe(val_df[["Date", "Total_WFO", "pred", "naive"]].reset_index(drop=True))

        # ------------------------------------------------
        # 11) Forecast Plot
        # ------------------------------------------------
        with st.expander("📊 Forecast Plot & Model Contributions"):

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=train_df["Date"], y=train_df["Total_WFO"],
                                      mode="lines", name="Training Actual", line=dict(color="royalblue")))
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

            # Always show model contribution lines
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
                mode="lines", name="Poly (short_term)",
                line=dict(color="orange")
            ))
            fig2.add_trace(go.Scatter(
                x=preds["Date"], y=preds["long_term_pred"],
                mode="lines", name="Prophet (long_term)",
                line=dict(color="purple")
            ))

            # Final blended forecast
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
        # 12) Contribution & full forecast table
        # ------------------------------------------------
        with st.expander("📋 Contribution Table & Full Forecast"):

            # Always show contribution table
            contrib = preds[["Date", "short_term_pred", "long_term_pred", "final_pred"]].copy()
            contrib["poly_contrib"] = best_w * contrib["short_term_pred"]
            contrib["prophet_contrib"] = (1 - best_w) * contrib["long_term_pred"]

            st.markdown("### Model Contribution Table")
            st.dataframe(contrib)

            st.markdown("### Full future forecast (business days)")
            st.dataframe(preds.reset_index(drop=True))

        combined_outputs[loc] = preds


    # --------------------------------------------------------
    # 13) Combined output across locations (if multiple selected)
    # --------------------------------------------------------
    # if combined_outputs:

    #     st.markdown("## 🌍 Combined Forecast Across Locations")

    #     merged = None
    #     for loc_name, df in combined_outputs.items():
    #         col_name = f"final_{loc_name}"
    #         temp = df[["Date", "final_pred"]].rename(columns={"final_pred": col_name})

    #         if merged is None:
    #             merged = temp
    #         else:
    #             merged = merged.merge(temp, on="Date", how="outer")

    #     st.dataframe(merged.reset_index(drop=True))

