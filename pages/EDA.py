
import os
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EDA - WFO Forecasting", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")
st.write("Analyse factors affecting **WFO and Seating Capacity** for **Chennai** and **Bangalore** based on workforce, attendance, events, holidays and weather.")

repo_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(repo_root, "data")

attendance_paths = {
    "Chennai": os.path.join(data_dir, "Chennai_all_months_clean.csv"),
    "Bangalore": os.path.join(data_dir, "Bangalore_all_months_clean.csv"),
}

workforce_path = os.path.join(data_dir, "Masked_Capstone_Team7_Data.xlsx")

missing_att = [loc for loc, p in attendance_paths.items() if not os.path.exists(p)]
if missing_att:
    st.warning(
        "⚠️ Could not find attendance files for: "
        + ", ".join(missing_att)
        + ". Place the combined CSVs in the `data` folder."
    )

if not os.path.exists(workforce_path):
    st.warning(
        f"⚠️ Could not find workforce file: `{workforce_path}`. "
        "Copy `Masked_Capstone_Team7_Data.xlsx` into the `data` folder."
    )
    st.stop()


def standardise_attendance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "Employee ID": "Employee",
        "Employee_Id": "Employee",
        "Emp ID": "Employee",
        "Attendance_Status": "Status",
        "Attendance Date": "Date",
        "Base Location": "Location",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)
    if "Employee_Name" in df.columns:
        df["Full_Name"] = df["Employee_Name"].astype(str).str.strip()
    elif {"First Name", "Last Name"}.issubset(df.columns):
        df["Full_Name"] = (
            df["First Name"].astype(str).str.strip()
            + " "
            + df["Last Name"].astype(str).str.strip()
        )
    else:
        if "Employee" in df.columns:
            df["Full_Name"] = df["Employee"].astype(str)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
    return df


@st.cache_data
def load_all_data(att_paths: dict, wf_path: str):
    attendance_by_loc = {}
    for loc, path in att_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = standardise_attendance(df)
            attendance_by_loc[loc] = df
        else:
            attendance_by_loc[loc] = None
    xls = pd.ExcelFile(wf_path)
    wf = pd.read_excel(xls, "Workforce")
    events = pd.read_excel(xls, "Events")
    seating = pd.read_excel(xls, "Seating")
    holidays = pd.read_excel(xls, "Holidays")
    return attendance_by_loc, wf, events, seating, holidays


attendance_by_loc, wf, events, seating, holidays = load_all_data(
    attendance_paths, workforce_path
)

available_locations = [
    loc for loc, df in attendance_by_loc.items()
    if df is not None and not df.empty
]
if not available_locations:
    st.error("No attendance data loaded for any location.")
    st.stop()


def get_daily_wfo_wfh(df_att, loc_name: str):
    if df_att is None or df_att.empty:
        return None
    cols_needed = {"Date", "Status", "Employee"}
    if not cols_needed.issubset(df_att.columns):
        return None
    df = df_att.copy()
    if "Location" in df.columns:
        df = df[
            df["Location"].astype(str).str.lower().str.contains(loc_name.lower())
        ]
    if df.empty:
        return None
    df["Status_clean"] = df["Status"].astype(str).str.upper()
    df = df[df["Status_clean"] == "WFO"]
    if df.empty:
        return None
    daily = (
        df.groupby("Date")["Employee"]
        .nunique()
        .reset_index(name="WFO_Count")
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Weekday"] = daily["Date"].dt.day_name()
    daily["Month"] = daily["Date"].dt.to_period("M").astype(str)
    daily["Year"] = daily["Date"].dt.year
    daily["Quarter"] = daily["Date"].dt.to_period("Q").astype(str)
    return daily


def get_seating_monthly_vs_wfo(daily_df, seating_df, loc_name: str):
    if daily_df is None or daily_df.empty:
        return None
    if seating_df is None or seating_df.empty:
        return None
    if "Location" not in seating_df.columns:
        return None
    seat = seating_df.copy()
    value_cols = [c for c in seat.columns if c != "Location"]
    seat_long = seat.melt(
        id_vars="Location",
        value_vars=value_cols,
        var_name="Month_Start",
        value_name="Seats",
    )
    seat_long["Month_Start"] = pd.to_datetime(seat_long["Month_Start"], errors="coerce")
    seat_long["Month"] = seat_long["Month_Start"].dt.to_period("M").astype(str)
    seat_loc = seat_long[
        seat_long["Location"].astype(str).str.lower().str.contains(loc_name.lower())
    ]
    monthly = (
        daily_df.groupby("Month")["WFO_Count"]
        .agg(["mean", "max"])
    )
    monthly.columns = ["Avg_WFO", "Max_WFO"]
    monthly = monthly.reset_index()
    merged = pd.merge(
        monthly,
        seat_loc[["Month", "Seats"]],
        on="Month",
        how="left",
    )
    merged["Utilization_Avg"] = merged["Avg_WFO"] / merged["Seats"]
    merged["Utilization_Peak"] = merged["Max_WFO"] / merged["Seats"]
    merged["Exceeds_Seats_Max"] = merged["Max_WFO"] > merged["Seats"]
    return merged


def get_joiners_exits_vs_wfo(daily_df, wf_df, loc_name: str):
    if daily_df is None or daily_df.empty:
        return None
    if wf_df is None or wf_df.empty:
        return None
    if "Location" not in wf_df.columns:
        return None
    wf_loc = wf_df[
        wf_df["Location"].astype(str).str.lower().str.contains(loc_name.lower())
    ].copy()
    if wf_loc.empty:
        return None
    if "Date_of_joining" in wf_loc.columns:
        wf_loc["Date_of_joining"] = pd.to_datetime(
            wf_loc["Date_of_joining"], errors="coerce"
        )
    if "Date_of_exit" in wf_loc.columns:
        wf_loc["Date_of_exit"] = pd.to_datetime(
            wf_loc["Date_of_exit"], errors="coerce"
        )
    join = wf_loc.dropna(subset=["Date_of_joining"]).copy()
    join["Month"] = join["Date_of_joining"].dt.to_period("M").astype(str)
    join_m = (
        join.groupby("Month")["EmployeeID"]
        .nunique()
        .reset_index(name="Joiners")
    )
    exit_df = wf_loc.dropna(subset=["Date_of_exit"]).copy()
    exit_df["Month"] = exit_df["Date_of_exit"].dt.to_period("M").astype(str)
    exit_m = (
        exit_df.groupby("Month")["EmployeeID"]
        .nunique()
        .reset_index(name="Exits")
    )
    monthly_wfo = (
        daily_df.groupby("Month")["WFO_Count"]
        .mean()
        .reset_index(name="Avg_WFO")
    )
    merged = monthly_wfo.merge(join_m, on="Month", how="left").merge(
        exit_m, on="Month", how="left"
    )
    merged[["Joiners", "Exits"]] = (
        merged[["Joiners", "Exits"]].fillna(0).astype(int)
    )
    merged["Net_Change"] = merged["Joiners"] - merged["Exits"]
    return merged


def get_events_vs_wfo(daily_df, events_df, loc_name: str):
    if daily_df is None or daily_df.empty:
        return None
    if events_df is None or events_df.empty:
        return None
    if "Location" not in events_df.columns:
        return None
    ev_loc = events_df[
        events_df["Location"].astype(str).str.lower().str.contains(loc_name.lower())
    ].copy()
    if ev_loc.empty:
        return None
    ev_loc["Date"] = pd.to_datetime(ev_loc["Date"], errors="coerce")
    merged = ev_loc.merge(
        daily_df[["Date", "WFO_Count", "Weekday", "Month"]],
        on="Date",
        how="left",
    )
    daily_base = (
        daily_df.groupby("Weekday")["WFO_Count"]
        .mean()
        .reset_index(name="Baseline_WFO_Weekday")
    )
    merged = merged.merge(daily_base, on="Weekday", how="left")
    merged["Uplift_WFO"] = merged["WFO_Count"] - merged["Baseline_WFO_Weekday"]
    merged["EventLabel"] = (
        merged["Date"].dt.strftime("%d-%b-%Y") + " – " + merged["Event_Type"]
    )
    return merged


def get_holiday_vs_wfo(daily_df, holidays_df, loc_name: str):
    if daily_df is None or daily_df.empty:
        return None
    if holidays_df is None or holidays_df.empty:
        return None
    if "Date" not in holidays_df.columns:
        return None
    hol = holidays_df.copy()
    if "Location" in hol.columns:
        hol = hol[
            hol["Location"].astype(str)
            .str.lower()
            .str.contains(loc_name.lower())
        ]
    hol["Date"] = pd.to_datetime(hol["Date"], errors="coerce")
    if hol.empty:
        return None
    min_d, max_d = daily_df["Date"].min(), daily_df["Date"].max()
    hol = hol[(hol["Date"] >= min_d) & (hol["Date"] <= max_d)]
    merged = daily_df.merge(
        hol[["Date", "Name", "Classification"]],
        on="Date",
        how="left",
    )
    merged["Classification"] = merged["Classification"].fillna("Non-Holiday")
    return merged


def get_holiday_window_vs_wfo(daily_df, holidays_df, loc_name: str):
    if daily_df is None or daily_df.empty:
        return None
    if holidays_df is None or holidays_df.empty:
        return None
    if "Date" not in holidays_df.columns:
        return None
    hol = holidays_df.copy()
    if "Location" in hol.columns:
        hol = hol[
            hol["Location"].astype(str)
            .str.lower()
            .str.contains(loc_name.lower())
        ]
    hol["Date"] = pd.to_datetime(hol["Date"], errors="coerce")
    if hol.empty:
        return None
    all_rows = []
    for _, row in hol.iterrows():
        d = row["Date"]
        name = row.get("Name", "")
        cls = row.get("Classification", "")
        before = d - pd.Timedelta(days=1)
        after = d + pd.Timedelta(days=1)
        all_rows.append({"Date": before, "Day_Type": "Before_Holiday", "Holiday_Name": name, "Holiday_Classification": cls, "Ref_Holiday_Date": d})
        all_rows.append({"Date": d, "Day_Type": "Holiday", "Holiday_Name": name, "Holiday_Classification": cls, "Ref_Holiday_Date": d})
        all_rows.append({"Date": after, "Day_Type": "After_Holiday", "Holiday_Name": name, "Holiday_Classification": cls, "Ref_Holiday_Date": d})
    win_df = pd.DataFrame(all_rows)
    merged = win_df.merge(
        daily_df[["Date", "WFO_Count"]],
        on="Date",
        how="left",
    )
    merged = merged.dropna(subset=["WFO_Count"])
    if merged.empty:
        return None
    return merged


def build_open_meteo_archive_url(lat, lon, start, end, daily_vars, timezone):
    base = "https://archive-api.open-meteo.com/v1/archive"
    daily_param = ",".join(daily_vars)
    tz = quote(timezone, safe="")
    return (
        f"{base}?latitude={lat}&longitude={lon}"
        f"&start_date={start.date()}&end_date={end.date()}"
        f"&daily={daily_param}&timezone={tz}"
    )


def get_weather_vs_wfo(daily_df, loc_name: str, root_dir: str):
    if daily_df is None or daily_df.empty:
        return None
    loc_lower = loc_name.lower()
    if loc_lower == "chennai":
        lat, lon = 13.0827, 80.2707
        subdir = "output_chennai"
        fname = "weather_chennai_2024_2025.csv"
    elif loc_lower in ["bangalore", "bengaluru"]:
        lat, lon = 12.9716, 77.5946
        subdir = "output_bangalore"
        fname = "weather_bangalore_2024_2025.csv"
    else:
        return None
    start = daily_df["Date"].min().normalize()
    end = daily_df["Date"].max().normalize()
    weather_dir = os.path.join(root_dir, subdir)
    os.makedirs(weather_dir, exist_ok=True)
    cache_path = os.path.join(weather_dir, fname)
    if os.path.exists(cache_path):
        wdf = pd.read_csv(cache_path)
        wdf["Date"] = pd.to_datetime(wdf["Date"])
    else:
        timezone = "Asia/Kolkata"
        daily_vars = ["precipitation_sum"]
        url = build_open_meteo_archive_url(lat, lon, start, end, daily_vars, timezone)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            j = resp.json()
            daily = j.get("daily", {})
            if not daily or "time" not in daily:
                return None
            wdf = pd.DataFrame(daily).rename(columns={"time": "Date"})
            wdf["Date"] = pd.to_datetime(wdf["Date"])
            wdf.to_csv(cache_path, index=False)
        except Exception:
            return None
    wdf["Month"] = wdf["Date"].values.astype("datetime64[M]")
    monthly_weather = (
        wdf.groupby("Month")["precipitation_sum"]
        .sum()
        .reset_index()
        .rename(columns={"precipitation_sum": "Total_Precip_mm"})
    )
    mw = daily_df.copy()
    mw["Month"] = mw["Date"].values.astype("datetime64[M]")
    mwfo = mw.groupby("Month")["WFO_Count"].mean().reset_index(name="Avg_WFO")
    df_weather = mwfo.merge(monthly_weather, on="Month", how="left")
    df_weather["Month"] = df_weather["Month"].dt.to_period("M").astype(str)
    return df_weather


def get_quarterly_summary(daily_df, seating_df, loc_name: str):
    if daily_df is None or daily_df.empty:
        return None
    q = (
        daily_df.groupby("Quarter")["WFO_Count"]
        .agg(["mean", "max"])
        .reset_index()
        .rename(columns={"mean": "Avg_WFO", "max": "Max_WFO"})
    )
    if seating_df is None or seating_df.empty or "Location" not in seating_df.columns:
        q["Seats"] = np.nan
        return q
    seat = seating_df.copy()
    value_cols = [c for c in seat.columns if c != "Location"]
    seat_long = seat.melt(
        id_vars="Location",
        value_vars=value_cols,
        var_name="Month_Start",
        value_name="Seats",
    )
    seat_long["Month_Start"] = pd.to_datetime(seat_long["Month_Start"], errors="coerce")
    seat_long["Quarter"] = seat_long["Month_Start"].dt.to_period("Q").astype(str)
    seat_loc = seat_long[
        seat_long["Location"].astype(str).str.lower().str.contains(loc_name.lower())
    ]
    seat_q = (
        seat_loc.groupby("Quarter")["Seats"]
        .mean()
        .reset_index()
    )
    q = q.merge(seat_q, on="Quarter", how="left")
    q["Utilization_Avg"] = q["Avg_WFO"] / q["Seats"]
    q["Utilization_Peak"] = q["Max_WFO"] / q["Seats"]
    return q


def get_monthly_correlation_df(loc_name, attendance_by_loc, seating_df, wf_df, events_df, holidays_df, root_dir):
    df_att = attendance_by_loc.get(loc_name)
    daily = get_daily_wfo_wfh(df_att, loc_name)
    if daily is None or daily.empty:
        return None
    base = (
        daily.groupby("Month")["WFO_Count"]
        .agg(["mean", "max"])
    )
    base.columns = ["Avg_WFO", "Max_WFO"]
    base = base.reset_index()
    df = base.copy()
    seat_month = get_seating_monthly_vs_wfo(daily, seating_df, loc_name)
    if seat_month is not None and not seat_month.empty:
        extra = seat_month[["Month", "Seats", "Utilization_Avg", "Utilization_Peak"]]
        df = df.merge(extra, on="Month", how="left")
    je_month = get_joiners_exits_vs_wfo(daily, wf_df, loc_name)
    if je_month is not None and not je_month.empty:
        extra = je_month[["Month", "Joiners", "Exits", "Net_Change"]]
        df = df.merge(extra, on="Month", how="left")
    if events_df is not None and not events_df.empty and "Location" in events_df.columns:
        ev_loc = events_df[
            events_df["Location"].astype(str).str.lower().str.contains(loc_name.lower())
        ].copy()
        if not ev_loc.empty:
            ev_loc["Date"] = pd.to_datetime(ev_loc["Date"], errors="coerce")
            ev_loc["Month"] = ev_loc["Date"].dt.to_period("M").astype(str)
            monthly_events = (
                ev_loc.groupby("Month")["Event_Type"]
                .count()
                .reset_index(name="Num_Events")
            )
            df = df.merge(monthly_events, on="Month", how="left")
    if holidays_df is not None and not holidays_df.empty and "Date" in holidays_df.columns:
        hol_loc = holidays_df.copy()
        if "Location" in hol_loc.columns:
            hol_loc = hol_loc[
                hol_loc["Location"].astype(str).str.lower().str.contains(loc_name.lower())
            ]
        if not hol_loc.empty:
            hol_loc["Date"] = pd.to_datetime(hol_loc["Date"], errors="coerce")
            hol_loc["Month"] = hol_loc["Date"].dt.to_period("M").astype(str)
            hol_count = (
                hol_loc.groupby("Month")["Name"]
                .count()
                .reset_index(name="Num_Holidays")
            )
            df = df.merge(hol_count, on="Month", how="left")
    weather = get_weather_vs_wfo(daily, loc_name, root_dir)
    if weather is not None and not weather.empty:
        extra = weather[["Month", "Total_Precip_mm"]]
        df = df.merge(extra, on="Month", how="left")
    df = df.sort_values("Month")
    return df


with st.expander("📅 Overall WFO Trend over Time", expanded=True):
    loc_overall = st.radio(
        "Select Location",
        available_locations,
        index=0,
        horizontal=True,
        key="loc_overall",
    )
    df_att_loc = attendance_by_loc.get(loc_overall)
    daily = get_daily_wfo_wfh(df_att_loc, loc_overall)
    if daily is None or daily.empty:
        st.info(
            "Could not compute daily WFO counts. Check that Date, Status and Employee columns exist for this location."
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["WFO_Count"],
                mode="lines",
                name="WFO"
            )
        )
        fig.update_layout(
            title=f"Daily WFO Count – {loc_overall}",
            xaxis_title="Date",
            yaxis_title="Unique WFO Employees"
        )
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg daily WFO", f"{daily['WFO_Count'].mean():.1f}")
        with col2:
            st.metric("Max WFO in a day", int(daily["WFO_Count"].max()))
        with col3:
            st.metric("Number of active days", len(daily))


with st.expander("📆 Weekday Pattern of WFO"):
    loc_weekday = st.radio(
        "Select Location",
        available_locations,
        index=0,
        horizontal=True,
        key="loc_weekday",
    )
    df_att_loc = attendance_by_loc.get(loc_weekday)
    daily = get_daily_wfo_wfh(df_att_loc, loc_weekday)
    if daily is None or daily.empty:
        st.info(
            "Could not compute weekday pattern for this location."
        )
    else:
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_summary = (
            daily.groupby("Weekday")["WFO_Count"]
            .mean()
            .reindex(weekday_order)
            .dropna()
            .reset_index(name="Avg_WFO")
        )
        fig1 = px.bar(
            weekday_summary,
            x="Weekday",
            y="Avg_WFO",
            title=f"Average WFO by Weekday – {loc_weekday}",
        )
        fig1.update_layout(
            xaxis_title="Weekday",
            yaxis_title="Average Unique WFO Employees"
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.dataframe(weekday_summary)


with st.expander("🪑 Seating Capacity vs WFO and Utilization"):
    loc_seat = st.radio(
        "Select Location",
        available_locations,
        index=0,
        horizontal=True,
        key="loc_seat",
    )
    df_att_loc = attendance_by_loc.get(loc_seat)
    daily = get_daily_wfo_wfh(df_att_loc, loc_seat)
    seat_month = get_seating_monthly_vs_wfo(daily, seating, loc_seat)
    if seat_month is None or seat_month.empty:
        st.info("Seating or WFO data not available for this analysis.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=seat_month["Month"],
                y=seat_month["Seats"],
                name="Seats",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=seat_month["Month"],
                y=seat_month["Max_WFO"],
                mode="lines+markers",
                name="Max WFO",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=f"Monthly Seating Capacity vs Max WFO – {loc_seat}",
            xaxis_title="Month",
            yaxis=dict(title="Seats"),
            yaxis2=dict(
                title="Max WFO",
                overlaying="y",
                side="right",
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        fig_util = px.bar(
            seat_month,
            x="Month",
            y="Utilization_Avg",
            title=f"Average Seat Utilization (based on Avg WFO) – {loc_seat}",
        )
        fig_util.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_util, use_container_width=True)
        overflow_months = seat_month[seat_month["Exceeds_Seats_Max"]]
        if not overflow_months.empty:
            st.subheader("Months where Max WFO exceeded Seats")
            st.dataframe(overflow_months[["Month", "Max_WFO", "Seats", "Utilization_Peak"]])


with st.expander("👥 Workforce Growth, Attrition vs WFO"):
    loc_je = st.radio(
        "Select Location",
        available_locations,
        index=0,
        horizontal=True,
        key="loc_je",
    )
    df_att_loc = attendance_by_loc.get(loc_je)
    daily = get_daily_wfo_wfh(df_att_loc, loc_je)
    je_month = get_joiners_exits_vs_wfo(daily, wf, loc_je)
    if je_month is None or je_month.empty:
        st.info("Workforce or WFO data not available for this analysis.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=je_month["Month"],
                y=je_month["Joiners"],
                name="Joiners",
            )
        )
        fig.add_trace(
            go.Bar(
                x=je_month["Month"],
                y=je_month["Exits"],
                name="Exits",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=je_month["Month"],
                y=je_month["Avg_WFO"],
                mode="lines+markers",
                name="Avg WFO",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=f"Monthly Joiners/Exits vs Avg WFO – {loc_je}",
            xaxis_title="Month",
            yaxis=dict(title="Headcount (Joiners/Exits)"),
            yaxis2=dict(
                title="Avg WFO",
                overlaying="y",
                side="right",
            ),
            barmode="group",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(je_month)


with st.expander("🎉 Events, Leadership Visits, Intern Batches vs WFO"):
    loc_ev = st.radio(
        "Select Location",
        available_locations,
        index=0,
        horizontal=True,
        key="loc_ev",
    )
    df_att_loc = attendance_by_loc.get(loc_ev)
    daily = get_daily_wfo_wfh(df_att_loc, loc_ev)
    ev_wfo = get_events_vs_wfo(daily, events, loc_ev)
    if ev_wfo is None or ev_wfo.empty:
        st.info("No events mapped to WFO for this location and period.")
    else:
        ev_plot = ev_wfo.dropna(subset=["WFO_Count"]).copy()
        if ev_plot.empty:
            st.info("No WFO data available on event days.")
        else:
            fig1 = px.bar(
                ev_plot,
                x="EventLabel",
                y="WFO_Count",
                color="Event_Type",
                title=f"WFO Count on Event Days by Event Type – {loc_ev}",
            )
            fig1.update_layout(
                xaxis_title="Event",
                yaxis_title="Unique WFO Employees",
                xaxis_tickangle=45,
            )
            st.plotly_chart(fig1, use_container_width=True)
            agg_uplift = (
                ev_plot.groupby("Event_Type")["Uplift_WFO"]
                .mean()
                .reset_index()
            )
            fig2 = px.bar(
                agg_uplift,
                x="Event_Type",
                y="Uplift_WFO",
                title="Average WFO Uplift vs Baseline by Event Type",
            )
            fig2.update_traces(opacity=0.8)
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(ev_plot[["Date", "Event_Type", "EventLabel", "WFO_Count", "Baseline_WFO_Weekday", "Uplift_WFO"]])


with st.expander("🏖️ Holidays, Festive Seasons vs WFO"):
    loc_hol = st.radio(
        "Select Location",
        available_locations,
        index=0,
        horizontal=True,
        key="loc_hol",
    )
    df_att_loc = attendance_by_loc.get(loc_hol)
    daily = get_daily_wfo_wfh(df_att_loc, loc_hol)
    hol_daily = get_holiday_vs_wfo(daily, holidays, loc_hol)
    hol_window = get_holiday_window_vs_wfo(daily, holidays, loc_hol)
    if hol_daily is None or hol_daily.empty:
        st.info("Holiday or WFO data not available.")
    else:
        hol_summary = (
            hol_daily.groupby("Classification")["WFO_Count"]
            .mean()
            .reset_index(name="Avg_WFO")
        )
        fig1 = px.bar(
            hol_summary,
            x="Classification",
            y="Avg_WFO",
            title=f"Average WFO by Holiday Classification – {loc_hol}",
        )
        fig1.update_layout(
            xaxis_title="Classification",
            yaxis_title="Average Unique WFO Employees",
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.dataframe(hol_summary)
    if hol_window is None or hol_window.empty:
        st.info("No valid WFO data around holidays.")
    else:
        win_summary = (
            hol_window.groupby("Day_Type")["WFO_Count"]
            .mean()
            .reset_index(name="Avg_WFO")
        )
        fig2 = px.bar(
            win_summary,
            x="Day_Type",
            y="Avg_WFO",
            title="Average WFO Before, On and After Holidays (All Holidays)",
        )
        fig2.update_layout(
            xaxis_title="Day Type",
            yaxis_title="Average Unique WFO Employees",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(win_summary)
        per_hol = (
            hol_window.groupby(["Holiday_Name", "Day_Type"])["WFO_Count"]
            .mean()
            .reset_index()
        )
        pivot = per_hol.pivot(index="Holiday_Name", columns="Day_Type", values="WFO_Count").reset_index()
        for col in ["Before_Holiday", "Holiday", "After_Holiday"]:
            if col not in pivot.columns:
                pivot[col] = np.nan
        order = (
            hol_window.groupby("Holiday_Name")["Ref_Holiday_Date"]
            .min()
            .sort_values()
            .index
        )
        pivot = pivot.set_index("Holiday_Name").loc[order].reset_index()
        fig3 = go.Figure()
        fig3.add_trace(
            go.Bar(
                x=pivot["Holiday_Name"],
                y=pivot["Before_Holiday"],
                name="Before Holiday"
            )
        )
        fig3.add_trace(
            go.Bar(
                x=pivot["Holiday_Name"],
                y=pivot["Holiday"],
                name="On Holiday"
            )
        )
        fig3.add_trace(
            go.Bar(
                x=pivot["Holiday_Name"],
                y=pivot["After_Holiday"],
                name="After Holiday"
            )
        )
        fig3.update_layout(
            barmode="group",
            title="WFO Count Before, On, and After Each Holiday",
            xaxis_title="Holiday",
            yaxis_title="Average WFO Count",
            xaxis_tickangle=45,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(pivot)


with st.expander("🌧️ Weather vs WFO"):
    loc_weather = st.radio(
        "Select Location",
        available_locations,
        index=0,
        horizontal=True,
        key="loc_weather",
    )
    df_att_loc = attendance_by_loc.get(loc_weather)
    daily = get_daily_wfo_wfh(df_att_loc, loc_weather)
    weather_wfo = get_weather_vs_wfo(daily, loc_weather, repo_root)
    if weather_wfo is None or weather_wfo.empty:
        st.info("Weather data is currently implemented only for Chennai and Bangalore or not available for the selected period.")
    else:
        tmp = weather_wfo.dropna(subset=["Avg_WFO", "Total_Precip_mm"]).copy()
        if (
            len(tmp) < 2
            or tmp["Avg_WFO"].std() == 0
            or tmp["Total_Precip_mm"].std() == 0
        ):
            corr = np.nan
        else:
            corr = float(
                np.corrcoef(tmp["Avg_WFO"], tmp["Total_Precip_mm"])[0, 1]
            )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=weather_wfo["Month"],
                y=weather_wfo["Avg_WFO"],
                name="Avg WFO",
                mode="lines+markers",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=weather_wfo["Month"],
                y=weather_wfo["Total_Precip_mm"],
                name="Total Precip (mm)",
                mode="lines+markers",
                yaxis="y2",
            )
        )
        if not np.isnan(corr):
            title_corr = f"Monthly Avg WFO vs Rainfall (r = {corr:.3f})"
        else:
            title_corr = "Monthly Avg WFO vs Rainfall"
        fig.update_layout(
            title=title_corr,
            xaxis_title="Month",
            yaxis=dict(title="Average Unique WFO Employees"),
            yaxis2=dict(
                title="Total Precipitation (mm)",
                overlaying="y",
                side="right",
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)


with st.expander("📈 Quarterly Seat Utilization Summary"):
    loc_q = st.radio(
        "Select Location",
        available_locations,
        index=0,
        horizontal=True,
        key="loc_q",
    )
    df_att_loc = attendance_by_loc.get(loc_q)
    daily = get_daily_wfo_wfh(df_att_loc, loc_q)
    q_summary = get_quarterly_summary(daily, seating, loc_q)
    if q_summary is None or q_summary.empty:
        st.info("No quarterly summary available.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=q_summary["Quarter"],
                y=q_summary["Avg_WFO"],
                name="Avg WFO",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=q_summary["Quarter"],
                y=q_summary["Seats"],
                name="Seats",
                yaxis="y2",
                mode="lines+markers",
            )
        )
        fig.update_layout(
            title=f"Quarterly Avg WFO vs Seats – {loc_q}",
            xaxis_title="Quarter",
            yaxis=dict(title="Avg WFO"),
            yaxis2=dict(
                title="Seats",
                overlaying="y",
                side="right",
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        fig_util_q = px.bar(
            q_summary,
            x="Quarter",
            y="Utilization_Avg",
            title=f"Quarterly Average Seat Utilization – {loc_q}",
        )
        fig_util_q.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_util_q, use_container_width=True)
        st.dataframe(q_summary)


with st.expander("📊 Correlation Matrix of Monthly Factors"):
    loc_corr = st.radio(
        "Select Location",
        available_locations,
        index=0,
        horizontal=True,
        key="loc_corr",
    )
    corr_df = get_monthly_correlation_df(loc_corr, attendance_by_loc, seating, wf, events, holidays, repo_root)
    if corr_df is None or corr_df.empty:
        st.info("Not enough data to compute correlations.")
    else:
        num_df = corr_df.drop(columns=["Month"])
        num_df = num_df.dropna(axis=1, how="all")
        if num_df.shape[1] < 2:
            st.info("Not enough numeric factors to compute a correlation matrix.")
        else:
            corr_matrix = num_df.corr()
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="Correlation"),
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                )
            )
            fig.update_layout(
                title=f"Correlation Matrix of Monthly Factors – {loc_corr}",
                xaxis_title="Factors",
                yaxis_title="Factors",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(corr_df)

st.caption(
    "EDA covers WFO trends, weekday patterns, seating utilization, workforce changes, events, holidays, weather (Chennai & Bangalore) and quarterly plus correlation summaries for Chennai and Bangalore."
)

