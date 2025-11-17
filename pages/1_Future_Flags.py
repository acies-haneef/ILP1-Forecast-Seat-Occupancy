# pages/1_Future_Flags.py
import streamlit as st
import pandas as pd
from datetime import date

from utils.forecasting import load_future_flags, save_future_flags


st.set_page_config(page_title="Future Flags", page_icon="⚙️", layout="wide")
st.title("⚙️ Configure Future Inputs (Bangalore / Chennai)")


# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------

def load_initial_values():
    blr = pd.read_csv("data/master_bangalore.csv")
    che = pd.read_csv("data/master_chennai.csv")

    blr["Date"] = pd.to_datetime(blr["Date"], dayfirst=True, errors="coerce")
    che["Date"] = pd.to_datetime(che["Date"], dayfirst=True, errors="coerce")

    blr = blr.dropna(subset=["Date"])
    che = che.dropna(subset=["Date"])

    blr_last = blr.sort_values("Date").iloc[-1]
    che_last = che.sort_values("Date").iloc[-1]

    return {
        "blr_workforce": int(blr_last["Headcount"]),
        "che_workforce": int(che_last["Headcount"]),
        "blr_seats": int(blr_last["Seating_Capacity"]),
        "che_seats": int(che_last["Seating_Capacity"])
    }


if "future_flags" not in st.session_state:
    st.session_state.future_flags = load_future_flags()

if "edit_index" not in st.session_state:
    st.session_state.edit_index = None

# Initialize workforce + seating
if "workforce_blr" not in st.session_state:
    init = load_initial_values()
    st.session_state.workforce_blr = init["blr_workforce"]
    st.session_state.workforce_che = init["che_workforce"]
    st.session_state.seats_blr = init["blr_seats"]
    st.session_state.seats_che = init["che_seats"]


# ---------------------------------------------------------
# Helper: Convert flags to dataframe
# ---------------------------------------------------------
def get_df():
    if not st.session_state.future_flags:
        return pd.DataFrame(columns=["Date", "Location", "Type", "Count", "Event_Name"])
    df = pd.DataFrame(st.session_state.future_flags)
    df["orig_index"] = df.index
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    return df.sort_values(["Date", "Location"]).reset_index(drop=True)


# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------
st.markdown("### 📊 Current Workforce & Seating Capacity")

col1, col2 = st.columns(2)
with col1:
    st.metric("Bangalore Workforce", st.session_state.workforce_blr)
    st.metric("Bangalore Seating Capacity", st.session_state.seats_blr)
with col2:
    st.metric("Chennai Workforce", st.session_state.workforce_che)
    st.metric("Chennai Seating Capacity", st.session_state.seats_che)

st.markdown("---")


# ---------------------------------------------------------
# INPUT EXPANDERS
# ---------------------------------------------------------
st.header("➕ Add New Inputs")

# Seating capacity update
with st.expander("🏢 Seating Capacity", expanded=False):
    loc = st.selectbox("Location", ["Bangalore", "Chennai"], key="seat_loc")
    current = st.session_state.seats_blr if loc == "Bangalore" else st.session_state.seats_che
    newcap = st.number_input("New Seating Capacity", min_value=0, value=current)

    if st.button("💾 Update Capacity"):
        if loc == "Bangalore":
            st.session_state.seats_blr = newcap
        else:
            st.session_state.seats_che = newcap

        # Save NO persist here because seating is not stored in CSV — only flags.
        st.success(f"Seat capacity updated for {loc}")


# Hiring / Exit
with st.expander("👥 Hiring / Exit", expanded=False):

    c1, c2, c3, c4 = st.columns(4)
    with c1: d = st.date_input("Date", value=date.today())
    with c2: loc = st.selectbox("Location", ["Bangalore", "Chennai"], key="he_loc")
    with c3: typ = st.selectbox("Type", ["Hiring", "Exit"])
    with c4: cnt = st.number_input("Count", min_value=0, step=1)

    if st.button("➕ Add Hiring/Exit"):
        rec = {"Date": d, "Location": loc, "Type": typ, "Count": cnt, "Event_Name": ""}
        st.session_state.future_flags.append(rec)

        # Save persistently
        save_future_flags(st.session_state.future_flags)

        # Update workforce
        # if loc == "Bangalore":
        #     st.session_state.workforce_blr += cnt if typ == "Hiring" else -cnt
        # else:
        #     st.session_state.workforce_che += cnt if typ == "Hiring" else -cnt

        st.success(f"{typ} added for {loc}.")


# Holidays / Events
with st.expander("📅 Holidays & Events", expanded=False):

    c1, c2, c3, c4 = st.columns(4)
    with c1: d = st.date_input("Event Date")
    with c2: loc = st.selectbox("Event Location", ["Bangalore", "Chennai"], key="ev_loc")
    with c3: typ = st.selectbox("Type", ["Mandatory", "Restricted", "Event"])
    with c4: name = st.text_input("Holiday/Event Name")

    if st.button("➕ Add Holiday/Event"):
        st.session_state.future_flags.append({
            "Date": d, "Location": loc, "Type": typ, "Count": 0, "Event_Name": name
        })
        save_future_flags(st.session_state.future_flags)
        st.success(f"{typ} added for {loc}.")


st.markdown("---")


# ---------------------------------------------------------
# REVIEW TABLES
# ---------------------------------------------------------
df = get_df()
st.header("📋 Review Flags")

if df.empty:
    st.info("No flags yet.")
else:
    people = df[df.Type.isin(["Hiring", "Exit"])].reset_index(drop=True)
    events = df[df.Type.isin(["Mandatory", "Restricted", "Event"])].reset_index(drop=True)

    # ---------------------- Hiring Table ----------------------
    st.subheader("👥 Hiring / Exit")
    cols = st.columns([2, 2, 2, 2, 1, 1])
    cols[0].markdown("**Date**")
    cols[1].markdown("**Location**")
    cols[2].markdown("**Type**")
    cols[3].markdown("**Count**")
    cols[4].markdown("**Edit**")
    cols[5].markdown("**Delete**")

    for display_i, row in people.iterrows():
        true_idx = row.name
        # TRUE index inside list
        # true_idx = df.index[df["Date"].eq(row.Date) &
        #                      df["Location"].eq(row.Location) &
        #                      df["Type"].eq(row.Type) &
        #                      df["Count"].eq(row.Count)].tolist()[0]

        rc = st.columns([2, 2, 2, 2, 1, 1])
        rc[0].write(row.Date.date())
        rc[1].write(row.Location)
        rc[2].write(row.Type)
        rc[3].write(row.Count)

        if rc[4].button("✏️", key=f"edit_{true_idx}"):
            st.session_state.edit_index = true_idx

        if rc[5].button("❌", key=f"del_{true_idx}"):

            # ❌ DO NOT TOUCH WORKFORCE ANYMORE
            orig = row.orig_index
            st.session_state.future_flags.pop(orig)
            save_future_flags(st.session_state.future_flags)
            st.rerun()

    st.markdown("---")

    # ---------------------- Events Table ----------------------
    st.subheader("📅 Holidays & Events")
    cols = st.columns([2, 2, 2, 2, 1, 1])
    cols[0].markdown("**Date**")
    cols[1].markdown("**Location**")
    cols[2].markdown("**Type**")
    cols[3].markdown("**Event Name**")
    cols[4].markdown("**Edit**")
    cols[5].markdown("**Delete**")

    for display_i, row in events.iterrows():
        true_idx = row.name
        # true_idx = df.index[df["Date"].eq(row.Date) &
        #                      df["Location"].eq(row.Location) &
        #                      df["Type"].eq(row.Type) &
        #                      df["Event_Name"].eq(row.Event_Name)].tolist()[0]

        rc = st.columns([2, 2, 2, 2, 1, 1])
        rc[0].write(row.Date.date())
        rc[1].write(row.Location)
        rc[2].write(row.Type)
        rc[3].write(row.Event_Name)

        if rc[4].button("✏️", key=f"edit_evt_{true_idx}"):
            st.session_state.edit_index = true_idx

        if rc[5].button("❌", key=f"del_evt_{true_idx}"):
            st.session_state.future_flags.pop(true_idx)
            save_future_flags(st.session_state.future_flags)
            st.rerun()



# ---------------------------------------------------------
# EDIT MODAL
# ---------------------------------------------------------
if st.session_state.edit_index is not None:

    @st.dialog("Edit Flag")
    def edit_modal():
        idx = st.session_state.edit_index

        # Read latest
        df = get_df()

        if idx >= len(df):
            st.error("Record no longer exists.")
            st.session_state.edit_index = None
            st.rerun()

        row = df.iloc[idx]

        st.write(f"Editing record for {row.Date.date()}")

        new_loc = st.selectbox(
            "Location",
            ["Bangalore", "Chennai"],
            index=["Bangalore", "Chennai"].index(row.Location)
        )

        # --------------- Hiring / Exit ----------------
        if row.Type in ["Hiring", "Exit"]:
            new_type = st.selectbox("Type", ["Hiring", "Exit"],
                                    index=["Hiring", "Exit"].index(row.Type))
            new_count = st.number_input("Count", value=row.Count, min_value=0)

            if st.button("Save"):
                orig = row.orig_index
                st.session_state.future_flags[orig] = {
                    "Date": row.Date.date(),
                    "Location": new_loc,
                    "Type": new_type,
                    "Count": new_count,
                    "Event_Name": ""
                }
                save_future_flags(st.session_state.future_flags)
                st.session_state.edit_index = None
                st.rerun()

        # --------------- Holiday/Event ---------------
        else:
            new_type = st.selectbox(
                "Type",
                ["Mandatory", "Restricted", "Event"],
                index=["Mandatory", "Restricted", "Event"].index(row.Type)
            )
            new_name = st.text_input("Event Name", value=row.Event_Name)

            if st.button("Save"):
                st.session_state.future_flags[idx] = {
                    "Date": row.Date.date(),
                    "Location": new_loc,
                    "Type": new_type,
                    "Count": 0,
                    "Event_Name": new_name
                }
                save_future_flags(st.session_state.future_flags)
                st.session_state.edit_index = None
                st.rerun()

        if st.button("Delete"):
            st.session_state.future_flags.pop(idx)
            save_future_flags(st.session_state.future_flags)
            st.session_state.edit_index = None
            st.rerun()

    edit_modal()

