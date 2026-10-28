"""Streamlit interface for the Med-Reserve patient resource-planning project."""
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from med_reserve.database import (
    delete_released_assessments, get_assessments, get_resources, initialize_database, release_allocation,
    save_assessment, update_capacities,
)
from med_reserve.modeling import FEATURES, LABELS, load_model

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "med_reserve.sqlite3"
MODEL_PATH = ROOT / "med_reserve_model.pkl"

st.set_page_config(page_title="Med-Reserve", page_icon="🏥", layout="wide")
st.title("🏥 Med-Reserve")
st.caption("Patient assessment and bed availability")

try:
    initialize_database(DB_PATH)
except sqlite3.Error as error:
    st.error(f"Could not initialize the local database: {error}")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model file is missing. From this project folder, run python train_model.py once.")
    st.stop()


@st.cache_resource
def get_model(path, modified):
    return load_model(path)


try:
    model = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as error:
    st.error(f"Could not load the local model: {error}")
    st.stop()


def create_assessment(person_name, age, spo2, heart_rate, comorbidities):
    inputs = pd.DataFrame([[age, spo2, heart_rate, comorbidities]], columns=FEATURES)
    prediction = int(model.predict(inputs)[0])
    confidence = float(max(model.predict_proba(inputs)[0]))
    return save_assessment(
        DB_PATH,
        person_name=person_name,
        age=age,
        spo2=spo2,
        heart_rate=heart_rate,
        comorbidities=comorbidities,
        predicted_level=prediction,
        predicted_label=LABELS[prediction],
        confidence=confidence,
    )


try:
    resources = {row["name"]: row for row in get_resources(DB_PATH)}
except sqlite3.Error as error:
    st.error(f"Could not read bed availability: {error}")
    st.stop()

entry_col, capacity_col = st.columns([1.0, 1.0], gap="large")

with entry_col:
    st.subheader("Add a person")
    with st.form("assessment_form", clear_on_submit=True):
        person_name = st.text_input("Name", max_chars=80, placeholder="Enter a name")
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        spo2 = st.number_input("SpO₂ (%)", min_value=70, max_value=100, value=94)
        heart_rate = st.number_input("Heart rate (BPM)", min_value=40, max_value=160, value=82)
        comorbidities = st.selectbox("Comorbidity count", options=[0, 1, 2, 3])
        submitted = st.form_submit_button("Add person", type="primary")

    if submitted:
        try:
            record = create_assessment(person_name, age, spo2, heart_rate, comorbidities)
            st.session_state["last_assessment_id"] = record["id"]
            st.rerun()
        except ValueError as error:
            st.error(str(error))
        except (OSError, RuntimeError, sqlite3.Error) as error:
            st.error(f"Could not save the person: {error}")

with capacity_col:
    st.subheader("Bed and ICU capacity")
    with st.form("capacity_form"):
        icu_capacity = st.number_input(
            "Total ICU slots", min_value=1, max_value=250, value=int(resources["ICU"]["capacity"])
        )
        general_capacity = st.number_input(
            "Total general beds", min_value=1, max_value=1000, value=int(resources["GENERAL"]["capacity"])
        )
        capacity_saved = st.form_submit_button("Save capacities")

    if capacity_saved:
        try:
            update_capacities(DB_PATH, icu_capacity=icu_capacity, general_capacity=general_capacity)
            st.success("Bed capacities updated.")
        except ValueError as error:
            st.error(str(error))
        except sqlite3.Error as error:
            st.error(f"Could not update bed capacities: {error}")

    try:
        resources = {row["name"]: row for row in get_resources(DB_PATH)}
    except sqlite3.Error as error:
        st.error(f"Could not refresh bed availability: {error}")
        st.stop()

    def capacity_circle(label, resource, color):
        item = resources[resource]
        available = int(item["available"])
        capacity = int(item["capacity"])
        percentage = round((available / capacity) * 100) if capacity else 0
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; align-items:center; padding:18px 8px;">
              <div role="img" aria-label="{label}: {available} of {capacity} available"
                   style="width:156px;height:156px;border-radius:50%;
                   background:conic-gradient({color} {percentage}%, #30343b {percentage}% 100%);
                   display:flex;align-items:center;justify-content:center;">
                <div style="width:116px;height:116px;border-radius:50%;background:#0e1117;
                     display:flex;flex-direction:column;align-items:center;justify-content:center;">
                  <span style="font-size:27px;font-weight:700;color:#f4f4f4;">{available} / {capacity}</span>
                  <span style="font-size:13px;color:#c7c9ce;">available</span>
                </div>
              </div>
              <div style="font-size:18px;font-weight:600;margin-top:10px;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    circle_icu, circle_general = st.columns(2)
    with circle_icu:
        capacity_circle("ICU slots", "ICU", "#ef6b73")
    with circle_general:
        capacity_circle("General beds", "GENERAL", "#35bd84")
    st.caption("Available capacity updates when a person is admitted or discharged.")

deleted_count = st.session_state.pop("deleted_released_count", None)
if deleted_count is not None:
    st.success(f"Removed {deleted_count} released record(s) from saved history.")

last_id = st.session_state.pop("last_assessment_id", None)
if last_id is not None:
    latest = next((row for row in get_assessments(DB_PATH) if row["id"] == last_id), None)
    if latest:
        st.success(
            f'Added {latest["person_name"]}. Risk category: {latest["predicted_label"]}; '
            f'admission status: {latest["allocation_status"]}.'
        )

st.divider()
st.subheader("Admissions and history")
try:
    records = get_assessments(DB_PATH)
except sqlite3.Error as error:
    st.error(f"Could not read saved people: {error}")
    st.stop()

if not records:
    st.info("No people have been added yet.")
else:
    history = pd.DataFrame(records)
    history["Name"] = history["person_name"]
    legacy_names = history["Name"].eq("")
    history.loc[legacy_names, "Name"] = history.loc[legacy_names, "id"].map(
        lambda item: f"Legacy record #{item}"
    )
    history["created_at"] = pd.to_datetime(history["created_at"], errors="coerce", utc=True)
    columns = [
        "id", "Name", "age", "spo2", "heart_rate", "comorbidities",
        "predicted_label", "suggested_resource", "allocation_status", "created_at",
    ]

    released_name = st.session_state.pop("released_person_name", None)
    if released_name:
        st.success(f"{released_name} moved to Released history and the bed was returned.")

    active_tab, released_tab, not_admitted_tab = st.tabs(
        ["Active admissions", "Released history", "Not admitted"]
    )

    with active_tab:
        active = history[history["allocation_status"] == "allocated"]
        if active.empty:
            st.info("There are no active admissions.")
        else:
            st.caption("People with a currently allocated bed.")
            st.dataframe(active[columns], use_container_width=True, hide_index=True)
            names_by_id = dict(zip(active["id"].astype(int), active["Name"]))
            discharge_id = st.selectbox(
                "Select an active person to discharge",
                options=list(names_by_id),
                format_func=lambda item: f"{names_by_id[item]} (record #{item})",
            )
            if st.button("Discharge selected person"):
                try:
                    selected_name = names_by_id[discharge_id]
                    if release_allocation(DB_PATH, discharge_id):
                        st.session_state["released_person_name"] = selected_name
                        st.rerun()
                    st.info("This person is no longer an active admission.")
                except sqlite3.Error as error:
                    st.error(f"Could not discharge the selected person: {error}")

    with released_tab:
        released = history[history["allocation_status"] == "released"]
        if released.empty:
            st.info("No released admissions yet.")
        else:
            st.caption("Released records stay here. You can create a new assessment from a previous record.")
            st.dataframe(released[columns], use_container_width=True, hide_index=True)

            released_records = [row for row in records if row["allocation_status"] == "released"]
            released_by_id = {int(row["id"]): row for row in released_records}
            fresh_id = st.selectbox(
                "Choose a released person for a fresh record",
                options=list(released_by_id),
                format_func=lambda item: (
                    f'{released_by_id[item]["person_name"] or f"Legacy record #{item}"} '
                    f'(record #{item})'
                ),
            )
            previous = released_by_id[fresh_id]
            st.markdown("**Create a fresh admission record**")
            with st.form(f"fresh_record_form_{fresh_id}"):
                fresh_name = st.text_input(
                    "Name", value=previous["person_name"], max_chars=80,
                    placeholder="Enter a name", key=f"fresh_name_{fresh_id}",
                )
                fresh_age = st.number_input(
                    "Age", min_value=1, max_value=100, value=int(previous["age"]),
                    key=f"fresh_age_{fresh_id}",
                )
                fresh_spo2 = st.number_input(
                    "SpO₂ (%)", min_value=70, max_value=100, value=int(previous["spo2"]),
                    key=f"fresh_spo2_{fresh_id}",
                )
                fresh_heart_rate = st.number_input(
                    "Heart rate (BPM)", min_value=40, max_value=160, value=int(previous["heart_rate"]),
                    key=f"fresh_heart_rate_{fresh_id}",
                )
                fresh_comorbidities = st.selectbox(
                    "Comorbidity count", options=[0, 1, 2, 3],
                    index=int(previous["comorbidities"]), key=f"fresh_comorbidities_{fresh_id}",
                )
                fresh_submitted = st.form_submit_button("Create fresh admission record")

            if fresh_submitted:
                try:
                    fresh_record = create_assessment(
                        fresh_name, fresh_age, fresh_spo2, fresh_heart_rate, fresh_comorbidities
                    )
                    st.session_state["last_assessment_id"] = fresh_record["id"]
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
                except (OSError, RuntimeError, sqlite3.Error) as error:
                    st.error(f"Could not create a fresh record: {error}")

            st.markdown("**Remove records from Released history**")
            st.caption("Removing a released record permanently deletes it from the saved list.")
            remove_ids = st.multiselect(
                "Select released record(s) to remove",
                options=list(released_by_id),
                format_func=lambda item: (
                    f'{released_by_id[item]["person_name"] or f"Legacy record #{item}"} '
                    f'(record #{item})'
                ),
                key="released_records_to_remove",
            )
            if st.button(
                "Remove selected released records",
                disabled=not remove_ids,
                key="remove_released_records_button",
            ):
                try:
                    st.session_state["deleted_released_count"] = delete_released_assessments(
                        DB_PATH, remove_ids
                    )
                    st.rerun()
                except sqlite3.Error as error:
                    st.error(f"Could not remove the released records: {error}")

    with not_admitted_tab:
        not_admitted = history[history["allocation_status"] == "capacity_full"]
        if not_admitted.empty:
            st.info("No assessments are waiting for an unavailable bed.")
        else:
            st.caption("Saved assessments that could not receive a bed because capacity was full.")
            st.dataframe(not_admitted[columns], use_container_width=True, hide_index=True)



