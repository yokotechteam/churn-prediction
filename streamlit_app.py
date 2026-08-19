"""Streamlit interface for an individual customer churn prediction."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app import DEFAULT_CUSTOMER, FIELD_GROUPS, load_model, normalise_customer, score_customer


st.set_page_config(
    page_title="Churn Lens",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


FIELD_LABELS = {
    "customerid": "Customer reference",
    "tenure": "Tenure (months)",
    "monthlycharges": "Monthly charges",
    "totalcharges": "Total charges to date",
    **{
        field["name"]: field["label"]
        for group in FIELD_GROUPS
        for field in group["fields"]
    },
}


@st.cache_resource(show_spinner="Loading the prediction model…")
def initialise_model() -> None:
    """Warm the process-local model cache before a user requests a score."""
    load_model()


def default_value(field_name: str) -> str:
    """Get a Streamlit-widget-friendly default without pre-filling an ID."""
    if field_name == "customerid":
        return ""
    return str(DEFAULT_CUSTOMER[field_name])


def select_field(field: dict[str, Any]) -> str:
    """Render one model-backed dropdown and return its machine-readable value."""
    choices = [value for value, _ in field["options"]]
    labels = dict(field["options"])
    current_value = default_value(field["name"])
    return st.selectbox(
        field["label"],
        options=choices,
        index=choices.index(current_value),
        format_func=labels.get,
        key=field["name"],
    )


def render_group(group: dict[str, Any], payload: dict[str, Any]) -> None:
    """Render a field group in two native Streamlit columns."""
    st.subheader(group["title"])
    st.caption(group["description"])

    fields = group["fields"]
    for start in range(0, len(fields), 2):
        columns = st.columns(2)
        for column, field in zip(columns, fields[start : start + 2]):
            with column:
                payload[field["name"]] = select_field(field)

    if group["title"] == "Customer profile":
        payload["tenure"] = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=120,
            value=int(DEFAULT_CUSTOMER["tenure"]),
            step=1,
            key="tenure",
        )

    if group["title"] == "Plan and billing":
        charges, total = st.columns(2)
        with charges:
            payload["monthlycharges"] = st.number_input(
                "Monthly charges (USD)",
                min_value=0.0,
                max_value=10_000.0,
                value=float(DEFAULT_CUSTOMER["monthlycharges"]),
                step=0.01,
                format="%.2f",
                key="monthlycharges",
            )
        with total:
            payload["totalcharges"] = st.number_input(
                "Total charges to date (USD)",
                min_value=0.0,
                max_value=1_000_000.0,
                value=float(DEFAULT_CUSTOMER["totalcharges"]),
                step=0.01,
                format="%.2f",
                key="totalcharges",
            )


def render_result(result: dict[str, Any] | None, errors: dict[str, str]) -> None:
    """Show a readable result or the next useful action in the right column."""
    st.subheader("Prediction result")

    if errors:
        st.error("We could not run the prediction yet. Review the validation details below.")
        with st.expander("Validation details", expanded=True):
            for field_name, message in errors.items():
                st.write(f"{FIELD_LABELS.get(field_name, field_name)}: {message}")
        return

    if not result:
        st.info("Complete the customer details, then select **Calculate churn risk**.")
        st.caption("The result will include a probability, risk band, and suggested human follow-up.")
        return

    risk_message = f"{result['risk_level']} risk — {result['percentage']:.1f}% estimated likelihood of churn"
    if result["risk_tone"] == "high":
        st.error(risk_message, icon="⚠️")
    elif result["risk_tone"] == "moderate":
        st.warning(risk_message, icon="⚠️")
    else:
        st.success(risk_message, icon="✅")

    st.metric("Estimated likelihood of churn", f"{result['percentage']:.1f}%")
    st.progress(result["percentage"] / 100)
    st.caption(f"Decision threshold: {result['threshold']:.0%}")

    if result["customer_id"]:
        st.caption(f"Customer reference: {result['customer_id']}")

    st.info(result["recommendation"], icon="💡")
    st.caption("This is a probability estimate from historical data, not a certainty or an automated decision.")

    with st.expander("View score data"):
        st.json(result)


def main() -> None:
    initialise_model()
    st.title("Churn Lens")
    st.caption("Customer retention decision support")
    st.write(
        "Enter a current account profile to estimate churn risk and guide a human follow-up. "
        "The app does not save submitted customer details."
    )

    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "validation_errors" not in st.session_state:
        st.session_state.validation_errors = {}

    input_column, result_column = st.columns((1.6, 1), gap="large")

    with input_column:
        with st.form("customer_prediction", border=True):
            st.subheader("Customer details")
            st.caption("The customer reference is optional and only labels the result; it is not a model feature.")
            payload: dict[str, Any] = {
                "customerid": st.text_input(
                    "Customer reference (optional)",
                    value=default_value("customerid"),
                    max_chars=80,
                    key="customerid",
                )
            }

            for group in FIELD_GROUPS:
                st.divider()
                render_group(group, payload)

            submitted = st.form_submit_button("Calculate churn risk", type="primary", use_container_width=True)

    if submitted:
        customer, errors = normalise_customer(payload)
        st.session_state.validation_errors = errors
        st.session_state.prediction = None if errors else score_customer(customer)

    with result_column:
        with st.container(border=True):
            render_result(st.session_state.prediction, st.session_state.validation_errors)


if __name__ == "__main__":
    main()
