"""Framework-independent validation and scoring for the churn predictor."""

from __future__ import annotations

import pickle
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Any


MODEL_PATH = Path(__file__).with_name("model_C=10.bin")
CHURN_THRESHOLD = 0.50


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, Any]:
    """Load the supplied vectorizer and classifier once per Python process."""
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


# Values deliberately match the categories used when the saved model was
# trained. Labels are only for the interface; the values go to DictVectorizer.
FIELD_GROUPS = (
    {
        "title": "Customer profile",
        "description": "A few basic details help place the account in context.",
        "fields": (
            {
                "name": "gender",
                "label": "Gender",
                "options": (("female", "Female"), ("male", "Male")),
            },
            {
                "name": "seniorcitizen",
                "label": "Senior citizen",
                "options": (("0", "No"), ("1", "Yes")),
            },
            {
                "name": "partner",
                "label": "Has a partner",
                "options": (("no", "No"), ("yes", "Yes")),
            },
            {
                "name": "dependents",
                "label": "Has dependents",
                "options": (("no", "No"), ("yes", "Yes")),
            },
        ),
    },
    {
        "title": "Services",
        "description": "Choose the customer’s active telecommunications services.",
        "fields": (
            {
                "name": "phoneservice",
                "label": "Phone service",
                "options": (("no", "No"), ("yes", "Yes")),
            },
            {
                "name": "multiplelines",
                "label": "Multiple lines",
                "options": (
                    ("no", "No"),
                    ("yes", "Yes"),
                    ("no_phone_service", "No phone service"),
                ),
            },
            {
                "name": "internetservice",
                "label": "Internet service",
                "options": (("dsl", "DSL"), ("fiber_optic", "Fiber optic"), ("no", "No internet")),
            },
            {
                "name": "onlinesecurity",
                "label": "Online security",
                "options": (
                    ("no", "No"),
                    ("yes", "Yes"),
                    ("no_internet_service", "No internet service"),
                ),
            },
            {
                "name": "onlinebackup",
                "label": "Online backup",
                "options": (
                    ("no", "No"),
                    ("yes", "Yes"),
                    ("no_internet_service", "No internet service"),
                ),
            },
            {
                "name": "deviceprotection",
                "label": "Device protection",
                "options": (
                    ("no", "No"),
                    ("yes", "Yes"),
                    ("no_internet_service", "No internet service"),
                ),
            },
            {
                "name": "techsupport",
                "label": "Tech support",
                "options": (
                    ("no", "No"),
                    ("yes", "Yes"),
                    ("no_internet_service", "No internet service"),
                ),
            },
            {
                "name": "streamingtv",
                "label": "Streaming TV",
                "options": (
                    ("no", "No"),
                    ("yes", "Yes"),
                    ("no_internet_service", "No internet service"),
                ),
            },
            {
                "name": "streamingmovies",
                "label": "Streaming movies",
                "options": (
                    ("no", "No"),
                    ("yes", "Yes"),
                    ("no_internet_service", "No internet service"),
                ),
            },
        ),
    },
    {
        "title": "Plan and billing",
        "description": "Enter the current agreement and charges.",
        "fields": (
            {
                "name": "contract",
                "label": "Contract",
                "options": (
                    ("month-to-month", "Month-to-month"),
                    ("one_year", "One year"),
                    ("two_year", "Two years"),
                ),
            },
            {
                "name": "paperlessbilling",
                "label": "Paperless billing",
                "options": (("no", "No"), ("yes", "Yes")),
            },
            {
                "name": "paymentmethod",
                "label": "Payment method",
                "options": (
                    ("bank_transfer_(automatic)", "Bank transfer (automatic)"),
                    ("credit_card_(automatic)", "Credit card (automatic)"),
                    ("electronic_check", "Electronic check"),
                    ("mailed_check", "Mailed check"),
                ),
            },
        ),
    },
)

MODEL_FIELDS = (
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
)

INTERNET_ADD_ON_FIELDS = (
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
)

SELECTED_VALUES = {
    field["name"]: {value for value, _ in field["options"]}
    for group in FIELD_GROUPS
    for field in group["fields"]
}

DEFAULT_CUSTOMER = {
    "customerid": "CUST-2048",
    "gender": "male",
    "seniorcitizen": 1,
    "partner": "yes",
    "dependents": "yes",
    "tenure": 32,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "fiber_optic",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "mailed_check",
    "monthlycharges": 93.95,
    "totalcharges": 2861.45,
}


def normalise_customer(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate browser/API input and return model-ready values plus field errors."""
    errors: dict[str, str] = {}
    customer: dict[str, Any] = {}

    customer_id = str(payload.get("customerid", "")).strip()
    if len(customer_id) > 80:
        errors["customerid"] = "Customer reference must be 80 characters or fewer."
    customer["customerid"] = customer_id

    for field_name, allowed_values in SELECTED_VALUES.items():
        value = str(payload.get(field_name, "")).strip().lower()
        if value not in allowed_values:
            errors[field_name] = "Choose one of the available options."
        else:
            customer[field_name] = value

    customer["seniorcitizen"] = _number_value(
        payload.get("seniorcitizen"), "seniorcitizen", errors, integer=True, minimum=0, maximum=1
    )
    customer["tenure"] = _number_value(
        payload.get("tenure"), "tenure", errors, integer=True, minimum=0, maximum=120
    )
    customer["monthlycharges"] = _number_value(
        payload.get("monthlycharges"), "monthlycharges", errors, minimum=0, maximum=10_000
    )
    customer["totalcharges"] = _number_value(
        payload.get("totalcharges"), "totalcharges", errors, minimum=0, maximum=1_000_000
    )
    _validate_service_combinations(customer, errors)

    return customer, errors


def _number_value(
    raw_value: Any,
    field_name: str,
    errors: dict[str, str],
    *,
    integer: bool = False,
    minimum: float,
    maximum: float,
) -> int | float | None:
    """Parse a bounded number while returning a friendly field-level error."""
    try:
        value = int(str(raw_value).strip()) if integer else float(str(raw_value).strip())
    except (TypeError, ValueError):
        errors[field_name] = "Enter a valid whole number." if integer else "Enter a valid number."
        return None

    if not isfinite(value):
        errors[field_name] = "Enter a finite whole number." if integer else "Enter a finite number."
        return None

    if value < minimum or value > maximum:
        label = field_name.replace("charges", " charges").replace("citizen", " citizen")
        errors[field_name] = f"{label.title()} must be between {minimum:g} and {maximum:g}."
        return None
    return value


def _validate_service_combinations(customer: dict[str, Any], errors: dict[str, str]) -> None:
    """Reject impossible service combinations instead of scoring corrupt account data."""
    phone_service = customer.get("phoneservice")
    multiple_lines = customer.get("multiplelines")
    if phone_service == "no" and multiple_lines != "no_phone_service":
        errors["multiplelines"] = "Choose 'No phone service' when phone service is not active."
    elif phone_service == "yes" and multiple_lines == "no_phone_service":
        errors["multiplelines"] = "Choose Yes or No when phone service is active."

    internet_service = customer.get("internetservice")
    for field_name in INTERNET_ADD_ON_FIELDS:
        value = customer.get(field_name)
        if internet_service == "no" and value != "no_internet_service":
            errors[field_name] = "Choose 'No internet service' when internet service is not active."
        elif internet_service in {"dsl", "fiber_optic"} and value == "no_internet_service":
            errors[field_name] = "Choose Yes or No when internet service is active."


def score_customer(customer: dict[str, Any]) -> dict[str, Any]:
    """Score the validated customer and package a UI/API-safe response."""
    model_features = {field_name: customer[field_name] for field_name in MODEL_FIELDS}
    dv, model = load_model()
    vector = dv.transform([model_features])
    probability = float(model.predict_proba(vector)[0, 1])
    churn = probability >= CHURN_THRESHOLD

    if probability >= 0.70:
        risk_level = "High"
        recommendation = "Prioritise a personal retention conversation this week."
    elif churn:
        risk_level = "Moderate"
        recommendation = "Review the account at the next customer touchpoint."
    else:
        risk_level = "Lower"
        recommendation = "Maintain the usual service and renewal check-ins."

    return {
        "customer_id": customer.get("customerid") or None,
        "probability": round(probability, 4),
        "percentage": round(probability * 100, 1),
        "churn": churn,
        "risk_level": risk_level,
        "risk_tone": risk_level.lower(),
        "threshold": CHURN_THRESHOLD,
        "recommendation": recommendation,
    }
