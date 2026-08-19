# Churn Lens project brief

This document is for a mixed interview audience. Start with the section that matches the person in front of you; the same product is described without assuming every reader needs the same depth.

## One-minute overview

Churn Lens is a lightweight Streamlit decision-support application for a customer-retention team. A team member enters the current profile, subscribed services, contract, and charges for one telecommunications customer. The app uses the supplied classification model to estimate the probability that the customer will churn, then presents a risk band and a suggested human follow-up.

The important distinction is that it recommends attention, not an automated outcome. A retention manager still decides whether and how to contact the customer.

## For a CEO or product leader

### The business problem

Retention teams often have more accounts to review than time. A simple, consistent signal helps them decide which conversations to prioritise first.

### What this version proves

- A frontline user can get from account facts to a readable risk signal in one screen.
- The result is intentionally actionable: lower, moderate, or high risk is paired with a proportional follow-up prompt.
- The form is designed around information a telecom retention team can normally retrieve from an account record.
- Streamlit lets a small team iterate on the workflow without maintaining a separate front-end application.
- The app does not retain submitted customer information, which keeps this demonstration focused and limits unnecessary data handling.

### How to demo it in 45 seconds

1. Open the app and point out the grouped fields: profile, services, and billing.
2. Explain that the optional customer reference is for the operator’s orientation only—it does not influence the prediction.
3. Submit the sample account details. Show the probability, risk band, and suggested next step in the right-hand panel.
4. Change a plan or service characteristic and submit again. Explain that the app recalculates the score rather than claiming that one field is a causal explanation.
5. Close by stating that the score prompts a human review; it is not a reason to treat a customer differently automatically.

### What would make it production-ready

Before using the score to direct live retention investment, define the target outcome, measure precision/recall and calibration on a recent holdout set, agree on the operating threshold and intervention cost, monitor performance by customer segment, and establish data governance. Those measures are not present in this compact inference demo.

## For a developer or technical interviewer

### Architecture

```text
Streamlit widgets and session state
             |
             v
  validation + normalisation in app.py
             |
             v
 DictVectorizer -> LogisticRegression model
             |
             v
 Streamlit metrics, risk state, and guidance
```

`streamlit_app.py` owns the interaction flow. `app.py` remains framework-independent: it normalises input, validates it, loads the model once per Python process with `lru_cache`, and packages a plain-Python score result. This separation makes the model layer straightforward to unit test or reuse in a future service.

The checked-in artifact contains a `DictVectorizer` and a scikit-learn `LogisticRegression` classifier. Its serialized metadata identifies scikit-learn 1.3.1, so the project pins that runtime version for repeatable deserialization.

### Streamlit interaction flow

1. Streamlit displays grouped native widgets inside one form.
2. A user submits the form with **Calculate churn risk**.
3. The app validates and normalises the widget values before it ever calls the model.
4. A valid score is kept in `st.session_state` and rendered in the right-hand result panel; invalid input produces a readable validation summary.
5. Streamlit retains widget values across reruns, so the user can correct data without rebuilding the form.

There are no template files, handwritten CSS, JavaScript bundles, or browser-side model calls. Streamlit supplies the responsive layout and accessible native controls.

### Input contract

The model consumes these account fields:

- Customer profile: `gender`, `seniorcitizen`, `partner`, `dependents`, `tenure`
- Services: `phoneservice`, `multiplelines`, `internetservice`, `onlinesecurity`, `onlinebackup`, `deviceprotection`, `techsupport`, `streamingtv`, `streamingmovies`
- Billing: `contract`, `paperlessbilling`, `paymentmethod`, `monthlycharges`, `totalcharges`

`customerid` is accepted as optional presentation metadata, but is removed before the model is called. Numeric values have deliberately generous server-side bounds, every categorical value is checked against the categories represented by the model artifact, and incompatible service combinations are rejected. This avoids silently scoring malformed data as an all-zero unknown category.

### UX and accessibility choices

- Native Streamlit labels and controls make the form keyboard-operable without custom client-side code.
- The form uses two columns where there is enough space and Streamlit naturally stacks content on narrower screens.
- The result uses a metric, progress indicator, risk-state alert, threshold context, and suggested next step rather than a raw probability alone.
- Validation feedback appears in a dedicated result panel and entered values remain available for correction.
- The result avoids false precision. It presents a rounded percentage and policy-level recommendation rather than pretending to know causal drivers.

### Relevant files

| File | Responsibility |
| --- | --- |
| `streamlit_app.py` | Native Streamlit form, result panel, session state, and cached model warm-up |
| `app.py` | Model loading, validation, and score packaging with no web-framework dependency |
| `model_C=10.bin` | Supplied vectorizer and logistic-regression artifact |
| `tests/test_app.py` | Core prediction tests and a Streamlit `AppTest` UI smoke test |

## Model and decision policy

The classifier returns a churn probability. This application marks a result as `churn: true` at a 0.50 threshold. The interface calls scores below that threshold “Lower,” scores from 0.50 up to 0.70 “Moderate,” and scores of 0.70 or above “High.” The 0.70 division is a presentation band; it does not change the classifier itself.

The thresholds and recommendations are product policy, not model facts. In a real rollout, they should be chosen jointly by retention operations, finance, and data science based on intervention capacity, customer value, false-positive cost, and validated model performance.

## Limitations to say out loud

- This repository is an inference application, not a reproducible training pipeline. It does not contain the training data, feature-generation job, or evaluation report.
- No current accuracy, calibration, drift, or fairness metric can be claimed from this code alone.
- `seniorcitizen` is an age-related proxy and deserves explicit fairness review before any production use.
- The recommendation is deliberately generic because this app does not calculate feature attribution or causal impact.
- There is no authentication, audit log, database, rate limiting, monitoring, or secrets-management layer. The form is intentionally stateless for this demo.
- This version is a human-facing Streamlit application, not a public JSON API. A separate authenticated service would be appropriate only when a machine-to-machine integration is actually needed.

## Sensible next engineering steps

1. Add a versioned training pipeline, data schema checks, and an evaluation report with calibration and segment analysis.
2. Add authentication, safe request logging, rate limits, and observability before handling real customer data.
3. Persist only an appropriate, consented audit record; define retention and deletion rules before storing customer data.
4. Add model version, score timestamp, and drift monitoring to each production result.
5. Extract the already framework-independent scoring layer into an authenticated API only if another system needs programmatic access.
6. Replace generic recommendations with tested playbooks and explainable, reviewed reason codes where appropriate.

## Questions this implementation answers well

**Why Streamlit?** It lets a data or product team build and adjust an internal prediction workflow in Python without carrying a separate HTML, CSS, JavaScript, or front-end build-toolchain burden. The framework’s native widgets cover the interaction this app needs.

**Why keep `app.py` separate from `streamlit_app.py`?** The UI reruns as users interact with it. Keeping validation and scoring in a plain Python module makes the business logic deterministic, independently testable, and portable to another interface later.

**Why validate if the model can vectorize unknown values?** A vectorizer might accept an unknown category by producing no matching feature. That can hide upstream data errors and produce a misleading score. The app instead rejects invalid categories clearly.

**Why not call a high score “will churn”?** Classifiers estimate probability from historical patterns. Treating a probability as certainty is both statistically wrong and poor customer practice.

**How would you deploy it?** Run `streamlit_app.py` through the `Procfile`, pin the artifact-compatible Python dependencies, inject a platform port through `PORT`, and add the production controls listed above before using real customer data.
