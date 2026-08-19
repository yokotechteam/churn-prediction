# Churn Lens

Churn Lens is a small, interview-ready Streamlit application that turns current telecommunications account details into an individual churn-risk estimate. It uses only native Streamlit widgets for the user experience—there are no hand-written HTML, CSS, or JavaScript assets to maintain.

## What a user can do

1. Enter a customer’s profile, services, contract, and billing details.
2. Select **Calculate churn risk**.
3. Read the estimated probability, risk band, and suggested human follow-up.

The app does not persist submitted customer details. A customer reference is optional and labels the result only; it is not a model feature.

## Run locally

The checked-in `.python-version` targets Python 3.10. `scikit-learn==1.3.1` is pinned because the supplied model artifact was saved with that version.

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL Streamlit prints (normally [http://localhost:8501](http://localhost:8501)). To choose a port explicitly, append `--server.port 9000`.

## Project structure

| File | Responsibility |
| --- | --- |
| `streamlit_app.py` | Native Streamlit form, result panel, session state, and model warm-up |
| `app.py` | Framework-independent input validation, model loading, and scoring logic |
| `model_C=10.bin` | Supplied DictVectorizer and logistic-regression model artifact |
| `tests/test_app.py` | Prediction-logic tests and a Streamlit UI smoke test |

## Validation and tests

The scorer validates categorical values against the model’s known categories, rejects impossible service combinations, and bounds numeric inputs before scoring. Streamlit presents any validation feedback beside the result area without losing the user’s entered values.

```bash
python -m unittest discover -s tests -v
```

## Deployment

The included `Procfile` starts Streamlit in headless mode and binds to the platform-provided `PORT`:

```text
web: streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```

## Interview guide

Read [the project brief](docs/PROJECT_BRIEF.md) for a CEO-level value story, developer-level architecture explanation, short demo narrative, limitations, and next steps. It is deliberately honest about what this repository does and does not demonstrate.
