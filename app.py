from flask import Flask
import pickle
# import matplotlib.plot as plt

model_file = "model_C=10.bin"
with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

customer = {
    "customerid": "0111-klbqg",
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
    "totalcharges": 2861.45
}

app = Flask("app")


@app.route("/predict", methods=["GET"])
def predict():
    user_feature = dv.transform([customer])
    pred_label = model.predict_proba(user_feature)[0, 1]
    return {
        "probability": pred_label,
        "churn": bool(pred_label >= 0.5)
    }


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9000)
