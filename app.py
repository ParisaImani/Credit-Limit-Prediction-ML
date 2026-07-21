from fastapi import FastAPI
import joblib
import pandas as pd
from iqr_clipper import IQRClipper

app = FastAPI()

model = joblib.load("C:\\Users\\Sharif\\CreditPrediction\\models\\credit_limit_model.pkl")


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "predicted_credit_limit": float(prediction[0])
    }