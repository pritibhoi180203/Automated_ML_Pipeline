from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    Fare: float

@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API"
    }

@app.post("/predict")
def predict(passenger: Passenger):

    features = [[
        passenger.Pclass,
        passenger.Sex,
        passenger.Age,
        passenger.Fare
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0]),
        "result": "Survived" if prediction[0] == 1 else "Not Survived"
    }