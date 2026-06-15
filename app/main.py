from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message":"Model Running"}