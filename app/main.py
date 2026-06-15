from fastapi import FastAPI
import joblib

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Version 2 deployed successfully"}