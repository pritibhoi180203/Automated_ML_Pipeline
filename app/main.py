from fastapi import FastAPI
import joblib

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Automated ML Pipeline Running Successfully"}