from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from typing import List, Dict

# Load model and class names
try:
    model = joblib.load("model.joblib")
    class_names = ["Benign", "Malignant"]  # 0=Benign, 1=Malignant
except Exception as e:
    print(f"Error loading model: {e}")
    raise

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Breast Cancer model API"}


@app.post("/predict")
async def predict(data: Dict[str, List[float]]):
    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return {"predicted_class": class_names[prediction]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
