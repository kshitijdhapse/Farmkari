import uvicorn
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the pickled model
with open("crop_predictor.pkl", "rb") as f:
    crop_predictor = pickle.load(f)

# Define the Pydantic model for input data
class Crops(BaseModel):
    N: float
    P: float
    K: float
    temp: float
    hum: float
    pH: float
    rain: float

@app.get('/')
def root():
    return {'message': 'Welcome to the Crop Prediction API'}

@app.post('/predict', response_model=dict)
def predict_crop(data: Crops):
    """Route to make predictions using the model."""
    try:
        # Access data attributes directly from the Pydantic model
        prediction = crop_predictor.predict([[data.N, data.P, data.K, data.temp, data.hum, data.pH, data.rain]])
        return {'prediction': prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
