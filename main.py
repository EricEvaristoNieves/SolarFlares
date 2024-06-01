# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
from pydantic.typing import Literal
import json

app = FastAPI()

class Item(BaseModel):
    modified_Zurich_class_B: float
    modified_Zurich_class_C: float
    modified_Zurich_class_D: float
    modified_Zurich_class_E: float
    modified_Zurich_class_F: float
    modified_Zurich_class_H: float
    largest_spot_size_A: float
    largest_spot_size_H: float
    largest_spot_size_K: float
    largest_spot_size_R: float
    largest_spot_size_S: float
    largest_spot_size_X: float
    spot_distribution_C: float
    spot_distribution_I: float
    spot_distribution_O: float
    spot_distribution_X: float
    activity: float
    evolution: float
    previous_24_hour_flare_activity: float
    historically_complex: float
    became_complex_on_this_pass: float
    area: float
    area_of_largest_spot: float        

@app.post("/predict")
async def predict(features: Item):
    try:
        # Load your pre-trained machine learning model
        model = joblib.load('src/models/model.joblib')
        
        #data = json.loads(features)        
        #df = pd.json_normalize(data)

        # Prepare input features for prediction        
        features_df = pd.DataFrame(features.__dict__, index=[0])

        # Make predictions
        prediction = model.predict(features_df)

        # Return the prediction as JSON response
        return {"predicted class": str(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.2", port=8000)
