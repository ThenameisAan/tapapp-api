from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI()

# --- Load the model and scaler when the API starts ---
# IMPORTANT: The code will look in multiple locations
model = None
scaler = None

# Define the expected input features from Flutter
FEATURE_ORDER = [
    'BoatSpeed (m/s)', 'BoatAngle (deg)', 'WindSpeed (m/s)', 'WindAngle (deg)',
    'AccelX (m/s^2)', 'AccelY (m/s^2)', 'AccelZ (m/s^2)',
    'GyroX (rad/s)', 'GyroY (rad/s)', 'GyroZ (rad/s)',
    'MagX (uT)', 'MagY (uT)', 'MagZ (uT)'
]

@app.on_event("startup")
async def load_model_and_scaler():
    global model, scaler
    
    # List of possible locations for model files
    model_paths = [
        "rf_timetoline_model.joblib",  # Current directory (Render default)
        "app/rf_timetoline_model.joblib",  # app subdirectory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_timetoline_model.joblib"),  # Same directory as this file
    ]
    
    scaler_paths = [
        "timetoline_scaler.joblib",  # Current directory (Render default)
        "app/timetoline_scaler.joblib",  # app subdirectory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "timetoline_scaler.joblib"),  # Same directory as this file
    ]
    
    # Print the current directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"This file is located at: {os.path.abspath(__file__)}")
    
    # Try loading the model
    for path in model_paths:
        try:
            print(f"Attempting to load model from: {path}")
            model = joblib.load(path)
            print(f"SUCCESS: Model loaded from {path}")
            break
        except Exception as e:
            print(f"FAILED: Could not load model from {path}: {e}")
    
    # Try loading the scaler
    for path in scaler_paths:
        try:
            print(f"Attempting to load scaler from: {path}")
            scaler = joblib.load(path)
            print(f"SUCCESS: Scaler loaded from {path}")
            break
        except Exception as e:
            print(f"FAILED: Could not load scaler from {path}: {e}")
    
    # Final check
    if model is None:
        print("ERROR: Could not load model from any location")
    if scaler is None:
        print("ERROR: Could not load scaler from any location")

class PredictionInput(BaseModel):
    # List of 13 feature values. Flutter should send them in the FEATURE_ORDER.
    # Allowing None for features that might sometimes be unavailable from sensors.
    features: list[float | None] 

class PredictionOutput(BaseModel):
    predicted_time_to_line_seconds: float | None # The model's output
    error_message: str | None = None # Optional error message

@app.get("/")
def read_root():
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    
    return {
        "message": "TapApp API for Time to Burn Prediction is running",
        "model_status": model_status,
        "scaler_status": scaler_status
    }

@app.post("/predict_time_to_line_raw/", response_model=PredictionOutput)
async def predict_raw_time_to_line(input_data: PredictionInput):
    if model is None or scaler is None:
        print("ERROR:    Model or scaler not available for prediction.")
        raise HTTPException(status_code=503, detail="Model or scaler not loaded on server.")

    if len(input_data.features) != len(FEATURE_ORDER):
        error_msg = f"Expected {len(FEATURE_ORDER)} features, got {len(input_data.features)}"
        print(f"ERROR:    {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    # Handle potential None values in features before scaling.
    # A simple strategy is to replace None with 0.0.
    # This MUST match how missing values were handled (if at all) during model training.
    # If your model was trained on data with no NaNs after preprocessing, ensure no NaNs are passed here.
    processed_features = []
    for i, val in enumerate(input_data.features):
        if val is None:
            print(f"Warning: Feature '{FEATURE_ORDER[i]}' is None, using 0.0 as fallback.")
            processed_features.append(0.0) 
        else:
            processed_features.append(val)
    
    raw_features_array = np.array(processed_features).reshape(1, -1)

    try:
        # Create a DataFrame with correct column names for the scaler
        # This is crucial if your scaler was fit on a DataFrame and is sensitive to feature order/names.
        features_df = pd.DataFrame(raw_features_array, columns=FEATURE_ORDER)
        
        # Scale the features
        scaled_features_array = scaler.transform(features_df)
        
        # Make a prediction
        prediction = model.predict(scaled_features_array)
        predicted_time_seconds = float(prediction[0])

        # The model predicts "TimeToBurn (s)" which is DistanceToLine / BoatSpeed.
        # This is effectively the "predicted time to reach line from current position at current speed".
        # Ensure it's not negative.
        if predicted_time_seconds < 0:
            predicted_time_seconds = 0.0 
        
        print(f"INFO:     Prediction successful: {predicted_time_seconds:.2f} seconds")
        return PredictionOutput(predicted_time_to_line_seconds=predicted_time_seconds)

    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(f"ERROR:    {error_msg}")
        # Consider logging the input_data.features as well for debugging
        print(f"DEBUG:    Input features received: {input_data.features}")
        raise HTTPException(status_code=500, detail=error_msg)
