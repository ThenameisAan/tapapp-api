# Tapapp/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd # Pandas is needed if your scaler expects a DataFrame with column names

app = FastAPI()

# --- Load the model and scaler when the API starts ---
MODEL_PATH = "rf_timetoline_model.joblib"
SCALER_PATH = "timetoline_scaler.joblib"
model = None
scaler = None

# Define the expected input features from Flutter
# IMPORTANT: The order here MUST match the order your Flutter app sends them
# AND the order your model was trained on in the notebook.
FEATURE_ORDER = [
    'BoatSpeed (m/s)', 'BoatAngle (deg)', 'WindSpeed (m/s)', 'WindAngle (deg)',
    'AccelX (m/s^2)', 'AccelY (m/s^2)', 'AccelZ (m/s^2)',
    'GyroX (rad/s)', 'GyroY (rad/s)', 'GyroZ (rad/s)',
    'MagX (uT)', 'MagY (uT)', 'MagZ (uT)'
]

@app.on_event("startup")
async def load_model_and_scaler():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("INFO:     Model and scaler loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR:    Model or scaler file not found. Searched for '{MODEL_PATH}' and '{SCALER_PATH}'.")
        print("          Ensure these files are in the same directory as main.py.")
        model = None # Ensure they are None if loading fails
        scaler = None
    except Exception as e:
        print(f"ERROR:    Error loading model or scaler: {e}")
        model = None
        scaler = None

class PredictionInput(BaseModel):
    # List of 13 feature values. Flutter should send them in the FEATURE_ORDER.
    # Allowing None for features that might sometimes be unavailable from sensors.
    features: list[float | None] 

class PredictionOutput(BaseModel):
    predicted_time_to_line_seconds: float | None # The model's output
    error_message: str | None = None # Optional error message

@app.get("/")
def read_root():
    return {"message": "TapApp API for Time to Burn Prediction is running"}

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

# You can keep your existing /calibrate_boat and /calculate_ttb endpoints if they are still needed
# for other functionalities or if you plan to use them later.
# If not, you can remove them to simplify the API.

# Example of how you might have other endpoints:
# @app.post("/calibrate_boat/")
# async def calibrate_boat_endpoint(data: dict):
#     # Your existing calibration logic
#     return {"message": "Calibration data received - placeholder"}

# @app.post("/calculate_ttb/")
# async def calculate_ttb_endpoint(data: dict):
#     # Your existing TTB calculation logic (if different from the raw prediction)
#     return {"time_to_burn": 120, "boat_speed": 5.0} # Placeholder
