# main.py

import pandas as pd
import joblib

# Load your trained model only
model = joblib.load("used_car_price_model.pkl")

def predict_price(input_data: dict) -> float:
    """
    Predict the car price using the trained model.
    (No model_features.pkl and no label_encoders.pkl needed)
    """

    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # --------- Handle Car_Age if needed ----------
    if "CarAge" in df.columns:
        df = df.rename(columns={"CarAge": "Car_Age"})

    # --------- Encode categorical columns ----------
    # Same simple encoding approach as before
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    # Make prediction
    prediction = model.predict(df)[0]
    return round(prediction, 2)


