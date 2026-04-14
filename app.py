from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load model + features
with open("model.pkl", "rb") as f:
    model, selected_features = pickle.load(f)

@app.post("/predict")
def predict(data: dict):

    # Convert input into dataframe
    input_df = pd.DataFrame([data])

    # Ensure correct column order
    input_df = input_df[selected_features]

    prediction = model.predict(input_df)[0]

    labels = ["Low", "Medium", "High"]

    return {"burnout_level": labels[prediction]}