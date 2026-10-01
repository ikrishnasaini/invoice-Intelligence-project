import joblib
import pandas as pd
from pathlib import Path

# Use a project-relative path so renaming the top-level folder doesn't break loading.
MODEL_PATH = Path(__file__).parent / "Regression model training" / "models" / "predict_freight_model.pkl"

def load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # joblib.load accepts a path-like directly
    model = joblib.load(model_path)
    return model

FEATURE_COLUMNS = ["Dollars"]

def predict_freight_cost(input_data):
    model = load_model()
    input_df = pd.DataFrame(input_data)

    missing_columns = [col for col in FEATURE_COLUMNS if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required features: {missing_columns}")

    input_df = input_df[FEATURE_COLUMNS]
    predictions = model.predict(input_df)
    result_df = input_df.copy()
    result_df["Predicted_Freight"] = predictions.round().astype(float)
    return result_df

if __name__ == "__main__":
    sample_data = {
        "Quantity": [1200, 800, 400, 200],
        "Dollars": [18500.0, 9000.0, 3000.0, 2000.0],
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)
