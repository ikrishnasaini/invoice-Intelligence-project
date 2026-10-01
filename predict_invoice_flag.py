import joblib
import pandas as pd
from pathlib import Path

# Use project-relative path so renaming top-level folder doesn't break loading
MODEL_PATH = Path(__file__).parent / "Classification Model Training" / "models" / "predict_flag_invoice.pkl"
FEATURE_COLUMNS = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_invoice_flag(input_data):
    model = load_model()
    input_df = pd.DataFrame(input_data)

    missing_columns = [col for col in FEATURE_COLUMNS if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required features: {missing_columns}")

    input_df = input_df[FEATURE_COLUMNS]
    predictions = model.predict(input_df.values)
    input_df = input_df.copy()
    input_df["Predicted_Flag"] = predictions.round().astype(int)
    return input_df


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [100],
        "invoice_dollars": [2500.0],
        "Freight": [150.0],
        "total_item_quantity": [95],
        "total_item_dollars": [2350.0],
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)

