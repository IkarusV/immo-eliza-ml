import joblib
import pandas as pd
import os

# grab the preprocessing function and feature lists from train.py
from train import preprocess_new_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def load_model(model_name="best_model"):
    """Load a saved model from the model folder.

    Input:
        model_name: filename without extension (default: "best_model").
    """
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no model found at {path}, run train.py first")
    # joblib.load rebuilds the exact same sklearn object that was saved
    model = joblib.load(path)
    print(f"loaded model from {path}")
    return model


def load_preprocessors():
    """Load the saved preprocessing objects (imputer, scaler, encoder)."""
    path = os.path.join(MODEL_DIR, "preprocessors.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no preprocessors found at {path}, run train.py first")
    preprocessors = joblib.load(path)
    print(f"loaded preprocessors from {path}")
    return preprocessors


def predict(model, preprocessors, data):
    """Predict prices for one or more properties.

    Input:
        model: trained sklearn model.
        preprocessors: dict from load_preprocessors().
        data: dict or list of dicts with property info.
    """
    # turn dict(s) into a dataframe so the model can read it
    df = pd.DataFrame(data if isinstance(data, list) else [data])

    # apply the same preprocessing that was used during training
    X = preprocess_new_data(df, preprocessors)

    # get the predicted prices
    predictions = model.predict(X)
    return predictions


def main():
    # load the best model and the preprocessing objects we saved during training
    model = load_model("best_model")
    preprocessors = load_preprocessors()

    # some made-up properties to test with
    test_properties = [
        {
            "property_type": "apartment",
            "province": "brussels",
            "property_state": "Good",
            "livable_surface": 75,
            "bedroom_count": 2,
            "total_surface": 75,
            "build_year": 2000,
            "garage": 0,
            "terrace": 1,
            "swimming_pool": 0,
            "energy_consumption_kWh/m2/year": 200,
            "preschool_distance_m": 500,
            "train_station_distance_m": 800,
            "supermarket_distance_m": 300,
            "nearest_city_distance_km": 2.0,
            "latitude": 50.85,
            "longitude": 4.35,
        },
        {
            "property_type": "house",
            "province": "antwerp",
            "property_state": "Normal",
            "livable_surface": 180,
            "bedroom_count": 4,
            "total_surface": 400,
            "build_year": 1990,
            "garage": 1,
            "terrace": 1,
            "swimming_pool": 0,
            "energy_consumption_kWh/m2/year": 350,
            "preschool_distance_m": 1200,
            "train_station_distance_m": 2000,
            "supermarket_distance_m": 600,
            "nearest_city_distance_km": 10.0,
            "latitude": 51.22,
            "longitude": 4.40,
        },
    ]

    # run predictions and print results
    predictions = predict(model, preprocessors, test_properties)

    print("\n--- predictions ---")

    for i, prop in enumerate(test_properties):
        print(f"\n{prop['property_type']} in {prop['province']}")
        print(f"  {prop['livable_surface']}m2, {prop['bedroom_count']} bedrooms")
        print(f"  predicted price: {predictions[i]:,.0f} EUR")


if __name__ == "__main__":
    main()
