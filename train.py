import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# config

DATA_PATH = r"C:\Aithing\MyownCoddedthing\AGPT CODEX\Becode\PreparationDatanalysis\immo-eliza-turtles-analysis\data\cleaned\clean_dataframe.json"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

TARGET = "price"

NUMERIC_FEATURES = [
    "livable_surface",
    "bedroom_count",
    "total_surface",
    "build_year",
    "garage",
    "terrace",
    "swimming_pool",
    "energy_consumption_kWh/m2/year",
    "preschool_distance_m",
    "train_station_distance_m",
    "supermarket_distance_m",
    "nearest_city_distance_km",
    "latitude",
    "longitude",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "province",
    "property_state",
]


# Step 1 load data

def load_data(path):
    """Read the json file into a dataframe.

    Input:
        path: string path to the json file.
    """
    df = pd.read_json(path)
    print(f"loaded {len(df)} rows")
    return df


# Step 2 preprocess data

def preprocess_data(df):
    """Clean up the dataframe so models can use it.

    Input:
        df: raw pandas DataFrame.
    Returns:
        X, y, preprocessors dict.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    # fill missing numbers with the median
    num_imputer = SimpleImputer(strategy="median")
    X[NUMERIC_FEATURES] = num_imputer.fit_transform(X[NUMERIC_FEATURES])

    # fill missing text with "unknown"
    cat_imputer = SimpleImputer(strategy="constant", fill_value="unknown")
    X[CATEGORICAL_FEATURES] = cat_imputer.fit_transform(X[CATEGORICAL_FEATURES])

    # scale numbers so they're all on the same range
    scaler = StandardScaler()
    X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])

    # turn text categories into 0/1 columns
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_cats = encoder.fit_transform(X[CATEGORICAL_FEATURES])

    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_columns, index=X.index)

    # swap old text columns for the new 0/1 columns
    X = X.drop(columns=CATEGORICAL_FEATURES)
    X = pd.concat([X, encoded_df], axis=1)

    # save these so predict.py can reuse the same transformations
    preprocessors = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "scaler": scaler,
        "encoder": encoder,
    }

    return X, y, preprocessors


def preprocess_new_data(df, preprocessors):
    """Same preprocessing but for new data. Uses transform() instead of fit_transform().

    Input:
        df: pandas DataFrame with the same columns as training data.
        preprocessors: dict of fitted preprocessing objects from preprocess_data().
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    # same 3 steps as preprocess_data but with .transform() (no re-learning, just apply)
    X[NUMERIC_FEATURES] = preprocessors["num_imputer"].transform(X[NUMERIC_FEATURES])
    X[CATEGORICAL_FEATURES] = preprocessors["cat_imputer"].transform(X[CATEGORICAL_FEATURES])
    X[NUMERIC_FEATURES] = preprocessors["scaler"].transform(X[NUMERIC_FEATURES])

    # turn text categories into 0/1 columns using the same encoding as training
    encoded_cats = preprocessors["encoder"].transform(X[CATEGORICAL_FEATURES])
    encoded_columns = preprocessors["encoder"].get_feature_names_out(CATEGORICAL_FEATURES)
    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_columns, index=X.index)

    # swap old text columns for the new 0/1 columns
    X = X.drop(columns=CATEGORICAL_FEATURES)
    X = pd.concat([X, encoded_df], axis=1)

    return X


# Step 3 train model

def train_model(model, X_train, y_train):
    """Fit the model on training data.

    Input:
        model: sklearn model object.
        X_train: features dataframe.
        y_train: target series.
    """
    model.fit(X_train, y_train)
    return model


# Step 4 evaluate model

def evaluate_model(model, X, y):
    """Predict on X, compare to real y, return the scores.

    Input:
        model: trained sklearn model.
        X: features dataframe.
        y: target series.
    """
    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# Step 5 cross-validation (test on 5 different splits instead of just 1)

def cross_validate_models(models, X_train, y_train):
    """Run 5-fold cross-validation on each model and print the scores.

    Input:
        models: dict of {name: sklearn model}.
        X_train: features dataframe.
        y_train: target series.
    """
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION (5-fold)")
    print("=" * 50)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        print(f"\n{name}:")
        print(f"  scores per fold: {[f'{s:.4f}' for s in scores]}")
        print(f"  mean R2: {scores.mean():.4f} (+/- {scores.std():.4f})")


# Step 6 hyperparameter tuning (testing different settings on RF and XGBoost)

def tune_hyperparameters(X_train, y_train):
    """Try different parameter combos with GridSearchCV and return the best models.

    Input:
        X_train: features dataframe.
        y_train: target series.
    Returns:
        dict of {name: best trained model}.
    """
    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING")
    print("=" * 50)

    # --- random forest ---
    print("\ntuning random_forest...")

    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
    }

    rf_grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=rf_params,
        scoring="r2",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )

    rf_grid.fit(X_train, y_train)

    print(f"  best params: {rf_grid.best_params_}")
    print(f"  best R2: {rf_grid.best_score_:.4f}")

    # --- xgboost ---
    print("\ntuning xgboost...")

    xgb_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.05, 0.1, 0.2],
    }

    xgb_grid = GridSearchCV(
        estimator=XGBRegressor(random_state=42, verbosity=0),
        param_grid=xgb_params,
        scoring="r2",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )

    xgb_grid.fit(X_train, y_train)

    print(f"  best params: {xgb_grid.best_params_}")
    print(f"  best R2: {xgb_grid.best_score_:.4f}")

    tuned_models = {
        "random_forest_tuned": rf_grid.best_estimator_,
        "xgboost_tuned": xgb_grid.best_estimator_,
    }

    return tuned_models


# Step 7 save model

def save_model(obj, name):
    """Save a python object to a .joblib file.

    Input:
        obj: any python object (model, scaler, etc).
        name: filename without extension.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(obj, path)
    print(f"saved to {path}")


# main

def main():
    df = load_data(DATA_PATH)

    # drop price_per_m2 because it's calculated from price (would let the model cheat)
    if "price_per_m2" in df.columns:
        df = df.drop(columns=["price_per_m2"])

    X, y, preprocessors = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"train: {len(X_train)} rows, test: {len(X_test)} rows")

    # create our 3 models
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgboost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }

    # train and evaluate each model
    results = {}

    for name, model in models.items():
        print(f"\ntraining {name}...")

        train_model(model, X_train, y_train)

        train_metrics = evaluate_model(model, X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)

        results[name] = {
            "train": train_metrics,
            "test": test_metrics,
            "model": model,
        }

        print(f"  TRAIN -> R2: {train_metrics['R2']:.4f} | MAE: {train_metrics['MAE']:.0f} | RMSE: {train_metrics['RMSE']:.0f}")
        print(f"  TEST  -> R2: {test_metrics['R2']:.4f} | MAE: {test_metrics['MAE']:.0f} | RMSE: {test_metrics['RMSE']:.0f}")

        # if train is way better than test, the model memorized instead of learning
        r2_gap = train_metrics["R2"] - test_metrics["R2"]
        if r2_gap > 0.1:
            print(f"  WARNING: probably overfitting (R2 gap: {r2_gap:.4f})")
        else:
            print(f"  looks ok, no major overfitting (R2 gap: {r2_gap:.4f})")

    best_name = max(results, key=lambda k: results[k]["test"]["R2"])
    print(f"\n--- best model (before tuning): {best_name} (R2: {results[best_name]['test']['R2']:.4f}) ---")

    # cross-validation on all 3 models
    cross_validate_models(models, X_train, y_train)

    # hyperparameter tuning on RF and XGBoost
    tuned_models = tune_hyperparameters(X_train, y_train)

    # evaluate tuned models on test set
    print("\n" + "=" * 50)
    print("TUNED MODELS vs ORIGINAL")
    print("=" * 50)

    for name, model in tuned_models.items():
        test_metrics = evaluate_model(model, X_test, y_test)
        train_metrics = evaluate_model(model, X_train, y_train)

        results[name] = {
            "train": train_metrics,
            "test": test_metrics,
            "model": model,
        }

        print(f"\n{name}:")
        print(f"  TRAIN -> R2: {train_metrics['R2']:.4f} | MAE: {train_metrics['MAE']:.0f} | RMSE: {train_metrics['RMSE']:.0f}")
        print(f"  TEST  -> R2: {test_metrics['R2']:.4f} | MAE: {test_metrics['MAE']:.0f} | RMSE: {test_metrics['RMSE']:.0f}")

    # pick the best overall model
    best_name = max(results, key=lambda k: results[k]["test"]["R2"])
    print(f"\n--- overall best model: {best_name} (R2: {results[best_name]['test']['R2']:.4f}) ---")

    # save everything
    save_model(preprocessors, "preprocessors")

    for name, result in results.items():
        save_model(result["model"], name)

    save_model(results[best_name]["model"], "best_model")


if __name__ == "__main__":
    main()
