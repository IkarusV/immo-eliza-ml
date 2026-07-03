# Immo Eliza - Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![pandas](https://img.shields.io/badge/pandas-1.5+-lightblue.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

## Description

This is my first machine learning project. The goal is to predict real estate prices in Belgium using different regression models. It builds on a dataset of **15,746 property listings** scraped from Immovlan during a previous group project. 

By providing details about a property (like the size, location, and number of rooms), the model estimates its market price in EUR.

---

## The Machine Learning Pipeline

### 1. Preprocessing
To prepare the raw scraped dataset for training, the pipeline performs the following steps:
*   **Handling Missing Values**:
    *   *Numerical features*: Filled with the **median** value of that column (using `SimpleImputer`). The median is used instead of the mean because property prices and sizes have large outliers that would skew the average.
    *   *Categorical features*: Filled with the text `"unknown"` so we don't have to discard rows with missing categorical descriptions (like property state).
*   **Feature Scaling**: Standardized numerical features using `StandardScaler` to bring them onto the same scale. This prevents the model from thinking a feature like distance (measured in thousands of meters) is automatically more important than the number of bedrooms just because the numbers are larger.
*   **Categorical Encoding**: Converted text categories into binary columns using `OneHotEncoder(handle_unknown="ignore")` so the models can read them.
*   **Leakage Prevention**: The fitted preprocessors are saved as a dictionary using `joblib`. This allows the prediction script to apply the exact same scaling and encoding parameters to new data without retraining them on new samples.

### 2. Features Used

| Type | Feature Name in Dataset | Description & Unit |
| :--- | :--- | :--- |
| **Target** | `price` | Selling price of the property (EUR) |
| **Numeric** | `livable_surface` | Living area (m²) |
| | `bedroom_count` | Number of bedrooms |
| | `total_surface` | Total land/property area (m²) |
| | `build_year` | Year of construction |
| | `garage` | Number/presence of garages |
| | `terrace` | Number/presence of terraces |
| | `swimming_pool` | Presence of a swimming pool (0 or 1) |
| | `energy_consumption_kWh/m2/year` | EPC energy consumption rating |
| | `preschool_distance_m` | Distance to the nearest preschool (meters) |
| | `train_station_distance_m` | Distance to the nearest train station (meters) |
| | `supermarket_distance_m` | Distance to the nearest supermarket (meters) |
| | `nearest_city_distance_km` | Distance to the nearest city center (kilometers) |
| | `latitude` | GPS latitude coordinate |
| | `longitude` | GPS longitude coordinate |
| **Categorical**| `property_type` | Type of property (e.g., house, apartment) |
| | `province` | Belgian province where it is located |
| | `property_state` | Condition of the property (e.g., Good, As new, To restore) |

---

## Model Evaluation & Results

I tested three different models:
1.  **Linear Regression**: Used as a baseline model.
2.  **Random Forest**: An ensemble of 100 decision trees voting on the final price.
3.  **XGBoost**: Gradient boosted trees that learn sequentially to correct errors.

To get reliable scores, all models were evaluated using **5-fold cross-validation**, and the Random Forest and XGBoost models were tuned using **GridSearchCV** to find the best settings.

### Performance Comparison

| Model | Train $R^2$ | Test $R^2$ | Test MAE | Overfitting Gap (Train - Test $R^2$) |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression | 0.52 | 0.04 | ~138,267 EUR | 0.48 |
| Random Forest | 0.95 | 0.78 | ~93,957 EUR | 0.17 |
| XGBoost | 0.96 | 0.76 | ~93,745 EUR | 0.20 |
| **Random Forest (Tuned)** | **0.95** | **0.79** | **~93,615 EUR** | **0.16** |
| **XGBoost (Tuned)** | **0.92** | **0.78** | **~92,162 EUR** | **0.14** |

*Note: The best overall model based on the highest Test $R^2$ score is the **Tuned Random Forest** ($R^2 = 0.79$), which is saved as `best_model.joblib`.*

### My Observations & Insights

*   **Why Linear Regression did so poorly**: The test $R^2$ was only 0.04. This is because real estate prices are highly non-linear. For example, location premium (latitude and longitude) isn't linear moving slightly north doesn't mean prices go up in a straight line; it depends on specific expensive cities and neighborhoods in the dataset.
*   **Overfitting**: Tree-based models tend to memorize the training data. Limiting hyper-parameters during tuning (like reducing tree depth in XGBoost to 6 and tuning learning rates) helped lower the overfitting gap from 0.20 to 0.14, making the model generalize much better.
*   **Typical Error**: The Mean Absolute Error (MAE) is around 92k–93k EUR. This is quite high, but this is partly because our dataset includes very expensive houses (over 1 million EUR) where the model's prediction errors are much larger.
*   **Data Limitations**: Many properties in our scraped dataset have `unknown` for `property_state` or missing construction years. Having more complete records on the property's condition would likely improve the prediction accuracy.

---

## Adapting to Other Datasets (Reusability)

This project was built with a step-by-step approach rather than utilizing a single compact Scikit-Learn `Pipeline`. Since this is a learning experience, keeping the steps (data loading, numeric imputation, categorical imputation, scaling, encoding) separated into explicit, sequential functions makes the code easy to read and understand. This design makes it simple to copy specific blocks or reuse the script structure for totally different prediction tasks (such as gaming console prices).

### What you would need to change:

1.  **Configure Features in `train.py`**:
    At the top of `train.py`, update the config variables:
    *   Change `DATA_PATH` to point to your new dataset (e.g., a CSV or JSON of gaming console sales).
    *   Change `TARGET` to your new target column (e.g., `price`).
    *   List your new numeric features in `NUMERIC_FEATURES` (e.g., `release_year`, `storage_GB`, `controller_count`).
    *   List your new categorical features in `CATEGORICAL_FEATURES` (e.g., `brand`, `color`, `condition`).
2.  **Imputation & Preprocessing**:
    *   In `preprocess_data()`, if your new dataset does not need median imputation (e.g., if there are no outliers and you want to use the mean or most frequent value), adjust the `SimpleImputer(strategy="...")` parameter.
    *   If you don't want to use "unknown" for missing text, adjust the categorical imputer `fill_value`.
3.  **Adjust Test Properties in `predict.py`**:
    *   In the `main()` function of `predict.py`, update the `test_properties` list of dictionaries with dummy examples that use the exact same feature keys as your new dataset.
    *   Update the print statements at the end of `predict.py` to display the appropriate output terms for your new task.

---

## Installation & Setup

1.  **Clone the repository** and navigate to the project directory.
2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    ```
3.  **Activate the virtual environment**:
    *   **Windows (PowerShell)**:
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    *   **Windows (Command Prompt)**:
        ```cmd
        .venv\Scripts\activate.bat
        ```
    *   **macOS / Linux**:
        ```bash
        source .venv/bin/activate
        ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Training and Evaluation
Run the training script to process the data, perform model training, run cross-validation and hyperparameter search, and save the resulting models:
```bash
python train.py
```
This script creates the `model/` folder and saves:
*   `preprocessors.joblib` (our imputers, scaler, and encoder)
*   `best_model.joblib` (the model with the highest $R^2$ score on the test set)
*   Individual trained models (e.g., `linear_regression.joblib`, `xgboost_tuned.joblib`, etc.)

### Making Predictions
To make predictions on new properties, run:
```bash
python predict.py
```
This loads `best_model.joblib` and `preprocessors.joblib` to estimate prices for a list of dummy properties.

---

## Repository Structure

```text
immo-eliza-ml/
├── model/                 # Saved models and preprocessors (.joblib files, ignored by git)
├── predict.py             # Inference script to predict prices for new properties
├── train.py               # Main training script (preprocessing, training, tuning, and evaluation)
├── .gitignore             # Git ignore patterns
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```
