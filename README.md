# Immo Eliza - Machine Learning

## Description

This is a solo project to predict real estate prices in Belgium using machine learning. It builds on the dataset from my previous group project where we scraped and cleaned 15,746 property listings from Immovlan.

The goal is simple: give the model some property details (surface, bedrooms, location, etc.) and it predicts the price.

## Approach

### Preprocessing

The raw data needed some cleanup before the models could use it:

- **Missing values**: filled with the median for numbers, and "unknown" for text categories (using sklearn's `SimpleImputer`).
- **Scaling**: standardized all numeric features so they're on the same scale (using `StandardScaler`). Without this, the model would think surface area matters way more than garage just because the numbers are bigger.
- **Encoding**: turned text categories like "apartment" or "brussels" into 0/1 columns (using `OneHotEncoder`). Models can't read text, only numbers.

All preprocessing objects are saved so `predict.py` can apply the exact same transformations to new data.

### Features used

| Type | Features |
| --- | --- |
| Numeric | livable_surface, bedroom_count, total_surface, build_year, garage, terrace, swimming_pool, energy_consumption, preschool_distance, train_station_distance, supermarket_distance, nearest_city_distance, latitude, longitude |
| Categorical | property_type, province, property_state |
| Target | price |

### Models tested

Three models:

1. **Linear Regression** — the simplest model, draws a straight line through the data. Used as a baseline.
2. **Random Forest** — 100 decision trees that each see a random part of the data, then vote on the price.
3. **XGBoost** — gradient boosting, builds trees one after another where each new tree tries to fix the mistakes of the previous one.

All three were also evaluated with **5-fold cross-validation** to get more reliable scores, and Random Forest + XGBoost were tuned with **GridSearchCV** to find better settings.

## Results

| Model | Train R² | Test R² | Test MAE | Overfitting? |
| --- | --- | --- | --- | --- |
| Linear Regression | 0.52 | 0.04 | 138,267 EUR | Yes (R² gap: 0.47) |
| Random Forest | 0.95 | 0.78 | 93,957 EUR | Some (R² gap: 0.17) |
| XGBoost | 0.96 | 0.76 | 93,745 EUR | Some (R² gap: 0.19) |
| **Random Forest (tuned)** | 0.95 | 0.79 | 93,615 EUR | Some (R² gap: 0.16) |
| **XGBoost (tuned)** | 0.92 | 0.78 | 92,162 EUR | Less (R² gap: 0.14) |

**Best model: Random Forest (tuned)** with R² = 0.79 on test data.

Best hyperparameters found by GridSearchCV:

- Random Forest: 200 trees, no max depth limit
- XGBoost: learning_rate=0.05, max_depth=6, 300 boosting rounds

Linear regression was terrible on this dataset (R² = 0.04 on test), which makes sense because property prices don't follow a straight line — they depend on many factors in non-linear ways.

Random Forest and XGBoost both show some overfitting (train scores way higher than test scores), but their test performance is decent. After hyperparameter tuning, XGBoost's overfitting dropped noticeably (R² gap went from 0.19 to 0.14). The model is off by about 92-94k EUR on average, which could still be improved with more data or even more tunning, or filtering some more feature of the data.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Train the models:

```bash
python train.py
```

This will preprocess the data, train all 3 models, print the evaluation scores, and save everything to the `model/` folder.

Predict with dummy data:

```bash
python predict.py
```

This loads the best saved model and predicts prices for some test properties.

## Repository Structure

```text
immo-eliza-ml/
  .gitignore
  requirements.txt
  README.md
  train.py          # preprocessing + training + evaluation
  predict.py         # load model and predict on new data
  model/             # saved models and preprocessors (.joblib files)
```

## Timeline

| Day | Work |
| --- | --- |
| Monday | Started the project, explored the dataset, reviewed the teacher's ML notebooks |
| Tuesday | Created the Github repo, started coding the training pipeline |
| Wednesday | Main coding day, built preprocessing, training, and evaluation |
| Thursday | Cleaning up code, optimizing, adding comments |
| Friday | Working on STAR method presentation, final review |
