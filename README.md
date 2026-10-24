# House Price Prediction

A machine learning application that estimates house prices from property characteristics using Python and Scikit-learn. The project combines reusable data-processing pipelines, regression model comparison, and a Streamlit interface for individual and batch predictions.

> **Dataset:** The included Pakistan-style housing data is synthetic and intended for demonstration. Predictions and evaluation results do not represent actual property values or housing-market performance.

## Overview

This project demonstrates a complete regression workflow, from validating raw data to evaluating a trained model and making predictions through an interactive application.

The model uses property features such as covered area, city, bedrooms, bathrooms, age, condition, and distance from the city center to estimate a price in Pakistani rupees.

## Features

- **Data preparation:** Validate schemas, convert data types, remove duplicate records, and handle invalid or missing values.
- **Reusable pipelines:** Combine imputation, scaling, categorical encoding, and regression.
- **Model comparison:** Evaluate a median-price baseline, Ridge regression, and Random Forest regression.
- **Reliable evaluation:** Select the model using training-only cross-validation before evaluating the held-out test set.
- **Interactive predictions:** Estimate prices for individual properties or upload a CSV for batch predictions.
- **Dataset exploration:** Filter records, inspect distributions, and download filtered data.
- **Evaluation reports:** Review MAE, RMSE, R², prediction plots, and residuals.

## Technologies

| Area | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib |
| Application interface | Streamlit |
| Model persistence | Joblib |
| Testing | Pytest |

## Evaluation Results

The included experiment used **1,500 cleaned synthetic property records**, with **1,200 for training** and **300 for testing**.

| Metric | Result |
|---|---:|
| Selected model | Random Forest |
| Test MAE | PKR 2,783,639 |
| Test RMSE | PKR 4,140,835 |
| Test R² | 0.906 |
| Median baseline MAE | PKR 10,699,467 |

Random Forest achieved the lowest mean absolute error during five-fold cross-validation on the training set.

These results demonstrate performance on the included synthetic dataset. They do not establish predictive performance on real property transactions.

![Actual versus predicted prices and residual distribution](reports/evaluation.png)

## Project Structure

```text
house-price-prediction/
├── app.py
├── train.py
├── generate_data.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   └── modeling.py
├── data/
│   ├── house_prices.csv
│   └── prediction_template.csv
├── models/
│   └── house_price_model.joblib  # Created during training
├── reports/
│   ├── metrics.json
│   ├── evaluation.png
│   └── test_predictions.csv
├── notebooks/
│   └── house_analysis.ipynb
├── tests/
│   └── test_data.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── requirements-dev.txt
├── requirements-tested.txt
├── VALIDATION.md
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12 or 3.13
- pip
- VS Code or another Python editor

No Node.js installation or separate frontend server is required.

### Windows Setup

Open PowerShell in the project directory and run each command after the previous one completes:

```powershell
py -3.13 -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe train.py

.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS and Linux Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python train.py
streamlit run app.py
```

Open the address displayed in the terminal, usually **http://localhost:8501**.

Training creates the model artifact and evaluation reports. On subsequent launches, run the Streamlit command from the same environment.

## Application Pages

| Page | Purpose |
|---|---|
| Predict a House | Enter property details and view a price estimate |
| Model Evaluation | Review test metrics, cross-validation results, and diagnostic plots |
| Dataset Explorer | Inspect, filter, and export property records |
| Batch Prediction | Upload property features and download predictions |

A batch prediction template is available at `data/prediction_template.csv`.

## Dataset

Each record represents one property. Prices use **PKR**, covered area uses **square feet**, and distance uses **kilometers**.

| Column | Description |
|---|---|
| `property_id` | Optional identifier; excluded from model features |
| `area_sqft` | Covered area |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_years` | Property age |
| `parking_spaces` | Number of parking spaces |
| `distance_to_center_km` | Distance from the city center |
| `city` | Property location |
| `property_type` | Property category, such as House or Apartment |
| `condition` | Property condition |
| `price_pkr` | Target price; required for training only |

The demonstration dataset includes Islamabad, Lahore, Karachi, Peshawar, and Mardan. It intentionally contains missing values, duplicate records, and an invalid age to demonstrate data cleaning.

## Modeling Approach

1. **Validate and clean:** Check required columns, normalize values, remove duplicate feature vectors, and discard invalid targets.
2. **Split the data:** Reserve 20% of the cleaned records for final testing.
3. **Build pipelines:** Apply numeric imputation and scaling, categorical imputation and encoding, and regression.
4. **Compare candidates:** Use five-fold cross-validation within the training set.
5. **Select and evaluate:** Fit the best candidate on the training split and evaluate it on the held-out test set.
6. **Save outputs:** Export the model, metrics, test predictions, and diagnostic plots.

Imputers, scalers, and encoders are fitted only on training data within each fold. The test set is not used for model selection.

## Using a Real Dataset

Prepare a CSV that matches the documented schema and units. Record its source, collection period, license, and any transformations applied.

Save it as `data/real_houses.csv`, then run:

```powershell
.\.venv\Scripts\python.exe train.py `
  --data data/real_houses.csv `
  --source-label "Dataset source and collection period; prices in PKR"
```

Restart the application after training.

- Uploading a CSV in the Dataset Explorer previews the data; it does not retrain the model.
- Update `CATEGORIES` in `src/data.py` if your dataset uses different cities, property types, or conditions.
- Review repeated property listings and conflicting prices before training.
- Consider chronological or grouped splits when records are related by sale date, property, or neighborhood.

## Testing

Install the development dependencies and run the tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover schema validation, invalid numeric inputs, prediction-row preservation, unseen categories, and training-only imputation.

See `VALIDATION.md` for the checks performed and their limitations.

## Deployment

The application can run on a Python host that supports Streamlit.

Before deployment:

1. Train the model using the Python and Scikit-learn versions intended for the host.
2. Include the trusted model artifact and matching evaluation reports.
3. Pin the tested dependency versions.
4. Set the application entry point to `app.py`.

Model artifacts are excluded from Git by default. If your deployment requires the model in the repository, add it explicitly:

```bash
git add -f models/house_price_model.joblib
```

Only load trusted Joblib artifacts. Do not accept arbitrary model files from users.

## Limitations

- The bundled data is synthetic and follows a programmed price relationship.
- Important real-world factors, including neighborhood detail, sale date, amenities, and market conditions, are not modeled.
- Predictions outside the training distribution may be unreliable.
- MAE describes average error; it is not a confidence interval for an individual prediction.
- R² is a regression metric, not classification accuracy.

This project is an educational demonstration and should not be used as a property appraisal tool.

## Future Improvements

- Train on licensed real property transaction data.
- Add geographic, neighborhood, and time-based features.
- Compare tuned gradient-boosting models.
- Evaluate errors across locations and price segments.
- Develop and assess prediction intervals.
- Track dataset versions and model experiments.

## References

- [Scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
