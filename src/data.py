"""Schema validation and deterministic cleaning; imputation belongs in the pipeline."""
import numpy as np
import pandas as pd

NUMERIC = ['area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'parking_spaces', 'distance_to_center_km']
CATEGORICAL = ['city', 'property_type', 'condition']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'price_pkr'
BOUNDS = {'area_sqft': (200, 20000), 'bedrooms': (1, 20), 'bathrooms': (1, 20),
          'age_years': (0, 150), 'parking_spaces': (0, 10), 'distance_to_center_km': (0, 150)}
CATEGORIES = {'city': ['Islamabad', 'Lahore', 'Karachi', 'Peshawar', 'Mardan'],
              'property_type': ['House', 'Apartment'], 'condition': ['Fair', 'Good', 'Excellent']}


def clean_data(raw: pd.DataFrame, training: bool = True) -> tuple[pd.DataFrame, dict]:
    """Reject incompatible schemas and clean values without learning dataset statistics.

    Training drops duplicate houses and rows without valid targets. Prediction preserves
    row order and count. Numeric out-of-range values become missing for train-only imputation.
    """
    data = raw.copy()
    data.columns = [str(c).strip().lower().replace(' ', '_') for c in data.columns]
    if data.columns.duplicated().any():
        raise ValueError('Column names must be unique after normalization.')
    required = FEATURES + ([TARGET] if training else [])
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    if data.empty:
        raise ValueError('The CSV has no data rows.')
    original_count = len(data)
    for column, (low, high) in BOUNDS.items():
        values = pd.to_numeric(data[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        values = values.where(values.between(low, high))
        if column in ['bedrooms', 'bathrooms', 'parking_spaces']:
            values = values.where(values.isna() | (values % 1 == 0))
        data[column] = values
    for column in CATEGORICAL:
        values = data[column].astype('string').str.strip().str.title()
        data[column] = values.replace('', pd.NA).astype(object).where(values.notna(), np.nan)
    missing_features = int(data[FEATURES].isna().sum().sum())
    duplicates = invalid_targets = 0
    if training:
        # Same feature vector cannot appear on both sides of a random split.
        before = len(data)
        data = data.drop_duplicates(subset=FEATURES, keep='first')
        duplicates = before - len(data)
        data[TARGET] = pd.to_numeric(data[TARGET], errors='coerce')
        valid = np.isfinite(data[TARGET]) & (data[TARGET] > 0)
        invalid_targets = int((~valid).sum())
        data = data.loc[valid]
        if len(data) < 50:
            raise ValueError('At least 50 valid, distinct houses are required for training.')
        if data[FEATURES].isna().all().any():
            raise ValueError('A required feature is entirely missing or invalid.')
    report = {'input_rows': original_count, 'output_rows': len(data),
              'duplicates_removed': duplicates, 'invalid_targets_removed': invalid_targets,
              'missing_or_invalid_feature_cells': missing_features}
    return data.reset_index(drop=True), report
