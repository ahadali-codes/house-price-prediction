import numpy as np
import pandas as pd
import pytest
from src.data import FEATURES, clean_data
from src.modeling import make_pipeline
from sklearn.linear_model import Ridge


def sample():
    return pd.DataFrame([dict(area_sqft=1500, bedrooms=3, bathrooms=2, age_years=4,
        parking_spaces=1, distance_to_center_km=5, city='Mardan', property_type='House', condition='Good')])


def test_missing_column_is_clear():
    with pytest.raises(ValueError, match='Missing columns: city'):
        clean_data(sample().drop(columns='city'), training=False)


def test_prediction_preserves_rows_and_marks_invalid():
    data = pd.concat([sample(), sample()], ignore_index=True)
    data.loc[0, 'area_sqft'] = -2
    cleaned, _ = clean_data(data, training=False)
    assert len(cleaned) == 2
    assert np.isnan(cleaned.loc[0, 'area_sqft'])


def test_unseen_categories_and_training_only_imputation():
    train = pd.concat([sample(), sample(), sample()], ignore_index=True)
    train['area_sqft'] = [1000., 2000., np.nan]
    model = make_pipeline(Ridge()).fit(train[FEATURES], [1e7, 2e7, 1.5e7])
    numeric = model.named_steps['preprocess'].named_transformers_['numeric']
    assert numeric.named_steps['impute'].statistics_[0] == 1500
    test = sample()
    test['city'] = 'Unseen city'
    assert np.isfinite(model.predict(test[FEATURES])).all()
    assert numeric.named_steps['impute'].statistics_[0] == 1500
