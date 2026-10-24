"""Train-only preprocessing and reproducible regression model selection."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import NUMERIC, CATEGORICAL, FEATURES, TARGET


def make_pipeline(estimator):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessing = ColumnTransformer([('numeric', numeric, NUMERIC),
                                       ('categorical', categorical, CATEGORICAL)])
    return Pipeline([('preprocess', preprocessing), ('regressor', estimator)])


def train_and_evaluate(data):
    """Choose the model on 5-fold training CV, then evaluate once on a held-out 20%."""
    train, test = train_test_split(data, test_size=.2, random_state=42)
    X, y = train[FEATURES], train[TARGET]
    candidates = {'Median baseline': DummyRegressor(strategy='median'),
                  'Ridge regression': Ridge(alpha=10),
                  'Random forest': RandomForestRegressor(n_estimators=120, min_samples_leaf=2,
                                                         random_state=42, n_jobs=1)}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    models = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        errors = -cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error')
        scores.append({'model': name, 'cv_mae_pkr': float(errors.mean()),
                       'cv_std_pkr': float(errors.std())})
        models[name] = pipeline
    best = min(scores, key=lambda row: row['cv_mae_pkr'])['model']
    model = models[best].fit(X, y)
    predicted = model.predict(test[FEATURES])
    baseline = np.repeat(y.median(), len(test))
    metrics = {'selected_model': best, 'train_rows': len(train), 'test_rows': len(test),
               'mae_pkr': float(mean_absolute_error(test[TARGET], predicted)),
               'rmse_pkr': float(np.sqrt(mean_squared_error(test[TARGET], predicted))),
               'r2': float(r2_score(test[TARGET], predicted)),
               'baseline_mae_pkr': float(mean_absolute_error(test[TARGET], baseline)),
               'cross_validation': scores, 'random_seed': 42,
               'split': '80/20; five-fold CV only within training set'}
    predictions = test.copy()
    predictions['predicted_price_pkr'] = predicted
    predictions['residual_pkr'] = predictions[TARGET] - predicted
    predictions['absolute_error_pkr'] = predictions['residual_pkr'].abs()
    return model, metrics, predictions
