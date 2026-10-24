# Validation

Executed in the build environment:
- Generated and cleaned the included dataset: 1,510 input records; 1,500 distinct valid houses.
- Trained all three pipelines with five-fold cross-validation and evaluated the selected model on 300 held-out records.
- Three Pytest checks passed: missing schema, invalid numeric input/row preservation, and unseen categories with training-only imputation.
- Streamlit AppTest loaded all four pages and executed a single-house prediction successfully.

See reports/metrics.json for actual metrics and dataset SHA-256. See requirements-tested.txt for the directly used package versions. Windows setup instructions are supplied, but Windows and remote deployment were not executed here. No browser screenshot verification was performed. The model artifact is excluded from the ZIP to avoid cross-version serialization issues; run python train.py to create it in your own environment.
