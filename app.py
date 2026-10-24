"""Streamlit interface for the locally trained house-price pipeline."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from src.data import FEATURES, CATEGORIES, clean_data

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='House Price Prediction', page_icon='🏠', layout='wide')
st.markdown("""<style>
.stApp {background:#f7f6f2;color:#1d2926;}
h1,h2,h3 {letter-spacing:-.025em;}
[data-testid="stMetric"] {background:white;padding:22px;border:1px solid #dedfd8;border-radius:10px;}
.stButton>button {border-radius:8px;min-height:46px;}
p, label {font-size:17px!important;}
.block-container {max-width:1200px;padding-top:3rem;}
</style>""", unsafe_allow_html=True)
st.caption('APPLIED MACHINE LEARNING / REGRESSION LAB')
st.title('House Price Prediction')
st.write('Explore how property features relate to price, inspect evaluation results, and try a prediction.')
page = st.sidebar.radio('Workspace', ['Predict a house', 'Model evaluation', 'Dataset explorer', 'Batch prediction'])
st.sidebar.caption('Python · Pandas · NumPy · Scikit-learn')
model_path = ROOT / 'models/house_price_model.joblib'
if not model_path.exists():
    st.info('Train the model first. Run this command from the project folder:')
    st.code('python train.py')
    st.stop()

@st.cache_resource
def load_model(modified: int):
    # Only load this local model produced by train.py, never untrusted model uploads.
    return joblib.load(model_path)

try:
    bundle = load_model(model_path.stat().st_mtime_ns)
except Exception:
    st.error('The saved model cannot be loaded. Run python train.py in this environment to recreate it.')
    st.stop()
model, metrics = bundle['pipeline'], bundle['metrics']
st.caption('Training source: ' + metrics['source'])
st.warning('Educational prediction only. The included data is fictional and does not establish real property values.')

if page == 'Predict a house':
    left, right = st.columns([1.5, 1], gap='large')
    with left:
        st.subheader('Property details')
        with st.form('house'):
            a, b = st.columns(2)
            with a:
                city = st.selectbox('City', CATEGORIES['city'])
                area = st.number_input('Covered area (square feet)', 200, 20000, 1800, 50)
                bedrooms = st.number_input('Bedrooms', 1, 20, 3)
                bathrooms = st.number_input('Bathrooms', 1, 20, 3)
                age = st.number_input('Property age (years)', 0, 150, 5)
            with b:
                kind = st.selectbox('Property type', CATEGORIES['property_type'])
                condition = st.selectbox('Condition', CATEGORIES['condition'], index=1)
                parking = st.number_input('Parking spaces', 0, 10, 1)
                distance = st.number_input('Distance to city center (km)', 0.0, 150.0, 8.0, .5)
            submitted = st.form_submit_button('Estimate price', type='primary', use_container_width=True)
    with right:
        st.subheader('Model estimate')
        if submitted:
            row = pd.DataFrame([dict(city=city, area_sqft=area, bedrooms=bedrooms, bathrooms=bathrooms,
                age_years=age, parking_spaces=parking, distance_to_center_km=distance,
                property_type=kind, condition=condition)])
            cleaned, _ = clean_data(row, training=False)
            price = float(model.predict(cleaned[FEATURES])[0])
            st.metric('Predicted price (PKR)', f'{price:,.0f}')
            st.caption(f'{price / 1e6:,.2f} million PKR · {metrics["selected_model"]}')
            if not 550 <= area <= 4999 or age > 39 or distance > 35 or bedrooms > 8:
                st.warning('Some inputs are outside the included demo training ranges. The estimate may be unreliable.')
            st.info(f'Test MAE: PKR {metrics["mae_pkr"]:,.0f}. This is an average error, not a prediction interval.')
        else:
            st.info('Enter property details and select Estimate price.')
        st.markdown('**What affects an estimate?** Area, city, property type, condition, age, bathrooms, bedrooms, parking, and distance are provided to the model. These associations do not establish causation.')

elif page == 'Model evaluation':
    st.subheader('An untouched test set')
    a, b, c = st.columns(3)
    a.metric('MAE · PKR', f'{metrics["mae_pkr"]:,.0f}')
    b.metric('RMSE · PKR', f'{metrics["rmse_pkr"]:,.0f}')
    c.metric('R²', f'{metrics["r2"]:.3f}')
    st.write(f'{metrics["train_rows"]:,} training rows · {metrics["test_rows"]:,} test rows · Selected: {metrics["selected_model"]}')
    st.caption('MAE is average absolute error. RMSE penalizes large errors more. R² is not a percentage accuracy score and can be negative.')
    st.image(str(ROOT / 'reports/evaluation.png'))
    st.subheader('Training-only cross-validation')
    st.dataframe(pd.DataFrame(metrics['cross_validation']), hide_index=True, use_container_width=True)
    st.write(f'Median baseline test MAE: PKR {metrics["baseline_mae_pkr"]:,.0f}')
    st.download_button('Download test predictions', (ROOT / 'reports/test_predictions.csv').read_bytes(),
                       'test_predictions.csv', 'text/csv')

elif page == 'Dataset explorer':
    st.subheader('Explore a CSV')
    uploaded = st.file_uploader('Optional training dataset', type=['csv'])
    try:
        raw = pd.read_csv(uploaded if uploaded is not None else ROOT / 'data/house_prices.csv')
        data, report = clean_data(raw)
        if uploaded is not None:
            st.info('Preview only: uploading does not retrain the model. Use the training command in README to retrain.')
        cities = st.multiselect('Cities', sorted(data.city.dropna().unique()))
        filtered = data[data.city.isin(cities)] if cities else data
        a, b, c = st.columns(3)
        a.metric('Rows shown', len(filtered))
        b.metric('Median price · PKR', f'{filtered.price_pkr.median():,.0f}')
        c.metric('Median area · sq ft', f'{filtered.area_sqft.median():,.0f}')
        st.bar_chart(filtered.groupby('city').price_pkr.median(), color='#1d554a')
        st.dataframe(filtered, hide_index=True, use_container_width=True)
        with st.expander('Cleaning report'):
            st.json(report)
        st.download_button('Download filtered CSV', filtered.to_csv(index=False), 'filtered_houses.csv', 'text/csv')
    except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
        st.error(f'Cannot read this dataset: {exc}')

else:
    st.subheader('Predict multiple houses')
    st.write('Upload the same feature columns as the training CSV. A price column is not required.')
    template = pd.read_csv(ROOT / 'data/prediction_template.csv')
    st.download_button('Download input template', template.to_csv(index=False), 'prediction_template.csv', 'text/csv')
    upload = st.file_uploader('Prediction CSV', type=['csv'])
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            cleaned, report = clean_data(raw, training=False)
            results = raw.copy()
            results['predicted_price_pkr'] = model.predict(cleaned[FEATURES])
            st.dataframe(results, hide_index=True, use_container_width=True)
            st.caption('Invalid or missing feature values are imputed using training data statistics. Unseen categories are encoded as unknown.')
            st.json(report)
            st.download_button('Download predictions', results.to_csv(index=False), 'house_predictions.csv', 'text/csv')
        except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f'Cannot predict from this CSV: {exc}')
