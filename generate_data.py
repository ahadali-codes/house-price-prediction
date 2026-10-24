"""Generate fictional Pakistan-style housing data. Prices are NOT market observations."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(output: Path, rows: int = 1500, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    city = rng.choice(['Islamabad', 'Lahore', 'Karachi', 'Peshawar', 'Mardan'], rows)
    area = rng.integers(550, 5000, rows)
    bedrooms = np.clip(np.rint(area / 800 + rng.normal(0, .6, rows)), 1, 8).astype(int)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, rows), 1, 9)
    age = rng.integers(0, 40, rows)
    parking = rng.integers(0, 4, rows)
    distance = rng.uniform(1, 35, rows).round(1)
    kind = rng.choice(['House', 'Apartment'], rows)
    condition = rng.choice(['Fair', 'Good', 'Excellent'], rows)
    rates = pd.Series(city).map({'Islamabad': 13500, 'Lahore': 10500, 'Karachi': 12000,
                                 'Peshawar': 7500, 'Mardan': 5500}).to_numpy()
    quality = pd.Series(condition).map({'Fair': .86, 'Good': 1, 'Excellent': 1.14}).to_numpy()
    price = (area * rates * quality * np.where(kind == 'House', 1.12, .9)
             * np.exp(-age * .006 - distance * .009)
             + parking * 450000 + bathrooms * 300000)
    price = np.maximum(1000000, price + rng.normal(0, .1 * price)).round(-4)
    data = pd.DataFrame({'property_id': [f'HP{i:05}' for i in range(rows)], 'city': city,
        'area_sqft': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'age_years': age,
        'parking_spaces': parking, 'distance_to_center_km': distance,
        'property_type': kind, 'condition': condition, 'price_pkr': price})
    # Deliberate imperfections demonstrate cleaning without altering target provenance.
    data['area_sqft'] = data['area_sqft'].astype(float)
    data.loc[rng.choice(rows, 30, replace=False), 'area_sqft'] = np.nan
    data.loc[rng.choice(rows, 20, replace=False), 'condition'] = np.nan
    data.loc[0, 'age_years'] = -5
    data = pd.concat([data, data.iloc[10:20]], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f'Saved {len(data)} synthetic records to {output}')

if __name__ == '__main__':
    generate(Path(__file__).parent / 'data/house_prices.csv')
