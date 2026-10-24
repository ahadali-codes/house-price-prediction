"""CLI: python train.py [--data path.csv --source-label 'Dataset provenance']."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from src.data import clean_data
from src.modeling import train_and_evaluate

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/house_prices.csv')
    parser.add_argument('--source-label', default=None)
    args = parser.parse_args()
    source = args.source_label or ('Synthetic demonstration data; not market observations'
        if args.data.resolve() == (ROOT / 'data/house_prices.csv').resolve()
        else 'User-provided CSV; provenance not independently verified')
    try:
        raw = pd.read_csv(args.data)
        data, cleaning = clean_data(raw)
        model, metrics, predictions = train_and_evaluate(data)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(1, f'Training failed: {exc}\n')
    metrics.update({'source': source, 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    'cleaning': cleaning, 'sklearn_version': sklearn.__version__})
    (ROOT / 'models').mkdir(exist_ok=True)
    (ROOT / 'reports').mkdir(exist_ok=True)
    joblib.dump({'pipeline': model, 'metrics': metrics}, ROOT / 'models/house_price_model.joblib')
    (ROOT / 'reports/metrics.json').write_text(json.dumps(metrics, indent=2))
    predictions.to_csv(ROOT / 'reports/test_predictions.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    axes[0].scatter(predictions.price_pkr / 1e6, predictions.predicted_price_pkr / 1e6,
                    color='#1d554a', alpha=.5, s=16)
    limits = [min(predictions.price_pkr.min(), predictions.predicted_price_pkr.min()) / 1e6,
              max(predictions.price_pkr.max(), predictions.predicted_price_pkr.max()) / 1e6]
    axes[0].plot(limits, limits, '--', color='#bf783a')
    axes[0].set(xlabel='Actual price (million PKR)', ylabel='Predicted price (million PKR)',
                title='Held-out test predictions')
    axes[1].hist(predictions.residual_pkr / 1e6, bins=25, color='#1d554a', edgecolor='white')
    axes[1].set(xlabel='Actual − predicted (million PKR)', ylabel='Houses', title='Residual distribution')
    fig.suptitle(source, fontsize=10)
    fig.savefig(ROOT / 'reports/evaluation.png', dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
