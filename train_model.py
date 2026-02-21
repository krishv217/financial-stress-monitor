"""
ML Model Training — Phase 6

Trains Linear Regression AND Random Forest across 5 prediction horizons:
  1w, 2w, 4w, 8w, 12w future FRED stress index

Features : 5 theme sentiment scores + current fred_score
Target   : fred_score_{horizon}_future

Prints a comparison table of Test R², RMSE, MAE for every model+horizon.
Saves the best model per horizon to data/model_{horizon}.pkl
Also saves data/model.pkl (best 2w model) for dashboard compatibility.
Writes current-week predictions to model_predictions.csv (2w horizon).

Usage:
  python train_model.py        # train all horizons, both models
  python train_model.py --eval # print comparison table only (no save)
  python train_model.py --horizon 2w   # train a single horizon
"""

import os
import csv
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv

load_dotenv()

WEEKLY_CSV = 'data/weekly_sentiment_scores.csv'
PREDICTIONS_CSV = 'data/model_predictions.csv'

FEATURE_COLS = [
    'monetary_policy_score',
    'credit_debt_score',
    'banking_liquidity_score',
    'inflation_growth_score',
    'geopolitical_external_score',
    'fred_score',
]

HORIZONS = [
    ('1w',  'fred_score_1w_future'),
    ('2w',  'fred_score_2w_future'),
    ('4w',  'fred_score_4w_future'),
    ('8w',  'fred_score_8w_future'),
    ('12w', 'fred_score_12w_future'),
]

MODELS = [
    ('LinearRegression', lambda: LinearRegression()),
    ('LassoCV',          lambda: LassoCV(cv=5, max_iter=5000)),
    ('RandomForest',     lambda: RandomForestRegressor(n_estimators=100, random_state=42)),
]

TRAIN_SPLIT = 0.8   # fraction of labeled data used for training


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_weekly():
    if not os.path.exists(WEEKLY_CSV) or os.path.getsize(WEEKLY_CSV) == 0:
        return pd.DataFrame()
    df = pd.read_csv(WEEKLY_CSV)
    df['week_start'] = pd.to_datetime(df['week_start'])
    all_numeric = FEATURE_COLS + [col for _, col in HORIZONS]
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('week_start').reset_index(drop=True)


def model_pkl_path(horizon_label, model_name):
    slug = model_name.lower().replace(' ', '_')
    return f'data/model_{horizon_label}_{slug}.pkl'


def load_predictions():
    if not os.path.exists(PREDICTIONS_CSV) or os.path.getsize(PREDICTIONS_CSV) == 0:
        return pd.DataFrame(columns=[
            'week_start', 'horizon', 'model_name',
            'predicted_fred_score', 'actual_fred_score',
            'prediction_error', 'model_version',
        ])
    df = pd.read_csv(PREDICTIONS_CSV)
    df['week_start'] = pd.to_datetime(df['week_start'])
    return df


def next_model_version():
    df = load_predictions()
    if df.empty or 'model_version' not in df.columns:
        return 'v1'
    versions = df['model_version'].dropna().tolist()
    nums = []
    for v in versions:
        try:
            nums.append(int(str(v).lstrip('v')))
        except ValueError:
            pass
    return f'v{max(nums) + 1}' if nums else 'v1'


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_one(df, target_col, model_instance):
    """
    Train a single model for one horizon.
    Returns (fitted_model, metrics_dict) or (None, {}) if not enough data.
    """
    labeled = df.dropna(subset=FEATURE_COLS + [target_col])
    if len(labeled) < 10:
        return None, {'error': f'only {len(labeled)} labeled rows'}

    n_train = max(5, int(len(labeled) * TRAIN_SPLIT))
    train_df = labeled.iloc[:n_train]
    test_df  = labeled.iloc[n_train:]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[target_col].values

    model_instance.fit(X_train, y_train)

    metrics = {
        'n_train':  len(train_df),
        'n_test':   len(test_df),
        'train_r2': r2_score(y_train, model_instance.predict(X_train)),
        'cutoff':   train_df['week_start'].iloc[-1].strftime('%Y-%m-%d'),
    }

    if not test_df.empty:
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df[target_col].values
        y_pred = model_instance.predict(X_test)
        metrics['test_r2']   = r2_score(y_test, y_pred)
        metrics['test_rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['test_mae']  = float(np.mean(np.abs(y_test - y_pred)))

    return model_instance, metrics


def train_all_horizons(df, target_horizons=None, save=True):
    """
    Train both models at every horizon. Print comparison table.
    Returns dict keyed by (horizon_label, model_name) -> (model, metrics).
    """
    if target_horizons:
        horizons = [(lbl, col) for lbl, col in HORIZONS if lbl in target_horizons]
    else:
        horizons = HORIZONS

    results = {}

    print(f'\n{"Horizon":<8} {"Model":<22} {"Train rows":>10} {"Test rows":>9} '
          f'{"Train R²":>9} {"Test R²":>9} {"RMSE":>8} {"MAE":>8}')
    print('-' * 90)

    best_per_horizon = {}   # horizon_label -> (model_name, model, r2)

    for h_label, h_col in horizons:
        if h_col not in df.columns:
            print(f'{h_label:<8}  *** column {h_col} missing — run pipeline phase 4 first ***')
            continue

        for m_name, m_factory in MODELS:
            model, metrics = _train_one(df, h_col, m_factory())
            results[(h_label, m_name)] = (model, metrics)

            if 'error' in metrics:
                print(f'{h_label:<8} {m_name:<22} {metrics["error"]}')
                continue

            test_r2   = metrics.get('test_r2',   float('nan'))
            test_rmse = metrics.get('test_rmse', float('nan'))
            test_mae  = metrics.get('test_mae',  float('nan'))

            print(f'{h_label:<8} {m_name:<22} {metrics["n_train"]:>10} '
                  f'{metrics["n_test"]:>9} {metrics["train_r2"]:>9.3f} '
                  f'{test_r2:>9.3f} {test_rmse:>8.4f} {test_mae:>8.4f}')

            # Track best model per horizon by test R²
            if not np.isnan(test_r2):
                prev_r2 = best_per_horizon.get(h_label, (None, None, float('-inf')))[2]
                if test_r2 > prev_r2:
                    best_per_horizon[h_label] = (m_name, model, test_r2)

    print('-' * 90)

    if save:
        os.makedirs('data', exist_ok=True)
        for h_label, (m_name, model, _) in best_per_horizon.items():
            path = model_pkl_path(h_label, m_name)
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f'  Saved best {h_label} model ({m_name}) -> {path}')

        # Also save data/model.pkl = best 2w model (dashboard compatibility)
        if '2w' in best_per_horizon:
            _, model_2w, _ = best_per_horizon['2w']
            with open('data/model.pkl', 'wb') as f:
                pickle.dump(model_2w, f)
            print(f'  Saved data/model.pkl  ({best_per_horizon["2w"][0]}, 2w)')

    return results, best_per_horizon


# ---------------------------------------------------------------------------
# Prediction (2w horizon, best model)
# ---------------------------------------------------------------------------

def predict_current_weeks(df, model, model_version, horizon_label='2w',
                          target_col='fred_score_2w_future'):
    """
    Generate predictions for weeks that have features but no target yet.
    Appends to model_predictions.csv.
    """
    has_features = df[FEATURE_COLS].notna().all(axis=1)
    needs_prediction = df[has_features & df[target_col].isna()]

    if needs_prediction.empty:
        print('No current weeks to predict.')
        return

    existing = load_predictions()
    existing_keys = set(
        zip(existing['week_start'].astype(str), existing.get('horizon', ''))
    )

    new_rows = []
    for _, row in needs_prediction.iterrows():
        week_str = row['week_start'].strftime('%Y-%m-%d')
        if (week_str, horizon_label) in existing_keys:
            continue
        X = np.array([[row[c] for c in FEATURE_COLS]])
        pred = float(model.predict(X)[0])
        new_rows.append({
            'week_start': week_str,
            'horizon': horizon_label,
            'model_name': type(model).__name__,
            'predicted_fred_score': round(pred, 4),
            'actual_fred_score': '',
            'prediction_error': '',
            'model_version': model_version,
        })

    if not new_rows:
        print('No new predictions to write.')
        return

    file_exists = os.path.exists(PREDICTIONS_CSV) and os.path.getsize(PREDICTIONS_CSV) > 0
    with open(PREDICTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'week_start', 'horizon', 'model_name',
            'predicted_fred_score', 'actual_fred_score',
            'prediction_error', 'model_version',
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f'Wrote {len(new_rows)} new prediction(s) to {PREDICTIONS_CSV}')


def backfill_actuals(df, target_col='fred_score_2w_future'):
    """
    Fill in actual_fred_score and prediction_error for predictions where
    the actual FRED value is now known.
    """
    if not os.path.exists(PREDICTIONS_CSV) or os.path.getsize(PREDICTIONS_CSV) == 0:
        return

    preds = pd.read_csv(PREDICTIONS_CSV)
    preds['week_start'] = pd.to_datetime(preds['week_start'])

    weekly_lookup = {}
    for _, row in df.iterrows():
        if pd.notna(row.get(target_col)) and row.get(target_col) != '':
            weekly_lookup[row['week_start']] = float(row[target_col])

    updated = 0
    for idx, pred_row in preds.iterrows():
        if str(pred_row.get('actual_fred_score', '')).strip() != '':
            continue
        week = pred_row['week_start']
        if week in weekly_lookup:
            actual = weekly_lookup[week]
            predicted = float(pred_row['predicted_fred_score'])
            preds.at[idx, 'actual_fred_score'] = round(actual, 4)
            preds.at[idx, 'prediction_error']  = round(actual - predicted, 4)
            updated += 1

    if updated:
        preds.to_csv(PREDICTIONS_CSV, index=False)
        print(f'Backfilled actuals for {updated} prediction(s).')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train stress-index prediction models')
    parser.add_argument('--eval', action='store_true',
                        help='Print comparison table only (no save, no predict)')
    parser.add_argument('--horizon', nargs='+',
                        choices=[lbl for lbl, _ in HORIZONS],
                        help='Train specific horizon(s) only (default: all)')
    args = parser.parse_args()

    df = load_weekly()
    if df.empty:
        print('No data in weekly_sentiment_scores.csv. Run pipeline.py first.')
        return

    print(f'Loaded {len(df)} weeks from {WEEKLY_CSV}')

    results, best_per_horizon = train_all_horizons(
        df,
        target_horizons=args.horizon,
        save=(not args.eval),
    )

    if args.eval:
        return

    # Predict current (unresolved) weeks using the best 2w model
    if '2w' in best_per_horizon:
        _, model_2w, _ = best_per_horizon['2w']
        version = next_model_version()
        print(f'\nModel version: {version}')
        predict_current_weeks(df, model_2w, version)
        backfill_actuals(df)

    print('\nTraining complete.')


if __name__ == '__main__':
    main()