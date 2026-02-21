"""
ML Model Training — Phase 6

Trains a linear regression model:
  features : 5 theme sentiment scores + current fred_score
  target   : fred_score_2w_future

Train/test split is chronological: train on 2019-2024, test on 2025-present.
Saves the trained model to data/model.pkl for use by the dashboard.
Writes current-week predictions to model_predictions.csv.

Usage:
  python train_model.py        # train and predict
  python train_model.py --eval # print evaluation metrics only
"""

import os
import csv
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv

load_dotenv()

WEEKLY_CSV = 'data/weekly_sentiment_scores.csv'
PREDICTIONS_CSV = 'data/model_predictions.csv'
MODEL_PKL = 'data/model.pkl'

FEATURE_COLS = [
    'monetary_policy_score',
    'credit_debt_score',
    'banking_liquidity_score',
    'inflation_growth_score',
    'geopolitical_external_score',
    'fred_score',
]
TARGET_COL = 'fred_score_2w_future'

TRAIN_SPLIT = 0.8   # fraction of labeled data used for training


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_weekly():
    if not os.path.exists(WEEKLY_CSV) or os.path.getsize(WEEKLY_CSV) == 0:
        return pd.DataFrame()
    df = pd.read_csv(WEEKLY_CSV)
    df['week_start'] = pd.to_datetime(df['week_start'])
    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('week_start').reset_index(drop=True)


def load_predictions():
    if not os.path.exists(PREDICTIONS_CSV) or os.path.getsize(PREDICTIONS_CSV) == 0:
        return pd.DataFrame(columns=[
            'week_start', 'predicted_fred_score', 'actual_fred_score',
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

def train(df):
    """
    Train linear regression on rows that have both features and target.
    Returns (model, train_df, test_df, metrics_dict).
    """
    labeled = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    if len(labeled) < 10:
        print(f'Not enough labeled data ({len(labeled)} rows). Need at least 10.')
        return None, None, None, {}

    n_train = max(5, int(len(labeled) * TRAIN_SPLIT))
    train_df = labeled.iloc[:n_train]
    test_df = labeled.iloc[n_train:]

    cutoff_date = train_df['week_start'].iloc[-1].strftime('%Y-%m-%d')
    print(f'Train/test split: {n_train} train / {len(test_df)} test  (cutoff: {cutoff_date})')

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    metrics = {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'train_r2': r2_score(y_train, model.predict(X_train)),
    }

    if not test_df.empty:
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df[TARGET_COL].values
        y_pred = model.predict(X_test)
        metrics['test_r2'] = r2_score(y_test, y_pred)
        metrics['test_rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['test_mae'] = float(np.mean(np.abs(y_test - y_pred)))

    return model, train_df, test_df, metrics


def print_metrics(model, metrics):
    print(f'\nModel coefficients:')
    for feat, coef in zip(FEATURE_COLS, model.coef_):
        print(f'  {feat:<35} {coef:+.4f}')
    print(f'  intercept                           {model.intercept_:+.4f}')
    print(f'\nTraining rows : {metrics["n_train"]}')
    print(f'Train R²      : {metrics["train_r2"]:.3f}')
    if 'test_r2' in metrics:
        print(f'Test rows     : {metrics["n_test"]}')
        print(f'Test R²       : {metrics["test_r2"]:.3f}')
        print(f'Test RMSE     : {metrics["test_rmse"]:.4f}')
        print(f'Test MAE      : {metrics["test_mae"]:.4f}')


def save_model(model):
    os.makedirs('data', exist_ok=True)
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {MODEL_PKL}')


def load_model():
    if not os.path.exists(MODEL_PKL):
        return None
    with open(MODEL_PKL, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_current_weeks(df, model, model_version):
    """
    Generate predictions for weeks that have features but no fred_score_2w_future yet.
    Writes new rows (or updates existing) in model_predictions.csv.
    """
    # Rows with all features but missing target (current / near-current weeks)
    has_features = df[FEATURE_COLS].notna().all(axis=1)
    needs_prediction = df[has_features & df[TARGET_COL].isna()]

    if needs_prediction.empty:
        print('No current weeks to predict.')
        return

    existing = load_predictions()
    existing_weeks = set(existing['week_start'].astype(str).tolist())

    new_rows = []
    for _, row in needs_prediction.iterrows():
        week_str = row['week_start'].strftime('%Y-%m-%d')
        if week_str in existing_weeks:
            continue   # already have a prediction for this week
        X = np.array([[row[c] for c in FEATURE_COLS]])
        pred = float(model.predict(X)[0])
        new_rows.append({
            'week_start': week_str,
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
            'week_start', 'predicted_fred_score', 'actual_fred_score',
            'prediction_error', 'model_version',
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f'Wrote {len(new_rows)} new prediction(s) to {PREDICTIONS_CSV}')


def backfill_actuals(df):
    """
    Fill in actual_fred_score and prediction_error for predictions where
    the actual FRED value is now known.
    """
    if not os.path.exists(PREDICTIONS_CSV) or os.path.getsize(PREDICTIONS_CSV) == 0:
        return

    preds = pd.read_csv(PREDICTIONS_CSV)
    preds['week_start'] = pd.to_datetime(preds['week_start'])

    # Build lookup: week_start -> actual fred_score_2w_future
    # fred_score_2w_future in weekly_sentiment_scores is the actual future FRED
    # The prediction was made for that future week, so we need the FRED of that week
    # Actually: for week W, we predicted fred_score_2w_future.
    # The "actual" is found by looking at the weekly row for W and reading fred_score_2w_future
    weekly_lookup = {}
    for _, row in df.iterrows():
        if pd.notna(row.get(TARGET_COL)) and row.get(TARGET_COL) != '':
            weekly_lookup[row['week_start']] = float(row[TARGET_COL])

    updated = 0
    for idx, pred_row in preds.iterrows():
        if str(pred_row.get('actual_fred_score', '')).strip() != '':
            continue  # already filled
        week = pred_row['week_start']
        if week in weekly_lookup:
            actual = weekly_lookup[week]
            predicted = float(pred_row['predicted_fred_score'])
            preds.at[idx, 'actual_fred_score'] = round(actual, 4)
            preds.at[idx, 'prediction_error'] = round(actual - predicted, 4)
            updated += 1

    if updated:
        preds.to_csv(PREDICTIONS_CSV, index=False)
        print(f'Backfilled actuals for {updated} prediction(s).')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true',
                        help='Print metrics only (no save, no predict)')
    args = parser.parse_args()

    df = load_weekly()
    if df.empty:
        print('No data in weekly_sentiment_scores.csv. Run pipeline.py first.')
        return

    print(f'Loaded {len(df)} weeks from {WEEKLY_CSV}')

    model, train_df, test_df, metrics = train(df)
    if model is None:
        return

    print_metrics(model, metrics)

    if args.eval:
        return

    save_model(model)

    version = next_model_version()
    print(f'\nModel version: {version}')

    predict_current_weeks(df, model, version)
    backfill_actuals(df)

    print('\nTraining complete.')


if __name__ == '__main__':
    main()
