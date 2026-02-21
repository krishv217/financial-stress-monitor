"""
Initialize the four pipeline CSV files with headers only.
Run this once at the start — safe to re-run (won't overwrite existing files).
"""
import csv
import os

os.makedirs('data', exist_ok=True)

FILES = {
    'data/raw_articles.csv': [
        'article_id', 'source', 'query_number', 'week_start',
        'publication_date', 'headline', 'abstract', 'section', 'url',
    ],
    'data/classified_articles.csv': [
        'article_id', 'source', 'query_number', 'week_start',
        'publication_date', 'headline', 'abstract', 'section', 'url',
        'is_relevant', 'stress_theme', 'stress_direction',
        'magnitude_score', 'prompt_version',
    ],
    'data/weekly_sentiment_scores.csv': [
        'week_start', 'fred_score', 'fred_score_2w_future',
        'monetary_policy_score', 'credit_debt_score',
        'banking_liquidity_score', 'inflation_growth_score',
        'geopolitical_external_score', 'total_articles', 'prompt_version',
    ],
    'data/model_predictions.csv': [
        'week_start', 'predicted_fred_score', 'actual_fred_score',
        'prediction_error', 'model_version',
    ],
}

for path, headers in FILES.items():
    if os.path.exists(path):
        print(f"  exists  {path}")
    else:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
        print(f"  created {path}")

print("\nReady.")
