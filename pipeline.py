"""
Data Pipeline — Phases 2-4

Phase 2: Dedup + relevance filter -> classified_articles.csv (is_relevant)
Phase 3: LLM classification -> classified_articles.csv (theme/direction/magnitude)
Phase 4: Weekly aggregation -> weekly_sentiment_scores.csv
         Populate fred_score_2w_future for all historical weeks

Usage:
  python pipeline.py              # run all phases
  python pipeline.py --phase 2    # dedup + relevance only
  python pipeline.py --phase 3    # classify only (assumes phase 2 done)
  python pipeline.py --phase 4    # aggregate only (assumes phase 3 done)
  python pipeline.py --prompt-version v2  # use v2 prompt, reclassify
"""

import os
import csv
import argparse
import unicodedata
import re
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from classifier import filter_relevance, classify_articles, PROMPT_VERSION
from fred_data import fetch_fred_data, save_fred_data, load_fred_data

load_dotenv()

RAW_CSV = 'data/raw_articles.csv'
CLASSIFIED_CSV = 'data/classified_articles.csv'
WEEKLY_CSV = 'data/weekly_sentiment_scores.csv'

CLASSIFIED_FIELDS = [
    'article_id', 'source', 'query_number', 'week_start',
    'publication_date', 'headline', 'abstract', 'section', 'url',
    'is_relevant', 'stress_theme', 'stress_direction',
    'magnitude_score', 'prompt_version',
]

WEEKLY_FIELDS = [
    'week_start', 'fred_score', 'fred_score_2w_future',
    'monetary_policy_score', 'credit_debt_score',
    'banking_liquidity_score', 'inflation_growth_score',
    'geopolitical_external_score', 'total_articles', 'prompt_version',
]

THEME_COLUMNS = {
    'monetary_policy_risk': 'monetary_policy_score',
    'credit_debt_risk': 'credit_debt_score',
    'banking_liquidity_risk': 'banking_liquidity_score',
    'inflation_growth_risk': 'inflation_growth_score',
    'geopolitical_external_risk': 'geopolitical_external_score',
}


# ---------------------------------------------------------------------------
# Phase 2: Dedup + relevance filter
# ---------------------------------------------------------------------------

def _normalize(text):
    """Lowercase, strip punctuation — for headline similarity comparison."""
    text = unicodedata.normalize('NFKD', text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_raw_articles():
    if not os.path.exists(RAW_CSV) or os.path.getsize(RAW_CSV) == 0:
        return []
    with open(RAW_CSV, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_classified_ids():
    """Return set of article_ids already written to classified_articles.csv."""
    if not os.path.exists(CLASSIFIED_CSV) or os.path.getsize(CLASSIFIED_CSV) == 0:
        return set()
    with open(CLASSIFIED_CSV, encoding='utf-8') as f:
        return {row['article_id'] for row in csv.DictReader(f)}


def run_phase2(prompt_ver=PROMPT_VERSION):
    """Dedup across queries, run relevance filter, write to classified_articles.csv."""
    print('\n=== Phase 2: Dedup + Relevance Filter ===')

    raw = load_raw_articles()
    if not raw:
        print('No raw articles found. Run collect_nyt.py first.')
        return

    already_classified = load_classified_ids()

    # Deduplicate: keep first occurrence of each normalized headline
    seen_norm = set()
    deduped = []
    for art in raw:
        norm = _normalize(art['headline'])
        if norm and norm not in seen_norm:
            seen_norm.add(norm)
            deduped.append(art)

    print(f'Raw articles   : {len(raw)}')
    print(f'After dedup    : {len(deduped)}')

    # Only process articles not already in classified_articles.csv
    to_filter = [a for a in deduped if a['article_id'] not in already_classified]
    print(f'New to process : {len(to_filter)}')

    if not to_filter:
        print('Nothing new to filter.')
        return

    # Run relevance filter
    relevance_map = filter_relevance(to_filter, batch_size=30)

    # Write results
    file_exists = os.path.exists(CLASSIFIED_CSV) and os.path.getsize(CLASSIFIED_CSV) > 0
    with open(CLASSIFIED_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CLASSIFIED_FIELDS)
        if not file_exists:
            writer.writeheader()
        for art in to_filter:
            row = {k: art.get(k, '') for k in CLASSIFIED_FIELDS}
            row['is_relevant'] = relevance_map.get(art['article_id'], 'yes')
            row['stress_theme'] = ''
            row['stress_direction'] = ''
            row['magnitude_score'] = ''
            row['prompt_version'] = ''
            writer.writerow(row)

    yes_count = sum(1 for v in relevance_map.values() if v == 'yes')
    print(f'Relevant: {yes_count}/{len(to_filter)}')
    print('Phase 2 complete.')


# ---------------------------------------------------------------------------
# Phase 3: LLM classification
# ---------------------------------------------------------------------------

def run_phase3(prompt_ver=PROMPT_VERSION):
    """Classify relevant articles that don't yet have a stress_theme."""
    print('\n=== Phase 3: LLM Classification ===')

    if not os.path.exists(CLASSIFIED_CSV) or os.path.getsize(CLASSIFIED_CSV) == 0:
        print('classified_articles.csv is empty. Run phase 2 first.')
        return

    df = pd.read_csv(CLASSIFIED_CSV, dtype=str).fillna('')

    # Articles that are relevant but not yet classified (or need reclassification)
    to_classify = df[
        (df['is_relevant'] == 'yes') &
        ((df['stress_theme'] == '') | (df['prompt_version'] != prompt_ver))
    ]

    print(f'Articles needing classification: {len(to_classify)}')
    if to_classify.empty:
        print('Nothing to classify.')
        return

    articles = to_classify[['article_id', 'headline', 'abstract']].to_dict('records')
    cls_map = classify_articles(articles, batch_size=20)

    # Update DataFrame
    for art_id, cls in cls_map.items():
        mask = df['article_id'] == art_id
        df.loc[mask, 'stress_theme'] = cls['theme']
        df.loc[mask, 'stress_direction'] = cls['direction']
        df.loc[mask, 'magnitude_score'] = str(cls['magnitude'])
        df.loc[mask, 'prompt_version'] = prompt_ver

    df.to_csv(CLASSIFIED_CSV, index=False)
    print(f'Phase 3 complete. {len(cls_map)} articles classified with {prompt_ver}.')


# ---------------------------------------------------------------------------
# Phase 4: Weekly aggregation
# ---------------------------------------------------------------------------

def _magnitude_weighted_score(group, theme):
    """
    Magnitude-weighted net stress score for one theme within a weekly group.
    +magnitude for increasing, -magnitude for decreasing, 0 for neutral.
    """
    theme_arts = group[group['stress_theme'] == theme]
    score = 0
    for _, row in theme_arts.iterrows():
        try:
            mag = int(row['magnitude_score']) if row['magnitude_score'] else 1
        except (ValueError, TypeError):
            mag = 1
        direction = row['stress_direction']
        if direction == 'increasing':
            score += mag
        elif direction == 'decreasing':
            score -= mag
    return score


def _get_fred_series():
    """
    Load or fetch FRED data; return a pandas Series indexed by datetime.
    """
    try:
        fred_df = load_fred_data()
    except FileNotFoundError:
        print('Fetching FRED data...')
        fred_df = fetch_fred_data(start_date='2019-01-01')
        save_fred_data(fred_df)

    fred_df = fred_df.copy()
    fred_df['date'] = pd.to_datetime(fred_df['date'])
    fred_df = fred_df.set_index('date')['value'].sort_index()
    return fred_df


def _nearest_fred(fred_series, date_str, max_days=10):
    """
    Return the FRED value closest to date_str within max_days, or ''.
    FRED publishes on Fridays; article weeks may start on any day.
    """
    target = pd.to_datetime(date_str)
    diffs = abs(fred_series.index - target)
    idx = diffs.argmin()
    if diffs[idx].days <= max_days:
        return float(fred_series.iloc[idx])
    return ''


def run_phase4(prompt_ver=PROMPT_VERSION):
    """Aggregate classified articles into weekly_sentiment_scores.csv."""
    print('\n=== Phase 4: Weekly Aggregation ===')

    if not os.path.exists(CLASSIFIED_CSV) or os.path.getsize(CLASSIFIED_CSV) == 0:
        print('classified_articles.csv is empty. Run phases 2-3 first.')
        return

    df = pd.read_csv(CLASSIFIED_CSV, dtype=str).fillna('')
    relevant = df[(df['is_relevant'] == 'yes') & (df['stress_theme'] != '')]

    if relevant.empty:
        print('No classified relevant articles yet.')
        return

    fred_series = _get_fred_series()

    # Group by week_start
    weekly_rows = []
    for week_start, group in relevant.groupby('week_start'):
        row = {
            'week_start': week_start,
            'fred_score': _nearest_fred(fred_series, week_start),
            'fred_score_2w_future': '',   # filled in below
            'prompt_version': prompt_ver,
            'total_articles': len(group),
        }
        for theme, col in THEME_COLUMNS.items():
            row[col] = _magnitude_weighted_score(group, theme)
        weekly_rows.append(row)

    # Sort by date
    weekly_rows.sort(key=lambda r: r['week_start'])

    # Populate fred_score_2w_future using nearest-date lookup 2 weeks ahead
    for row in weekly_rows:
        d = datetime.strptime(row['week_start'], '%Y-%m-%d')
        future_str = (d + timedelta(weeks=2)).strftime('%Y-%m-%d')
        row['fred_score_2w_future'] = _nearest_fred(fred_series, future_str)

    # Write weekly_sentiment_scores.csv (full overwrite on each aggregation run)
    with open(WEEKLY_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=WEEKLY_FIELDS)
        writer.writeheader()
        writer.writerows(weekly_rows)

    print(f'Wrote {len(weekly_rows)} weeks to {WEEKLY_CSV}')
    filled = sum(1 for r in weekly_rows if r['fred_score_2w_future'] != '')
    print(f'fred_score_2w_future populated for {filled}/{len(weekly_rows)} weeks')
    print('Phase 4 complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run pipeline phases 2-4')
    parser.add_argument('--phase', type=int, choices=[2, 3, 4],
                        help='Run a specific phase only (default: all)')
    parser.add_argument('--prompt-version', default=PROMPT_VERSION,
                        help='Prompt version label (default: v1)')
    args = parser.parse_args()
    pv = args.prompt_version

    if args.phase == 2:
        run_phase2(pv)
    elif args.phase == 3:
        run_phase3(pv)
    elif args.phase == 4:
        run_phase4(pv)
    else:
        run_phase2(pv)
        run_phase3(pv)
        run_phase4(pv)
        print('\nAll phases complete.')


if __name__ == '__main__':
    main()
