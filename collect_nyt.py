"""
NYT Historical Article Collector — Phases 1 (data collection)

Samples every other week from START_DATE to today (~163 weeks).
Fires 10 queries x 3 pages = 30 API calls per week, 7s sleep between calls.
Estimated full runtime: ~6 hours single machine; ~2 hours across 3 machines.
Safe to interrupt and resume.

Usage:
  python collect_nyt.py           # full run
  python collect_nyt.py --test    # 2 weeks only (smoke test)
"""

import os
import csv
import time
import argparse
import itertools
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NYT_BASE_URL = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'

# Load up to 7 API keys: NYT_API_KEY_1 … NYT_API_KEY_7.
# Falls back to NYT_API_KEY if numbered keys are not set.
_raw_keys = [os.getenv(f'NYT_API_KEY_{i}') for i in range(1, 8)]
NYT_API_KEYS = [k for k in _raw_keys if k] or [os.getenv('NYT_API_KEY')]
NYT_API_KEYS = [k for k in NYT_API_KEYS if k]  # drop any remaining None

# Rotate through keys on every call.
# With N keys each key is called every N requests, so safe sleep = ceil(6/N).
# Single key: 7s. Two keys: 3.5s. Five keys: 1.4s. Seven keys: 1.0s.
SLEEP_SECS = max(1.0, 7 / len(NYT_API_KEYS))
_key_cycle = itertools.cycle(NYT_API_KEYS)

# 10 queries covering all 5 stress themes.
# Kept to 3 terms each — NYT Article Search API returns null results for
# longer OR chains combined with historical date ranges.
QUERIES = [
    'federal reserve OR interest rates OR monetary policy',  # Q1  monetary policy
    'inflation OR consumer prices OR stagflation',           # Q2  inflation/growth
    'recession OR economic growth OR GDP',                   # Q3  macro conditions
    'unemployment OR labor market OR jobs',                  # Q4  macro data
    'yield curve OR bond market OR credit markets',          # Q5  credit/debt
    'debt ceiling OR sovereign debt OR corporate bonds',     # Q6  credit/debt
    'bank failure OR banking crisis OR financial stability', # Q7  banking/liquidity
    'liquidity OR deposit insurance OR FDIC',               # Q8  banking/liquidity
    'tariffs OR trade war OR sanctions',                     # Q9  geopolitical
    'stock market OR market volatility OR financial markets', # Q10 equity/broad
]

START_DATE = datetime(2003, 1, 3)
RAW_CSV = 'data/raw_articles.csv'
FIELDNAMES = [
    'article_id', 'source', 'query_number', 'week_start',
    'publication_date', 'headline', 'abstract', 'section', 'url',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_target_weeks():
    """Generate weekly Monday dates from START_DATE through today."""
    weeks, d, today = [], START_DATE, datetime.now()
    while d <= today:
        weeks.append(d)
        d += timedelta(weeks=1)
    return weeks


def load_collected_weeks():
    """Return set of week_start strings already present in raw_articles.csv."""
    if not os.path.exists(RAW_CSV) or os.path.getsize(RAW_CSV) == 0:
        return set()
    with open(RAW_CSV, encoding='utf-8') as f:
        return {row['week_start'] for row in csv.DictReader(f)}


def fetch_page(query, begin_date, end_date, page):
    """Fetch one page from NYT Article Search API; retries on rate limits."""
    params = {
        'q': query,
        'begin_date': begin_date,
        'end_date': end_date,
        'page': page,
        'api-key': next(_key_cycle),
    }
    while True:
        try:
            r = requests.get(NYT_BASE_URL, params=params, timeout=30)
            if r.status_code == 429:
                print('      [429] rate limit — rotating key and waiting 60s...')
                params['api-key'] = next(_key_cycle)
                time.sleep(60)
                continue
            r.raise_for_status()
            data = r.json()
            if 'fault' in data:
                msg = data['fault'].get('faultstring', 'rate limit')
                print(f'      [fault] {msg} — rotating key and waiting 60s...')
                params['api-key'] = next(_key_cycle)
                time.sleep(60)
                continue
            if data.get('status') != 'OK':
                print(f"      [error] status={data.get('status')}")
                return []
            return data.get('response', {}).get('docs') or []
        except requests.exceptions.RequestException as e:
            print(f'      [request error] {e} — waiting 15s...')
            time.sleep(15)


# Q1-Q4, Q9-Q10 are high-volume: 2 pages. Q5-Q8 are low-volume: 1 page.
QUERY_PAGES = {1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2}


def collect_week(week_start):
    """
    Collect articles for one week using per-query page limits.
    High-volume queries (Q1-Q4, Q9-Q10): 2 pages. Low-volume (Q5-Q8): 1 page.
    Returns deduplicated list of article dicts.
    """
    end = week_start + timedelta(days=6)
    begin_str = week_start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')
    week_str = week_start.strftime('%Y-%m-%d')

    articles = []
    seen_headlines = set()
    counter = 1

    for q_num, query in enumerate(QUERIES, 1):
        pages = QUERY_PAGES[q_num]
        for page in range(pages):
            label = f'Q{q_num} page {page + 1}/{pages}'
            print(f'    {label} ({begin_str} to {end_str}) ...', end=' ', flush=True)

            docs = fetch_page(query, begin_str, end_str, page)
            added = 0

            for doc in docs:
                headline = (doc.get('headline') or {}).get('main', '').strip()
                if not headline or headline in seen_headlines:
                    continue
                seen_headlines.add(headline)

                articles.append({
                    'article_id': f'NYT-{begin_str}-{str(counter).zfill(3)}',
                    'source': 'NYT',
                    'query_number': q_num,
                    'week_start': week_str,
                    'publication_date': (doc.get('pub_date') or '')[:10],
                    'headline': headline,
                    'abstract': (doc.get('abstract') or '').replace('\n', ' ').strip(),
                    'section': doc.get('section_name') or '',
                    'url': doc.get('web_url') or '',
                })
                counter += 1
                added += 1

            print(f'{added} articles')
            time.sleep(SLEEP_SECS)

    return articles


def append_to_csv(articles):
    """Append article dicts to raw_articles.csv (creates header if new file)."""
    if not articles:
        return
    file_exists = os.path.exists(RAW_CSV) and os.path.getsize(RAW_CSV) > 0
    with open(RAW_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(articles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Collect NYT articles for pipeline')
    parser.add_argument('--test', action='store_true',
                        help='Collect 2 weeks only (smoke test)')
    args = parser.parse_args()

    if not NYT_API_KEYS:
        raise SystemExit('No NYT API keys found. Set NYT_API_KEY_1 … NYT_API_KEY_5 (or NYT_API_KEY) in .env')
    print(f'API keys loaded : {len(NYT_API_KEYS)}')
    print(f'Sleep per call  : {SLEEP_SECS:.1f}s')

    os.makedirs('data', exist_ok=True)

    weeks = get_target_weeks()
    collected = load_collected_weeks()
    remaining = [w for w in weeks if w.strftime('%Y-%m-%d') not in collected]
    remaining = remaining[::-1]  # collect most recent weeks first (backwards to 2019)

    if args.test:
        remaining = remaining[:2]
        print('-- TEST MODE: 2 weeks --')

    est_min = len(remaining) * len(QUERIES) * 3 * SLEEP_SECS / 60
    print(f'Target weeks : {len(weeks)}')
    print(f'Already done : {len(collected)}')
    print(f'Remaining    : {len(remaining)}')
    print(f'Est. time    : {est_min:.0f} min')
    print()

    total_articles = 0
    for i, week in enumerate(remaining, 1):
        print(f'[{i}/{len(remaining)}] Week {week.strftime("%Y-%m-%d")}')
        articles = collect_week(week)
        append_to_csv(articles)
        total_articles += len(articles)
        print(f'    => {len(articles)} written  (running total: {total_articles})')

    print(f'\nCollection complete. {total_articles} new articles written to {RAW_CSV}')


if __name__ == '__main__':
    main()
