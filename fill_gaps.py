"""
Gap-fill collector — targets sparse/missing weeks in raw_articles.csv.

Collects API articles for every week between GAP_START and GAP_END where
the existing raw_articles.csv has fewer than MIN_ARTICLES from the NYT API
(source='NYT').  Kaggle-only weeks count as uncollected.

Appends results to data/raw_articles.csv.  Phase 2 dedup will remove any
headline overlap with existing Kaggle rows.

Usage:
  python fill_gaps.py           # fill 2023-05-22 to 2025-07-28
  python fill_gaps.py --test    # first 2 gap weeks only
"""

import os
import csv
import time
import argparse
import itertools
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config — same queries/keys as collect_nyt.py
# ---------------------------------------------------------------------------

NYT_BASE_URL = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'

_raw_keys = [os.getenv(f'NYT_API_KEY_{i}') for i in range(1, 8)]
NYT_API_KEYS = [k for k in _raw_keys if k] or [os.getenv('NYT_API_KEY')]
NYT_API_KEYS = [k for k in NYT_API_KEYS if k]
SLEEP_SECS   = max(1.0, 7 / len(NYT_API_KEYS))
_key_cycle   = itertools.cycle(NYT_API_KEYS)

QUERIES = [
    'federal reserve OR interest rates OR monetary policy',
    'inflation OR consumer prices OR stagflation',
    'recession OR economic growth OR GDP',
    'unemployment OR labor market OR jobs',
    'yield curve OR bond market OR credit markets',
    'debt ceiling OR sovereign debt OR corporate bonds',
    'bank failure OR banking crisis OR financial stability',
    'liquidity OR deposit insurance OR FDIC',
    'tariffs OR trade war OR sanctions',
    'stock market OR market volatility OR financial markets',
]
QUERY_PAGES = {1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2}

GAP_START    = datetime(2023, 5, 22)
GAP_END      = datetime(2025, 7, 28)
MIN_ARTICLES = 30   # weeks with fewer API-sourced articles than this get refetched

RAW_CSV    = 'data/raw_articles.csv'
FIELDNAMES = [
    'article_id', 'source', 'query_number', 'week_start',
    'publication_date', 'headline', 'abstract', 'section', 'url',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_gap_weeks():
    """Return list of Mondays between GAP_START and GAP_END inclusive."""
    weeks, d = [], GAP_START
    while d <= GAP_END:
        weeks.append(d)
        d += timedelta(weeks=1)
    return weeks


def load_api_covered_weeks():
    """Return set of week_start strings that already have >= MIN_ARTICLES API rows."""
    if not os.path.exists(RAW_CSV) or os.path.getsize(RAW_CSV) == 0:
        return set()
    df = pd.read_csv(RAW_CSV, dtype=str, usecols=['source', 'week_start']).fillna('')
    api_rows = df[df['source'] == 'NYT']
    counts = api_rows.groupby('week_start').size()
    return set(counts[counts >= MIN_ARTICLES].index)


def fetch_page(query, begin_date, end_date, page):
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
                print('      [429] rate limit -- rotating key and waiting 60s...')
                params['api-key'] = next(_key_cycle)
                time.sleep(60)
                continue
            r.raise_for_status()
            data = r.json()
            if 'fault' in data:
                msg = data['fault'].get('faultstring', 'rate limit')
                print(f'      [fault] {msg} -- rotating key and waiting 60s...')
                params['api-key'] = next(_key_cycle)
                time.sleep(60)
                continue
            if data.get('status') != 'OK':
                print(f"      [error] status={data.get('status')}")
                return []
            return data.get('response', {}).get('docs') or []
        except requests.exceptions.RequestException as e:
            print(f'      [request error] {e} -- waiting 15s...')
            time.sleep(15)


def collect_week(week_start):
    end = week_start + timedelta(days=6)
    begin_str = week_start.strftime('%Y%m%d')
    end_str   = end.strftime('%Y%m%d')
    week_str  = week_start.strftime('%Y-%m-%d')

    articles = []
    seen_headlines = set()
    counter = 1

    for q_num, query in enumerate(QUERIES, 1):
        pages = QUERY_PAGES[q_num]
        for page in range(pages):
            label = f'Q{q_num} page {page+1}/{pages}'
            print(f'    {label} ({begin_str}-{end_str}) ...', end=' ', flush=True)

            docs = fetch_page(query, begin_str, end_str, page)
            added = 0
            for doc in docs:
                headline = (doc.get('headline') or {}).get('main', '').strip()
                if not headline or headline in seen_headlines:
                    continue
                seen_headlines.add(headline)
                articles.append({
                    'article_id':       f'NYT-{begin_str}-{str(counter).zfill(6)}',
                    'source':           'NYT',
                    'query_number':     q_num,
                    'week_start':       week_str,
                    'publication_date': (doc.get('pub_date') or '')[:10],
                    'headline':         headline,
                    'abstract':         (doc.get('abstract') or '').replace('\n', ' ').strip(),
                    'section':          doc.get('section_name') or '',
                    'url':              doc.get('web_url') or '',
                })
                counter += 1
                added += 1
            print(f'{added} articles')
            time.sleep(SLEEP_SECS)

    return articles


def append_to_csv(articles):
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
    parser = argparse.ArgumentParser(description='Fill sparse/missing gap weeks in raw_articles.csv')
    parser.add_argument('--test', action='store_true', help='Collect first 2 gap weeks only')
    args = parser.parse_args()

    if not NYT_API_KEYS:
        raise SystemExit('No NYT API keys found.')
    print(f'API keys loaded : {len(NYT_API_KEYS)}')
    print(f'Sleep per call  : {SLEEP_SECS:.1f}s')
    print(f'Min articles    : {MIN_ARTICLES} (weeks below this threshold get refetched)')

    os.makedirs('data', exist_ok=True)

    all_gap_weeks  = get_gap_weeks()
    covered        = load_api_covered_weeks()
    to_collect     = [w for w in all_gap_weeks if w.strftime('%Y-%m-%d') not in covered]
    # Most recent first so if quota runs out we have the latest data
    to_collect     = to_collect[::-1]

    if args.test:
        to_collect = to_collect[:2]
        print('-- TEST MODE: 2 weeks --')

    est_calls = len(to_collect) * sum(QUERY_PAGES.values())
    est_min   = est_calls * SLEEP_SECS / 60
    print(f'\nGap range      : {GAP_START.date()} to {GAP_END.date()}')
    print(f'Total gap weeks: {len(all_gap_weeks)}')
    print(f'Already covered: {len(all_gap_weeks) - len(to_collect)}')
    print(f'To collect     : {len(to_collect)}')
    print(f'Est. API calls : {est_calls}')
    print(f'Est. time      : {est_min:.0f} min\n')

    total_articles = 0
    for i, week in enumerate(to_collect, 1):
        print(f'[{i}/{len(to_collect)}] Week {week.strftime("%Y-%m-%d")}')
        articles = collect_week(week)
        append_to_csv(articles)
        total_articles += len(articles)
        print(f'    => {len(articles)} written  (running total: {total_articles})')

    print(f'\nGap fill complete. {total_articles} new articles written to {RAW_CSV}')


if __name__ == '__main__':
    main()
