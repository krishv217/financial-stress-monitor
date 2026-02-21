"""
News Data Module
Fetches financial news from NewsAPI (recent) and GDELT (historical)
"""

import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from gdeltdoc import GdeltDoc, Filters

load_dotenv()

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Financial stress keywords for querying
FINANCIAL_KEYWORDS = [
    'federal reserve',
    'inflation',
    'credit crisis',
    'recession',
    'banking stress',
    'yield curve',
    'liquidity crisis',
    'interest rates',
    'financial markets',
    'stock market crash'
]


def fetch_newsapi_headlines(keywords=None, days_back=30, language='en', page_size=100):
    """
    Fetch recent news headlines from NewsAPI.

    Args:
        keywords: List of keywords to search (default: FINANCIAL_KEYWORDS)
        days_back: How many days back to fetch (max 30 for free tier)
        language: Language code (default: 'en')
        page_size: Number of results per page (max 100)

    Returns:
        pandas DataFrame with columns: date, headline, source, url
    """
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY not found in environment variables")

    if keywords is None:
        keywords = FINANCIAL_KEYWORDS

    # NewsAPI free tier can only go back 30 days
    days_back = min(days_back, 30)

    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    # Build query string (OR operation)
    query = ' OR '.join([f'"{kw}"' for kw in keywords])

    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': language,
        'sortBy': 'publishedAt',
        'pageSize': page_size,
        'apiKey': NEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'ok':
            raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")

        articles = data.get('articles', [])

        if not articles:
            print("No articles found from NewsAPI")
            return pd.DataFrame(columns=['date', 'headline', 'source', 'url'])

        # Parse into standardized format
        news_data = []
        for article in articles:
            news_data.append({
                'date': pd.to_datetime(article.get('publishedAt')),
                'headline': article.get('title', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'url': article.get('url', '')
            })

        df = pd.DataFrame(news_data)
        df = df.dropna(subset=['headline'])
        df = df[df['headline'] != '']

        print(f"Fetched {len(df)} articles from NewsAPI")
        return df

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching NewsAPI data: {e}")


def fetch_gdelt_headlines(keywords=None, start_date='2015-01-01', end_date=None, max_records=5000):
    """
    Fetch historical news headlines from GDELT.

    Args:
        keywords: List of keywords to search (default: FINANCIAL_KEYWORDS)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (default: today)
        max_records: Maximum number of records to fetch

    Returns:
        pandas DataFrame with columns: date, headline, source, url
    """
    if keywords is None:
        keywords = FINANCIAL_KEYWORDS

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Initialize GDELT doc client
    gd = GdeltDoc()

    all_articles = []

    # Query for each keyword (GDELT works better with individual keywords)
    for keyword in keywords[:3]:  # Limit to first 3 keywords to avoid rate limits
        try:
            print(f"Fetching GDELT data for '{keyword}'...")

            # Search articles using Filters object (gdeltdoc >= 1.4)
            f = Filters(
                keyword=keyword,
                start_date=start_date,
                end_date=end_date
            )
            articles = gd.article_search(f)
            time.sleep(2)  # Avoid GDELT rate limits

            if articles is not None and not articles.empty:
                all_articles.append(articles)
                print(f"  Found {len(articles)} articles")

        except Exception as e:
            print(f"  Warning: Error fetching GDELT data for '{keyword}': {e}")
            continue

    if not all_articles:
        print("No articles found from GDELT")
        return pd.DataFrame(columns=['date', 'headline', 'source', 'url'])

    # Combine all results
    combined_df = pd.concat(all_articles, ignore_index=True)

    # Standardize column names
    df = pd.DataFrame({
        'date': pd.to_datetime(combined_df.get('seendate', combined_df.get('date', ''))),
        'headline': combined_df.get('title', ''),
        'source': combined_df.get('domain', 'GDELT'),
        'url': combined_df.get('url', '')
    })

    # Clean up
    df = df.dropna(subset=['headline'])
    df = df[df['headline'] != '']
    df = df.drop_duplicates(subset=['headline'])

    print(f"Total GDELT articles after deduplication: {len(df)}")
    return df


def save_news_data(df, filename):
    """
    Save news data to CSV file.

    Args:
        df: pandas DataFrame with news data
        filename: output CSV filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} news articles to {filename}")


def load_news_data(filename):
    """
    Load news data from CSV file.

    Args:
        filename: CSV filename to load

    Returns:
        pandas DataFrame with news data
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    return df


if __name__ == '__main__':
    # Test NewsAPI
    print("=" * 60)
    print("Testing NewsAPI...")
    print("=" * 60)
    try:
        recent_news = fetch_newsapi_headlines(days_back=7)
        print(f"\nSample headlines:")
        print(recent_news.head())
        save_news_data(recent_news, 'data/news_recent.csv')
    except Exception as e:
        print(f"Error with NewsAPI: {e}")

    # Test GDELT
    print("\n" + "=" * 60)
    print("Testing GDELT...")
    print("=" * 60)
    try:
        # Test with small date range
        historical_news = fetch_gdelt_headlines(
            start_date='2024-01-01',
            end_date='2024-01-31',
            max_records=100
        )
        print(f"\nSample headlines:")
        print(historical_news.head())
        save_news_data(historical_news, 'data/news_historical_test.csv')
    except Exception as e:
        print(f"Error with GDELT: {e}")
