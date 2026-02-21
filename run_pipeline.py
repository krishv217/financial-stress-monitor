"""
Complete Data Pipeline Runner
Orchestrates data collection, classification, and storage
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd

from fred_data import fetch_fred_data, save_fred_data, load_fred_data
from news_data import fetch_newsapi_headlines, fetch_gdelt_headlines, save_news_data
from classifier import classify_news_dataframe


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_environment():
    """Check if all required environment variables are set."""
    print_section("Environment Check")

    required_vars = ['FRED_API_KEY', 'NEWS_API_KEY', 'ANTHROPIC_API_KEY']
    missing = []

    for var in required_vars:
        if os.getenv(var):
            print(f"✓ {var} found")
        else:
            print(f"✗ {var} missing")
            missing.append(var)

    if missing:
        print(f"\nError: Missing environment variables: {', '.join(missing)}")
        print("Please create a .env file with your API keys.")
        print("See .env.example for the required format.")
        return False

    print("\n✓ All environment variables configured")
    return True


def fetch_fred_pipeline():
    """Fetch FRED data."""
    print_section("Step 1: Fetching FRED Data")

    try:
        # Check if we already have recent data
        try:
            existing_df = load_fred_data()
            last_date = existing_df['date'].max()
            days_old = (datetime.now() - last_date).days

            if days_old < 7:
                print(f"✓ Recent FRED data exists (last update: {last_date.strftime('%Y-%m-%d')})")
                response = input("Fetch fresh data anyway? (y/n): ").lower()
                if response != 'y':
                    print("Using existing FRED data")
                    return existing_df
        except FileNotFoundError:
            print("No existing FRED data found")

        # Fetch historical data from 2015
        print("Fetching FRED STLFSI4 data from 2015 to present...")
        fred_df = fetch_fred_data(start_date='2015-01-01')

        # Save to file
        save_fred_data(fred_df)

        print(f"\n✓ FRED data collected: {len(fred_df)} observations")
        print(f"  Date range: {fred_df['date'].min().strftime('%Y-%m-%d')} to {fred_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Current stress score: {fred_df['value'].iloc[-1]:.4f}")

        return fred_df

    except Exception as e:
        print(f"✗ Error fetching FRED data: {e}")
        return None


def fetch_news_pipeline():
    """Fetch news data from multiple sources."""
    print_section("Step 2: Fetching News Data")

    all_news = []

    # Fetch recent news from NewsAPI
    print("\nFetching recent news from NewsAPI (last 30 days)...")
    try:
        newsapi_df = fetch_newsapi_headlines(days_back=30)
        if not newsapi_df.empty:
            all_news.append(newsapi_df)
            save_news_data(newsapi_df, 'data/news_recent.csv')
            print(f"✓ NewsAPI: {len(newsapi_df)} articles")
    except Exception as e:
        print(f"✗ NewsAPI error: {e}")

    # Fetch recent news from GDELT (last 90 days — GDELT DOC 2.0 limit)
    print("\nFetching recent news from GDELT (last 90 days)...")

    try:
        gdelt_start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        gdelt_end = datetime.now().strftime('%Y-%m-%d')

        gdelt_df = fetch_gdelt_headlines(
            start_date=gdelt_start,
            end_date=gdelt_end,
            max_records=1000
        )

        if not gdelt_df.empty:
            all_news.append(gdelt_df)
            save_news_data(gdelt_df, 'data/news_historical.csv')
            print(f"✓ GDELT: {len(gdelt_df)} articles")
        else:
            print("No articles from GDELT")

    except Exception as e:
        print(f"✗ GDELT error: {e}")

    # Combine all news
    if all_news:
        combined_news = pd.concat(all_news, ignore_index=True)
        combined_news = combined_news.drop_duplicates(subset=['headline'])
        combined_news = combined_news.sort_values('date')

        print(f"\n✓ Total news collected: {len(combined_news)} unique articles")
        print(f"  Date range: {combined_news['date'].min().strftime('%Y-%m-%d')} to {combined_news['date'].max().strftime('%Y-%m-%d')}")

        return combined_news
    else:
        print("✗ No news data collected")
        return None


def classify_news_pipeline(news_df):
    """Classify news headlines using Claude API."""
    print_section("Step 3: Classifying News Headlines")

    if news_df is None or news_df.empty:
        print("✗ No news data to classify")
        return None

    # Check if already classified
    if 'theme' in news_df.columns and 'direction' in news_df.columns:
        print("News data already contains classifications")
        response = input("Re-classify all headlines? (y/n): ").lower()
        if response != 'y':
            print("Using existing classifications")
            return news_df

    print(f"\nClassifying {len(news_df)} headlines using Claude API...")
    print("This may take several minutes depending on the number of articles...")
    print("(Processing in batches of 20 headlines)")

    try:
        classified_df = classify_news_dataframe(news_df, batch_size=20)

        # Save classified data
        save_news_data(classified_df, 'data/news_classified.csv')

        print(f"\n✓ Classification complete")
        print("\nTheme distribution:")
        print(classified_df['theme'].value_counts())
        print("\nDirection distribution:")
        print(classified_df['direction'].value_counts())

        return classified_df

    except Exception as e:
        print(f"✗ Classification error: {e}")
        print("Saving unclassified data for manual inspection...")
        save_news_data(news_df, 'data/news_unclassified.csv')
        return None


def main():
    """Run the complete data pipeline."""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║          FINANCIAL STRESS MONITOR - DATA PIPELINE              ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Confirm before proceeding
    print("\nThis pipeline will:")
    print("  1. Fetch FRED financial stress data (2015-present)")
    print("  2. Fetch news headlines from NewsAPI and GDELT")
    print("  3. Classify headlines using Claude API (may incur API costs)")
    print("\nEstimated time: 15-30 minutes")

    response = input("\nProceed? (y/n): ").lower()
    if response != 'y':
        print("Pipeline cancelled")
        sys.exit(0)

    start_time = datetime.now()

    # Step 1: Fetch FRED data
    fred_df = fetch_fred_pipeline()

    if fred_df is None:
        print("\n✗ Pipeline failed: Could not fetch FRED data")
        sys.exit(1)

    # Step 2: Fetch news data
    news_df = fetch_news_pipeline()

    if news_df is None:
        print("\n✗ Pipeline failed: Could not fetch news data")
        sys.exit(1)

    # Step 3: Classify news
    classified_df = classify_news_pipeline(news_df)

    # Summary
    print_section("Pipeline Complete")

    elapsed = (datetime.now() - start_time).total_seconds() / 60

    print(f"\n✓ Pipeline completed successfully in {elapsed:.1f} minutes")
    print("\nData files created:")
    print("  - data/fred_historical.csv")
    print("  - data/news_recent.csv")
    print("  - data/news_historical.csv")
    if classified_df is not None:
        print("  - data/news_classified.csv")

    print("\nNext steps:")
    print("  1. Run: streamlit run app.py")
    print("  2. Open browser to http://localhost:8501")
    print("  3. Explore the dashboard!")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
