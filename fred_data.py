"""
FRED API Integration Module
Fetches St. Louis Fed Financial Stress Index (STLFSI4) data
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv('FRED_API_KEY')
FRED_BASE_URL = 'https://api.stlouisfed.org/fred/series/observations'
SERIES_ID = 'STLFSI4'


def fetch_fred_data(start_date='2015-01-01', end_date=None):
    """
    Fetch FRED STLFSI4 time series data.

    Args:
        start_date: Start date in YYYY-MM-DD format (default: 2015-01-01)
        end_date: End date in YYYY-MM-DD format (default: today)

    Returns:
        pandas DataFrame with columns: date, value
    """
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY not found in environment variables")

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    params = {
        'series_id': SERIES_ID,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date
    }

    try:
        response = requests.get(FRED_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if 'observations' not in data:
            raise ValueError(f"Unexpected API response format: {data}")

        # Parse observations into DataFrame
        observations = data['observations']
        df = pd.DataFrame(observations)

        # Keep only date and value columns
        df = df[['date', 'value']].copy()

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Convert value to numeric, replacing '.' with NaN
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Drop any rows with missing values
        df = df.dropna()

        print(f"Fetched {len(df)} observations from {df['date'].min()} to {df['date'].max()}")

        return df

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching FRED data: {e}")


def save_fred_data(df, filename='data/fred_historical.csv'):
    """
    Save FRED data to CSV file.

    Args:
        df: pandas DataFrame with FRED data
        filename: output CSV filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")


def load_fred_data(filename='data/fred_historical.csv'):
    """
    Load FRED data from CSV file.

    Args:
        filename: CSV filename to load

    Returns:
        pandas DataFrame with FRED data
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_current_stress_score(df):
    """
    Get the most recent stress score from FRED data.

    Args:
        df: pandas DataFrame with FRED data

    Returns:
        tuple: (date, value) of most recent observation
    """
    if df.empty:
        return None, None

    latest = df.loc[df['date'].idxmax()]
    return latest['date'], latest['value']


if __name__ == '__main__':
    # Test the module
    print("Fetching FRED data...")
    fred_df = fetch_fred_data()

    print("\nFirst few rows:")
    print(fred_df.head())

    print("\nLast few rows:")
    print(fred_df.tail())

    print("\nSummary statistics:")
    print(fred_df['value'].describe())

    # Save to file
    save_fred_data(fred_df)

    # Get current score
    current_date, current_score = get_current_stress_score(fred_df)
    print(f"\nCurrent stress score: {current_score:.4f} (as of {current_date.strftime('%Y-%m-%d')})")
