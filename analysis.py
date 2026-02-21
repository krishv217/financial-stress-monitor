"""
Analysis Module
Performs lead-lag correlation analysis between news sentiment and FRED stress index
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def aggregate_news_by_week(news_df, date_column='date', direction_column='direction'):
    """
    Aggregate news sentiment by week.

    Args:
        news_df: DataFrame with classified news
        date_column: Name of date column
        direction_column: Name of direction column

    Returns:
        DataFrame with weekly aggregated sentiment scores
    """
    if news_df.empty:
        return pd.DataFrame(columns=['week_start', 'sentiment_score', 'article_count'])

    df = news_df.copy()

    # Create week start date (Monday)
    df['week_start'] = df[date_column].dt.to_period('W').apply(lambda r: r.start_time)

    # Calculate weekly sentiment scores
    weekly_data = []

    for week, group in df.groupby('week_start'):
        increasing = (group[direction_column] == 'increasing').sum()
        decreasing = (group[direction_column] == 'decreasing').sum()
        total = len(group)

        # Sentiment score: (% increasing - % decreasing)
        if total > 0:
            sentiment_score = (increasing - decreasing) / total
        else:
            sentiment_score = 0.0

        weekly_data.append({
            'week_start': week,
            'sentiment_score': sentiment_score,
            'article_count': total,
            'increasing_count': increasing,
            'decreasing_count': decreasing
        })

    weekly_df = pd.DataFrame(weekly_data)
    weekly_df = weekly_df.sort_values('week_start')

    return weekly_df


def align_fred_weekly(fred_df, date_column='date', value_column='value'):
    """
    Convert FRED data to weekly format (if not already).

    Args:
        fred_df: DataFrame with FRED data
        date_column: Name of date column
        value_column: Name of value column

    Returns:
        DataFrame with weekly FRED data
    """
    if fred_df.empty:
        return pd.DataFrame(columns=['week_start', 'fred_value'])

    df = fred_df.copy()

    # Create week start date
    df['week_start'] = df[date_column].dt.to_period('W').apply(lambda r: r.start_time)

    # If multiple values per week, take the last one
    weekly_df = df.groupby('week_start')[value_column].last().reset_index()
    weekly_df.columns = ['week_start', 'fred_value']

    return weekly_df


def calculate_lead_lag_correlation(news_weekly, fred_weekly, max_lag_weeks=4):
    """
    Calculate cross-correlation at various lag values.

    Tests if news sentiment changes lead FRED index changes.

    Args:
        news_weekly: DataFrame with weekly news sentiment (columns: week_start, sentiment_score)
        fred_weekly: DataFrame with weekly FRED data (columns: week_start, fred_value)
        max_lag_weeks: Maximum lag to test (in weeks)

    Returns:
        DataFrame with correlation coefficients at each lag
    """
    # Merge datasets on week_start
    merged = pd.merge(news_weekly, fred_weekly, on='week_start', how='inner')

    if len(merged) < max_lag_weeks + 1:
        print(f"Warning: Not enough data points ({len(merged)}) for lag analysis")
        return pd.DataFrame(columns=['lag_weeks', 'correlation', 'n_observations'])

    merged = merged.sort_values('week_start')

    correlations = []

    for lag in range(0, max_lag_weeks + 1):
        if lag == 0:
            # No lag - contemporaneous correlation
            corr_data = merged[['sentiment_score', 'fred_value']].corr()
            corr = corr_data.iloc[0, 1]
            n_obs = len(merged)
        else:
            # Lag: news at time t vs FRED at time t+lag
            # (Does news sentiment predict future FRED values?)
            news_series = merged['sentiment_score'].values[:-lag]
            fred_series = merged['fred_value'].values[lag:]

            if len(news_series) > 0 and len(fred_series) > 0:
                corr = np.corrcoef(news_series, fred_series)[0, 1]
                n_obs = len(news_series)
            else:
                corr = np.nan
                n_obs = 0

        correlations.append({
            'lag_weeks': lag,
            'correlation': corr,
            'n_observations': n_obs
        })

    corr_df = pd.DataFrame(correlations)
    return corr_df


def detect_divergence(news_weekly, fred_weekly, threshold=0.3):
    """
    Detect periods where news sentiment and FRED index diverge.

    Args:
        news_weekly: DataFrame with weekly news sentiment
        fred_weekly: DataFrame with weekly FRED data
        threshold: Minimum normalized difference to flag as divergence

    Returns:
        DataFrame with divergence flags
    """
    merged = pd.merge(news_weekly, fred_weekly, on='week_start', how='inner')

    if merged.empty:
        return pd.DataFrame(columns=['week_start', 'sentiment_score', 'fred_value', 'divergence'])

    # Normalize both to 0-1 scale for comparison
    sentiment_norm = (merged['sentiment_score'] - merged['sentiment_score'].min()) / (
        merged['sentiment_score'].max() - merged['sentiment_score'].min() + 1e-10
    )

    fred_norm = (merged['fred_value'] - merged['fred_value'].min()) / (
        merged['fred_value'].max() - merged['fred_value'].min() + 1e-10
    )

    # Calculate difference
    diff = np.abs(sentiment_norm - fred_norm)

    merged['divergence'] = diff > threshold
    merged['divergence_magnitude'] = diff

    return merged


def fit_narrative_to_fred_regression(news_weekly, fred_weekly, max_lag_weeks=4):
    """
    Fit OLS linear regression: FRED(t + best_lag) = slope * sentiment(t) + intercept.

    Selects the lag with the highest absolute cross-correlation, then fits a
    least-squares line over the full overlapping history at that lag.

    Returns:
        dict with keys slope, intercept, r_squared, best_lag, n_observations
        or None if there are fewer than 5 aligned observations.
    """
    corr_df = calculate_lead_lag_correlation(news_weekly, fred_weekly, max_lag_weeks)
    if corr_df.empty:
        return None

    best_lag = int(corr_df.loc[corr_df['correlation'].abs().idxmax(), 'lag_weeks'])

    merged = pd.merge(news_weekly, fred_weekly, on='week_start', how='inner').sort_values('week_start')

    if best_lag == 0:
        X = merged['sentiment_score'].values
        y = merged['fred_value'].values
    else:
        X = merged['sentiment_score'].values[:-best_lag]
        y = merged['fred_value'].values[best_lag:]

    mask = ~(np.isnan(X) | np.isnan(y))
    X, y = X[mask], y[mask]

    if len(X) < 5:
        return None

    slope, intercept = np.polyfit(X, y, 1)

    y_pred = slope * X + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': r_squared,
        'best_lag': best_lag,
        'n_observations': int(len(X)),
    }


def generate_analysis_summary(corr_df):
    """
    Generate plain English summary of correlation analysis.

    Args:
        corr_df: DataFrame with correlation results

    Returns:
        String with analysis summary
    """
    if corr_df.empty:
        return "Insufficient data for analysis."

    # Find lag with highest correlation
    max_corr_row = corr_df.loc[corr_df['correlation'].idxmax()]
    lag_weeks = int(max_corr_row['lag_weeks'])
    max_corr = max_corr_row['correlation']

    summary = f"""Lead-Lag Analysis Summary:

The strongest correlation ({max_corr:.3f}) occurs at a {lag_weeks}-week lag.
"""

    if lag_weeks == 0:
        summary += "This suggests news sentiment and FRED stress index move contemporaneously."
    elif lag_weeks == 1:
        summary += "This suggests news sentiment shifts tend to precede FRED index movements by approximately 1 week."
    else:
        summary += f"This suggests news sentiment shifts tend to precede FRED index movements by approximately {lag_weeks} weeks."

    if max_corr > 0.5:
        summary += "\n\nThe correlation is strong, indicating a meaningful relationship."
    elif max_corr > 0.3:
        summary += "\n\nThe correlation is moderate, indicating a noticeable relationship."
    else:
        summary += "\n\nThe correlation is weak, suggesting limited predictive power."

    return summary


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing analysis module with synthetic data...")
    print("=" * 60)

    # Create synthetic news data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    news_data = {
        'date': dates,
        'headline': ['Test headline'] * len(dates),
        'direction': np.random.choice(['increasing', 'decreasing', 'neutral'], len(dates))
    }
    news_df = pd.DataFrame(news_data)

    # Create synthetic FRED data (weekly)
    fred_dates = pd.date_range('2023-01-01', '2024-01-01', freq='W-MON')
    fred_data = {
        'date': fred_dates,
        'value': np.random.randn(len(fred_dates))
    }
    fred_df = pd.DataFrame(fred_data)

    # Aggregate news by week
    print("\nAggregating news by week...")
    news_weekly = aggregate_news_by_week(news_df)
    print(f"Created {len(news_weekly)} weekly observations")
    print(news_weekly.head())

    # Align FRED data weekly
    print("\nAligning FRED data weekly...")
    fred_weekly = align_fred_weekly(fred_df)
    print(f"Created {len(fred_weekly)} weekly observations")
    print(fred_weekly.head())

    # Calculate correlations
    print("\nCalculating lead-lag correlations...")
    corr_df = calculate_lead_lag_correlation(news_weekly, fred_weekly, max_lag_weeks=4)
    print(corr_df)

    # Generate summary
    print("\n" + "=" * 60)
    summary = generate_analysis_summary(corr_df)
    print(summary)

    # Detect divergence
    print("\n" + "=" * 60)
    print("Detecting divergence periods...")
    divergence_df = detect_divergence(news_weekly, fred_weekly)
    divergent_weeks = divergence_df[divergence_df['divergence'] == True]
    print(f"Found {len(divergent_weeks)} divergent weeks out of {len(divergence_df)}")
