"""
LLM Classification Module
Uses Claude API to classify news headlines by stress theme and direction
"""

import os
import json
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import time

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Stress themes
STRESS_THEMES = [
    'credit_risk',
    'inflation_risk',
    'liquidity_risk',
    'geopolitical_risk',
    'banking_risk',
    'none'
]

# Stress directions
STRESS_DIRECTIONS = [
    'increasing',
    'decreasing',
    'neutral'
]


def create_classification_prompt(headlines):
    """
    Create a structured prompt for headline classification.

    Args:
        headlines: List of headline strings

    Returns:
        Formatted prompt string
    """
    headlines_json = json.dumps(headlines, indent=2)

    prompt = f"""You are a financial analyst tasked with classifying news headlines for financial stress monitoring.

For each headline below, determine:
1. The primary stress theme (if any)
2. The stress direction (whether it suggests stress is increasing, decreasing, or neutral)

**Stress Themes:**
- credit_risk: Issues related to creditworthiness, defaults, debt problems
- inflation_risk: Inflation concerns, price increases, monetary policy tightening
- liquidity_risk: Cash flow problems, market liquidity issues, funding stress
- geopolitical_risk: International conflicts, trade tensions, political instability
- banking_risk: Banking sector problems, bank failures, financial institution stress
- none: Not related to financial stress

**Stress Directions:**
- increasing: Headline suggests stress is rising or worsening
- decreasing: Headline suggests stress is declining or improving
- neutral: Headline is informational without clear directional signal

Return your analysis as a JSON array with one object per headline, in the same order as the input.

**Headlines to classify:**
{headlines_json}

Respond ONLY with valid JSON in this exact format:
[
  {{"theme": "inflation_risk", "direction": "increasing"}},
  {{"theme": "banking_risk", "direction": "decreasing"}},
  ...
]"""

    return prompt


def classify_headlines_batch(headlines, batch_size=20, model="claude-haiku-4-5-20251001"):
    """
    Classify headlines using Claude API in batches.

    Args:
        headlines: List of headline strings
        batch_size: Number of headlines per API call
        model: Claude model to use

    Returns:
        List of dicts with 'theme' and 'direction' keys
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    all_classifications = []

    # Process in batches
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        print(f"Classifying headlines {i + 1} to {min(i + batch_size, len(headlines))}...")

        prompt = create_classification_prompt(batch)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract JSON from response
            response_text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json or ```) and last line (```)
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            # Try to parse JSON
            classifications = json.loads(response_text)

            if not isinstance(classifications, list):
                raise ValueError("Response is not a JSON array")

            if len(classifications) != len(batch):
                print(f"Warning: Expected {len(batch)} classifications, got {len(classifications)}")
                # Pad with defaults or truncate to match batch size
                while len(classifications) < len(batch):
                    classifications.append({"theme": "none", "direction": "neutral"})
                classifications = classifications[:len(batch)]

            all_classifications.extend(classifications)

            # Rate limiting - small delay between batches
            time.sleep(1)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response text: {response_text[:500]}")
            # Add None classifications for failed batch
            all_classifications.extend([{"theme": "none", "direction": "neutral"}] * len(batch))

        except Exception as e:
            print(f"Error calling Claude API: {e}")
            # Add None classifications for failed batch
            all_classifications.extend([{"theme": "none", "direction": "neutral"}] * len(batch))

    return all_classifications


def classify_news_dataframe(df, headline_column='headline', batch_size=20):
    """
    Classify headlines in a DataFrame and add classification columns.

    Args:
        df: pandas DataFrame with news data
        headline_column: Name of column containing headlines
        batch_size: Batch size for API calls

    Returns:
        DataFrame with added 'theme' and 'direction' columns
    """
    if df.empty:
        df['theme'] = []
        df['direction'] = []
        return df

    headlines = df[headline_column].tolist()

    print(f"Classifying {len(headlines)} headlines...")
    classifications = classify_headlines_batch(headlines, batch_size=batch_size)

    # Add classifications to DataFrame
    df = df.copy()
    df['theme'] = [c.get('theme', 'none') for c in classifications]
    df['direction'] = [c.get('direction', 'neutral') for c in classifications]

    print(f"Classification complete.")
    print(f"Theme distribution:")
    print(df['theme'].value_counts())
    print(f"\nDirection distribution:")
    print(df['direction'].value_counts())

    return df


def calculate_stress_score(df, date_column='date', direction_column='direction'):
    """
    Calculate aggregate stress score by counting directional sentiment.

    Args:
        df: DataFrame with classified news
        date_column: Name of date column
        direction_column: Name of direction column

    Returns:
        Float between -1 and 1 representing stress sentiment
        (positive = stress increasing, negative = stress decreasing)
    """
    if df.empty:
        return 0.0

    direction_counts = df[direction_column].value_counts()

    increasing = direction_counts.get('increasing', 0)
    decreasing = direction_counts.get('decreasing', 0)
    total = len(df)

    if total == 0:
        return 0.0

    # Score: (% increasing - % decreasing)
    score = (increasing - decreasing) / total

    return score


if __name__ == '__main__':
    # Test with sample headlines
    test_headlines = [
        "Federal Reserve raises interest rates to combat inflation",
        "Banking crisis eases as regulators step in",
        "Stock market rallies on positive economic data",
        "Credit default rates surge to decade high",
        "Central bank provides emergency liquidity to markets",
        "Inflation shows signs of cooling in latest report",
        "Geopolitical tensions escalate in Middle East",
        "Tech stocks lead market gains"
    ]

    print("Testing classifier with sample headlines...")
    print("=" * 60)

    # Create test DataFrame
    test_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=len(test_headlines)),
        'headline': test_headlines,
        'source': 'Test'
    })

    # Classify
    classified_df = classify_news_dataframe(test_df)

    print("\nClassified headlines:")
    print(classified_df[['headline', 'theme', 'direction']])

    # Calculate stress score
    score = calculate_stress_score(classified_df)
    print(f"\nOverall stress score: {score:.3f}")
