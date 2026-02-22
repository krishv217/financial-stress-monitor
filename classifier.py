"""
LLM Classification Module — Pipeline v2

Handles two classification tasks:
  1. Relevance filter  — is this article about macro/financial markets? (yes/no)
  2. Stress classifier — theme (5 categories) + direction + magnitude (1-3)

Also retains the legacy classify_news_dataframe() for backward compatibility
with run_pipeline.py.
"""

import os
import json
import time
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
RELEVANCE_MODEL    = 'claude-haiku-4-5-20251001'   # fast yes/no filter
CLASSIFY_MODEL     = 'claude-sonnet-4-6'            # nuanced multi-theme classification
MODEL              = CLASSIFY_MODEL                 # legacy alias
PROMPT_VERSION     = 'v3'

STRESS_THEMES = [
    'monetary_policy_risk',
    'credit_debt_risk',
    'banking_liquidity_risk',
    'inflation_growth_risk',
    'geopolitical_external_risk',
    'none',
]

STRESS_DIRECTIONS = ['increasing', 'decreasing', 'neutral']


def _client():
    if not ANTHROPIC_API_KEY:
        raise ValueError('ANTHROPIC_API_KEY not set')
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def _strip_fences(text):
    """Strip markdown code fences from LLM JSON response."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# Task 1: Relevance filter
# ---------------------------------------------------------------------------

def _relevance_prompt(items):
    """
    items: list of {'headline': ..., 'abstract': ...}
    Returns prompt asking for yes/no per article.
    """
    numbered = '\n'.join(
        f'{i+1}. {it["headline"]}. {it["abstract"]}'
        for i, it in enumerate(items)
    )
    return f"""For each article below, answer: is this article relevant to macroeconomic conditions or financial markets?
Answer only "yes" or "no" for each article.

{numbered}

Return a JSON array of strings, one per article, in order. Example: ["yes","no","yes"]
Respond ONLY with valid JSON."""


def filter_relevance(articles, batch_size=30):
    """
    articles: list of dicts with 'article_id', 'headline', 'abstract'
    Returns dict mapping article_id -> 'yes' | 'no'
    """
    client = _client()
    results = {}

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        print(f'  Relevance filter {i+1}-{min(i+batch_size, len(articles))}/{len(articles)}')
        prompt = _relevance_prompt(batch)

        try:
            resp = client.messages.create(
                model=RELEVANCE_MODEL,
                max_tokens=1024,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = _strip_fences(resp.content[0].text)
            answers = json.loads(raw)
            if not isinstance(answers, list):
                raise ValueError('not a list')
            # Pad / truncate to batch size
            while len(answers) < len(batch):
                answers.append('yes')
            answers = answers[:len(batch)]
        except Exception as e:
            print(f'    relevance filter error: {e} — defaulting to yes')
            answers = ['yes'] * len(batch)

        for art, ans in zip(batch, answers):
            results[art['article_id']] = 'yes' if str(ans).lower().startswith('y') else 'no'

        time.sleep(1)

    return results


# ---------------------------------------------------------------------------
# Task 2: Stress classification
# ---------------------------------------------------------------------------

def _classify_prompt(items):
    """
    items: list of {'headline': ..., 'abstract': ...}
    Returns classification prompt requesting 1-3 themes per article.
    """
    numbered = '\n'.join(
        f'{i+1}. {it["headline"]}. {it["abstract"]}'
        for i, it in enumerate(items)
    )
    return f"""You are a financial stress analyst calibrated to predict the St. Louis Fed Financial Stress Index (STLFSI4). For each article identify ALL relevant stress themes (1-3 maximum, most relevant first).

CRITICAL RULE — DIRECTION MUST REFLECT REALIZED US FINANCIAL STRESS, NOT NARRATIVE ANXIETY:
Score "increasing" ONLY if the event has a direct, near-term transmission mechanism into US financial system stress metrics (credit spreads, bank funding costs, interbank rates, systemic risk indicators). News coverage of a risk event alone is NOT sufficient — the event must plausibly move those metrics.

STRESS THEMES:
- monetary_policy_risk: Fed decisions, interest rate changes, Fed language shifts, QT/QE
- credit_debt_risk: credit tightening, yield spread widening, debt ceiling breach risk, corporate/sovereign default risk
- banking_liquidity_risk: bank failures, FDIC intervention, deposit flight, interbank funding stress
- inflation_growth_risk: CPI/PCE surprises, GDP contraction, recession confirmation, large unemployment spike
- geopolitical_external_risk: ONLY if there is a direct financial transmission channel (e.g. energy supply shock raising credit costs, sanctions freezing dollar liquidity, trade war measurably tightening US credit conditions). General war coverage, tariff announcements, or sanctions news WITHOUT a clear financial market impact = score as "neutral", not "increasing"
- none: not related to financial stress

STRESS DIRECTION (per theme):
- increasing: stress rising with a clear transmission to US financial system metrics
- decreasing: stress falling / conditions improving
- neutral: informational, potential risk discussed but no confirmed financial market impact yet

MAGNITUDE (per theme) — be aggressive, do not default to 1:
- 1: routine coverage, data in-line with expectations, minor developments
- 2: notable surprise, moderate market reaction, worth monitoring
- 3: major confirmed event with immediate financial system impact

ALWAYS score magnitude 3 for:
- Any Fed rate decision, pivot, or significant forward guidance shift
- Any bank failure, emergency liquidity action, or FDIC intervention
- Recession confirmation, GDP contraction, or large unemployment spike
- Debt ceiling breach risk, sovereign credit downgrade, or debt crisis signal
- Market crash, circuit breaker, or systemic risk event
- Sanctions or trade actions that have already caused measurable credit spread widening or dollar funding stress

DO NOT score magnitude 3 for:
- Tariff announcements or geopolitical events where financial market impact is speculative
- Routine Fed speeches reiterating existing policy
- War/conflict coverage without confirmed financial transmission

Articles:
{numbered}

Return a JSON array with one entry per article. Each entry is an array of 1-3 classification objects ordered by relevance:
[
  [
    {{"theme": "monetary_policy_risk", "direction": "increasing", "magnitude": 3}},
    {{"theme": "banking_liquidity_risk", "direction": "increasing", "magnitude": 2}}
  ],
  [
    {{"theme": "inflation_growth_risk", "direction": "decreasing", "magnitude": 1}}
  ],
  ...
]
Respond ONLY with valid JSON."""


def classify_articles(articles, batch_size=10):
    """
    articles: list of dicts with 'article_id', 'headline', 'abstract'
    Returns dict mapping article_id -> list of {theme, direction, magnitude} dicts (1-3 per article)
    """
    client = _client()
    results = {}

    _default = [{'theme': 'none', 'direction': 'neutral', 'magnitude': 1}]

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        print(f'  Classifying {i+1}-{min(i+batch_size, len(articles))}/{len(articles)}')
        prompt = _classify_prompt(batch)

        try:
            resp = client.messages.create(
                model=CLASSIFY_MODEL,
                max_tokens=4096,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = _strip_fences(resp.content[0].text)
            classifications = json.loads(raw)
            if not isinstance(classifications, list):
                raise ValueError('not a list')
            while len(classifications) < len(batch):
                classifications.append(_default)
            classifications = classifications[:len(batch)]
        except Exception as e:
            print(f'    classification error: {e} — using defaults')
            classifications = [_default] * len(batch)

        for art, cls_list in zip(batch, classifications):
            # Ensure it's a list; single-dict responses are wrapped
            if isinstance(cls_list, dict):
                cls_list = [cls_list]
            cleaned = []
            for cls in cls_list[:3]:
                theme = cls.get('theme', 'none')
                direction = cls.get('direction', 'neutral')
                try:
                    magnitude = max(1, min(3, int(cls.get('magnitude', 1))))
                except (ValueError, TypeError):
                    magnitude = 1
                if theme not in STRESS_THEMES:
                    theme = 'none'
                if direction not in STRESS_DIRECTIONS:
                    direction = 'neutral'
                cleaned.append({'theme': theme, 'direction': direction, 'magnitude': magnitude})
            results[art['article_id']] = cleaned if cleaned else _default

        time.sleep(1)

    return results


# ---------------------------------------------------------------------------
# Legacy interface (used by run_pipeline.py)
# ---------------------------------------------------------------------------

def _legacy_prompt(headlines):
    headlines_json = json.dumps(headlines, indent=2)
    return f"""You are a financial analyst classifying news headlines for financial stress monitoring.

For each headline, determine:
1. stress_theme: credit_risk | inflation_risk | liquidity_risk | geopolitical_risk | banking_risk | none
2. direction: increasing | decreasing | neutral

Return a JSON array with one object per headline in the same order.

Headlines:
{headlines_json}

Respond ONLY with valid JSON:
[
  {{"theme": "inflation_risk", "direction": "increasing"}},
  ...
]"""


def classify_headlines_batch(headlines, batch_size=20, model=MODEL):
    client = _client()
    all_cls = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        print(f'Classifying headlines {i+1} to {min(i+batch_size, len(headlines))}...')
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{'role': 'user', 'content': _legacy_prompt(batch)}],
            )
            raw = _strip_fences(resp.content[0].text)
            cls = json.loads(raw)
            if not isinstance(cls, list):
                raise ValueError('not a list')
            if len(cls) != len(batch):
                print(f'Warning: Expected {len(batch)} classifications, got {len(cls)}')
                while len(cls) < len(batch):
                    cls.append({'theme': 'none', 'direction': 'neutral'})
                cls = cls[:len(batch)]
            all_cls.extend(cls)
            time.sleep(1)
        except Exception as e:
            print(f'Error: {e}')
            all_cls.extend([{'theme': 'none', 'direction': 'neutral'}] * len(batch))

    return all_cls


def classify_news_dataframe(df, headline_column='headline', batch_size=20):
    if df.empty:
        df['theme'] = []
        df['direction'] = []
        return df
    headlines = df[headline_column].tolist()
    print(f'Classifying {len(headlines)} headlines...')
    cls = classify_headlines_batch(headlines, batch_size=batch_size)
    df = df.copy()
    df['theme'] = [c.get('theme', 'none') for c in cls]
    df['direction'] = [c.get('direction', 'neutral') for c in cls]
    print('Classification complete.')
    return df


def calculate_stress_score(df, date_column='date', direction_column='direction'):
    if df.empty:
        return 0.0
    counts = df[direction_column].value_counts()
    return (counts.get('increasing', 0) - counts.get('decreasing', 0)) / max(len(df), 1)
