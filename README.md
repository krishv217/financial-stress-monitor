# Financial Stress Monitor

A data pipeline and ML system that predicts the St. Louis Fed Financial Stress Index (FRED STLFSI4) using LLM-classified NYT news sentiment. Models are trained across 5 prediction horizons (1–12 weeks) using Linear Regression, Lasso, and Random Forest.

## Overview

The system collects NYT articles related to financial stress, classifies them by theme and direction using Claude AI, aggregates weekly sentiment scores, and trains ML models to predict future FRED stress index values. The core research question: **how far into the future can news-derived sentiment predict measurable financial stress?**

## Project Structure

```
financial-stress-monitor/
├── collect_nyt.py        # Phase 1: NYT Article Search API collection
├── pipeline.py           # Phases 2-4: dedup, relevance filter, classify, aggregate
├── classifier.py         # Claude AI relevance filter + multi-theme classification
├── fred_data.py          # FRED API integration (STLFSI4)
├── train_model.py        # Phase 6: multi-horizon ML training + comparison
├── app.py                # Streamlit dashboard
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
├── .gitignore
└── data/
    ├── raw_articles.csv              # Raw NYT articles (Phase 1 output)
    ├── classified_articles.csv       # Deduped, filtered, classified articles
    ├── weekly_sentiment_scores.csv   # Weekly aggregated scores + FRED targets
    ├── fred_historical.csv           # FRED STLFSI4 time series
    ├── model_predictions.csv         # Current-week predictions
    ├── model.pkl                     # Best 2w model (dashboard use)
    └── model_{horizon}_{name}.pkl    # Best model per horizon (e.g. model_4w_randomforest.pkl)
```

## Setup

### Prerequisites

- Python 3.8+
- API keys for FRED, Anthropic, and NYT (up to 7 keys supported)

### Installation

```bash
cd financial-stress-monitor
pip install -r requirements.txt
```

### API Key Configuration

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

`.env` format:
```
FRED_API_KEY=your_fred_key
ANTHROPIC_API_KEY=your_anthropic_key

NYT_API_KEY_1=your_nyt_key_1
NYT_API_KEY_2=your_nyt_key_2
# ... up to NYT_API_KEY_7
```

**FRED API key:** [fred.stlouisfed.org](https://fred.stlouisfed.org) → My Account → API Keys

**Anthropic API key:** [console.anthropic.com](https://console.anthropic.com) → API Keys

**NYT API keys:** [developer.nytimes.com](https://developer.nytimes.com) → Apps → Create App
- Each key allows ~4,000 calls/day (IP-level shared quota)
- Multiple keys rotate automatically with a 1-second floor between calls

---

## Pipeline

Run each phase in order, or run all at once:

### Phase 1 — Collect NYT Articles

```bash
python collect_nyt.py
```

Collects articles from 10 financial stress queries going backwards from the most recent week. Outputs to `data/raw_articles.csv`. Supports up to 7 rotating API keys.

- Queries cover: monetary policy, credit/debt, banking/liquidity, inflation/growth, geopolitical risk
- Per-query page limits: Q1–Q4, Q9–Q10 = 2 pages; Q5–Q8 = 1 page (16 API calls/week)
- At ~4,000 calls/day, processes roughly 250 weeks/day

### Phase 2 — Dedup + Relevance Filter

```bash
python pipeline.py --phase 2
```

Deduplicates articles by normalized headline, then uses Claude AI in batches of 30 to filter for financial stress relevance. Outputs to `data/classified_articles.csv`.

### Phase 3 — LLM Multi-Theme Classification

```bash
python pipeline.py --phase 3
```

Classifies each relevant article into 1–3 stress themes with direction (increasing/decreasing/neutral) and magnitude (1–3). Uses Claude Haiku in batches of 10.

**Themes:**
- `monetary_policy_risk` — Fed policy, interest rates, tightening/easing
- `credit_debt_risk` — corporate/consumer debt, defaults, spreads
- `banking_liquidity_risk` — bank stress, deposit flows, liquidity crises
- `inflation_growth_risk` — CPI, GDP, recession signals
- `geopolitical_external_risk` — trade wars, sanctions, global shocks

### Phase 4 — Weekly Aggregation

```bash
python pipeline.py --phase 4
```

Groups classified articles by week and computes magnitude-weighted net sentiment scores per theme. Populates 5 future FRED score columns via nearest-date lookup.

**FRED horizon columns written:**
| Column | Offset |
|---|---|
| `fred_score_1w_future` | +1 week |
| `fred_score_2w_future` | +2 weeks |
| `fred_score_4w_future` | +4 weeks |
| `fred_score_8w_future` | +8 weeks |
| `fred_score_12w_future` | +12 weeks |

### Run All Phases

```bash
python pipeline.py
# or with a specific prompt version:
python pipeline.py --prompt-version v2
```

---

## ML Training

```bash
python train_model.py
```

Trains **3 models × 5 horizons = 15 combinations** and prints a comparison table:

```
Horizon  Model             Train rows  Test rows  Train R²   Test R²     RMSE      MAE
----------------------------------------------------------------------------------------
1w       LinearRegression        ...        ...     0.xxx     0.xxx    0.xxxx   0.xxxx
1w       LassoCV                 ...        ...     0.xxx     0.xxx    0.xxxx   0.xxxx
1w       RandomForest            ...        ...     0.xxx     0.xxx    0.xxxx   0.xxxx
2w       ...
```

**Models:**
- `LinearRegression` — OLS baseline
- `LassoCV` — L1-regularized regression with automatic alpha selection (5-fold CV); can zero out uninformative features
- `RandomForest` — 100-tree ensemble, captures non-linear interactions

**Features:** `monetary_policy_score`, `credit_debt_score`, `banking_liquidity_score`, `inflation_growth_score`, `geopolitical_external_score`, `fred_score`

**Train/test split:** chronological 80/20

The best model per horizon (by test R²) is saved to `data/model_{horizon}_{name}.pkl`. The best 2-week model is also saved to `data/model.pkl` for dashboard use.

**Options:**
```bash
python train_model.py --eval              # print table only, no save
python train_model.py --horizon 2w 4w    # train specific horizons only
```

---

## Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Key Design Decisions

**Weekly aggregation** — FRED publishes weekly; weekly grouping reduces noise and aligns prediction targets naturally.

**Multi-theme classification** — Each article can contribute to 1–3 themes, preventing misattribution when an article spans multiple risk categories. Primary theme (highest magnitude) is stored in flat columns; full list stored as JSON in `stress_themes`.

**Magnitude-weighted scoring** — Net score per theme = sum of (+magnitude for increasing, −magnitude for decreasing) across all articles in the week. Neutral articles contribute 0.

**5 prediction horizons** — Allows comparison of short-term (1–2w) vs medium-term (4–12w) predictability, which is the core research question.

**Nearest-date FRED lookup** — FRED publishes on Fridays; article weeks may start on any day. A ±10-day nearest-match prevents missed joins.

**Backwards collection** — Articles are collected from most recent week backwards, so if the daily quota is exhausted, the most recent (most relevant) data is collected first.

---

## Troubleshooting

**NYT 429 rate limit errors**
- All keys from the same IP share a ~4,000 call/day quota
- Wait for midnight ET reset, or have a group member collect from a different network

**`classified_articles.csv` missing `stress_themes` column**
- Older rows from before v2 classifier will have an empty `stress_themes` field
- Phase 4 aggregation falls back to single-theme columns for those rows

**FRED values missing for recent weeks**
- FRED data is fetched up to the current date; the most recent 1–12 weeks will have empty future columns since those FRED readings don't exist yet
- These rows are excluded from ML training (no target label) but can be predicted by the model

**`model.pkl` not found when running dashboard**
- Run `python train_model.py` after completing pipeline phases 2–4

---

## Credits

- [FRED STLFSI4](https://fred.stlouisfed.org/series/STLFSI4) — St. Louis Fed Financial Stress Index
- [NYT Article Search API](https://developer.nytimes.com/docs/articlesearch-product/1/overview) — News source
- [Claude API](https://anthropic.com) — LLM classification (Haiku)
- [Streamlit](https://streamlit.io) — Dashboard framework
- [scikit-learn](https://scikit-learn.org) — ML models