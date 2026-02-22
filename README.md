# Financial Stress Monitor

A data pipeline and ML system that predicts the St. Louis Fed Financial Stress Index (FRED STLFSI4) using LLM-classified NYT news sentiment. Models are trained across 5 prediction horizons (1–12 weeks) using Linear Regression, LassoCV, Random Forest, XGBoost, and LightGBM.

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

Groups classified articles by week — snapping all week_start dates to Friday to align with FRED's publication cadence — and computes magnitude-weighted net sentiment scores per theme. Populates 5 future FRED score columns via nearest-date lookup.

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

Trains **5 models × 5 horizons = 25 combinations** and prints a comparison table:

```
Horizon  Model             Train rows  Test rows  Train R²   Test R²     RMSE      MAE
----------------------------------------------------------------------------------------
1w       LinearRegression        ...        ...     0.xxx     0.xxx    0.xxxx   0.xxxx
1w       LassoCV                 ...        ...     0.xxx     0.xxx    0.xxxx   0.xxxx
1w       RandomForest            ...        ...     0.xxx     0.xxx    0.xxxx   0.xxxx
1w       XGBoost                 ...        ...     0.xxx     0.xxx    0.xxxx   0.xxxx
1w       LightGBM                ...        ...     0.xxx     0.xxx    0.xxxx   0.xxxx
2w       ...
```

**Models:**
- `LinearRegression` — OLS baseline
- `LassoCV` — L1-regularized regression with automatic alpha selection (5-fold CV); zeros out uninformative features
- `RandomForest` — 100-tree ensemble, captures non-linear interactions
- `XGBoost` — gradient-boosted trees (max_depth=3, L1+L2 regularization)
- `LightGBM` — leaf-wise gradient boosting (max_depth=3); best performer at 4w–12w horizons

**Features (11 total):** 5 current-week theme scores + `fred_score` + 5 lagged (previous week) theme scores

**Target:** ΔFRED = `fred_score_{horizon}_future − fred_score` (change, not level); predictions are converted back to absolute FRED for display

**Train/test split:** chronological 90/10 (~1,226 train weeks, ~136 test weeks)

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

**Delta-FRED target** — Models predict the *change* in FRED (ΔFRED = future − current) rather than the absolute level. This forces the model to learn from sentiment rather than just extrapolating autocorrelation, and lets LassoCV/LightGBM zero out the `fred_score` feature when it adds no incremental signal.

**Lagged features** — Previous week's 5 theme scores are appended as additional features, giving the model access to momentum in sentiment signals.

**5 prediction horizons** — Allows comparison of short-term (1–2w) vs medium-term (4–12w) predictability, which is the core research question.

**Friday normalization** — All week_start dates are snapped to the following Friday before aggregation, aligning the Kaggle dataset (Monday weeks) with the NYT API dataset (Friday weeks) and FRED's publication cadence.

**Nearest-date FRED lookup** — FRED publishes on Fridays. A ±10-day nearest-match prevents missed joins at holiday weeks.

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
- [Claude API](https://anthropic.com) — LLM classification (Haiku relevance filter, Haiku multi-theme classifier)
- [Streamlit](https://streamlit.io) — Dashboard framework
- [scikit-learn](https://scikit-learn.org) — LinearRegression, LassoCV, RandomForest
- [XGBoost](https://xgboost.readthedocs.io) — Gradient-boosted trees
- [LightGBM](https://lightgbm.readthedocs.io) — Leaf-wise gradient boosting
