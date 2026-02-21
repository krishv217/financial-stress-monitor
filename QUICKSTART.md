# Quick Start Guide

Get the Financial Stress Monitor running in 5 minutes.

## 1. Install Dependencies

```bash
cd financial-stress-monitor
pip install -r requirements.txt
```

## 2. Set Up API Keys

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your keys
# Get keys from:
# - FRED: https://fred.stlouisfed.org (free account)
# - NewsAPI: https://newsapi.org (free account)
# - Anthropic: https://console.anthropic.com (paid API)
```

## 3. Run the Data Pipeline

```bash
python run_pipeline.py
```

This will:
- Fetch FRED data (2015-present)
- Fetch news from NewsAPI and GDELT
- Classify headlines with Claude AI
- Takes ~15-30 minutes on first run

## 4. Launch Dashboard

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

## Testing Individual Modules

```bash
# Test just FRED data
python fred_data.py

# Test just news fetching
python news_data.py

# Test just classification (uses sample data)
python classifier.py

# Test just analysis (uses synthetic data)
python analysis.py
```

## Troubleshooting

**"API Key not found"**
- Make sure `.env` file exists in project root
- Check variable names match: `FRED_API_KEY`, `NEWS_API_KEY`, `ANTHROPIC_API_KEY`

**"File not found" in dashboard**
- Run `python run_pipeline.py` first to generate data

**Classification taking too long**
- Reduce batch size in classifier.py
- Or fetch less historical data in run_pipeline.py

## Project Structure

```
financial-stress-monitor/
├── app.py                    # ← Main dashboard
├── run_pipeline.py           # ← Run this first
├── fred_data.py             # FRED API module
├── news_data.py             # News API module
├── classifier.py            # Claude classification
├── analysis.py              # Correlation analysis
├── requirements.txt         # Dependencies
├── .env                     # Your API keys (create this)
└── data/                    # Generated CSV files
```

## What Each File Does

- **run_pipeline.py**: Orchestrates everything - run this once to set up
- **app.py**: The Streamlit dashboard - run this to visualize
- **fred_data.py**: Fetches financial stress data from St. Louis Fed
- **news_data.py**: Fetches news from NewsAPI (recent) and GDELT (historical)
- **classifier.py**: Uses Claude to classify headlines by theme/direction
- **analysis.py**: Calculates correlations and detects divergences

## Next Steps

1. Customize keywords in `news_data.py` (FINANCIAL_KEYWORDS)
2. Tune classification prompt in `classifier.py`
3. Add visualizations in `app.py`
4. Schedule periodic updates with cron/Task Scheduler

## Cost Estimates

- **FRED API**: Free
- **NewsAPI**: Free (30 days history), $449/month for more
- **GDELT**: Free
- **Anthropic Claude**: ~$0.01-0.05 per 1000 headlines classified

For 5000 historical headlines: ~$0.25
For daily updates (100 headlines/day): ~$0.05/day
