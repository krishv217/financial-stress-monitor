# Financial Stress Monitor 📊

A real-time dashboard that monitors financial stress by combining the St. Louis Fed Financial Stress Index (FRED STLFSI4) with LLM-classified news sentiment to detect narrative-reality divergences.

## Overview

This system analyzes whether news narrative about financial stress leads, lags, or diverges from actual measured stress in the financial system. It uses Claude AI to classify news headlines by stress theme and direction, then correlates this with the FRED stress index.

## Features

- **Real-time FRED Stress Monitoring**: Fetches and displays the latest STLFSI4 financial stress index
- **AI-Powered News Classification**: Uses Claude API to classify headlines by stress theme and sentiment direction
- **Lead-Lag Analysis**: Calculates cross-correlations to identify if news sentiment predicts FRED movements
- **Divergence Detection**: Highlights periods when narrative and reality diverge
- **Interactive Dashboard**: Streamlit-based visualization with multiple analytical views

## Project Structure

```
financial-stress-monitor/
├── fred_data.py          # FRED API integration
├── news_data.py          # NewsAPI and GDELT integration
├── classifier.py         # Claude AI headline classification
├── analysis.py           # Lead-lag correlation analysis
├── app.py                # Streamlit dashboard
├── run_pipeline.py       # Complete data pipeline runner
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
├── .gitignore           # Git ignore rules
└── data/                # Data storage (CSV files)
    ├── fred_historical.csv
    ├── news_recent.csv
    ├── news_historical.csv
    └── news_classified.csv
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- API keys for:
  - FRED (St. Louis Fed)
  - NewsAPI
  - Anthropic Claude

### 2. Installation

```bash
# Navigate to project directory
cd financial-stress-monitor

# Install dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```
FRED_API_KEY=your_fred_api_key_here
NEWS_API_KEY=your_newsapi_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

#### Getting API Keys

**FRED API Key:**
- Create a free account at [https://fred.stlouisfed.org](https://fred.stlouisfed.org)
- Navigate to "My Account" → "API Keys"
- Request an API key

**NewsAPI Key:**
- Create a free account at [https://newsapi.org](https://newsapi.org)
- Find your API key in the dashboard
- Note: Free tier limited to 30 days of historical data

**Anthropic API Key:**
- Sign up at [https://console.anthropic.com](https://console.anthropic.com)
- Navigate to API Keys section
- Generate a new key

### 4. Initial Data Collection

Run the complete data pipeline to collect historical data:

```bash
python run_pipeline.py
```

This will:
1. Fetch FRED historical data (2015-present)
2. Fetch recent news from NewsAPI (last 30 days)
3. Fetch historical news from GDELT (2015-present)
4. Classify all headlines using Claude AI
5. Save processed data to CSV files

**Note:** The initial run may take 15-30 minutes depending on the amount of data and API rate limits.

## Usage

### Testing Individual Modules

Test each module independently:

```bash
# Test FRED data fetching
python fred_data.py

# Test news data fetching
python news_data.py

# Test headline classification
python classifier.py

# Test analysis functions
python analysis.py
```

### Running the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Dashboard Sections

1. **Current Stress Indicator**: Shows the latest FRED stress score with color-coded status
2. **12-Week Trend**: Line chart of recent stress index movements
3. **Current Narrative Sentiment**: Breakdown of recent news by theme and direction
4. **Divergence Indicator**: Dual-axis chart showing when news sentiment diverges from FRED
5. **Historical Lead-Lag Analysis**: Correlation analysis showing if news predicts FRED movements

## Data Pipeline

### Module Descriptions

**fred_data.py**
- Fetches STLFSI4 time series from FRED API
- Handles date parsing and data cleaning
- Saves to CSV for offline analysis

**news_data.py**
- Fetches recent headlines from NewsAPI (last 30 days)
- Fetches historical headlines from GDELT (2015-present)
- Standardizes output format across sources
- Uses predefined financial stress keywords

**classifier.py**
- Batches headlines for efficient API usage (20 per call)
- Uses structured prompts to get JSON responses
- Classifies by theme: credit_risk, inflation_risk, liquidity_risk, geopolitical_risk, banking_risk, none
- Classifies by direction: increasing, decreasing, neutral

**analysis.py**
- Aggregates daily news to weekly frequency
- Aligns news and FRED data by week
- Calculates cross-correlations at 0-4 week lags
- Detects divergence periods
- Generates plain English summaries

**app.py**
- Streamlit dashboard with 4 main sections
- Interactive filtering by date range
- Real-time divergence alerts
- Cached data loading for performance

## Key Design Decisions

### Why CSV instead of Database?
For a 36-hour hackathon timeline, CSV files provide:
- Zero setup overhead
- Easy inspection and debugging
- Sufficient performance for this data scale
- Simple version control and sharing

### Why Weekly Aggregation?
- FRED data is published weekly
- Reduces noise in daily news cycles
- More stable correlation analysis
- Aligns with typical financial reporting cycles

### Why Batch Classification?
- Reduces API costs (fewer calls)
- Improves throughput (20 headlines per call)
- Maintains Claude's context for consistent classification
- Includes rate limiting to avoid throttling

## Troubleshooting

### "API Key not found" errors
- Ensure `.env` file exists in the project root
- Check that variable names match exactly: `FRED_API_KEY`, `NEWS_API_KEY`, `ANTHROPIC_API_KEY`
- No quotes needed around values in `.env`

### NewsAPI "426 Upgrade Required"
- Free tier only provides 30 days of history
- Use GDELT for older data
- Consider upgrading NewsAPI plan if needed

### GDELT timeouts or errors
- GDELT can be slow for large date ranges
- Try smaller date ranges or fewer keywords
- The library sometimes has rate limits - add delays between calls

### Classification produces "none/neutral" for everything
- Check Anthropic API key validity
- Review Claude API usage limits
- Inspect the raw API response in classifier.py debug output
- Prompt may need tuning for specific news sources

### Streamlit dashboard shows "File not found"
- Run `python run_pipeline.py` first to generate data files
- Check that `data/` directory exists
- Verify CSV files are not empty

## Extending the System

### Adding New News Sources
1. Create fetch function in `news_data.py`
2. Return standardized DataFrame format (date, headline, source, url)
3. Update `run_pipeline.py` to include new source

### Adding New Stress Themes
1. Update `STRESS_THEMES` list in `classifier.py`
2. Update classification prompt with new theme definition
3. Re-run classification on historical data

### Adding New Visualizations
1. Add new section in `app.py`
2. Use Plotly for interactive charts
3. Follow existing pattern: data processing → visualization → insights

## Performance Optimization

- **Caching**: Streamlit caches data loading (1 hour TTL)
- **Batch Processing**: Headlines classified in groups of 20
- **Incremental Updates**: Only fetch new data since last run
- **Local Storage**: CSV files avoid repeated API calls during development

## Team Workflow Recommendation

**Person 1 - Data Pipeline:**
- Own `fred_data.py`, `news_data.py`, `run_pipeline.py`
- Focus on data quality and API reliability

**Person 2 - Analysis & Classification:**
- Own `classifier.py`, `analysis.py`
- Tune LLM prompts and correlation methods

**Person 3 - Frontend:**
- Own `app.py`
- Focus on visualization and user experience

## Credits

Built with:
- [Streamlit](https://streamlit.io) - Dashboard framework
- [Plotly](https://plotly.com) - Interactive visualizations
- [Claude API](https://anthropic.com) - LLM classification
- [FRED](https://fred.stlouisfed.org) - Financial stress data
- [NewsAPI](https://newsapi.org) - Recent news data
- [GDELT](https://gdeltproject.org) - Historical news data

## License

MIT License - feel free to use and modify for your projects.
