"""
Financial Stress Monitor Dashboard
Streamlit application for visualizing FRED stress index and news sentiment
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
from anthropic import Anthropic

# Resolve all data paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

from fred_data import load_fred_data, get_current_stress_score
from news_data import load_news_data
from analysis import (
    aggregate_news_by_week,
    align_fred_weekly,
    calculate_lead_lag_correlation,
    detect_divergence,
    generate_analysis_summary,
    fit_narrative_to_fred_regression,
)

# Page configuration
st.set_page_config(
    page_title="Financial Stress Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .big-metric {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .stress-low {
        color: #28a745;
    }
    .stress-medium {
        color: #ffc107;
    }
    .stress-high {
        color: #dc3545;
    }
    .divergence-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        margin: 1rem 0;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_all_data():
    """Load all data files."""
    try:
        fred_df = load_fred_data(os.path.join(DATA_DIR, 'fred_historical.csv'))
    except FileNotFoundError:
        st.error("FRED data not found. Please run fred_data.py first.")
        fred_df = pd.DataFrame()

    try:
        news_df = load_news_data(os.path.join(DATA_DIR, 'news_classified.csv'))
    except FileNotFoundError:
        st.warning("Classified news data not found. Loading recent news...")
        try:
            news_df = load_news_data(os.path.join(DATA_DIR, 'news_recent.csv'))
        except FileNotFoundError:
            st.error("No news data found. Please run news_data.py first.")
            news_df = pd.DataFrame()

    return fred_df, news_df


def get_stress_color(value):
    """Get color based on stress level."""
    if value < -0.5:
        return "stress-low"
    elif value < 0.5:
        return "stress-medium"
    else:
        return "stress-high"


@st.cache_data(ttl=86400)
def generate_fred_explanation(score, date_str, days_ago):
    """Generate plain-English interpretation of the current FRED score."""
    try:
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        prompt = f"""The St. Louis Fed Financial Stress Index (STLFSI4) currently reads {score:.4f} as of {date_str} ({days_ago} days ago).

The index is centered at 0. Negative = below-average stress (calm). Positive = above-average stress.
Historical reference points: normal range is roughly -1 to +1, 2008 financial crisis peaked ~5.0, COVID-19 spike ~6.0.

Write exactly 3 concise sentences for a macro analyst:
1. What this specific reading means right now
2. Historical context — is this high, low, or normal?
3. What market/credit/rate conditions typically produce readings at this level

Be specific and direct. No hedging."""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception:
        if score < -0.5:
            return f"STLFSI4 at {score:.4f} indicates below-average financial stress — markets are calm."
        elif score < 0.5:
            return f"STLFSI4 at {score:.4f} indicates moderate stress near the historical average."
        else:
            return f"STLFSI4 at {score:.4f} indicates above-average financial stress."


def compute_daily_predictions(news_df, last_fred_date):
    """Compute daily narrative stress scores for every day in the gap since last FRED reading.

    Generates a row for each calendar day from the day after last_fred_date through today,
    including days with no article data (has_data=False).
    """
    today = datetime.now().date()
    gap_start = last_fred_date.date() + timedelta(days=1)
    if gap_start > today:
        return pd.DataFrame()

    gap_days = pd.date_range(start=gap_start, end=today, freq='D')

    daily_rows = []
    for day in gap_days:
        day_date = day.date()

        # Find articles for this specific day
        if not news_df.empty and 'direction' in news_df.columns:
            day_news = news_df[news_df['date'].dt.date == day_date]
        else:
            day_news = pd.DataFrame()

        if not day_news.empty:
            inc = (day_news['direction'] == 'increasing').sum()
            dec = (day_news['direction'] == 'decreasing').sum()
            total = len(day_news)
            score = (inc - dec) / total if total > 0 else 0.0
            daily_rows.append({
                'date': pd.Timestamp(day_date),
                'narrative_score': score,
                'article_count': total,
                'headlines': day_news['headline'].tolist()[:5],
                'themes': day_news['theme'].value_counts().to_dict() if 'theme' in day_news.columns else {},
                'has_data': True
            })
        else:
            daily_rows.append({
                'date': pd.Timestamp(day_date),
                'narrative_score': None,
                'article_count': 0,
                'headlines': [],
                'themes': {},
                'has_data': False
            })

    return pd.DataFrame(daily_rows)


@st.cache_data(ttl=86400)
def generate_daily_summaries(daily_data_json):
    """Generate one-sentence AI summaries for each prediction day (batched)."""
    try:
        daily_data = json.loads(daily_data_json)
        if not daily_data:
            return {}
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        days_text = ""
        for d in daily_data:
            hl = '\n'.join(f'  - {h}' for h in d['headlines'])
            days_text += f"\n**{d['date']} | score: {d['narrative_score']:+.3f} | {d['article_count']} articles**\n{hl}\n"
        prompt = f"""For each date below, write ONE sentence for a macro analyst explaining what the day's financial news suggests about stress conditions. The narrative score ranges from -1 (all calming) to +1 (all stress-increasing).

{days_text}

Return ONLY valid JSON: {{"YYYY-MM-DD": "one sentence", ...}}"""
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)
    except Exception:
        return {}


def main():
    # Header
    st.title("📊 Financial Stress Monitor")
    st.markdown("**The FRED index tells you where stress has been. This tool tries to tell you where it's going.**")

    # Load data
    fred_df, news_df = load_all_data()

    if fred_df.empty:
        st.error("Cannot proceed without FRED data.")
        return

    # Fit narrative→FRED regression using full (unfiltered) history
    full_fred_df_raw, full_news_df_raw = load_all_data()
    _reg_news_weekly = aggregate_news_by_week(full_news_df_raw) if not full_news_df_raw.empty else pd.DataFrame()
    _reg_fred_weekly = align_fred_weekly(full_fred_df_raw) if not full_fred_df_raw.empty else pd.DataFrame()
    reg_params = fit_narrative_to_fred_regression(_reg_news_weekly, _reg_fred_weekly)

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range filter
    if not fred_df.empty:
        min_date = fred_df['date'].min().date()
        today = datetime.now().date()

        date_range = st.sidebar.date_input(
            "Date Range",
            value=(today - timedelta(days=90), today),
            min_value=min_date,
            max_value=today
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            fred_df = fred_df[(fred_df['date'].dt.date >= start_date) & (fred_df['date'].dt.date <= end_date)]
            if not news_df.empty:
                news_df = news_df[(news_df['date'].dt.date >= start_date) & (news_df['date'].dt.date <= end_date)]

    # =====================================================================
    # HERO SECTION: FRED (last week) vs Narrative (right now)
    # =====================================================================
    st.header("🎯 Then vs. Now")
    st.caption("FRED updates weekly — the narrative sentiment fills the gap with what's happening today.")

    has_news = not news_df.empty and 'theme' in news_df.columns and 'direction' in news_df.columns

    col_fred, col_vs, col_narrative = st.columns([2, 1, 2])

    # --- Left: FRED (lagging) ---
    with col_fred:
        st.markdown("#### FRED Stress Index")
        st.caption("Last official reading (lags ~1 week)")
        full_fred_df, _ = load_all_data()
        fred_date, fred_score = get_current_stress_score(full_fred_df)

        if fred_score is not None:
            stress_class = get_stress_color(fred_score)
            st.markdown(f'<div class="big-metric {stress_class}">{fred_score:.4f}</div>',
                        unsafe_allow_html=True)
            st.markdown(f"**As of:** {fred_date.strftime('%Y-%m-%d')}")
            days_ago = (datetime.now().date() - fred_date.date()).days
            st.caption(f"_{days_ago} days ago_")
            if fred_score < -0.5:
                st.success("Low stress")
            elif fred_score < 0.5:
                st.warning("Moderate stress")
            else:
                st.error("High stress")

    # --- Center: signal ---
    with col_vs:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if has_news:
            recent_news_all = news_df[news_df['date'] >= (news_df['date'].max() - timedelta(days=7))]
            inc = (recent_news_all['direction'] == 'increasing').sum()
            dec = (recent_news_all['direction'] == 'decreasing').sum()
            total = len(recent_news_all)
            narrative_score = (inc - dec) / total if total > 0 else 0.0

            if fred_score is not None:
                fred_regime = "calm" if fred_score < 0 else "stressed"
                narrative_regime = "calm" if narrative_score < 0 else "stressed"
                if fred_regime != narrative_regime:
                    st.markdown("### ⚠️")
                    st.markdown("**Diverging**")
                else:
                    st.markdown("### ✅")
                    st.markdown("**Aligned**")
        else:
            st.markdown("### →")

    # --- Right: Narrative (leading) ---
    with col_narrative:
        st.markdown("#### Narrative Stress Score")
        st.caption("Derived from today's news headlines (leads FRED)")
        if has_news:
            recent_news_all = news_df[news_df['date'] >= (news_df['date'].max() - timedelta(days=7))]
            inc = (recent_news_all['direction'] == 'increasing').sum()
            dec = (recent_news_all['direction'] == 'decreasing').sum()
            total = len(recent_news_all)
            narrative_score = (inc - dec) / total if total > 0 else 0.0
            narrative_class = get_stress_color(narrative_score - 0.5)  # shift scale: 0 = neutral

            st.markdown(f'<div class="big-metric {narrative_class}">{narrative_score:+.3f}</div>',
                        unsafe_allow_html=True)
            st.markdown(f"**Based on:** {total} articles")
            latest_news_date = news_df['date'].max()
            st.caption(f"_Latest article: {latest_news_date.strftime('%Y-%m-%d')}_")
            if narrative_score < -0.1:
                st.success("Narrative: calming")
            elif narrative_score < 0.1:
                st.warning("Narrative: neutral")
            else:
                st.error("Narrative: stress rising")
        else:
            st.info("No classified news. Run classifier.py.")

    # FRED plain-English explanation
    if fred_score is not None:
        days_ago = (datetime.now().date() - fred_date.date()).days
        with st.expander("📖 What does this FRED score mean?", expanded=True):
            explanation = generate_fred_explanation(fred_score, fred_date.strftime('%Y-%m-%d'), days_ago)
            st.markdown(explanation)

    # Divergence explanation banner
    if has_news and fred_score is not None:
        fred_regime = "calm" if fred_score < 0 else "stressed"
        narrative_regime = "calm" if narrative_score < 0 else "stressed"
        if fred_regime != narrative_regime and narrative_score > 0 and fred_score < 0:
            st.markdown("""
            <div class="divergence-alert">
                <strong>⚠️ DIVERGENCE DETECTED — Potential Leading Signal</strong><br>
                FRED says markets are calm, but financial news narratives are showing elevated stress.
                Historically, narrative stress has preceded FRED index movements by 1–4 weeks.
                This may be an early warning of an upcoming regime shift.
            </div>
            """, unsafe_allow_html=True)

    # ── Calibrated FRED Forecast ──────────────────────────────────────────
    if reg_params is not None and has_news:
        predicted_fred = reg_params['slope'] * narrative_score + reg_params['intercept']
        lag = reg_params['best_lag']
        r2 = reg_params['r_squared']
        n = reg_params['n_observations']

        st.markdown("### 🔮 Calibrated FRED Forecast")
        st.caption(
            f"OLS regression trained on {n} weeks of aligned data · "
            f"optimal lag = {lag} week(s) · R² = {r2:.2f}"
        )

        fc_col, fc_desc = st.columns([1, 2])
        with fc_col:
            forecast_class = get_stress_color(predicted_fred)
            st.markdown(
                f'<div class="big-metric {forecast_class}">{predicted_fred:+.4f}</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Estimated FRED reading in ~{lag} week(s)")

        with fc_desc:
            direction = "higher" if predicted_fred > fred_score else "lower"
            change = predicted_fred - fred_score
            st.markdown(
                f"The regression model predicts the next FRED release will read **{predicted_fred:+.4f}**, "
                f"**{abs(change):.4f} points {direction}** than the current official reading of {fred_score:.4f}."
            )
            if r2 < 0.25:
                st.warning(f"Low R² ({r2:.2f}) — only {n} weeks of training data. Forecast improves as more news history is collected.")
            elif r2 < 0.5:
                st.info(f"Moderate fit (R²={r2:.2f}). The model explains roughly {r2*100:.0f}% of historical FRED variance.")
            else:
                st.success(f"Strong fit (R²={r2:.2f}). The model explains {r2*100:.0f}% of historical FRED variance.")

    st.divider()

    # =====================================================================
    # SECTION 2: 12-week trend + daily predictions for the gap
    # =====================================================================
    st.header("📈 12-Week Trend + Daily Predictions")
    st.caption("Orange dots = daily narrative-based estimates for the days since the last FRED release. Click a day below to see what drove it.")

    full_fred_df, full_news_df = load_all_data()
    recent_12w = full_fred_df.nlargest(12, 'date').sort_values('date')
    last_fred_date = full_fred_df['date'].max()

    # Compute daily predictions for all gap days (always, not gated on has_news)
    daily_preds = compute_daily_predictions(full_news_df if not full_news_df.empty else pd.DataFrame(), last_fred_date)
    preds_with_data = daily_preds[daily_preds['has_data'] == True].copy() if not daily_preds.empty else pd.DataFrame()

    fig_trend = go.Figure()

    # FRED historical line
    fig_trend.add_trace(go.Scatter(
        x=recent_12w['date'],
        y=recent_12w['value'],
        mode='lines+markers',
        name='FRED Stress Index (weekly)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    # Daily prediction points for the gap — calibrated to FRED scale via regression
    if not preds_with_data.empty:
        if reg_params is not None:
            # Use regression to place orange diamonds on the actual FRED scale
            chart_scores = (
                reg_params['slope'] * preds_with_data['narrative_score'] + reg_params['intercept']
            )
            score_label = "Calibrated FRED Estimate"
        else:
            # Fall back to ad-hoc visual scaling when no regression is available
            fred_range = recent_12w['value'].max() - recent_12w['value'].min()
            fred_mid = recent_12w['value'].mean()
            chart_scores = preds_with_data['narrative_score'] * (fred_range / 2) + fred_mid
            score_label = "Narrative Estimate (unscaled)"

        fig_trend.add_trace(go.Scatter(
            x=preds_with_data['date'],
            y=chart_scores,
            mode='markers',
            name=f'Daily {score_label} (since last FRED)',
            marker=dict(color='#ff7f0e', size=14, symbol='diamond',
                        line=dict(color='white', width=1)),
            customdata=preds_with_data[['narrative_score', 'article_count']].values,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                f"{score_label}: %{{y:.4f}}<br>"
                "Raw narrative score: %{customdata[0]:+.3f}<br>"
                "Articles: %{customdata[1]}<br>"
                "<i>See cards below for detail</i><extra></extra>"
            )
        ))

        # Dashed connector from last FRED point to first calibrated prediction
        first_pred_y = chart_scores.iloc[0]
        last_fred_val = recent_12w.iloc[-1]['value']
        first_pred = preds_with_data.iloc[0]
        fig_trend.add_trace(go.Scatter(
            x=[last_fred_date, first_pred['date']],
            y=[last_fred_val, first_pred_y],
            mode='lines',
            name='',
            line=dict(color='#ff7f0e', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="Stress Index",
        hovermode='x unified',
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Daily prediction cards — one per day in the gap
    if not daily_preds.empty:
        st.subheader("Daily Breakdown — Gap Period")

        # Generate AI summaries only for days that have article data
        if not preds_with_data.empty:
            serializable = [
                {
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'narrative_score': row['narrative_score'],
                    'article_count': row['article_count'],
                    'headlines': row['headlines']
                }
                for _, row in preds_with_data.iterrows()
            ]
            summaries = generate_daily_summaries(json.dumps(serializable))
        else:
            summaries = {}

        for _, row in daily_preds.sort_values('date', ascending=False).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')

            if not row['has_data']:
                label = f"⬜ **{date_str}** — No article data captured"
                with st.expander(label):
                    st.caption("No financial news articles were captured for this day.")
                    st.caption("This is likely due to NewsAPI free-tier limitations — articles are only available for the current day when the pipeline runs. Re-run the pipeline daily to build up coverage.")
                continue

            score = row['narrative_score']
            if score > 0.1:
                icon = "🔴"
            elif score < -0.1:
                icon = "🟢"
            else:
                icon = "🟡"
            label = f"{icon} **{date_str}** — Narrative score: {score:+.3f} ({row['article_count']} articles)"
            with st.expander(label):
                summary = summaries.get(date_str, "")
                if summary:
                    st.markdown(f"**Summary:** {summary}")
                st.markdown("**Top headlines:**")
                for hl in row['headlines']:
                    st.markdown(f"- {hl}")
                if row['themes']:
                    theme_str = ", ".join(f"{k} ({v})" for k, v in row['themes'].items() if k != 'none')
                    if theme_str:
                        st.caption(f"Themes: {theme_str}")

    st.divider()

    # =====================================================================
    # SECTION 3: Current Narrative Breakdown
    # =====================================================================
    st.header("📰 Current Narrative Breakdown")

    if has_news:
        recent_news = news_df[news_df['date'] >= (news_df['date'].max() - timedelta(days=7))]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stress Themes")
            theme_counts = recent_news['theme'].value_counts()
            fig_themes = px.bar(
                x=theme_counts.index,
                y=theme_counts.values,
                labels={'x': 'Theme', 'y': 'Number of Articles'},
                color=theme_counts.values,
                color_continuous_scale='Reds'
            )
            fig_themes.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_themes, use_container_width=True)

        with col2:
            st.subheader("Sentiment Direction")
            direction_counts = recent_news['direction'].value_counts()
            fig_direction = px.pie(
                values=direction_counts.values,
                names=direction_counts.index,
                color=direction_counts.index,
                color_discrete_map={
                    'increasing': '#dc3545',
                    'neutral': '#6c757d',
                    'decreasing': '#28a745'
                }
            )
            fig_direction.update_layout(height=300)
            st.plotly_chart(fig_direction, use_container_width=True)

    else:
        st.info("No classified news data available. Run classifier.py to generate classifications.")

    st.divider()

    # =====================================================================
    # SECTION 4: Divergence Indicator
    # =====================================================================
    st.header("🔍 Narrative vs. FRED Divergence Over Time")

    if not news_df.empty and 'theme' in news_df.columns:
        # Aggregate data weekly
        news_weekly = aggregate_news_by_week(news_df)
        fred_weekly = align_fred_weekly(fred_df)

        # Detect divergence
        divergence_df = detect_divergence(news_weekly, fred_weekly, threshold=0.3)

        if not divergence_df.empty:
            # Normalize for plotting
            divergence_df['sentiment_normalized'] = (
                (divergence_df['sentiment_score'] - divergence_df['sentiment_score'].mean()) /
                (divergence_df['sentiment_score'].std() + 1e-10)
            )
            divergence_df['fred_normalized'] = (
                (divergence_df['fred_value'] - divergence_df['fred_value'].mean()) /
                (divergence_df['fred_value'].std() + 1e-10)
            )

            # Dual-axis chart
            fig_divergence = go.Figure()

            fig_divergence.add_trace(go.Scatter(
                x=divergence_df['week_start'],
                y=divergence_df['sentiment_normalized'],
                name='News Sentiment (normalized)',
                line=dict(color='#ff7f0e', width=2),
                yaxis='y1'
            ))

            fig_divergence.add_trace(go.Scatter(
                x=divergence_df['week_start'],
                y=divergence_df['fred_normalized'],
                name='FRED Index (normalized)',
                line=dict(color='#1f77b4', width=2),
                yaxis='y1'
            ))

            # Highlight divergence periods
            divergent_periods = divergence_df[divergence_df['divergence'] == True]
            if not divergent_periods.empty:
                fig_divergence.add_trace(go.Scatter(
                    x=divergent_periods['week_start'],
                    y=divergent_periods['sentiment_normalized'],
                    mode='markers',
                    name='Divergence Detected',
                    marker=dict(color='red', size=10, symbol='x')
                ))

            fig_divergence.update_layout(
                xaxis_title="Week",
                yaxis_title="Normalized Score",
                hovermode='x unified',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_divergence, use_container_width=True)

            # Divergence alert
            recent_divergence = divergence_df.nlargest(1, 'week_start')
            if not recent_divergence.empty and recent_divergence['divergence'].values[0]:
                st.markdown("""
                <div class="divergence-alert">
                    <strong>⚠️ DIVERGENCE DETECTED</strong><br>
                    Current narrative stress diverges from FRED index. This may indicate:
                    <ul>
                        <li>Market sentiment shift not yet reflected in systemic indicators</li>
                        <li>Media narrative disconnect from actual financial conditions</li>
                        <li>Early warning signal of upcoming stress changes</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # =====================================================================
    # SECTION 5: Historical Lead-Lag Analysis
    # =====================================================================
    st.header("📊 Historical Lead-Lag Analysis")

    if not news_df.empty and 'theme' in news_df.columns:
        news_weekly = aggregate_news_by_week(news_df)
        fred_weekly = align_fred_weekly(fred_df)

        corr_df = calculate_lead_lag_correlation(news_weekly, fred_weekly, max_lag_weeks=4)

        if not corr_df.empty:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Correlation by Lag")

                fig_corr = px.bar(
                    corr_df,
                    x='lag_weeks',
                    y='correlation',
                    labels={'lag_weeks': 'Lag (weeks)', 'correlation': 'Correlation Coefficient'},
                    color='correlation',
                    color_continuous_scale='RdYlGn',
                    range_color=[-1, 1]
                )
                fig_corr.update_layout(height=300)
                st.plotly_chart(fig_corr, use_container_width=True)

                st.dataframe(corr_df, hide_index=True, use_container_width=True)

            with col2:
                st.subheader("Interpretation")
                summary = generate_analysis_summary(corr_df)
                st.info(summary)

                # Key finding
                max_corr_row = corr_df.loc[corr_df['correlation'].abs().idxmax()]
                st.metric(
                    "Strongest Correlation",
                    f"{max_corr_row['correlation']:.3f}",
                    f"at {int(max_corr_row['lag_weeks'])} week lag"
                )

    # =====================================================================
    # Recent Headlines (Sidebar)
    # =====================================================================
    st.sidebar.header("Recent Headlines")

    if not news_df.empty:
        recent_headlines = news_df.nlargest(10, 'date')

        for _, row in recent_headlines.iterrows():
            with st.sidebar.expander(f"{row['date'].strftime('%Y-%m-%d')} - {row['source']}"):
                st.write(row['headline'])
                if 'theme' in row and 'direction' in row:
                    st.caption(f"Theme: {row['theme']} | Direction: {row['direction']}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit & Claude")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == '__main__':
    main()
