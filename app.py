"""
Financial Stress Monitor — Dashboard v2

5 panels reading from the 4 pipeline CSV files:
  1. FRED trend  (weekly_sentiment_scores.csv — last 12 weeks)
  2. Theme scores (weekly_sentiment_scores.csv — current week bar chart)
  3. Model prediction (model_predictions.csv — latest predicted FRED)
  4. Divergence alert (current FRED vs predicted change)
  5. Predicted vs actual over time (model_predictions.csv)
"""

import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEEKLY_CSV = 'data/weekly_sentiment_scores.csv'
PREDICTIONS_CSV = 'data/model_predictions.csv'
MODEL_PKL = 'data/model.pkl'

THEME_COLS = {
    'monetary_policy_score': 'Monetary Policy',
    'credit_debt_score': 'Credit & Debt',
    'banking_liquidity_score': 'Banking & Liquidity',
    'inflation_growth_score': 'Inflation & Growth',
    'geopolitical_external_score': 'Geopolitical & External',
}

st.set_page_config(
    page_title='Financial Stress Monitor',
    page_icon='📊',
    layout='wide',
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_weekly():
    if not os.path.exists(WEEKLY_CSV) or os.path.getsize(WEEKLY_CSV) == 0:
        return pd.DataFrame()
    df = pd.read_csv(WEEKLY_CSV)
    df['week_start'] = pd.to_datetime(df['week_start'])
    numeric_cols = list(THEME_COLS.keys()) + ['fred_score', 'fred_score_2w_future', 'total_articles']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('week_start').reset_index(drop=True)


@st.cache_data(ttl=300)
def load_predictions():
    if not os.path.exists(PREDICTIONS_CSV) or os.path.getsize(PREDICTIONS_CSV) == 0:
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_CSV)
    df['week_start'] = pd.to_datetime(df['week_start'])
    for col in ['predicted_fred_score', 'actual_fred_score', 'prediction_error']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('week_start').reset_index(drop=True)


def fred_color(value):
    if pd.isna(value):
        return '#888888'
    if value < -0.5:
        return '#22c55e'
    if value < 0.5:
        return '#f59e0b'
    return '#ef4444'


def fred_label(value):
    if pd.isna(value):
        return 'Unknown'
    if value < -0.5:
        return 'Low Stress'
    if value < 0.5:
        return 'Moderate'
    return 'High Stress'


# ---------------------------------------------------------------------------
# Panel 1: FRED Trend (12-week)
# ---------------------------------------------------------------------------

def panel_fred_trend(weekly):
    st.subheader('Panel 1 — Current Financial Stress (FRED STLFSI4)')

    if weekly.empty or weekly['fred_score'].isna().all():
        st.info('No FRED data yet. Run pipeline.py to populate.')
        return

    recent = weekly.dropna(subset=['fred_score']).tail(12)
    current_score = recent['fred_score'].iloc[-1]
    current_date = recent['week_start'].iloc[-1]
    prev_score = recent['fred_score'].iloc[-2] if len(recent) > 1 else None

    col1, col2, col3 = st.columns(3)
    col1.metric('Current FRED Score', f'{current_score:.4f}',
                delta=f'{current_score - prev_score:.4f}' if prev_score is not None else None)
    col2.metric('Status', fred_label(current_score))
    col3.metric('As of week', current_date.strftime('%Y-%m-%d'))

    colors = [fred_color(v) for v in recent['fred_score']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=recent['week_start'],
        y=recent['fred_score'],
        marker_color=colors,
        name='FRED Score',
        hovertemplate='%{x|%Y-%m-%d}<br>FRED: %{y:.4f}<extra></extra>',
    ))
    fig.add_hline(y=0, line_color='gray', line_dash='dash', line_width=1)
    fig.add_hline(y=-0.5, line_color='#22c55e', line_dash='dot', line_width=1,
                  annotation_text='calm threshold', annotation_position='bottom right')
    fig.add_hline(y=0.5, line_color='#ef4444', line_dash='dot', line_width=1,
                  annotation_text='stress threshold', annotation_position='top right')
    fig.update_layout(
        title='12-Week FRED Financial Stress Index History',
        xaxis_title='Week', yaxis_title='STLFSI4 Score',
        height=350, showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Panel 2: Theme Scores
# ---------------------------------------------------------------------------

def panel_theme_scores(weekly):
    st.subheader('Panel 2 — Current Week: Stress Theme Breakdown')

    theme_data = weekly.dropna(subset=list(THEME_COLS.keys()), how='all')
    if theme_data.empty:
        st.info('No theme scores yet. Run pipeline.py to populate.')
        return

    latest = theme_data.iloc[-1]
    week_label = latest['week_start'].strftime('%Y-%m-%d')
    total = int(latest.get('total_articles', 0) or 0)
    st.caption(f'Week of {week_label}  |  {total} articles classified')

    scores = {label: float(latest.get(col, 0) or 0)
              for col, label in THEME_COLS.items()}
    colors = ['#ef4444' if v > 0 else '#22c55e' if v < 0 else '#94a3b8'
              for v in scores.values()]

    fig = go.Figure(go.Bar(
        x=list(scores.values()),
        y=list(scores.keys()),
        orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: %{x:.1f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_color='gray', line_dash='dash', line_width=1)
    fig.update_layout(
        title='Magnitude-Weighted Net Stress Score by Theme (red=increasing, green=decreasing)',
        xaxis_title='Net Score',
        height=320,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Panel 3: Model Prediction
# ---------------------------------------------------------------------------

def panel_prediction(predictions, weekly):
    st.subheader('Panel 3 — Model Prediction: FRED in 2 Weeks')

    if predictions.empty:
        st.info('No predictions yet. Run train_model.py after pipeline.py.')
        return

    latest_pred = predictions.sort_values('week_start').iloc[-1]
    predicted = float(latest_pred['predicted_fred_score'])
    pred_week = latest_pred['week_start']
    model_ver = str(latest_pred.get('model_version', 'v1'))

    with_actuals = predictions.dropna(subset=['actual_fred_score', 'predicted_fred_score'])
    avg_error = float(with_actuals['prediction_error'].abs().mean()) if not with_actuals.empty else None

    col1, col2, col3 = st.columns(3)
    col1.metric('Predicted FRED (2w out)', f'{predicted:.4f}')
    col2.metric('Model version', model_ver)
    if avg_error is not None:
        col3.metric('Avg absolute error', f'±{avg_error:.4f}')
    else:
        col3.metric('Prediction week', pred_week.strftime('%Y-%m-%d'))

    current_fred = None
    if not weekly.empty:
        w = weekly.dropna(subset=['fred_score'])
        if not w.empty:
            current_fred = float(w['fred_score'].iloc[-1])

    if current_fred is not None:
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=predicted,
            delta={'reference': current_fred, 'valueformat': '.4f'},
            title={'text': f'Predicted FRED Score<br><sub>current: {current_fred:.4f}</sub>'},
            gauge={
                'axis': {'range': [-3, 3]},
                'bar': {'color': fred_color(predicted)},
                'steps': [
                    {'range': [-3, -0.5], 'color': '#dcfce7'},
                    {'range': [-0.5, 0.5], 'color': '#fef9c3'},
                    {'range': [0.5, 3], 'color': '#fee2e2'},
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 3},
                    'thickness': 0.75,
                    'value': current_fred,
                },
            },
        ))
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Panel 4: Divergence Alert
# ---------------------------------------------------------------------------

def panel_divergence(predictions, weekly):
    st.subheader('Panel 4 — Divergence Alert')

    if predictions.empty or weekly.empty:
        st.info('Need both weekly scores and predictions to show divergence.')
        return

    current_fred_row = weekly.dropna(subset=['fred_score'])
    if current_fred_row.empty:
        return

    current_fred = float(current_fred_row['fred_score'].iloc[-1])
    latest_pred = predictions.sort_values('week_start').iloc[-1]
    predicted = float(latest_pred['predicted_fred_score'])
    divergence = predicted - current_fred
    abs_div = abs(divergence)

    if abs_div < 0.3:
        color = '#22c55e'
        label = 'CALM — No Unusual Signal'
        desc = 'Model prediction aligns with current FRED. No elevated stress signal detected.'
    elif abs_div < 0.75:
        color = '#f59e0b'
        direction = 'rising' if divergence > 0 else 'falling'
        label = f'MODERATE DIVERGENCE — FRED trending {direction}'
        desc = (f'Model predicts FRED moving {direction} by {abs_div:.3f} over the next 2 weeks. '
                'Monitor for confirmation in incoming data.')
    else:
        color = '#ef4444'
        direction = 'RISING' if divergence > 0 else 'FALLING'
        label = f'SIGNIFICANT DIVERGENCE — FRED {direction}'
        desc = (f'Model predicts a meaningful FRED shift of {abs_div:.3f} over the next 2 weeks. '
                'This is an elevated early warning signal. Review current theme scores.')

    st.markdown(
        f'<div style="background:{color}22;border-left:6px solid {color};'
        f'padding:1rem 1.5rem;border-radius:4px;margin:0.5rem 0">'
        f'<h3 style="color:{color};margin:0">{label}</h3>'
        f'<p style="margin:0.5rem 0 0">{desc}</p>'
        f'<p style="margin:0.25rem 0 0;font-size:0.85rem;color:#666">'
        f'Current FRED: {current_fred:.4f} &nbsp;|&nbsp; '
        f'Predicted: {predicted:.4f} &nbsp;|&nbsp; '
        f'Delta: {divergence:+.4f}</p></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Panel 5: Predicted vs Actual
# ---------------------------------------------------------------------------

def panel_predicted_vs_actual(predictions):
    st.subheader('Panel 5 — Predicted vs Actual FRED Over Time')

    if predictions.empty:
        st.info('No predictions yet.')
        return

    with_actuals = predictions.dropna(subset=['actual_fred_score', 'predicted_fred_score'])

    if with_actuals.empty:
        st.caption('Predictions recorded — actuals will appear after 2 weeks have passed.')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions['week_start'],
            y=predictions['predicted_fred_score'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#3b82f6', dash='dash'),
        ))
        fig.update_layout(
            title='Model Predictions (awaiting actuals)',
            xaxis_title='Week', yaxis_title='FRED Score',
            height=350,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    rmse = float(np.sqrt((with_actuals['prediction_error'] ** 2).mean()))
    mae = float(with_actuals['prediction_error'].abs().mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=with_actuals['week_start'], y=with_actuals['actual_fred_score'],
        mode='lines+markers', name='Actual FRED',
        line=dict(color='#1e293b', width=2), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=with_actuals['week_start'], y=with_actuals['predicted_fred_score'],
        mode='lines+markers', name='Model Prediction',
        line=dict(color='#3b82f6', dash='dash', width=2),
        marker=dict(size=6, symbol='diamond'),
    ))
    # Shaded error band
    upper = with_actuals['predicted_fred_score'] + with_actuals['prediction_error'].abs()
    lower = with_actuals['predicted_fred_score'] - with_actuals['prediction_error'].abs()
    fig.add_trace(go.Scatter(
        x=pd.concat([with_actuals['week_start'], with_actuals['week_start'].iloc[::-1]]),
        y=pd.concat([upper, lower.iloc[::-1]]),
        fill='toself', fillcolor='rgba(59,130,246,0.1)',
        line=dict(color='rgba(255,255,255,0)'), name='Error band',
    ))
    fig.update_layout(
        title=f'Model Performance | RMSE: {rmse:.4f} | MAE: {mae:.4f} | n={len(with_actuals)}',
        xaxis_title='Week', yaxis_title='FRED Score',
        height=420,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander('Prediction history table'):
        display = with_actuals[['week_start', 'predicted_fred_score',
                                'actual_fred_score', 'prediction_error',
                                'model_version']].copy()
        display['week_start'] = display['week_start'].dt.strftime('%Y-%m-%d')
        st.dataframe(display.sort_values('week_start', ascending=False), use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title('Financial Stress Monitor')
    st.caption(
        'Narrative-driven early warning system for FRED STLFSI4 '
        '| NYT Article Search API + Claude classification + Linear Regression'
    )

    weekly = load_weekly()
    predictions = load_predictions()

    with st.sidebar:
        st.header('Data Status')
        if not weekly.empty:
            n_weeks = len(weekly)
            n_with_fred = int(weekly['fred_score'].notna().sum())
            n_classified = int(weekly['total_articles'].notna().sum())
            st.metric('Weeks in dataset', n_weeks)
            st.metric('Weeks with FRED', n_with_fred)
            st.metric('Weeks classified', n_classified)
            date_range = (
                f"{weekly['week_start'].min().strftime('%Y-%m-%d')} to "
                f"{weekly['week_start'].max().strftime('%Y-%m-%d')}"
            )
            st.caption(date_range)
        else:
            st.warning('No data yet.')
            st.markdown('''
**Run in order:**
```
python init_data.py
python collect_nyt.py   # ~2 hrs
python pipeline.py
python train_model.py
```
            ''')

        if not predictions.empty:
            st.divider()
            n_preds = len(predictions)
            n_actual = int(predictions['actual_fred_score'].notna().sum())
            st.metric('Predictions made', n_preds)
            st.metric('With actuals', n_actual)

        st.divider()
        if st.button('Refresh data'):
            st.cache_data.clear()
            st.rerun()

    panel_fred_trend(weekly)
    st.divider()

    left, right = st.columns(2)
    with left:
        panel_theme_scores(weekly)
    with right:
        panel_prediction(predictions, weekly)

    st.divider()
    panel_divergence(predictions, weekly)

    st.divider()
    panel_predicted_vs_actual(predictions)

    st.caption(
        f'Refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M")} | '
        'Data: NYT Article Search API (2019-present) + FRED STLFSI4 | '
        'Model: OLS linear regression on 5 magnitude-weighted theme scores + current FRED'
    )


if __name__ == '__main__':
    main()
