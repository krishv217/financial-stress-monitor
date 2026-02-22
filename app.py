"""
Financial Stress Monitor — Dashboard v3

Tabs:
  1. Live Monitor  — current FRED, theme scores, 2w prediction, divergence alert
  2. Model Analysis — 5-horizon charts (actual vs LR/Lasso/RF), feature importance,
                      R² table, per-week article explorer
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEEKLY_CSV      = 'data/weekly_sentiment_scores.csv'
PREDICTIONS_CSV = 'data/model_predictions.csv'
CLASSIFIED_CSV  = 'data/classified_articles.csv'

SENTIMENT_COLS = [
    'monetary_policy_score', 'credit_debt_score', 'banking_liquidity_score',
    'inflation_growth_score', 'geopolitical_external_score',
]
LAG_COLS = [f'lag1_{c}' for c in SENTIMENT_COLS]
FEATURE_COLS = SENTIMENT_COLS + ['fred_score'] + LAG_COLS
FEATURE_LABELS = [
    'Monetary Policy', 'Credit & Debt', 'Banking & Liquidity',
    'Inflation & Growth', 'Geopolitical', 'Current FRED',
    'Lag: Monetary Policy', 'Lag: Credit & Debt', 'Lag: Banking & Liquidity',
    'Lag: Inflation & Growth', 'Lag: Geopolitical',
]

THEME_COLS = {
    'monetary_policy_score':       'Monetary Policy',
    'credit_debt_score':           'Credit & Debt',
    'banking_liquidity_score':     'Banking & Liquidity',
    'inflation_growth_score':      'Inflation & Growth',
    'geopolitical_external_score': 'Geopolitical & External',
}

DAY_COLORS = {
    'Monday':    '#3b82f6',
    'Tuesday':   '#8b5cf6',
    'Wednesday': '#f59e0b',
    'Thursday':  '#ef4444',
    'Friday':    '#22c55e',
    'Saturday':  '#94a3b8',
    'Sunday':    '#64748b',
}
DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

HORIZONS = [
    ('1w',  'fred_score_1w_future',  '1 Week Ahead'),
    ('2w',  'fred_score_2w_future',  '2 Weeks Ahead'),
    ('4w',  'fred_score_4w_future',  '4 Weeks Ahead'),
    ('8w',  'fred_score_8w_future',  '8 Weeks Ahead'),
    ('12w', 'fred_score_12w_future', '12 Weeks Ahead'),
]

MODEL_DEFS = [
    ('LinearRegression', lambda: LinearRegression(),                                       '#3b82f6'),
    ('LassoCV',          lambda: LassoCV(cv=5, max_iter=5000),                            '#f59e0b'),
    ('RandomForest',     lambda: RandomForestRegressor(n_estimators=100, random_state=42), '#ef4444'),
]

TRAIN_SPLIT = 0.9

st.set_page_config(page_title='Financial Stress Monitor', page_icon='📊', layout='wide')


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_weekly():
    if not os.path.exists(WEEKLY_CSV) or os.path.getsize(WEEKLY_CSV) == 0:
        return pd.DataFrame()
    df = pd.read_csv(WEEKLY_CSV)
    df['week_start'] = pd.to_datetime(df['week_start'])
    all_num = SENTIMENT_COLS + ['fred_score'] + [h[1] for h in HORIZONS] + ['total_articles']
    for col in all_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('week_start').reset_index(drop=True)
    for col in SENTIMENT_COLS:
        df[f'lag1_{col}'] = df[col].shift(1)
    return df


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


@st.cache_data(ttl=600, show_spinner='Loading articles...')
def load_articles():
    if not os.path.exists(CLASSIFIED_CSV) or os.path.getsize(CLASSIFIED_CSV) == 0:
        return pd.DataFrame()
    needed = {'week_start', 'headline', 'abstract', 'url', 'stress_theme',
              'stress_direction', 'magnitude_score', 'is_relevant'}
    df = pd.read_csv(CLASSIFIED_CSV, dtype=str,
                     usecols=lambda c: c in needed).fillna('')
    df = df[df['is_relevant'] == 'yes'].copy()
    df['magnitude_score'] = pd.to_numeric(df['magnitude_score'], errors='coerce').fillna(1)
    return df.sort_values(['week_start', 'magnitude_score'], ascending=[True, False])


@st.cache_data(ttl=600, show_spinner='Computing daily sentiment...')
def load_daily_sentiment():
    """
    Compute a magnitude-weighted composite stress score for every calendar day
    that has at least one classified relevant article.

    Returns a DataFrame with columns:
      pub_date, week_friday, day_name, composite_score, article_count
    """
    if not os.path.exists(CLASSIFIED_CSV) or os.path.getsize(CLASSIFIED_CSV) == 0:
        return pd.DataFrame()

    needed = {'publication_date', 'is_relevant', 'stress_theme',
              'stress_themes', 'stress_direction', 'magnitude_score'}
    df = pd.read_csv(CLASSIFIED_CSV, dtype=str,
                     usecols=lambda c: c in needed).fillna('')
    df = df[(df['is_relevant'] == 'yes') & (df['stress_theme'] != '')].copy()
    df['magnitude_score'] = pd.to_numeric(df['magnitude_score'], errors='coerce').fillna(1)
    df['pub_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df = df.dropna(subset=['pub_date'])

    def _score(row):
        try:
            cls_list = json.loads(row['stress_themes']) if row['stress_themes'] else None
        except Exception:
            cls_list = None
        if not cls_list:
            cls_list = [{'direction': row['stress_direction'],
                         'magnitude': row['magnitude_score']}]
        s = 0
        for c in cls_list:
            try:
                mag = max(1, min(3, int(c.get('magnitude', 1))))
            except Exception:
                mag = 1
            d = c.get('direction', 'neutral')
            s += mag if d == 'increasing' else -mag if d == 'decreasing' else 0
        return s

    df['score'] = df.apply(_score, axis=1)

    daily = (
        df.groupby(df['pub_date'].dt.normalize())
        .agg(composite_score=('score', 'sum'),
             article_count=('score', 'count'))
        .reset_index()
        .rename(columns={'pub_date': 'pub_date'})
    )
    daily['pub_date']    = pd.to_datetime(daily['pub_date'])
    daily['day_name']    = daily['pub_date'].dt.day_name()
    daily['day_of_week'] = daily['pub_date'].dt.weekday
    daily['week_friday'] = daily['pub_date'].apply(
        lambda d: d + pd.Timedelta(days=(4 - d.weekday()) % 7)
    )
    return daily.sort_values('pub_date').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model training (cached — runs once per unique dataset)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner='Training models...')
def train_all_models():
    """Train all 15 model+horizon combos. Returns nested results dict."""
    weekly = load_weekly()
    results = {}
    if weekly.empty:
        return results

    for h_label, h_col, _ in HORIZONS:
        if h_col not in weekly.columns:
            continue
        delta_col = h_col + '_delta'
        work = weekly.copy()
        work[delta_col] = work[h_col] - work['fred_score']
        labeled = work.dropna(subset=FEATURE_COLS + [delta_col]).copy()
        if len(labeled) < 10:
            continue
        n_train = max(5, int(len(labeled) * TRAIN_SPLIT))
        train_df = labeled.iloc[:n_train]
        test_df  = labeled.iloc[n_train:]

        X_train, y_train = train_df[FEATURE_COLS].values, train_df[delta_col].values
        X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df[delta_col].values

        for m_name, m_factory, _ in MODEL_DEFS:
            model = m_factory()
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            test_preds  = model.predict(X_test) if len(test_df) > 0 else np.array([])

            if len(test_df) > 0:
                test_r2 = float(r2_score(y_test, test_preds))
                rmse    = float(np.sqrt(mean_squared_error(y_test, test_preds)))
                mae     = float(np.mean(np.abs(y_test - test_preds)))
            else:
                test_r2 = rmse = mae = float('nan')

            coef = (model.feature_importances_ if hasattr(model, 'feature_importances_')
                    else model.coef_ if hasattr(model, 'coef_') else np.zeros(len(FEATURE_COLS)))

            results[(h_label, m_name)] = {
                'train_r2':      float(r2_score(y_train, train_preds)),
                'test_r2':       test_r2,
                'rmse':          rmse,
                'mae':           mae,
                'coef':          coef.tolist(),
                'n_train':       n_train,
                'n_test':        len(test_df),
                'cutoff':        str(train_df['week_start'].iloc[-1].date()),
                'train_weeks':   train_df['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                'test_weeks':    test_df['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                'train_preds':   train_preds.tolist(),
                'test_preds':    test_preds.tolist(),
                'train_actuals': y_train.tolist(),
                'test_actuals':  y_test.tolist(),
            }
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fred_color(v):
    if pd.isna(v): return '#888888'
    return '#22c55e' if v < -0.5 else '#f59e0b' if v < 0.5 else '#ef4444'

def fred_label(v):
    if pd.isna(v): return 'Unknown'
    return 'Low Stress' if v < -0.5 else 'Moderate' if v < 0.5 else 'High Stress'


# ---------------------------------------------------------------------------
# Tab 1 panels
# ---------------------------------------------------------------------------

def panel_fred_trend(weekly):
    st.subheader('Current Financial Stress (FRED STLFSI4)')
    if weekly.empty or weekly['fred_score'].isna().all():
        st.info('No FRED data yet. Run pipeline.py to populate.')
        return
    recent = weekly.dropna(subset=['fred_score']).tail(12)
    cur  = float(recent['fred_score'].iloc[-1])
    prev = float(recent['fred_score'].iloc[-2]) if len(recent) > 1 else None
    c1, c2, c3 = st.columns(3)
    c1.metric('Current FRED Score', f'{cur:.4f}',
              delta=f'{cur - prev:.4f}' if prev is not None else None)
    c2.metric('Status', fred_label(cur))
    c3.metric('As of week', recent['week_start'].iloc[-1].strftime('%Y-%m-%d'))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=recent['week_start'], y=recent['fred_score'],
        marker_color=[fred_color(v) for v in recent['fred_score']],
        hovertemplate='%{x|%Y-%m-%d}<br>FRED: %{y:.4f}<extra></extra>',
    ))
    fig.add_hline(y=0,    line_color='gray',    line_dash='dash', line_width=1)
    fig.add_hline(y=-0.5, line_color='#22c55e', line_dash='dot',  line_width=1,
                  annotation_text='calm',   annotation_position='bottom right')
    fig.add_hline(y=0.5,  line_color='#ef4444', line_dash='dot',  line_width=1,
                  annotation_text='stress', annotation_position='top right')
    fig.update_layout(title='12-Week FRED Financial Stress History',
                      xaxis_title='Week', yaxis_title='STLFSI4', height=340,
                      showlegend=False,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)


def panel_theme_scores(weekly):
    st.subheader('Current Week: Stress Theme Breakdown')
    theme_data = weekly.dropna(subset=list(THEME_COLS.keys()), how='all')
    if theme_data.empty:
        st.info('No theme scores yet.')
        return
    latest = theme_data.iloc[-1]
    st.caption(f'Week of {latest["week_start"].strftime("%Y-%m-%d")}  |  '
               f'{int(latest.get("total_articles", 0) or 0)} articles classified')
    scores = {label: float(latest.get(col, 0) or 0) for col, label in THEME_COLS.items()}
    fig = go.Figure(go.Bar(
        x=list(scores.values()), y=list(scores.keys()), orientation='h',
        marker_color=['#ef4444' if v > 0 else '#22c55e' if v < 0 else '#94a3b8'
                      for v in scores.values()],
        hovertemplate='%{y}: %{x:.1f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_color='gray', line_dash='dash')
    fig.update_layout(title='Magnitude-Weighted Net Stress Score by Theme',
                      xaxis_title='Net Score', height=300,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)


def panel_prediction(predictions, weekly):
    st.subheader('Model Prediction: FRED in 2 Weeks')
    if predictions.empty:
        st.info('No predictions yet. Run train_model.py.')
        return
    latest = predictions.sort_values('week_start').iloc[-1]
    pred   = float(latest['predicted_fred_score'])
    w_act  = predictions.dropna(subset=['actual_fred_score', 'predicted_fred_score'])
    avg_err = float(w_act['prediction_error'].abs().mean()) if not w_act.empty else None
    c1, c2, c3 = st.columns(3)
    c1.metric('Predicted FRED (2w)', f'{pred:.4f}')
    c2.metric('Model version', str(latest.get('model_version', 'v1')))
    if avg_err is not None:
        c3.metric('Avg abs error', f'+-{avg_err:.4f}')
    if not weekly.empty:
        cur = float(weekly.dropna(subset=['fred_score'])['fred_score'].iloc[-1])
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta', value=pred,
            delta={'reference': cur, 'valueformat': '.4f'},
            title={'text': f'Predicted FRED<br><sub>current: {cur:.4f}</sub>'},
            gauge={
                'axis': {'range': [-3, 3]},
                'bar': {'color': fred_color(pred)},
                'steps': [{'range': [-3, -0.5], 'color': '#dcfce7'},
                          {'range': [-0.5, 0.5], 'color': '#fef9c3'},
                          {'range': [0.5, 3],    'color': '#fee2e2'}],
                'threshold': {'line': {'color': 'black', 'width': 3},
                              'thickness': 0.75, 'value': cur},
            },
        ))
        fig.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)


def panel_divergence(predictions, weekly):
    st.subheader('Divergence Alert')
    if predictions.empty or weekly.empty:
        return
    w = weekly.dropna(subset=['fred_score'])
    if w.empty:
        return
    cur  = float(w['fred_score'].iloc[-1])
    pred = float(predictions.sort_values('week_start').iloc[-1]['predicted_fred_score'])
    div, abs_div = pred - cur, abs(pred - cur)
    if abs_div < 0.3:
        color, label, desc = '#22c55e', 'CALM — No Unusual Signal', \
            'Model prediction aligns with current FRED. No elevated stress signal detected.'
    elif abs_div < 0.75:
        d = 'rising' if div > 0 else 'falling'
        color, label, desc = '#f59e0b', f'MODERATE DIVERGENCE — trending {d}', \
            f'Model predicts FRED moving {d} by {abs_div:.3f} over 2 weeks. Monitor for confirmation.'
    else:
        d = 'RISING' if div > 0 else 'FALLING'
        color, label, desc = '#ef4444', f'SIGNIFICANT DIVERGENCE — FRED {d}', \
            f'Model predicts a shift of {abs_div:.3f} over 2 weeks. Review current theme scores.'
    st.markdown(
        f'<div style="background:{color}22;border-left:6px solid {color};'
        f'padding:1rem 1.5rem;border-radius:4px">'
        f'<h3 style="color:{color};margin:0">{label}</h3>'
        f'<p style="margin:.5rem 0 0">{desc}</p>'
        f'<p style="font-size:.85rem;color:#666;margin:.25rem 0 0">'
        f'Current: {cur:.4f} | Predicted: {pred:.4f} | Delta: {div:+.4f}</p></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 2: Model Analysis helpers
# ---------------------------------------------------------------------------

def _metrics_table(model_results, h_label):
    rows = []
    best_r2 = max(
        (model_results[(h_label, m)]['test_r2']
         for m, _, _ in MODEL_DEFS if (h_label, m) in model_results
         and not np.isnan(model_results[(h_label, m)]['test_r2'])),
        default=float('-inf'),
    )
    for m_name, _, color in MODEL_DEFS:
        key = (h_label, m_name)
        if key not in model_results:
            continue
        res = model_results[key]
        r2 = res['test_r2']
        is_best = not np.isnan(r2) and r2 == best_r2
        rows.append({
            'Model':     ('* ' if is_best else '  ') + m_name.replace('LinearRegression', 'Linear').replace('RandomForest', 'RF'),
            'Train R2':  f'{res["train_r2"]:.3f}',
            'Test R2':   f'{r2:.3f}' if not np.isnan(r2) else 'N/A',
            'RMSE':      f'{res["rmse"]:.4f}' if not np.isnan(res["rmse"]) else 'N/A',
            'MAE':       f'{res["mae"]:.4f}' if not np.isnan(res["mae"]) else 'N/A',
            'n train':   res['n_train'],
            'n test':    res['n_test'],
        })
    if rows:
        st.caption('* = best test R² for this horizon')
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _feature_importance_chart(model_results, h_label):
    sel_model = st.radio(
        'Show feature importance for:',
        [m for m, _, _ in MODEL_DEFS if (h_label, m) in model_results],
        horizontal=True,
        key=f'imp_radio_{h_label}',
    )
    key = (h_label, sel_model)
    if key not in model_results:
        return
    coef = np.array(model_results[key]['coef'])
    is_rf = 'Forest' in sel_model
    vals = coef if is_rf else np.abs(coef)
    if vals.sum() > 0:
        vals = vals / vals.sum()

    imp_df = pd.DataFrame({'Feature': FEATURE_LABELS, 'Value': vals}) \
               .sort_values('Value', ascending=True)

    colors = ['#3b82f6' if v >= 0 else '#ef4444' for v in imp_df['Value']]
    fig = go.Figure(go.Bar(
        x=imp_df['Value'], y=imp_df['Feature'], orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: %{x:.3f}<extra></extra>',
    ))
    title = 'Feature Importance (RF)' if is_rf else 'Coefficients — abs normalized (LR/Lasso)'
    fig.update_layout(
        title=title, xaxis_title='Relative weight',
        height=260, margin=dict(l=0, r=10, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)


def _horizon_scatter(weekly, model_results, h_label, h_col, h_name):
    delta_col = h_col + '_delta'
    work = weekly.copy()
    work[delta_col] = work[h_col] - work['fred_score']
    labeled = work.dropna(subset=FEATURE_COLS + [delta_col])
    if labeled.empty:
        st.info(f'No labeled data for {h_name}.')
        return

    n_train   = max(5, int(len(labeled) * TRAIN_SPLIT))
    cutoff    = labeled['week_start'].iloc[n_train - 1]
    cutoff_str = str(cutoff.date())
    x0_str     = str(labeled['week_start'].iloc[0].date())

    # --- Full-width scatter chart ---
    fig = go.Figure()

    # Training region shading + split marker
    fig.add_vrect(
        x0=x0_str, x1=cutoff_str,
        fillcolor='rgba(200,200,200,0.12)', layer='below', line_width=0,
    )
    fig.add_shape(
        type='line', x0=cutoff_str, x1=cutoff_str, y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='#64748b', dash='dash', width=1.5),
    )
    fig.add_annotation(
        x=cutoff_str, y=1, xref='x', yref='paper',
        text='Test split', showarrow=False,
        xanchor='left', yanchor='top',
        font=dict(size=11, color='#64748b'),
        bgcolor='rgba(255,255,255,0.7)',
    )
    fig.add_annotation(
        x=x0_str, y=1, xref='x', yref='paper',
        text='Train period', showarrow=False,
        xanchor='left', yanchor='top',
        font=dict(size=11, color='#64748b'),
        bgcolor='rgba(255,255,255,0.7)',
    )

    # Actual Δfred
    actual_delta = labeled[delta_col].values
    fig.add_trace(go.Scatter(
        x=labeled['week_start'], y=actual_delta,
        mode='lines+markers', name='Actual ΔFRED',
        line=dict(color='#1e293b', width=2.5),
        marker=dict(size=5),
        hovertemplate='%{x|%Y-%m-%d}<br>Actual Δ: %{y:.4f}<extra></extra>',
    ))

    # Zero line for reference
    fig.add_hline(y=0, line_color='gray', line_dash='dash', line_width=1)

    for m_name, _, color in MODEL_DEFS:
        key = (h_label, m_name)
        if key not in model_results:
            continue
        res = model_results[key]
        short = m_name.replace('LinearRegression', 'Linear').replace('RandomForest', 'RF')

        # Train predictions (faded) — already in delta space
        fig.add_trace(go.Scatter(
            x=res['train_weeks'], y=res['train_preds'],
            mode='lines', name=f'{short} (train)',
            line=dict(color=color, width=1.5, dash='dot'),
            opacity=0.6, showlegend=False,
            hovertemplate=f'{short} train Δ: %{{y:.4f}}<extra></extra>',
        ))
        # Test predictions (bold dashed)
        if res['test_preds']:
            r2_str = f'{res["test_r2"]:.3f}' if not np.isnan(res['test_r2']) else 'N/A'
            fig.add_trace(go.Scatter(
                x=res['test_weeks'], y=res['test_preds'],
                mode='lines+markers', name=f'{short}  (test R²={r2_str})',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=7, symbol='diamond'),
                hovertemplate=f'%{{x|%Y-%m-%d}}<br>{short} Δ: %{{y:.4f}}<extra></extra>',
            ))

    fig.update_layout(
        title=f'Predicted vs Actual ΔFRED — {h_name}',
        xaxis_title='Week', yaxis_title='ΔFRED (change from current)',
        height=480,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.08, font_size=11),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Metrics + feature importance side by side below the chart ---
    col_metrics, col_imp = st.columns(2)
    with col_metrics:
        st.markdown('**Test Set Metrics**')
        _metrics_table(model_results, h_label)
    with col_imp:
        st.markdown('**Feature Weights**')
        _feature_importance_chart(model_results, h_label)


def panel_recent_articles(articles, weekly):
    """Show top articles from the most recent classified week."""
    st.subheader('Recent Articles Driving Sentiment')

    if articles.empty:
        st.info('Enable "Load articles" in the sidebar to see recent articles.')
        return

    # Find the most recent week that has both sentiment scores and articles
    available_weeks = sorted(articles['week_start'].unique(), reverse=True)
    if not available_weeks:
        st.caption('No articles found.')
        return

    latest_week = available_weeks[0]

    # Let user browse other recent weeks too
    week_options = available_weeks[:12]  # last 12 weeks
    selected = st.selectbox('Week:', week_options, index=0, key='live_week_sel')

    # Theme score summary for selected week
    if not weekly.empty:
        week_row = weekly[weekly['week_start'].dt.strftime('%Y-%m-%d') == selected]
        if not week_row.empty:
            r = week_row.iloc[0]
            cols = st.columns(len(THEME_COLS))
            for i, (col_name, label) in enumerate(THEME_COLS.items()):
                val = float(r.get(col_name, 0) or 0)
                cols[i].metric(label, f'{val:+.0f}',
                               delta_color='normal' if val != 0 else 'off')

    week_arts = articles[articles['week_start'] == selected].head(15)
    st.markdown(f'**Top articles — {selected}** ({len(week_arts)} shown, sorted by magnitude)')

    for _, art in week_arts.iterrows():
        theme     = art.get('stress_theme', '')
        direction = art.get('stress_direction', '')
        mag       = int(art.get('magnitude_score', 1))
        url       = art.get('url', '')
        headline  = art.get('headline', '')
        abstract  = art.get('abstract', '')

        dir_color = ('#ef4444' if direction == 'increasing'
                     else '#22c55e' if direction == 'decreasing' else '#94a3b8')
        stars = '★' * mag + '☆' * (3 - mag)

        with st.expander(headline[:110] or '(no headline)'):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'**Theme:** `{theme}`')
            c2.markdown(f'**Direction:** <span style="color:{dir_color};font-weight:600">'
                        f'{direction}</span>', unsafe_allow_html=True)
            c3.markdown(f'**Magnitude:** {stars}')
            if abstract:
                st.caption(abstract[:300])
            if url:
                st.markdown(f'[Read on NYT]({url})')


# ---------------------------------------------------------------------------
# Tab 3: Daily Signal
# ---------------------------------------------------------------------------

DAILY_HORIZON_OPTS = [
    ('1 Week Ahead',  'fred_score_1w_future'),
    ('2 Weeks Ahead', 'fred_score_2w_future'),
    ('4 Weeks Ahead', 'fred_score_4w_future'),
]

def panel_daily_sentiment_scatter(weekly):
    st.subheader('Daily Article Sentiment vs Weekly ΔFRED')
    st.caption(
        'Each point is one calendar day of articles. '
        'X = that day\'s composite stress score (sum of magnitude-weighted directions). '
        'Y = ΔFRED for the week that day belongs to. '
        'Color = day of week. Shows whether intra-week timing or article volume drives the signal.'
    )

    daily = load_daily_sentiment()
    if daily.empty:
        st.info('No article data. Enable "Load daily signal" in the sidebar first.')
        return

    if weekly.empty or 'fred_score_1w_future' not in weekly.columns:
        st.info('No weekly FRED data yet.')
        return

    # Horizon picker
    sel_h_label = st.radio(
        'ΔFRED horizon:', [h[0] for h in DAILY_HORIZON_OPTS],
        horizontal=True, key='daily_h_sel',
    )
    h_col = dict(DAILY_HORIZON_OPTS)[sel_h_label]

    # Build ΔFRED lookup: week_friday -> delta
    work = weekly.copy()
    if h_col not in work.columns:
        st.info(f'{h_col} not available.')
        return
    work['delta'] = work[h_col] - work['fred_score']
    delta_lookup = work.dropna(subset=['delta']).set_index('week_start')['delta']

    plot_df = daily.copy()
    plot_df['delta'] = plot_df['week_friday'].map(delta_lookup)
    plot_df = plot_df.dropna(subset=['delta']).copy()

    if plot_df.empty:
        st.info('No overlapping daily+FRED data found.')
        return

    # ── Scatter chart ────────────────────────────────────────────────────────
    fig = go.Figure()

    for day_name in DAY_ORDER:
        color = DAY_COLORS[day_name]
        sub = plot_df[plot_df['day_name'] == day_name]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['composite_score'],
            y=sub['delta'],
            mode='markers',
            name=day_name,
            marker=dict(color=color, size=6, opacity=0.55),
            hovertemplate=(
                f'<b>{day_name}</b><br>'
                'Date: %{customdata}<br>'
                'Sentiment: %{x:.1f}<br>'
                f'ΔFRED ({sel_h_label}): %{{y:.4f}}'
                '<extra></extra>'
            ),
            customdata=sub['pub_date'].dt.strftime('%Y-%m-%d'),
        ))

    # OLS trend line across all days
    x_all = plot_df['composite_score'].values
    y_all = plot_df['delta'].values
    if len(x_all) > 10:
        slope, intercept = np.polyfit(x_all, y_all, 1)
        x_range = np.linspace(x_all.min(), x_all.max(), 200)
        fig.add_trace(go.Scatter(
            x=x_range, y=slope * x_range + intercept,
            mode='lines',
            name=f'Trend  (slope={slope:.4f})',
            line=dict(color='#1e293b', width=2.5, dash='dash'),
            hoverinfo='skip',
        ))
        corr = float(np.corrcoef(x_all, y_all)[0, 1])
        st.caption(
            f'All days — r = **{corr:.3f}** | slope = {slope:.4f} | n = {len(x_all):,} days'
        )

    fig.add_hline(y=0, line_color='gray', line_dash='dot', line_width=1)
    fig.add_vline(x=0, line_color='gray', line_dash='dot', line_width=1)

    fig.update_layout(
        title=f'Daily Composite Sentiment vs ΔFRED ({sel_h_label})',
        xaxis_title='Daily Composite Stress Score',
        yaxis_title=f'ΔFRED — {sel_h_label}',
        height=520,
        hovermode='closest',
        legend=dict(title='Day of week', font_size=11,
                    orientation='v', x=1.01, y=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-day-of-week stats table ───────────────────────────────────────────
    st.markdown('**Correlation by Day of Week**')
    rows = []
    for day_name in DAY_ORDER:
        sub = plot_df[plot_df['day_name'] == day_name]
        if len(sub) < 5:
            continue
        r = float(np.corrcoef(sub['composite_score'], sub['delta'])[0, 1])
        rows.append({
            'Day':              day_name,
            'Days w/ articles': len(sub),
            'Avg sentiment':    round(float(sub['composite_score'].mean()), 2),
            'Avg |ΔFRED|':      round(float(sub['delta'].abs().mean()), 4),
            'Correlation r':    round(r, 3),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 main
# ---------------------------------------------------------------------------

def tab_model_analysis(weekly, model_results):
    st.subheader('5-Horizon Model Comparison')
    st.caption(
        'Each section: actual FRED (black) vs all 3 model predictions. '
        'Faded dotted lines = training period. Dashed bold lines = test period.'
    )

    for h_label, h_col, h_name in HORIZONS:
        with st.expander(f'{h_name}  ({h_label})', expanded=(h_label == '2w')):
            _horizon_scatter(weekly, model_results, h_label, h_col, h_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title('Financial Stress Monitor')
    st.caption(
        'Narrative-driven early warning system for FRED STLFSI4 | '
        'NYT articles classified by Claude AI | '
        'LinearRegression · LassoCV · RandomForest across 5 prediction horizons'
    )

    weekly      = load_weekly()
    predictions = load_predictions()

    with st.sidebar:
        st.header('Data Status')
        if not weekly.empty:
            st.metric('Weeks in dataset', len(weekly))
            st.metric('Weeks with FRED', int(weekly['fred_score'].notna().sum()))
            st.caption(
                f"{weekly['week_start'].min().strftime('%Y-%m-%d')} to "
                f"{weekly['week_start'].max().strftime('%Y-%m-%d')}"
            )
        else:
            st.warning('No data yet.')
        if not predictions.empty:
            st.divider()
            st.metric('Predictions made', len(predictions))
            st.metric('With actuals', int(predictions['actual_fred_score'].notna().sum()))
        st.divider()
        load_arts = st.checkbox('Show recent articles', value=False,
                                help='Reads classified_articles.csv (~25 MB). Slow on first load.')
        load_daily = st.checkbox('Show daily signal', value=False,
                                 help='Computes per-day sentiment from classified_articles.csv.')
        st.divider()
        if st.button('Refresh data'):
            st.cache_data.clear()
            st.rerun()

    tab1, tab2, tab3 = st.tabs(['Live Monitor', 'Model Analysis', 'Daily Signal'])

    with tab1:
        panel_fred_trend(weekly)
        st.divider()
        left, right = st.columns(2)
        with left:
            panel_theme_scores(weekly)
        with right:
            panel_prediction(predictions, weekly)
        st.divider()
        panel_divergence(predictions, weekly)
        if load_arts:
            st.divider()
            articles = load_articles()
            panel_recent_articles(articles, weekly)

    with tab2:
        if weekly.empty:
            st.info('Run pipeline.py and train_model.py first to populate data.')
        else:
            model_results = train_all_models()
            tab_model_analysis(weekly, model_results)

    with tab3:
        if not load_daily:
            st.info('Enable **"Show daily signal"** in the sidebar to load this tab.')
        else:
            panel_daily_sentiment_scatter(weekly)

    st.caption(
        f'Refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M")} | '
        'Data: NYT Article Search API + FRED STLFSI4 | '
        'Models: LinearRegression, LassoCV, RandomForest'
    )


if __name__ == '__main__':
    main()
