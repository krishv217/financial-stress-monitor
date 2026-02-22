"""
StressPress — Dashboard v3

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

st.set_page_config(page_title='StressPress', page_icon='📊', layout='wide')


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


# ---------------------------------------------------------------------------
# Model training (cached — runs once per unique dataset)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner='Training models...')
def train_all_models():
    """Train delta and absolute models for all horizons. Returns results dict."""
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

        X_train = train_df[FEATURE_COLS].values
        X_test  = test_df[FEATURE_COLS].values

        # Forecast row: most recent week with full features but no known future FRED
        future_rows = work[work[FEATURE_COLS].notna().all(axis=1) & work[delta_col].isna()]
        weeks_ahead = int(h_label.replace('w', ''))
        if not future_rows.empty:
            latest          = future_rows.iloc[-1]
            X_fc            = latest[FEATURE_COLS].values.reshape(1, -1)
            forecast_from   = str(latest['week_start'].date())
            forecast_target = (
                latest['week_start'] + pd.Timedelta(weeks=weeks_ahead)
            ).strftime('%Y-%m-%d')
        else:
            X_fc = forecast_from = forecast_target = None

        for m_name, m_factory, _ in MODEL_DEFS:
            shared = dict(
                n_train=n_train, n_test=len(test_df),
                cutoff=str(train_df['week_start'].iloc[-1].date()),
                train_weeks=train_df['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                test_weeks=test_df['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                forecast_from=forecast_from, forecast_target=forecast_target,
            )

            # --- Delta model ---
            y_train_d = train_df[delta_col].values
            y_test_d  = test_df[delta_col].values
            model_d   = m_factory()
            model_d.fit(X_train, y_train_d)
            tp_d = model_d.predict(X_train)
            ep_d = model_d.predict(X_test) if len(test_df) > 0 else np.array([])
            coef_d = (model_d.feature_importances_ if hasattr(model_d, 'feature_importances_')
                      else model_d.coef_ if hasattr(model_d, 'coef_')
                      else np.zeros(len(FEATURE_COLS)))
            results[(h_label, m_name)] = {
                **shared,
                'train_r2':      float(r2_score(y_train_d, tp_d)),
                'test_r2':       float(r2_score(y_test_d, ep_d)) if len(test_df) > 0 else float('nan'),
                'rmse':          float(np.sqrt(mean_squared_error(y_test_d, ep_d))) if len(test_df) > 0 else float('nan'),
                'mae':           float(np.mean(np.abs(y_test_d - ep_d))) if len(test_df) > 0 else float('nan'),
                'coef':          coef_d.tolist(),
                'train_preds':   tp_d.tolist(),
                'test_preds':    ep_d.tolist(),
                'train_actuals': y_train_d.tolist(),
                'test_actuals':  y_test_d.tolist(),
                'forecast_delta': float(model_d.predict(X_fc)[0]) if X_fc is not None else None,
            }

            # --- Absolute model (predicts fred_score_Nw_future directly) ---
            y_train_a = train_df[h_col].values
            y_test_a  = test_df[h_col].values
            model_a   = m_factory()
            model_a.fit(X_train, y_train_a)
            tp_a = model_a.predict(X_train)
            ep_a = model_a.predict(X_test) if len(test_df) > 0 else np.array([])
            coef_a = (model_a.feature_importances_ if hasattr(model_a, 'feature_importances_')
                      else model_a.coef_ if hasattr(model_a, 'coef_')
                      else np.zeros(len(FEATURE_COLS)))
            results[(h_label, m_name, 'abs')] = {
                **shared,
                'train_r2':      float(r2_score(y_train_a, tp_a)),
                'test_r2':       float(r2_score(y_test_a, ep_a)) if len(test_df) > 0 else float('nan'),
                'rmse':          float(np.sqrt(mean_squared_error(y_test_a, ep_a))) if len(test_df) > 0 else float('nan'),
                'mae':           float(np.mean(np.abs(y_test_a - ep_a))) if len(test_df) > 0 else float('nan'),
                'coef':          coef_a.tolist(),
                'train_preds':   tp_a.tolist(),
                'test_preds':    ep_a.tolist(),
                'train_actuals': y_train_a.tolist(),
                'test_actuals':  y_test_a.tolist(),
                'forecast_abs':  float(model_a.predict(X_fc)[0]) if X_fc is not None else None,
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

    model_results  = train_all_models()
    offset         = pd.Timedelta(weeks=1)
    last_fred_date = weekly.dropna(subset=['fred_score'])['week_start'].max()

    fig = go.Figure()

    # Actual FRED line — last 12 weeks, placed at target date (week_start + 1w)
    last_actual_x = last_actual_y = None
    for m_name, _, _ in MODEL_DEFS:
        key = ('1w', m_name, 'abs')
        if key not in model_results:
            continue
        res         = model_results[key]
        all_weeks   = res['train_weeks'] + res['test_weeks']
        all_actuals = res['train_actuals'] + res['test_actuals']
        pairs = [(pd.Timestamp(w) + offset, a) for w, a in zip(all_weeks, all_actuals)
                 if pd.Timestamp(w) + offset <= last_fred_date][-12:]
        if pairs:
            tx, ty = zip(*pairs)
            last_actual_x, last_actual_y = tx[-1], ty[-1]
            fig.add_trace(go.Scatter(
                x=list(tx), y=list(ty),
                mode='lines+markers', name='Actual FRED',
                line=dict(color='#1e293b', width=2.5),
                marker=dict(size=5),
                hovertemplate='%{x|%Y-%m-%d}<br>Actual FRED: %{y:.4f}<extra></extra>',
            ))
        break  # actuals identical across models

    # Dotted connector + CI cone + forecast star — Linear only on front page
    for m_name, _, color in [m for m in MODEL_DEFS if m[0] == 'LinearRegression']:
        key = ('1w', m_name, 'abs')
        if key not in model_results:
            continue
        res       = model_results[key]
        all_weeks = res['train_weeks'] + res['test_weeks']
        all_preds = res['train_preds'] + res['test_preds']
        future = [(pd.Timestamp(w) + offset, p) for w, p in zip(all_weeks, all_preds)
                  if pd.Timestamp(w) + offset > last_fred_date]
        if not future or last_actual_x is None:
            continue
        ft, fp = future[-1]
        short = m_name.replace('LinearRegression', 'Linear').replace('RandomForest', 'RF')

        # 95% CI from test set residuals
        ci_half = None
        if res['test_preds'] and res['test_actuals']:
            residuals = np.array(res['test_actuals']) - np.array(res['test_preds'])
            ci_half = 1.96 * float(np.std(residuals))

        # Shaded cone: narrows to actual point on left, opens to CI range on right
        if ci_half is not None:
            fig.add_trace(go.Scatter(
                x=[last_actual_x, ft, ft, last_actual_x],
                y=[last_actual_y, fp + ci_half, fp - ci_half, last_actual_y],
                fill='toself',
                fillcolor='rgba(245,158,11,0.12)',
                line=dict(width=0),
                showlegend=True,
                name=f'95% CI (±{ci_half:.4f})',
                hoverinfo='skip',
            ))

        # Dotted centre line
        fig.add_trace(go.Scatter(
            x=[last_actual_x, ft], y=[last_actual_y, fp],
            mode='lines', showlegend=False,
            line=dict(color=color, width=1.5, dash='dot'),
            hoverinfo='skip',
        ))
        # Forecast star
        fig.add_trace(go.Scatter(
            x=[ft], y=[fp],
            mode='markers', name=f'{short} forecast ({ft.strftime("%Y-%m-%d")})',
            marker=dict(size=14, symbol='star', color='#f59e0b',
                        line=dict(color=color, width=2)),
            hovertemplate=f'<b>{short} FORECAST</b><br>%{{x|%Y-%m-%d}}<br>Predicted FRED: %{{y:.4f}}<br>95% CI: [{fp - ci_half:.4f}, {fp + ci_half:.4f}]<extra></extra>' if ci_half else f'<b>{short} FORECAST</b><br>%{{x|%Y-%m-%d}}<br>Predicted FRED: %{{y:.4f}}<extra></extra>',
        ))

    fig.add_hline(y=0,    line_color='gray',    line_dash='dash', line_width=1)
    fig.add_hline(y=-0.5, line_color='#22c55e', line_dash='dot',  line_width=1,
                  annotation_text='calm',   annotation_position='bottom right')
    fig.add_hline(y=0.5,  line_color='#ef4444', line_dash='dot',  line_width=1,
                  annotation_text='stress', annotation_position='top right')
    fig.update_layout(title='Recent FRED History + Next Week Forecast (1w Model)',
                      xaxis_title='Week', yaxis_title='STLFSI4', height=340,
                      legend=dict(orientation='h', y=1.08, font_size=11),
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
    st.subheader('Model Prediction: FRED in 1 Week')
    if weekly.empty:
        st.info('No data yet. Run pipeline.py first.')
        return

    model_results  = train_all_models()
    offset         = pd.Timedelta(weeks=1)
    last_fred_date = weekly.dropna(subset=['fred_score'])['week_start'].max()

    # Use LinearRegression 1w abs model — find the most recent forecast point
    key = ('1w', 'LinearRegression', 'abs')
    if key not in model_results:
        st.info('1w model not trained yet.')
        return
    res       = model_results[key]
    all_weeks = res['train_weeks'] + res['test_weeks']
    all_preds = res['train_preds'] + res['test_preds']
    all_acts  = res['train_actuals'] + res['test_actuals']

    future = [(pd.Timestamp(w) + offset, p)
              for w, p in zip(all_weeks, all_preds)
              if pd.Timestamp(w) + offset > last_fred_date]
    if not future:
        st.info('No upcoming 1w prediction available.')
        return
    forecast_date, pred = future[-1]

    # Avg abs error from test set
    if res['test_preds'] and res['test_actuals']:
        avg_err = float(np.mean(np.abs(np.array(res['test_preds']) - np.array(res['test_actuals']))))
    else:
        avg_err = None

    cur = float(weekly.dropna(subset=['fred_score'])['fred_score'].iloc[-1])
    c1, c2, c3 = st.columns(3)
    c1.metric('Predicted FRED (1w)', f'{pred:.4f}')
    c2.metric('For week of', forecast_date.strftime('%Y-%m-%d'))
    if avg_err is not None:
        c3.metric('Avg abs error (test)', f'±{avg_err:.4f}')

    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta', value=pred,
        delta={'reference': cur, 'valueformat': '.4f'},
        title={'text': f'Predicted FRED (1w)<br><sub>current: {cur:.4f}</sub>'},
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

def _metrics_table(model_results, h_label, key_suffix=None):
    def _key(m):
        return (h_label, m, key_suffix) if key_suffix else (h_label, m)
    rows = []
    best_r2 = max(
        (model_results[_key(m)]['test_r2']
         for m, _, _ in MODEL_DEFS if _key(m) in model_results
         and not np.isnan(model_results[_key(m)]['test_r2'])),
        default=float('-inf'),
    )
    for m_name, _, color in MODEL_DEFS:
        key = _key(m_name)
        if key not in model_results:
            continue
        res = model_results[key]
        r2 = res['test_r2']
        is_best = not np.isnan(r2) and r2 == best_r2
        rows.append({
            'Model':    ('* ' if is_best else '  ') + m_name.replace('LinearRegression', 'Linear').replace('RandomForest', 'RF'),
            'Train R2': f'{res["train_r2"]:.3f}',
            'Test R2':  f'{r2:.3f}' if not np.isnan(r2) else 'N/A',
            'RMSE':     f'{res["rmse"]:.4f}' if not np.isnan(res["rmse"]) else 'N/A',
            'MAE':      f'{res["mae"]:.4f}' if not np.isnan(res["mae"]) else 'N/A',
            'n train':  res['n_train'],
            'n test':   res['n_test'],
        })
    if rows:
        st.caption('* = best test R² for this horizon')
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _feature_importance_chart(model_results, h_label, key_suffix=None):
    def _key(m):
        return (h_label, m, key_suffix) if key_suffix else (h_label, m)
    sel_model = st.radio(
        'Show feature importance for:',
        [m for m, _, _ in MODEL_DEFS if _key(m) in model_results],
        horizontal=True,
        key=f'imp_radio_{h_label}{"_abs" if key_suffix else ""}',
    )
    key = _key(sel_model)
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

    # Shift all dates to target date (feature week + N weeks ahead)
    weeks_ahead   = int(h_label.replace('w', ''))
    offset        = pd.Timedelta(weeks=weeks_ahead)
    cutoff_str    = str((cutoff + offset).date())
    x0_str        = str((labeled['week_start'].iloc[0] + offset).date())

    # Cap actual line at last concretely available FRED reading
    last_fred_date = weekly.dropna(subset=['fred_score'])['week_start'].max()
    actual_labeled = labeled[labeled['week_start'] + offset <= last_fred_date]

    # View toggle
    view_mode = st.radio(
        'View:',
        ['ΔFRED (change)', 'Absolute FRED'],
        horizontal=True,
        key=f'view_{h_label}',
    )
    show_delta = (view_mode == 'ΔFRED (change)')

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

    if show_delta:
        actual_y = actual_labeled[delta_col].values
        actual_name = 'Actual ΔFRED'
        actual_hover = '%{x|%Y-%m-%d}<br>Actual Δ: %{y:.4f}<extra></extra>'
        yaxis_title = 'ΔFRED (change from current)'
        chart_title = f'Predicted vs Actual ΔFRED — {h_name}'
    else:
        actual_y = actual_labeled[h_col].values
        actual_name = 'Actual FRED'
        actual_hover = '%{x|%Y-%m-%d}<br>Actual FRED: %{y:.4f}<extra></extra>'
        yaxis_title = 'FRED Stress Index'
        chart_title = f'Predicted vs Actual FRED — {h_name}'

    fig.add_trace(go.Scatter(
        x=actual_labeled['week_start'] + offset, y=actual_y,
        mode='lines+markers', name=actual_name,
        line=dict(color='#1e293b', width=2.5),
        marker=dict(size=5),
        hovertemplate=actual_hover,
    ))

    if show_delta:
        fig.add_hline(y=0, line_color='gray', line_dash='dash', line_width=1)

    for m_name, _, color in MODEL_DEFS:
        # Use delta or absolute model results depending on view
        key = (h_label, m_name) if show_delta else (h_label, m_name, 'abs')
        if key not in model_results:
            continue
        res = model_results[key]
        short = m_name.replace('LinearRegression', 'Linear').replace('RandomForest', 'RF')

        train_y = res['train_preds']
        test_y  = res['test_preds']
        if show_delta:
            train_hover = f'{short} train Δ: %{{y:.4f}}<extra></extra>'
            test_hover  = f'%{{x|%Y-%m-%d}}<br>{short} Δ: %{{y:.4f}}<extra></extra>'
        else:
            train_hover = f'{short} train FRED: %{{y:.4f}}<extra></extra>'
            test_hover  = f'%{{x|%Y-%m-%d}}<br>{short} FRED: %{{y:.4f}}<extra></extra>'

        # Shift prediction weeks to target date
        train_x = [(pd.Timestamp(w) + offset).strftime('%Y-%m-%d') for w in res['train_weeks']]
        test_x  = [(pd.Timestamp(w) + offset).strftime('%Y-%m-%d') for w in res['test_weeks']]

        # Train predictions (faded)
        fig.add_trace(go.Scatter(
            x=train_x, y=train_y,
            mode='lines', name=f'{short} (train)',
            line=dict(color=color, width=1.5, dash='dot'),
            opacity=0.6, showlegend=False,
            hovertemplate=train_hover,
        ))
        # Test predictions (bold dashed)
        if res['test_preds']:
            r2_str = f'{res["test_r2"]:.3f}' if not np.isnan(res['test_r2']) else 'N/A'
            fig.add_trace(go.Scatter(
                x=test_x, y=test_y,
                mode='lines+markers', name=f'{short}  (test R²={r2_str})',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=7, symbol='diamond'),
                hovertemplate=test_hover,
            ))

        # Live forecast star — not shown for 1w horizon
        if h_label != '1w':
            if show_delta and res.get('forecast_delta') is not None:
                star_y   = res['forecast_delta']
                fc_hover = (
                    f'<b>{short} FORECAST</b><br>'
                    f'Target date: {res["forecast_target"]}<br>'
                    f'Based on week of: {res["forecast_from"]}<br>'
                    f'Predicted ΔFRED: %{{y:.4f}}<extra></extra>'
                )
            elif not show_delta and res.get('forecast_abs') is not None:
                star_y   = res['forecast_abs']
                fc_hover = (
                    f'<b>{short} FORECAST</b><br>'
                    f'Target date: {res["forecast_target"]}<br>'
                    f'Based on week of: {res["forecast_from"]}<br>'
                    f'Predicted FRED: %{{y:.4f}}<extra></extra>'
                )
            else:
                star_y = None
            if star_y is not None:
                fig.add_trace(go.Scatter(
                    x=[res['forecast_target']],
                    y=[star_y],
                    mode='markers',
                    name=f'{short} forecast ({res["forecast_target"]})',
                    marker=dict(
                        size=16, symbol='star',
                        color='#f59e0b',
                        line=dict(color=color, width=2),
                    ),
                    hovertemplate=fc_hover,
                ))

    fig.update_layout(
        title=chart_title,
        xaxis_title='Week', yaxis_title=yaxis_title,
        height=480,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.08, font_size=11),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Metrics + feature importance side by side below the chart ---
    key_suffix = None if show_delta else 'abs'
    col_metrics, col_imp = st.columns(2)
    with col_metrics:
        st.markdown('**Test Set Metrics**')
        _metrics_table(model_results, h_label, key_suffix=key_suffix)
    with col_imp:
        st.markdown('**Feature Weights**')
        _feature_importance_chart(model_results, h_label, key_suffix=key_suffix)


def panel_recent_articles(articles, weekly):
    """Show top articles from the most recent classified week."""
    st.subheader('Recent Articles Driving Sentiment')

    if articles.empty:
        st.info('No classified articles found. Run pipeline.py to populate.')
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
# Tab 2 main
# ---------------------------------------------------------------------------

def tab_model_analysis(weekly, model_results):
    st.subheader('5-Horizon Model Comparison')
    st.caption(
        'Each section: actual FRED (black) vs all 3 model predictions. '
        'Faded dotted lines = training period. Dashed bold lines = test period.'
    )

    for h_label, h_col, h_name in HORIZONS:
        with st.expander(f'{h_name}  ({h_label})', expanded=True):
            _horizon_scatter(weekly, model_results, h_label, h_col, h_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title('StressPress')
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
        st.divider()
        if st.button('Refresh data'):
            st.cache_data.clear()
            st.rerun()

    tab1, tab2 = st.tabs(['Live Monitor', 'Model Analysis'])

    with tab1:
        panel_fred_trend(weekly)
        panel_divergence(predictions, weekly)
        st.divider()
        left, right = st.columns(2)
        with left:
            panel_theme_scores(weekly)
        with right:
            panel_prediction(predictions, weekly)
        st.divider()
        articles = load_articles()
        panel_recent_articles(articles, weekly)

    with tab2:
        if weekly.empty:
            st.info('Run pipeline.py and train_model.py first to populate data.')
        else:
            model_results = train_all_models()
            tab_model_analysis(weekly, model_results)

    st.caption(
        f'Refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M")} | '
        'Data: NYT Article Search API + FRED STLFSI4 | '
        'Models: LinearRegression, LassoCV, RandomForest'
    )


if __name__ == '__main__':
    main()
