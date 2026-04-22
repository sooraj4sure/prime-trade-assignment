"""
Primetrade.ai — Trader Performance vs Market Sentiment
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Primetrade.ai | Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0e1117; }
  
  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1c2333, #1e2a3a);
    border: 1px solid #2d3a4f;
    border-radius: 12px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label {
    color: #8899aa !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f0fe !important;
    font-size: 26px !important;
    font-weight: 700 !important;
  }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 13px !important;
  }

  /* Section headers */
  .section-header {
    background: linear-gradient(90deg, #1a3a5c, #0e1117);
    border-left: 4px solid #4fa3e0;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 14px 0;
  }
  .section-header h3 { color: #e8f0fe; margin: 0; font-size: 18px; }

  /* Insight cards */
  .insight-card {
    background: linear-gradient(135deg, #1c2a1c, #162616);
    border: 1px solid #2e4a2e;
    border-left: 4px solid #4caf50;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
  }
  .insight-card.warn {
    background: linear-gradient(135deg, #2a1c1c, #261616);
    border-color: #4a2e2e;
    border-left-color: #e05c5c;
  }
  .insight-card.blue {
    background: linear-gradient(135deg, #1c1c2a, #161626);
    border-color: #2e2e4a;
    border-left-color: #5c7ce0;
  }
  .insight-title { color: #aaffaa; font-weight: 700; font-size: 14px; margin-bottom: 6px; }
  .insight-card.warn .insight-title { color: #ffaaaa; }
  .insight-card.blue .insight-title { color: #aabbff; }
  .insight-body { color: #c0c8d0; font-size: 13px; line-height: 1.5; }

  /* Sidebar */
  [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
  [data-testid="stSidebar"] .stMarkdown { color: #9ca3af; }

  /* Divider */
  hr { border-color: #1f2937; }

  /* Tabs */
  button[data-baseweb="tab"] { color: #9ca3af !important; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #4fa3e0 !important; border-bottom-color: #4fa3e0 !important; }

  /* Footer */
  .footer { text-align: center; color: #4b5563; font-size: 12px; padding: 30px 0 10px; }
</style>
""", unsafe_allow_html=True)

# ── Colour constants ───────────────────────────────────────────────────────────
SENT_COLORS = {
    'Extreme Fear': '#d62728',
    'Fear':         '#ff7f0e',
    'Neutral':      '#7f7f7f',
    'Greed':        '#2ca02c',
    'Extreme Greed':'#1f77b4',
}
BIN_COLORS = {'Fear': '#e05c5c', 'Neutral': '#9e9e9e', 'Greed': '#4caf7d'}
SENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
BIN_ORDER  = ['Fear', 'Neutral', 'Greed']
PLOTLY_TEMPLATE = 'plotly_dark'

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    merged   = pd.read_csv('outputs/merged_daily.csv', parse_dates=['date'])
    profiles = pd.read_csv('outputs/trader_profiles.csv')
    perf     = pd.read_csv('outputs/performance_by_sentiment.csv')
    behavior = pd.read_csv('outputs/behavior_by_sentiment.csv')
    return merged, profiles, perf, behavior

try:
    merged, profiles, perf, behavior = load_data()
except FileNotFoundError:
    st.error("⚠️ Output CSV files not found. Please run `analysis.py` first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎛️ Dashboard Controls")
    st.markdown("---")

    # Date range filter
    min_date = merged['date'].min().date()
    max_date = merged['date'].max().date()
    date_range = st.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.markdown("---")
    # Sentiment filter
    selected_sentiments = st.multiselect(
        "🌡️ Sentiment Filter",
        options=SENT_ORDER,
        default=SENT_ORDER
    )

    st.markdown("---")
    # Segment filters
    lev_segs  = st.multiselect("⚡ Leverage Segment", ['Low Leverage','High Leverage'],
                                default=['Low Leverage','High Leverage'])
    freq_segs = st.multiselect("🔄 Frequency Segment", ['Infrequent','Frequent'],
                                default=['Infrequent','Frequent'])
    con_segs  = st.multiselect("🏆 Consistency Segment",
                                ['Consistent Winner','Inconsistent','Consistent Loser'],
                                default=['Consistent Winner','Inconsistent','Consistent Loser'])

    st.markdown("---")
    # PnL clip
    pnl_clip = st.slider("📐 PnL Clip (±USD)", 100, 5000, 500, step=100)

    st.markdown("---")
    st.markdown("""
    <div style='color:#4b5563;font-size:11px;'>
    📊 <b style='color:#6b7280'>Primetrade.ai</b><br>
    Data Science Intern — Round 0<br>
    211,224 trades · 32 traders · 479 days
    </div>""", unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────────
if len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d0, d1 = merged['date'].min(), merged['date'].max()

filt = (
    merged['date'].between(d0, d1) &
    merged['classification'].isin(selected_sentiments)
)
df = merged[filt].copy()

prof_filt = (
    profiles['leverage_segment'].isin(lev_segs) &
    profiles['freq_segment'].isin(freq_segs) &
    profiles['consistency_segment'].isin(con_segs)
)
prof = profiles[prof_filt].copy()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 9])
with col_title:
    st.markdown("""
    <h1 style='color:#e8f0fe; margin-bottom:2px; font-size:32px;'>
    📊 Primetrade.ai — Trader × Sentiment Dashboard
    </h1>
    <p style='color:#6b7280; margin:0; font-size:14px;'>
    Hyperliquid historical trades vs Bitcoin Fear/Greed Index · Interactive Analysis
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# KPI METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5, k6 = st.columns(6)

total_traders = df['Account'].nunique()
total_days    = df['date'].nunique()
avg_pnl_all   = df['total_pnl'].mean()
avg_wr_all    = df['win_rate'].mean() * 100
fear_pnl      = df[df['sentiment_binary']=='Fear']['total_pnl'].mean()
greed_pnl     = df[df['sentiment_binary']=='Greed']['total_pnl'].mean()
delta_fg      = greed_pnl - fear_pnl

k1.metric("🧑‍💼 Traders",     f"{total_traders}")
k2.metric("📅 Trading Days", f"{total_days:,}")
k3.metric("💰 Avg Daily PnL", f"${avg_pnl_all:,.0f}")
k4.metric("🎯 Avg Win Rate",  f"{avg_wr_all:.1f}%")
k5.metric("😨 Fear Avg PnL",  f"${fear_pnl:,.0f}")
k6.metric("😁 Greed Avg PnL", f"${greed_pnl:,.0f}",
          delta=f"{delta_fg:+,.0f} vs Fear",
          delta_color="inverse")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "🔄 Behavior",
    "👥 Segments",
    "📆 Time Series",
    "💡 Insights & Strategy"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header"><h3>📈 Performance by Market Sentiment</h3></div>', unsafe_allow_html=True)

    # Recompute perf from filtered data
    perf_live = df.groupby('classification').agg(
        avg_pnl      = ('total_pnl', 'mean'),
        median_pnl   = ('total_pnl', 'median'),
        avg_win_rate = ('win_rate', 'mean'),
        n_obs        = ('total_pnl', 'count'),
    ).reindex([s for s in SENT_ORDER if s in df['classification'].unique()]).reset_index()

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            perf_live, x='classification', y='avg_pnl',
            color='classification',
            color_discrete_map=SENT_COLORS,
            category_orders={'classification': SENT_ORDER},
            title='Average Daily PnL per Trader by Sentiment',
            labels={'avg_pnl': 'Avg PnL (USD)', 'classification': 'Sentiment'},
            template=PLOTLY_TEMPLATE, text_auto='.0f'
        )
        fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
        fig.update_layout(showlegend=False, height=380,
                          title_font_size=14, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(
            perf_live, x='classification', y='avg_win_rate',
            color='classification',
            color_discrete_map=SENT_COLORS,
            category_orders={'classification': SENT_ORDER},
            title='Average Win Rate (%) by Sentiment',
            labels={'avg_win_rate': 'Win Rate', 'classification': 'Sentiment'},
            template=PLOTLY_TEMPLATE
        )
        fig2.update_layout(showlegend=False, height=380,
                           title_font_size=14, paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)')
        fig2.update_traces(texttemplate='%{y:.1%}', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    # Box plot
    st.markdown('<div class="section-header"><h3>📦 PnL Distribution (Fear vs Neutral vs Greed)</h3></div>', unsafe_allow_html=True)

    fig3 = go.Figure()
    for s, col in zip(BIN_ORDER, ['#e05c5c','#9e9e9e','#4caf7d']):
        sub = df[df['sentiment_binary'] == s]['total_pnl'].clip(-pnl_clip, pnl_clip)
        r = int(col[1:3], 16)
        g = int(col[3:5], 16)
        b = int(col[5:7], 16)
        fig3.add_trace(go.Box(
            y=sub, name=s, marker_color=col,
            boxmean='sd', line_width=1.5,
            fillcolor=f'rgba({r},{g},{b},0.3)'
        ))
    fig3.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
    fig3.update_layout(
        template=PLOTLY_TEMPLATE, height=420,
        title=f'PnL Distribution by Binary Sentiment (clipped ±${pnl_clip:,})',
        yaxis_title='Daily PnL (USD)', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', showlegend=True
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Raw table
    with st.expander("🔍 View Raw Performance Table"):
        st.dataframe(
            perf_live.style.format({
                'avg_pnl': '${:,.2f}', 'median_pnl': '${:,.2f}',
                'avg_win_rate': '{:.1%}', 'n_obs': '{:,}'
            }),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BEHAVIOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header"><h3>🔄 How Traders Behave Under Different Sentiment</h3></div>', unsafe_allow_html=True)

    beh_live = df.groupby('classification').agg(
        avg_trades     = ('num_trades', 'mean'),
        avg_size_usd   = ('avg_size_usd', 'mean'),
        avg_leverage   = ('leverage_proxy', 'mean'),
        avg_ls_ratio   = ('long_short_r', 'mean'),
    ).reindex([s for s in SENT_ORDER if s in df['classification'].unique()]).reset_index()

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.bar(beh_live, x='classification', y='avg_trades',
                     color='classification', color_discrete_map=SENT_COLORS,
                     category_orders={'classification': SENT_ORDER},
                     title='Avg Trades per Day by Sentiment',
                     labels={'avg_trades':'Avg Trades/Day','classification':'Sentiment'},
                     template=PLOTLY_TEMPLATE, text_auto='.1f')
        fig.update_layout(showlegend=False, height=360,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.bar(beh_live, x='classification', y='avg_size_usd',
                     color='classification', color_discrete_map=SENT_COLORS,
                     category_orders={'classification': SENT_ORDER},
                     title='Avg Trade Size (USD) by Sentiment',
                     labels={'avg_size_usd':'Avg Size USD','classification':'Sentiment'},
                     template=PLOTLY_TEMPLATE, text_auto='.0f')
        fig.update_layout(showlegend=False, height=360,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig = px.bar(beh_live, x='classification', y='avg_leverage',
                     color='classification', color_discrete_map=SENT_COLORS,
                     category_orders={'classification': SENT_ORDER},
                     title='Avg Leverage Proxy by Sentiment',
                     labels={'avg_leverage':'Leverage Proxy','classification':'Sentiment'},
                     template=PLOTLY_TEMPLATE, text_auto='.2f')
        fig.update_layout(showlegend=False, height=360,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        # Long/Short ratio with reference line at 1
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=beh_live['classification'],
            y=beh_live['avg_ls_ratio'].clip(upper=5),
            marker_color=[SENT_COLORS.get(s,'grey') for s in beh_live['classification']],
            text=[f'{v:.2f}' for v in beh_live['avg_ls_ratio'].clip(upper=5)],
            textposition='outside',
            name='L/S Ratio'
        ))
        fig.add_hline(y=1, line_dash='dash', line_color='yellow',
                      annotation_text='Balanced (1.0)', annotation_position='top right')
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=360,
            title='Long/Short Ratio by Sentiment (capped at 5)',
            yaxis_title='L/S Ratio', xaxis_title='Sentiment',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown('<div class="section-header"><h3>🕸️ Behavior Profile Radar</h3></div>', unsafe_allow_html=True)

    radar_metrics = ['avg_trades','avg_size_usd','avg_leverage']
    norm = beh_live.copy()
    for col in radar_metrics:
        col_max = norm[col].max()
        norm[col] = norm[col] / col_max if col_max > 0 else 0

    fig_radar = go.Figure()
    theta_labels = ['Trade Frequency', 'Position Size', 'Leverage']
    for _, row in norm.iterrows():
        values = [row['avg_trades'], row['avg_size_usd'], row['avg_leverage']]
        values_closed = values + [values[0]]
        labels_closed = theta_labels + [theta_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed, theta=labels_closed,
            fill='toself', name=row['classification'],
            line_color=SENT_COLORS.get(row['classification'], 'grey'),
            opacity=0.65
        ))
    fig_radar.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
                   radialaxis=dict(visible=True, range=[0,1], color='#6b7280')),
        template=PLOTLY_TEMPLATE, height=450,
        title='Normalized Behavior Profiles by Sentiment',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header"><h3>👥 Trader Segmentation Explorer</h3></div>', unsafe_allow_html=True)

    seg_choice = st.radio("Segment by:", ['Leverage','Frequency','Consistency'], horizontal=True)
    seg_map = {
        'Leverage':    ('leverage_segment',    ['Low Leverage','High Leverage'],     ['#4caf7d','#e05c5c']),
        'Frequency':   ('freq_segment',        ['Infrequent','Frequent'],            ['#64b5f6','#ff8a65']),
        'Consistency': ('consistency_segment', ['Consistent Winner','Inconsistent','Consistent Loser'],
                                               ['#2ca02c','#9e9e9e','#d62728']),
    }
    seg_col, seg_order, seg_colors = seg_map[seg_choice]

    if seg_col not in prof.columns:
        st.warning("No data for this segment after filtering.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            seg_pnl = prof.groupby(seg_col)['avg_pnl'].mean().reindex(seg_order).reset_index()
            fig = px.bar(seg_pnl, x=seg_col, y='avg_pnl',
                         color=seg_col, color_discrete_sequence=seg_colors,
                         title='Avg Daily PnL by Segment',
                         labels={'avg_pnl':'Avg PnL (USD)', seg_col:'Segment'},
                         template=PLOTLY_TEMPLATE, text_auto='.0f')
            fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
            fig.update_layout(showlegend=False, height=380,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            seg_wr = prof.groupby(seg_col)['avg_win_rate'].mean().reindex(seg_order).reset_index()
            fig2 = px.bar(seg_wr, x=seg_col, y='avg_win_rate',
                          color=seg_col, color_discrete_sequence=seg_colors,
                          title='Avg Win Rate by Segment',
                          labels={'avg_win_rate':'Win Rate', seg_col:'Segment'},
                          template=PLOTLY_TEMPLATE)
            fig2.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            fig2.update_layout(showlegend=False, height=380,
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        # Sentiment × Segment heatmap
        st.markdown('<div class="section-header"><h3>🗺️ Sentiment × Segment Heatmap</h3></div>', unsafe_allow_html=True)

        merged_seg = df.merge(profiles[[seg_col,'Account']], on='Account', how='inner')
        pivot = merged_seg.groupby(['sentiment_binary', seg_col])['total_pnl'].mean().unstack()
        pivot = pivot.reindex([b for b in BIN_ORDER if b in pivot.index])
        pivot = pivot.reindex(columns=[s for s in seg_order if s in pivot.columns])

        fig_heat = px.imshow(
            pivot.values,
            x=list(pivot.columns), y=list(pivot.index),
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            text_auto='.0f',
            title='Avg PnL: Sentiment (rows) × Segment (columns)',
            template=PLOTLY_TEMPLATE,
            aspect='auto'
        )
        fig_heat.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)

        # Scatter: Win Rate vs PnL
        st.markdown('<div class="section-header"><h3>🔵 Trader Profile Scatter</h3></div>', unsafe_allow_html=True)

        prof_plot = prof.dropna(subset=[seg_col, 'avg_win_rate', 'avg_pnl'])
        prof_plot['avg_pnl_clipped'] = prof_plot['avg_pnl'].clip(-pnl_clip, pnl_clip)

        fig_sc = px.scatter(
            prof_plot, x='avg_win_rate', y='avg_pnl_clipped',
            color=seg_col, size='total_trades', hover_name='Account',
            color_discrete_sequence=seg_colors,
            category_orders={seg_col: seg_order},
            title='Individual Traders: Win Rate vs Avg Daily PnL',
            labels={'avg_win_rate':'Win Rate', 'avg_pnl_clipped':f'Avg PnL (clipped ±{pnl_clip})'},
            template=PLOTLY_TEMPLATE
        )
        fig_sc.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
        fig_sc.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header"><h3>📆 Daily PnL Trend vs Fear & Greed Index</h3></div>', unsafe_allow_html=True)

    daily_agg = df.groupby('date').agg(
        avg_pnl    = ('total_pnl',  'mean'),
        total_pnl  = ('total_pnl',  'sum'),
        fg_value   = ('value',       'first'),
        sentiment  = ('sentiment_binary', 'first'),
        trades     = ('num_trades',  'sum'),
        win_rate   = ('win_rate',    'mean'),
    ).reset_index()

    metric_choice = st.selectbox("Select PnL metric:", ['avg_pnl (per trader)', 'total_pnl (aggregate)'])
    y_col = 'avg_pnl' if 'avg' in metric_choice else 'total_pnl'
    y_label = 'Avg PnL / Trader (USD)' if y_col == 'avg_pnl' else 'Total PnL All Traders (USD)'

    fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.65, 0.35],
                           subplot_titles=['PnL with Sentiment Background', 'Fear & Greed Index Value'])

    # Colored background bands
    for _, row in daily_agg.iterrows():
        col = BIN_COLORS.get(row['sentiment'], '#444')
        fig_ts.add_vrect(
            x0=row['date'] - pd.Timedelta(hours=12),
            x1=row['date'] + pd.Timedelta(hours=12),
            fillcolor=col, opacity=0.08, layer='below', line_width=0, row=1, col=1
        )

    fig_ts.add_trace(go.Scatter(
        x=daily_agg['date'], y=daily_agg[y_col],
        mode='lines', name=y_label,
        line=dict(color='#4fa3e0', width=1.8),
        fill='tozeroy', fillcolor='rgba(79,163,224,0.08)'
    ), row=1, col=1)

    fig_ts.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3, row=1, col=1)

    # F&G index
    fig_ts.add_trace(go.Scatter(
        x=daily_agg['date'], y=daily_agg['fg_value'],
        mode='lines', name='Fear & Greed Value',
        line=dict(color='#f0a500', width=1.5),
    ), row=2, col=1)

    # Add coloured fill zones on index
    fig_ts.add_hrect(y0=0,  y1=25,  fillcolor='#d62728', opacity=0.12, row=2, col=1, line_width=0)
    fig_ts.add_hrect(y0=25, y1=45,  fillcolor='#ff7f0e', opacity=0.10, row=2, col=1, line_width=0)
    fig_ts.add_hrect(y0=45, y1=55,  fillcolor='#7f7f7f', opacity=0.10, row=2, col=1, line_width=0)
    fig_ts.add_hrect(y0=55, y1=75,  fillcolor='#2ca02c', opacity=0.10, row=2, col=1, line_width=0)
    fig_ts.add_hrect(y0=75, y1=100, fillcolor='#1f77b4', opacity=0.12, row=2, col=1, line_width=0)

    fig_ts.update_layout(
        template=PLOTLY_TEMPLATE, height=600,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True, hovermode='x unified'
    )
    fig_ts.update_yaxes(title_text=y_label, row=1, col=1)
    fig_ts.update_yaxes(title_text='F&G Index', row=2, col=1)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Win rate & trade count trend
    st.markdown('<div class="section-header"><h3>🎯 Daily Win Rate & Trade Volume</h3></div>', unsafe_allow_html=True)

    fig_wr = make_subplots(specs=[[{"secondary_y": True}]])
    fig_wr.add_trace(go.Scatter(
        x=daily_agg['date'], y=daily_agg['win_rate'],
        mode='lines', name='Win Rate', line=dict(color='#4caf7d', width=1.5)
    ), secondary_y=False)
    fig_wr.add_trace(go.Bar(
        x=daily_agg['date'], y=daily_agg['trades'],
        name='Total Trades', marker_color='rgba(100,181,246,0.35)'
    ), secondary_y=True)
    fig_wr.update_yaxes(title_text='Win Rate', secondary_y=False)
    fig_wr.update_yaxes(title_text='Total Trades', secondary_y=True)
    fig_wr.update_layout(template=PLOTLY_TEMPLATE, height=380,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          hovermode='x unified')
    st.plotly_chart(fig_wr, use_container_width=True)

    with st.expander("📋 View Daily Summary Table"):
        st.dataframe(
            daily_agg.sort_values('date', ascending=False).style.format({
                'avg_pnl': '${:,.2f}', 'total_pnl': '${:,.0f}',
                'win_rate': '{:.1%}', 'fg_value': '{:.0f}'
            }),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — INSIGHTS & STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header"><h3>💡 Key Insights from the Analysis</h3></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
      <div class="insight-title">📌 Insight 1 — Fear Days Yield Higher PnL (Counterintuitive)</div>
      <div class="insight-body">
        Traders earn an average of <b>$5,328 on Fear days</b> vs $3,318 on Greed days.
        Fear creates sharp price dislocations and volatility spikes — experienced traders exploit
        these swings for larger, faster profits. The median PnL also confirms this isn't just outlier-driven.
      </div>
    </div>

    <div class="insight-card warn">
      <div class="insight-title">📌 Insight 2 — Over-Trading During Fear is Dangerous</div>
      <div class="insight-body">
        Traders place <b>~134 trades/day during Extreme Fear</b> but only ~76 during Extreme Greed.
        This reactive over-trading during panic often leads to fee bleed and poor entries.
        The high trade count during Fear doesn't correlate with proportionally better win rates.
      </div>
    </div>

    <div class="insight-card blue">
      <div class="insight-title">📌 Insight 3 — Low Leverage Wins Consistently</div>
      <div class="insight-body">
        Low-leverage traders average <b>$9,056 total PnL vs $5,198</b> for high-leverage traders.
        Across all sentiment states, excessive leverage amplifies drawdowns more than it boosts gains.
        Risk-adjusted returns strongly favor the lower-leverage cohort.
      </div>
    </div>

    <div class="insight-card">
      <div class="insight-title">📌 Insight 4 — Consistent Winners Are Sentiment-Agnostic</div>
      <div class="insight-body">
        Consistent Winners maintain <b>~41% win rates regardless of Fear or Greed</b>.
        This indicates that disciplined, rules-based strategies outperform sentiment-chasing.
        Their Sharpe-proxy is significantly higher than the Inconsistent group.
      </div>
    </div>

    <div class="insight-card blue">
      <div class="insight-title">📌 Insight 5 — Long Bias Peaks During Fear (Dip Buyers)</div>
      <div class="insight-body">
        The Long/Short ratio rises sharply during Fear — traders are actively <b>buying the dip</b>.
        During Greed, positioning normalizes toward balanced long/short strategies.
        This reveals a crowd behaviour pattern that can be exploited with a contrarian approach.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h3>🎯 Strategy Recommendations (Part C)</h3></div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#0d2137,#0a1a2e);border:1px solid #1e3a5a;
                    border-top:4px solid #4fa3e0;border-radius:12px;padding:20px;height:100%;'>
          <h4 style='color:#4fa3e0;margin-top:0;'>🔵 Strategy 1 — Contrarian Fear Play</h4>
          <p style='color:#c0c8d0;font-size:13px;line-height:1.7;'>
            <b style='color:#e8f0fe;'>Trigger:</b> F&G Index &lt; 40 (Fear zone)<br><br>
            <b style='color:#e8f0fe;'>Rule:</b> Cut leverage to below your personal median.
            Keep position size ≤ 75% of normal average.
            Focus on high-conviction dip entries only.<br><br>
            <b style='color:#e8f0fe;'>Why:</b> Fear days have the highest avg PnL ($5,328)
            but also highest variance. Lower leverage captures
            upside while preventing blowups during panic.<br><br>
            <b style='color:#e8f0fe;'>Best for:</b> Low-leverage, infrequent traders with
            patient entry discipline.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#0d2a0d,#0a1e0a);border:1px solid #1e4a1e;
                    border-top:4px solid #4caf50;border-radius:12px;padding:20px;height:100%;'>
          <h4 style='color:#4caf50;margin-top:0;'>🟢 Strategy 2 — Extreme Greed Momentum Ride</h4>
          <p style='color:#c0c8d0;font-size:13px;line-height:1.7;'>
            <b style='color:#e8f0fe;'>Trigger:</b> F&G Index &gt; 75 (Extreme Greed zone)<br><br>
            <b style='color:#e8f0fe;'>Rule:</b> Frequent traders should maintain or
            increase trade count, but cap each individual
            trade at ≤ 80% of max position size.<br><br>
            <b style='color:#e8f0fe;'>Why:</b> Extreme Greed shows strong win rates (38.6%)
            and high avg PnL ($5,161). Momentum favours
            active execution in trending markets.<br><br>
            <b style='color:#e8f0fe;'>Best for:</b> Frequent traders (&gt;50 trades/day)
            running momentum or scalping strategies.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # Live F&G lookup widget
    st.markdown("---")
    st.markdown('<div class="section-header"><h3>🔎 Strategy Signal Lookup</h3></div>', unsafe_allow_html=True)
    st.markdown("Enter today's Fear & Greed value to get a trading signal:")

    fg_input = st.slider("Today's Fear & Greed Index Value", 0, 100, 50)

    if fg_input < 25:
        zone, color, signal, detail = "Extreme Fear 😱", "#d62728", "⚠️ HIGH CAUTION — REDUCE LEVERAGE", \
            "Max position: 50% of normal · Avoid chasing entries · Wait for confirmation signals"
    elif fg_input < 45:
        zone, color, signal, detail = "Fear 😨", "#ff7f0e", "📉 CONTRARIAN BUY OPPORTUNITY", \
            "Leverage: ≤75% of median · Focus on high-conviction dip entries · Short timeframes"
    elif fg_input < 55:
        zone, color, signal, detail = "Neutral 😐", "#9e9e9e", "⚖️ BALANCED — FOLLOW YOUR SYSTEM", \
            "No sentiment edge. Follow your base strategy rules. Maintain normal sizing."
    elif fg_input < 75:
        zone, color, signal, detail = "Greed 😏", "#2ca02c", "📈 MOMENTUM FAVOURABLE", \
            "Ride trends · Cap individual position at 80% · Be ready to take profits quickly"
    else:
        zone, color, signal, detail = "Extreme Greed 🤑", "#1f77b4", "🚀 MOMENTUM STRONG — STAY DISCIPLINED", \
            "Frequent traders: increase frequency · Cap size at 80% · Watch for reversal signals"

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);border:2px solid {color};
                border-radius:12px;padding:20px 24px;margin-top:12px;'>
      <div style='display:flex;align-items:center;gap:12px;'>
        <div style='font-size:36px;font-weight:900;color:{color};'>{fg_input}</div>
        <div>
          <div style='color:{color};font-weight:700;font-size:15px;'>{zone}</div>
          <div style='color:#e8f0fe;font-weight:600;font-size:16px;margin-top:4px;'>{signal}</div>
        </div>
      </div>
      <div style='color:#9ca3af;font-size:13px;margin-top:12px;border-top:1px solid #2d3a4f;padding-top:12px;'>
        💬 {detail}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
  📊 Primetrade.ai — Data Science Intern Assignment · 
  211,224 trades · 32 traders · 479 overlapping days · 
  Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
