"""
Primetrade.ai – Data Science Intern Assignment
Trader Performance vs Market Sentiment (Fear/Greed)
Author: Candidate Submission
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ─── Style ────────────────────────────────────────────────────────────────────
PALETTE = {
    'Extreme Fear': '#d62728',
    'Fear':         '#ff7f0e',
    'Neutral':      '#7f7f7f',
    'Greed':        '#2ca02c',
    'Extreme Greed':'#1f77b4',
    'Fear_combined':    '#e05c5c',
    'Greed_combined':   '#4caf7d',
    'Neutral_color':    '#9e9e9e',
}
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 140, 'savefig.bbox': 'tight',
                     'font.family': 'DejaVu Sans'})
CHART_DIR = '/home/claude/primetrade_analysis/charts'
OUTPUT_DIR = '/home/claude/primetrade_analysis/outputs'

# ══════════════════════════════════════════════════════════════════════════════
# PART A – DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PART A — DATA PREPARATION")
print("=" * 65)

# ── A1. Load ──────────────────────────────────────────────────────────────────
fg_raw = pd.read_csv('/mnt/user-data/uploads/fear_greed_index.csv')
tr_raw = pd.read_csv('/mnt/user-data/uploads/historical_data.csv')

print(f"\nFear/Greed  → {fg_raw.shape[0]:,} rows × {fg_raw.shape[1]} cols")
print(f"Trader Data → {tr_raw.shape[0]:,} rows × {tr_raw.shape[1]} cols")

# ── A2. Clean Fear/Greed ──────────────────────────────────────────────────────
fg = fg_raw.copy()
fg['date'] = pd.to_datetime(fg['date'])
fg = fg.sort_values('date').drop_duplicates('date').reset_index(drop=True)
fg['sentiment_binary'] = fg['classification'].map(
    lambda x: 'Fear' if 'Fear' in x else ('Greed' if 'Greed' in x else 'Neutral'))

print(f"\nFear/Greed date range : {fg['date'].min().date()} → {fg['date'].max().date()}")
print("Classification counts :")
print(fg['classification'].value_counts().to_string())

# ── A3. Clean Trader Data ─────────────────────────────────────────────────────
tr = tr_raw.copy()

# Parse date from "Timestamp IST" column (format: DD-MM-YYYY HH:MM)
tr['datetime'] = pd.to_datetime(tr['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
tr['date'] = tr['datetime'].dt.date
tr['date'] = pd.to_datetime(tr['date'])

# Numeric coercion
tr['Closed PnL'] = pd.to_numeric(tr['Closed PnL'], errors='coerce').fillna(0)
tr['Size USD']   = pd.to_numeric(tr['Size USD'],   errors='coerce').fillna(0)
tr['Fee']        = pd.to_numeric(tr['Fee'],        errors='coerce').fillna(0)
tr['Net PnL']    = tr['Closed PnL'] - tr['Fee']

# Long/Short flag
tr['is_long']  = tr['Direction'].isin(['Open Long',  'Buy', 'Long > Short', 'Close Short'])
tr['is_short'] = tr['Direction'].isin(['Open Short', 'Sell','Short > Long','Close Long'])
tr['is_close'] = tr['Direction'].isin(['Close Long', 'Close Short', 'Long > Short', 'Short > Long'])

# Winning trade flag (only on closing trades that have non-zero PnL)
tr['is_win']   = (tr['Closed PnL'] > 0)

print(f"\nTrader date range     : {tr['date'].min().date()} → {tr['date'].max().date()}")
print(f"Unique accounts       : {tr['Account'].nunique():,}")
print(f"Unique coins          : {tr['Coin'].nunique():,}")
print(f"Null datetimes        : {tr['datetime'].isna().sum()}")

# ── A4. Merge ─────────────────────────────────────────────────────────────────
# Daily aggregation per trader
def daily_trader_agg(df):
    g = df.groupby(['Account', 'date'])
    agg = g.agg(
        total_pnl      = ('Closed PnL', 'sum'),
        net_pnl        = ('Net PnL', 'sum'),
        num_trades     = ('Trade ID', 'count'),
        avg_size_usd   = ('Size USD', 'mean'),
        total_size_usd = ('Size USD', 'sum'),
        total_fee      = ('Fee', 'sum'),
        wins           = ('is_win', 'sum'),
        longs          = ('is_long', 'sum'),
        shorts         = ('is_short', 'sum'),
        closes         = ('is_close', 'sum'),
    ).reset_index()
    agg['win_rate']      = agg['wins'] / agg['num_trades'].clip(lower=1)
    agg['long_short_r']  = agg['longs'] / (agg['shorts'] + 1e-9)
    return agg

daily = daily_trader_agg(tr)

# Merge with sentiment
merged = daily.merge(
    fg[['date', 'value', 'classification', 'sentiment_binary']],
    on='date', how='inner')

print(f"\nMerged rows           : {merged.shape[0]:,}")
print(f"Date overlap range    : {merged['date'].min().date()} → {merged['date'].max().date()}")
print(f"Sentiment days covered: {merged['date'].nunique()}")

# ── A5. Leverage proxy ────────────────────────────────────────────────────────
# Hyperliquid doesn't expose raw leverage; proxy = position_size_usd / avg_size_usd
# We'll use total_size_usd relative to the trader's own median as a leverage proxy
trader_median_size = merged.groupby('Account')['total_size_usd'].transform('median').clip(lower=1)
merged['leverage_proxy'] = merged['total_size_usd'] / trader_median_size

merged.to_csv(f'{OUTPUT_DIR}/merged_daily.csv', index=False)
print(f"\n✅ Merged daily data saved → {OUTPUT_DIR}/merged_daily.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART B — ANALYSIS")
print("=" * 65)

# Helper – sentiment order & colors
SENT_ORDER   = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
BIN_ORDER    = ['Fear', 'Neutral', 'Greed']
SENT_COLORS  = [PALETTE[s] for s in SENT_ORDER]
BIN_COLORS   = [PALETTE['Fear_combined'], PALETTE['Neutral_color'], PALETTE['Greed_combined']]

# ── B1. Performance by Sentiment ──────────────────────────────────────────────
print("\n[B1] Performance (PnL, win rate) by sentiment…")

perf = merged.groupby('classification').agg(
    avg_pnl      = ('total_pnl', 'mean'),
    median_pnl   = ('total_pnl', 'median'),
    avg_win_rate = ('win_rate', 'mean'),
    n_obs        = ('total_pnl', 'count'),
).reindex(SENT_ORDER).reset_index()

print(perf.to_string(index=False))
perf.to_csv(f'{OUTPUT_DIR}/performance_by_sentiment.csv', index=False)

# Chart 1 – Avg PnL & Win Rate by sentiment
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 1 — Trader Performance by Market Sentiment', fontsize=14, fontweight='bold', y=1.01)

bars1 = axes[0].bar(perf['classification'], perf['avg_pnl'], color=SENT_COLORS, edgecolor='white', linewidth=0.8)
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title('Average Daily PnL per Trader')
axes[0].set_ylabel('Avg PnL (USD)')
axes[0].set_xlabel('Sentiment')
for bar, val in zip(bars1, perf['avg_pnl']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'${val:.2f}', ha='center', va='bottom', fontsize=9)

bars2 = axes[1].bar(perf['classification'], perf['avg_win_rate']*100, color=SENT_COLORS, edgecolor='white', linewidth=0.8)
axes[1].set_title('Average Win Rate (%)')
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_xlabel('Sentiment')
axes[1].set_ylim(0, 60)
for bar, val in zip(bars2, perf['avg_win_rate']*100):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart1_performance_by_sentiment.png')
plt.close()
print("  → chart1 saved")

# Chart 2 – PnL distribution box plot by binary sentiment
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Chart 2 — PnL Distribution: Fear vs Neutral vs Greed Days', fontsize=14, fontweight='bold')
data_bins = [merged[merged['sentiment_binary'] == s]['total_pnl'].clip(-500, 500) for s in BIN_ORDER]
bp = ax.boxplot(data_bins, patch_artist=True, notch=False, sym='',
                widths=0.5, medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], BIN_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticklabels(BIN_ORDER, fontsize=12)
ax.set_ylabel('Daily PnL per Trader (USD, clipped ±500)')
ax.set_xlabel('Sentiment Category')
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart2_pnl_distribution_boxplot.png')
plt.close()
print("  → chart2 saved")

# ── B2. Behavior by Sentiment ──────────────────────────────────────────────────
print("\n[B2] Trader behavior by sentiment…")

behavior = merged.groupby('classification').agg(
    avg_trades_per_day = ('num_trades', 'mean'),
    avg_size_usd       = ('avg_size_usd', 'mean'),
    avg_leverage_proxy = ('leverage_proxy', 'mean'),
    avg_long_short_r   = ('long_short_r', 'mean'),
).reindex(SENT_ORDER).reset_index()

print(behavior.to_string(index=False))
behavior.to_csv(f'{OUTPUT_DIR}/behavior_by_sentiment.csv', index=False)

# Chart 3 – Behavior heatmap
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Chart 3 — Trader Behavior Changes Across Market Sentiment', fontsize=14, fontweight='bold')

metrics = [
    ('avg_trades_per_day', 'Avg Trades per Day', axes[0,0]),
    ('avg_size_usd',       'Avg Trade Size (USD)', axes[0,1]),
    ('avg_leverage_proxy', 'Avg Leverage Proxy (rel.)', axes[1,0]),
    ('avg_long_short_r',   'Avg Long/Short Ratio', axes[1,1]),
]
for col, title, ax in metrics:
    vals = behavior[col].fillna(0)
    bars = ax.bar(behavior['classification'], vals, color=SENT_COLORS, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(SENT_ORDER, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart3_behavior_by_sentiment.png')
plt.close()
print("  → chart3 saved")

# Chart 4 – Long/Short Ratio by sentiment (bar)
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle('Chart 4 — Long/Short Bias by Market Sentiment', fontsize=14, fontweight='bold')
colors = [PALETTE[s] for s in SENT_ORDER]
bars = ax.bar(behavior['classification'], behavior['avg_long_short_r'].fillna(0),
              color=SENT_COLORS, edgecolor='white')
ax.axhline(1, color='black', linestyle='--', linewidth=1, label='Balanced (ratio=1)')
ax.set_ylabel('Long / Short Ratio')
ax.set_xlabel('Sentiment')
ax.legend()
for bar, val in zip(bars, behavior['avg_long_short_r'].fillna(0)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart4_long_short_ratio.png')
plt.close()
print("  → chart4 saved")

# ── B3. Trader Segmentation ───────────────────────────────────────────────────
print("\n[B3] Trader segmentation…")

trader_profile = merged.groupby('Account').agg(
    total_pnl        = ('total_pnl',       'sum'),
    avg_pnl          = ('total_pnl',       'mean'),
    std_pnl          = ('total_pnl',       'std'),
    avg_win_rate     = ('win_rate',         'mean'),
    avg_trades       = ('num_trades',       'mean'),
    total_trades     = ('num_trades',       'sum'),
    avg_size_usd     = ('avg_size_usd',     'mean'),
    avg_leverage     = ('leverage_proxy',   'mean'),
    avg_long_short   = ('long_short_r',     'mean'),
    num_active_days  = ('date',             'nunique'),
).reset_index()

trader_profile['std_pnl'] = trader_profile['std_pnl'].fillna(0)
trader_profile['sharpe_proxy'] = trader_profile['avg_pnl'] / (trader_profile['std_pnl'] + 1e-9)

# Segment 1: High vs Low Leverage
trader_profile['leverage_segment'] = pd.qcut(
    trader_profile['avg_leverage'], q=2, labels=['Low Leverage', 'High Leverage'])

# Segment 2: Frequent vs Infrequent
trader_profile['freq_segment'] = pd.qcut(
    trader_profile['avg_trades'], q=2, labels=['Infrequent', 'Frequent'])

# Segment 3: Consistent Winners / Losers / Inconsistent
def consistency_label(row):
    if row['avg_pnl'] > 0 and row['sharpe_proxy'] > 0.5:
        return 'Consistent Winner'
    elif row['avg_pnl'] < 0 and row['sharpe_proxy'] < -0.5:
        return 'Consistent Loser'
    else:
        return 'Inconsistent'
trader_profile['consistency_segment'] = trader_profile.apply(consistency_label, axis=1)

print("\nLeverage segments:")
print(trader_profile.groupby('leverage_segment')['avg_pnl'].describe()[['count','mean','50%']].round(2).to_string())
print("\nFrequency segments:")
print(trader_profile.groupby('freq_segment')['avg_pnl'].describe()[['count','mean','50%']].round(2).to_string())
print("\nConsistency segments:")
print(trader_profile.groupby('consistency_segment')[['avg_pnl','avg_win_rate','total_trades']].mean().round(2).to_string())

trader_profile.to_csv(f'{OUTPUT_DIR}/trader_profiles.csv', index=False)

# Chart 5 – Segments comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chart 5 — Trader Segments: PnL Comparison', fontsize=14, fontweight='bold')

seg_data = [
    ('leverage_segment',     ['Low Leverage', 'High Leverage'],  ['#4caf7d', '#e05c5c'],  axes[0], 'Leverage Segment'),
    ('freq_segment',         ['Infrequent',   'Frequent'],        ['#64b5f6', '#ff8a65'],  axes[1], 'Frequency Segment'),
    ('consistency_segment',  ['Consistent Winner','Inconsistent','Consistent Loser'],
                              ['#2ca02c','#7f7f7f','#d62728'],     axes[2], 'Consistency Segment'),
]
for col, order, colors_s, ax, title in seg_data:
    seg_pnl = trader_profile.groupby(col)['avg_pnl'].mean().reindex(order)
    ax.bar(seg_pnl.index, seg_pnl.values, color=colors_s, edgecolor='white', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Avg Daily PnL (USD)')
    ax.set_xticklabels(order, rotation=15, ha='right', fontsize=9)
    for i, (idx, val) in enumerate(seg_pnl.items()):
        ax.text(i, val + (0.05 if val >= 0 else -0.15),
                f'${val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart5_segments.png')
plt.close()
print("  → chart5 saved")

# Chart 6 – Sentiment × Segment heatmap (PnL)
merged2 = merged.merge(trader_profile[['Account','leverage_segment','freq_segment','consistency_segment']], on='Account')

pivot_lev = merged2.groupby(['sentiment_binary','leverage_segment'])['total_pnl'].mean().unstack()
pivot_freq = merged2.groupby(['sentiment_binary','freq_segment'])['total_pnl'].mean().unstack()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 6 — Avg PnL: Sentiment × Trader Segment (Heatmap)', fontsize=14, fontweight='bold')

sns.heatmap(pivot_lev.reindex(['Fear','Neutral','Greed']), annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=axes[0], linewidths=0.5)
axes[0].set_title('Leverage Segments')
axes[0].set_xlabel(''); axes[0].set_ylabel('Sentiment')

sns.heatmap(pivot_freq.reindex(['Fear','Neutral','Greed']), annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=axes[1], linewidths=0.5)
axes[1].set_title('Frequency Segments')
axes[1].set_xlabel(''); axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart6_sentiment_segment_heatmap.png')
plt.close()
print("  → chart6 saved")

# Chart 7 – Daily PnL trend with sentiment overlay
print("\n[B4] Daily aggregate PnL trend…")
daily_agg = merged.groupby('date').agg(
    total_pnl   = ('total_pnl', 'sum'),
    avg_pnl     = ('total_pnl', 'mean'),
    sentiment   = ('sentiment_binary', 'first'),
    fg_value    = ('value', 'first'),
).reset_index()

fig, ax1 = plt.subplots(figsize=(16, 5))
fig.suptitle('Chart 7 — Daily Aggregate PnL vs Fear/Greed Index', fontsize=14, fontweight='bold')

color_map = {'Fear': '#e05c5c', 'Neutral': '#aaaaaa', 'Greed': '#4caf7d'}
for _, row in daily_agg.iterrows():
    ax1.axvspan(row['date'] - pd.Timedelta(hours=12),
                row['date'] + pd.Timedelta(hours=12),
                alpha=0.15, color=color_map.get(row['sentiment'], '#aaaaaa'), linewidth=0)

ax1.plot(daily_agg['date'], daily_agg['avg_pnl'], color='#333333', linewidth=1.2, label='Avg PnL')
ax1.axhline(0, color='black', linewidth=0.6, linestyle='--')
ax1.set_ylabel('Avg Daily PnL per Trader (USD)', color='#333333')
ax1.set_xlabel('Date')

ax2 = ax1.twinx()
ax2.plot(daily_agg['date'], daily_agg['fg_value'], color='steelblue', linewidth=1.0, alpha=0.7, label='F&G Index')
ax2.set_ylabel('Fear & Greed Index Value', color='steelblue')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
patches = [mpatches.Patch(color='#e05c5c', alpha=0.4, label='Fear'),
           mpatches.Patch(color='#aaaaaa', alpha=0.4, label='Neutral'),
           mpatches.Patch(color='#4caf7d', alpha=0.4, label='Greed')]
ax1.legend(handles=lines1 + lines2 + patches, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart7_daily_pnl_sentiment_trend.png')
plt.close()
print("  → chart7 saved")

# Chart 8 – Win rate & trade count by sentiment (dual axis bar)
fig, ax1 = plt.subplots(figsize=(10, 5))
fig.suptitle('Chart 8 — Win Rate & Trade Frequency by Sentiment', fontsize=14, fontweight='bold')

x = np.arange(len(SENT_ORDER))
w = 0.35
wr = [perf[perf['classification']==s]['avg_win_rate'].values[0]*100 if len(perf[perf['classification']==s]) else 0 for s in SENT_ORDER]
tc = [behavior[behavior['classification']==s]['avg_trades_per_day'].values[0] if len(behavior[behavior['classification']==s]) else 0 for s in SENT_ORDER]

bars1 = ax1.bar(x - w/2, wr, w, label='Win Rate (%)', color=SENT_COLORS, alpha=0.8, edgecolor='white')
ax1.set_ylabel('Win Rate (%)')
ax1.set_ylim(0, 70)
ax1.set_xticks(x)
ax1.set_xticklabels(SENT_ORDER, rotation=12, ha='right')
ax1.set_xlabel('Sentiment')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + w/2, tc, w, label='Avg Trades/Day', color='#5b9bd5', alpha=0.7, edgecolor='white')
ax2.set_ylabel('Avg Trades per Day')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart8_winrate_tradecount.png')
plt.close()
print("  → chart8 saved")

# ── BONUS: Predictive Model ───────────────────────────────────────────────────
print("\n[BONUS] Simple predictive model…")

model_df = merged[['total_pnl','num_trades','avg_size_usd','leverage_proxy',
                    'long_short_r','win_rate','value','sentiment_binary']].dropna().copy()

# Target: next-day profitable (1) or not (0)
model_df = model_df.copy()
model_df['target'] = (model_df['total_pnl'] > 0).astype(int)

le = LabelEncoder()
model_df['sentiment_enc'] = le.fit_transform(model_df['sentiment_binary'])

features = ['num_trades','avg_size_usd','leverage_proxy','long_short_r','win_rate','value','sentiment_enc']
X = model_df[features]
y = model_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
print("  Model accuracy:", round(report['accuracy'], 3))
print("  F1 (class 1)  :", round(report['1']['f1-score'], 3))

# Feature importance chart
fi = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Chart 9 — Feature Importance: Predict Profitable Day', fontsize=13, fontweight='bold')
ax.barh(fi.index, fi.values, color='steelblue', edgecolor='white')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart9_feature_importance.png')
plt.close()
print("  → chart9 saved")

# ── BONUS: KMeans Clustering ──────────────────────────────────────────────────
print("\n[BONUS] KMeans clustering…")
cluster_features = ['avg_pnl','avg_win_rate','avg_trades','avg_leverage','avg_long_short']
clust_df = trader_profile[cluster_features].dropna()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clust_df)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
trader_profile_clean = trader_profile.dropna(subset=cluster_features).copy()
trader_profile_clean['cluster'] = kmeans.fit_predict(X_scaled)

cluster_summary = trader_profile_clean.groupby('cluster')[cluster_features].mean().round(3)
print(cluster_summary.to_string())

# Chart 10 – Cluster scatter
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle('Chart 10 — Trader Behavioral Clusters (PnL vs Win Rate)', fontsize=13, fontweight='bold')
cluster_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
cluster_labels = {0:'Cluster A', 1:'Cluster B', 2:'Cluster C', 3:'Cluster D'}
for cl in range(4):
    sub = trader_profile_clean[trader_profile_clean['cluster'] == cl]
    ax.scatter(sub['avg_win_rate']*100, sub['avg_pnl'].clip(-200, 200),
               alpha=0.5, s=30, color=cluster_colors[cl], label=f'Cluster {cl}')
ax.set_xlabel('Avg Win Rate (%)')
ax.set_ylabel('Avg Daily PnL (USD, clipped)')
ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
ax.legend()
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/chart10_clusters.png')
plt.close()
print("  → chart10 saved")

# ── Final summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY STATS")
print("=" * 65)
print("\nPerformance by Sentiment:")
print(perf[['classification','avg_pnl','median_pnl','avg_win_rate','n_obs']].to_string(index=False))
print("\nBehavior by Sentiment:")
print(behavior.to_string(index=False))

print("\n✅ All charts saved to:", CHART_DIR)
print("✅ All outputs saved to:", OUTPUT_DIR)
