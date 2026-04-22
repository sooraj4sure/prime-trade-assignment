#  Primetrade.ai — Data Science Intern Assignment
## Trader Performance vs Market Sentiment (Fear/Greed)

---

##  Repository Structure

```
primetrade_analysis/
│
├── primetrade_analysis.ipynb   ← Main analysis notebook (Parts A, B, C + Bonus)
├── analysis.py                 ← Standalone Python script (same analysis)
├── README.md                   ← This file
│
├── charts/
│   ├── chart1_performance_by_sentiment.png
│   ├── chart2_pnl_distribution_boxplot.png
│   ├── chart3_behavior_by_sentiment.png
│   ├── chart4_long_short_ratio.png
│   ├── chart5_segments.png
│   ├── chart6_sentiment_segment_heatmap.png
│   ├── chart7_daily_pnl_sentiment_trend.png
│   ├── chart8_winrate_tradecount.png
│   ├── chart9_feature_importance.png
│   └── chart10_clusters.png
│
└── outputs/
    ├── merged_daily.csv              ← Merged trader+sentiment daily data
    ├── performance_by_sentiment.csv  ← Aggregated performance metrics
    ├── behavior_by_sentiment.csv     ← Aggregated behavior metrics
    └── trader_profiles.csv           ← Per-trader profile with segments
```

---

##  Setup & How to Run

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbformat
```

### Data Files Required
Place these two CSV files in the same directory as the notebook:
- `fear_greed_index.csv`
- `historical_data.csv`

### Run Notebook
```bash
jupyter notebook primetrade_analysis.ipynb
```

### Run Script
```bash
python3 analysis.py
or run  
primetrade_analysis.ipynb file
```

---

##  Methodology

### Data Sources
| Dataset | Rows | Columns | Date Range |
|---------|------|---------|-----------|
| Fear/Greed Index | 2,644 | 4 | 2018-02-01 → 2025-05-02 |
| Hyperliquid Trader Data | 211,224 | 16 | 2023-05-01 → 2025-05-01 |

### Part A — Data Preparation
1. **Loaded** both datasets; confirmed zero missing values in either file
2. **Parsed** `Timestamp IST` (format `DD-MM-YYYY HH:MM`) to extract trade dates
3. **Created binary sentiment**: `Fear` (Extreme Fear + Fear), `Greed` (Greed + Extreme Greed), `Neutral`
4. **Engineered daily metrics** per trader:
   - `total_pnl`, `net_pnl`, `num_trades`, `avg_size_usd`, `win_rate`, `long_short_r`
   - `leverage_proxy` = trader's daily total_size_usd / their own median (relative leverage)
5. **Merged** on `date` → **2,340 trader-day records across 479 days**

### Part B — Analysis
- Grouped by `classification` (5-level) and `sentiment_binary` (3-level)
- Compared PnL, win rate, trade frequency, leverage, long/short ratio
- Segmented traders into: Low/High Leverage · Frequent/Infrequent · Consistent Winner/Loser/Inconsistent
- Cross-tabulated sentiment × segment via heatmaps

### Bonus
- **Random Forest Classifier** to predict whether a trader-day is profitable → **95.7% accuracy**
- **KMeans (k=4)** clustering to identify behavioral archetypes

---

##  Key Insights

### Insight 1 — Fear ≠ Bad Performance (Contrary to Intuition)
> Traders actually earn **higher average PnL on Fear days ($5,328)** than on Greed days ($3,318).
> Extreme Greed days also show strong PnL ($5,161) but with higher variability.
> Possible reason: Fear creates sharp price dislocations → larger, faster PnL swings for active traders.

### Insight 2 — Trade Frequency Drops as Greed Rises
> Traders place **~134 trades/day during Extreme Fear** but only ~76 during Extreme Greed.
> This is counterintuitive — one might expect more activity when markets are bullish.
> Interpretation: Fear/volatility triggers reactive over-trading; Greed leads to patient, larger bets.

### Insight 3 — Low-Leverage Traders Consistently Outperform
> Low leverage segment averages **$9,056 total PnL** vs $5,198 for high leverage.
> This holds across all sentiment states. Excessive leverage amplifies losses more than gains.

### Insight 4 — Consistent Winners Are Rare but Disciplined
> Only a subset of traders qualify as "Consistent Winners" (avg_pnl > 0 & Sharpe proxy > 0.5).
> They maintain ~41% win rates regardless of market sentiment — suggesting rules-based discipline.

### Insight 5 — Sentiment Shapes Long/Short Bias
> Long/Short ratio is elevated during Fear, meaning many traders **buy dips** during fear.
> During Greed, the ratio normalizes — traders are less directionally biased (more balanced strategies).

---

## Strategy Recommendations (Part C)

###  Strategy 1: Contrarian Fear Play — "Downsize & Buy the Dip"
> **Rule:** On Fear days (F&G value < 40), reduce leverage to below your personal median.
> Keep individual position size ≤ 75% of your normal average.
> **Why it works:** Fear days have the highest avg PnL but also highest variance.
> Smaller leverage captures the upside of dip-buying with reduced blowup risk.
> **Best for:** Low-leverage traders, infrequent traders with patient entries.

###  Strategy 2: Extreme Greed Momentum — "Ride the Wave with Frequency Caps"
> **Rule:** During Extreme Greed (F&G > 75), frequent traders should maintain or
> increase trade count but cap each individual trade to ≤ 80% of max size.
> **Why it works:** Extreme Greed days show strong win rates (38.6%) + high avg PnL.
> Momentum favors active execution. But individual over-sizing in frothy markets is risky.
> **Best for:** Frequent traders (>50 trades/day) with momentum-based strategies.

---

##  Charts Summary

| Chart | What It Shows |
|-------|--------------|
| Chart 1 | Avg PnL & Win Rate by all 5 sentiment levels |
| Chart 2 | PnL distribution boxplot (Fear/Neutral/Greed) |
| Chart 3 | 4-panel behavior grid (trades, size, leverage, L/S) |
| Chart 4 | Long/Short ratio by sentiment |
| Chart 5 | 3-segment PnL comparison |
| Chart 6 | Heatmap: Sentiment × Leverage/Frequency segment |
| Chart 7 | Time-series PnL trend with sentiment color overlay |
| Chart 8 | Win rate & trade count dual-axis bar chart |
| Chart 9 | Feature importance from Random Forest model |
| Chart 10 | KMeans cluster scatter (win rate vs avg PnL) |

---

*Submitted for Primetrade.ai Data Science Intern – Round 0*

---

##  Streamlit Dashboard

### Run the Dashboard
```bash
# Install dependencies
pip install streamlit plotly pandas numpy scikit-learn

# Run (from inside the primetrade_analysis folder)
streamlit run dashboard.py
```

### Dashboard Tabs
| Tab | What You Can Explore |
|-----|---------------------|
|  Performance | PnL & Win Rate by all 5 sentiment levels, box plots |
|  Behavior | Trade frequency, size, leverage, long/short ratio |
|  Segments | Heatmaps & scatter plots by Leverage / Frequency / Consistency |
|  Time Series | PnL trend with sentiment background overlay + F&G index |
|  Insights & Strategy | Key findings + live signal lookup by F&G value |

### Sidebar Controls
-  Date range filter
-  Sentiment filter (select which sentiment days to include)
-  Leverage / Frequency / Consistency segment filters
-  PnL clip range slider
