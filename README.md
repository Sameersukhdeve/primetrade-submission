# Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Intern Round-0 Assignment ...

---

## Overview

This project analyzes the relationship between Bitcoin Fear/Greed market sentiment and trader behavior/performance on the Hyperliquid DEX. The goal is to extract actionable patterns that could inform smarter trading strategies.

---

## Setup

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

### Data files
Place both CSV files in the project root directory:
| File | Source |
|---|---|
| `sentiment.csv` | Bitcoin Fear/Greed Index (provided) |
| `trades.csv` | Hyperliquid historical trades (provided) |

> **Note:** If either file is missing, the notebook automatically generates synthetic data so all cells run end-to-end for demonstration purposes.

### Run the notebook
```bash
jupyter notebook analysis_notebook.ipynb
```
Or run as a plain Python script:
```bash
python analysis.py
```

---

## Project Structure

```
trader_sentiment_analysis/
├── analysis_notebook.ipynb   # Main Jupyter notebook (submit this)
├── analysis.py               # Equivalent Python script
├── README.md                 # This file
└── charts/                   # Auto-generated output charts
    ├── chart1_pnl_distribution.png
    ├── chart2_behaviour_shifts.png
    ├── chart3_drawdown_heatmap.png
    ├── chart4_timeseries.png
    ├── chart5_6_winrate_leverage.png
    ├── chart7_trader_archetypes.png
    └── chart8_feature_importance.png
```

---

## Methodology

### Data Preparation (Part A)
- Loaded both datasets and documented shape, dtypes, missing values, and duplicates
- Converted timestamps to UTC-normalized dates for daily-level alignment
- Merged sentiment labels onto every trader-day observation
- Engineered daily metrics per trader: PnL, win rate, average leverage, long/short ratio, position size, trade frequency
- Computed a **drawdown proxy** as `cumulative PnL - rolling maximum cumulative PnL`

### Analysis (Part B)
- **Fear vs Greed comparison:** Welch two-sample t-tests on PnL, win rate, drawdown, leverage, and position size
- **Behaviour shifts:** Normalized bar-chart comparison of trading behaviour metrics across sentiment states
- **Trader segmentation:**
  - High vs Low leverage (split at median)
  - Frequent vs Infrequent traders (split at median trades/day)
  - Consistent Winners (win rate ≥ 50% AND net positive PnL) vs Inconsistent
- **K-Means clustering** (k=4) on 5 behavioral features to derive trader archetypes

### Predictive Model (Bonus)
- Random Forest classifier predicting **next-day profitability** (binary: profit / loss)
- Features: sentiment flag, leverage, trade count, long ratio, position size, lagged PnL, lagged win rate
- Train/test split 80/20, no data leakage (lag features prevent lookahead)

---

## Key Insights

| # | Insight | Evidence |
|---|---|---|
| 1 | **Fear days drive deeper drawdowns** | Drawdown significantly worse on Fear days (p < 0.05). High-lev traders most affected. |
| 2 | **Leverage amplifies sentiment risk** | High-lev segment shows larger win-rate drop on Fear vs Low-lev peers. |
| 3 | **Position-size paradox** | Average trade size is slightly *higher* on Fear days — reactive/revenge trading behaviour. |

---

## Strategy Recommendations (Part C)

### Strategy 1 — Sentiment-Adjusted Leverage Cap
**Target:** High-Leverage / "Aggressive Losers" archetype

| Sentiment | Rule |
|---|---|
| Fear | Cap leverage at 5x; reduce position size by 30% |
| Greed | Allow up to standard leverage limit |

**Rationale:** This archetype shows the clearest performance degradation under Fear. A dynamic cap directly reduces the amplification of downside volatility at the most vulnerable moments.

### Strategy 2 — Contrarian Frequency Signal
**Target:** Consistent Winners archetype

| Sentiment | Rule |
|---|---|
| Fear | Maintain or slightly increase trade frequency (contrarian edge) |
| Greed | Reduce long bias below 60%; favor mean-reversion setups |

**Rationale:** Consistent Winners retain a statistical edge even during Fear markets. Market over-reaction on Fear days creates mispricings. On Greed days, over-extended long positioning historically increases reversal risk.

---

## Charts

| Chart | Description |
|---|---|
| Chart 1 | PnL and win-rate violin plots — Fear vs Greed |
| Chart 2 | Normalised behaviour metrics comparison |
| Chart 3 | Drawdown heatmap by sentiment × leverage segment |
| Chart 4 | Rolling 7-day avg PnL time-series with sentiment background |
| Chart 5 | Win rate by leverage segment × sentiment |
| Chart 6 | Leverage distribution density — Fear vs Greed |
| Chart 7 | K-Means trader archetype scatter (leverage vs win rate, bubble = frequency) |
| Chart 8 | Random Forest feature importance for profitability prediction |

---

*Submitted for Primetrade.ai Data Science Intern Round-0*
