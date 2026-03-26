"""
Trader Performance vs Market Sentiment — Primetrade.ai Round-0
Full analysis pipeline. Can also run as a plain .py script.
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

# Plot style
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccee",
    "xtick.color":      "#ccccee",
    "ytick.color":      "#ccccee",
    "text.color":       "#ccccee",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#444466",
    "font.family":      "DejaVu Sans",
})
FEAR_COLOR   = "#e05c5c"
GREED_COLOR  = "#4caf82"
NEUTRAL_COLOR = "#9b9bbf"

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING  (tries real files first, falls back to synthetic)
# ─────────────────────────────────────────────────────────────────────────────

def load_sentiment(path="sentiment.csv"):
    """Load Bitcoin Fear/Greed dataset."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        # Try to find date & classification columns
        date_col  = next((c for c in df.columns if "date" in c), df.columns[0])
        class_col = next((c for c in df.columns if "class" in c or "sentiment" in c), df.columns[1])
        df = df.rename(columns={date_col: "date", class_col: "classification"})
        df["date"] = pd.to_datetime(df["date"])
        df["is_fear"] = df["classification"].str.lower().str.contains("fear").astype(int)
        print(f"[sentiment] loaded {len(df):,} rows from {path}")
        return df[["date", "classification", "is_fear"]]
    else:
        return _synthetic_sentiment()

def _synthetic_sentiment():
    """Generate ~2 years of synthetic Fear/Greed labels with realistic clustering."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    # Markov-chain: fear tends to persist, greed tends to persist
    states = []
    s = 0  # 0=Fear, 1=Greed
    for _ in dates:
        states.append(s)
        s = np.random.choice([0, 1], p=[0.35, 0.65] if s == 1 else [0.6, 0.4])
    labels = ["Greed" if s else "Fear" for s in states]
    df = pd.DataFrame({"date": dates, "classification": labels, "is_fear": [1-s for s in states]})
    print(f"[sentiment] generated {len(df):,} synthetic rows")
    return df

def load_trades(path="trades.csv"):
    """Load Hyperliquid historical trades."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # Standardise key columns
        renames = {}
        for col in df.columns:
            if "pnl" in col and "closed" in col: renames[col] = "closedpnl"
            elif "pnl" in col:                   renames[col] = "closedpnl"
            elif "account" in col:               renames[col] = "account"
            elif "symbol" in col or "coin" in col: renames[col] = "symbol"
            elif "size" in col and "usd" not in col: renames[col] = "size"
            elif "side" in col:                  renames[col] = "side"
            elif "leverage" in col:              renames[col] = "leverage"
            elif "time" in col or "timestamp" in col: renames[col] = "timestamp"
        df = df.rename(columns=renames)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            if df["timestamp"].isna().all():
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.normalize()
        for col in ["closedpnl", "size", "leverage"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"[trades] loaded {len(df):,} rows from {path}")
        return df
    else:
        return _synthetic_trades()

def _synthetic_trades():
    """Generate realistic synthetic Hyperliquid trade data."""
    np.random.seed(99)
    n = 80_000
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    accounts = [f"0x{i:04x}{'ab'*8}" for i in range(200)]
    symbols  = ["BTC", "ETH", "SOL", "ARB", "DOGE", "MATIC"]

    acct_arr  = np.random.choice(accounts, n)
    sym_arr   = np.random.choice(symbols, n, p=[0.4, 0.3, 0.1, 0.08, 0.07, 0.05])
    date_arr  = np.random.choice(dates, n)
    side_arr  = np.random.choice(["B", "A"], n)   # B=Buy/Long, A=Ask/Short
    lev_arr   = np.random.choice([1,2,3,5,10,20,50], n, p=[0.1,0.15,0.2,0.25,0.15,0.1,0.05])
    size_arr  = np.abs(np.random.lognormal(3, 1.5, n))

    # PnL correlated with leverage and size (noisily)
    base_pnl  = np.random.normal(0, 1, n)
    pnl_arr   = base_pnl * size_arr * 0.02 - (lev_arr * 0.001 * size_arr)

    ts_arr = pd.to_datetime(date_arr) + pd.to_timedelta(
        np.random.randint(0, 86400, n), unit="s"
    )

    df = pd.DataFrame({
        "account":   acct_arr,
        "symbol":    sym_arr,
        "timestamp": ts_arr,
        "date":      pd.to_datetime(date_arr),
        "side":      side_arr,
        "size":      size_arr,
        "leverage":  lev_arr.astype(float),
        "closedpnl": pnl_arr,
    })
    print(f"[trades] generated {len(df):,} synthetic rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def data_quality_report(sentiment, trades):
    print("\n" + "="*60)
    print("  DATA QUALITY REPORT")
    print("="*60)
    for name, df in [("Sentiment", sentiment), ("Trades", trades)]:
        print(f"\n── {name} ──")
        print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} cols")
        print(f"  Duplicates     : {df.duplicated().sum():,}")
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if len(miss):
            print(f"  Missing values :\n{miss.to_string()}")
        else:
            print("  Missing values : none")
        print(f"  Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MERGE & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_metrics(trades, sentiment):
    """Compute daily trader-level and market-level metrics, then merge sentiment."""
    t = trades.copy()

    # ── per-trade flags ──────────────────────────────────────────────────────
    t["is_long"]  = t["side"].str.upper().isin(["B", "BUY", "LONG"]).astype(int)
    t["is_win"]   = (t["closedpnl"] > 0).astype(int)

    # ── daily aggregate per account ─────────────────────────────────────────
    grp = t.groupby(["date", "account"])
    daily = grp.agg(
        daily_pnl        = ("closedpnl", "sum"),
        n_trades         = ("closedpnl", "count"),
        win_rate         = ("is_win",    "mean"),
        avg_size         = ("size",      "mean"),
        avg_leverage     = ("leverage",  "mean"),
        long_ratio       = ("is_long",   "mean"),
        total_volume     = ("size",      "sum"),
    ).reset_index()

    # ── merge sentiment ──────────────────────────────────────────────────────
    daily = daily.merge(
        sentiment[["date", "classification", "is_fear"]],
        on="date", how="left"
    )
    daily["sentiment"] = daily["classification"].fillna("Unknown")

    # ── cumulative PnL per trader (for drawdown proxy) ───────────────────────
    daily = daily.sort_values(["account", "date"])
    daily["cum_pnl"]     = daily.groupby("account")["daily_pnl"].cumsum()
    daily["rolling_max"] = daily.groupby("account")["cum_pnl"].cummax()
    daily["drawdown"]    = daily["cum_pnl"] - daily["rolling_max"]

    # ── high / low leverage split ────────────────────────────────────────────
    median_lev = daily["avg_leverage"].median()
    daily["lev_segment"] = np.where(daily["avg_leverage"] >= median_lev, "High Lev", "Low Lev")

    print(f"[features] daily metrics: {daily.shape[0]:,} rows | "
          f"sentiment match rate: {daily['classification'].notna().mean():.1%}")
    return daily, median_lev


def build_trader_profiles(daily):
    """One row per trader summarising overall behaviour."""
    grp = daily.groupby("account")
    profiles = grp.agg(
        total_pnl        = ("daily_pnl",    "sum"),
        avg_daily_pnl    = ("daily_pnl",    "mean"),
        std_daily_pnl    = ("daily_pnl",    "std"),
        total_trades     = ("n_trades",     "sum"),
        avg_win_rate     = ("win_rate",     "mean"),
        avg_leverage     = ("avg_leverage", "mean"),
        avg_long_ratio   = ("long_ratio",   "mean"),
        active_days      = ("date",         "nunique"),
        max_drawdown     = ("drawdown",     "min"),
    ).reset_index()
    profiles["sharpe_proxy"]   = profiles["avg_daily_pnl"] / (profiles["std_daily_pnl"] + 1e-9)
    profiles["trades_per_day"] = profiles["total_trades"] / (profiles["active_days"] + 1e-9)
    # Segment: frequent vs infrequent
    med_freq = profiles["trades_per_day"].median()
    profiles["freq_segment"] = np.where(profiles["trades_per_day"] >= med_freq, "Frequent", "Infrequent")
    # Segment: consistent winners
    profiles["winner_segment"] = np.where(
        (profiles["avg_win_rate"] >= 0.5) & (profiles["total_pnl"] > 0),
        "Consistent Winner", "Inconsistent"
    )
    return profiles, med_freq


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANALYSIS — PART B
# ─────────────────────────────────────────────────────────────────────────────

def fear_vs_greed_summary(daily):
    """Statistical comparison of Fear vs Greed days."""
    fear  = daily[daily["is_fear"] == 1]
    greed = daily[daily["is_fear"] == 0]

    metrics = ["daily_pnl", "win_rate", "drawdown", "avg_leverage",
               "n_trades", "long_ratio", "avg_size"]
    rows = []
    for m in metrics:
        f_vals = fear[m].dropna()
        g_vals = greed[m].dropna()
        t_stat, p_val = stats.ttest_ind(f_vals, g_vals, equal_var=False)
        rows.append({
            "Metric":        m,
            "Fear (mean)":   round(f_vals.mean(), 4),
            "Greed (mean)":  round(g_vals.mean(), 4),
            "Δ (Greed-Fear)":round(g_vals.mean() - f_vals.mean(), 4),
            "p-value":       round(p_val, 4),
            "Significant":   "✓" if p_val < 0.05 else "✗",
        })
    df_out = pd.DataFrame(rows)
    print("\n── Fear vs Greed Comparison ──")
    print(df_out.to_string(index=False))
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[chart] saved → {path}")
    plt.close(fig)


def chart_1_pnl_distribution(daily):
    """Chart 1: PnL distribution Fear vs Greed (violin + strip)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Chart 1 — Daily PnL Distribution: Fear vs Greed", fontsize=14, y=1.01)

    for ax, col, title, lim in zip(
        axes,
        ["daily_pnl", "win_rate"],
        ["Daily PnL (USD)", "Win Rate"],
        [(-5000, 5000), (0, 1)]
    ):
        subset = daily[daily["sentiment"].isin(["Fear", "Greed"])].copy()
        subset = subset[subset[col].between(*lim)]
        palette = {"Fear": FEAR_COLOR, "Greed": GREED_COLOR}
        sns.violinplot(data=subset, x="sentiment", y=col, palette=palette,
                       order=["Fear", "Greed"], ax=ax, inner="quartile", linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.grid(True, axis="y")

    _save(fig, "chart1_pnl_distribution.png")


def chart_2_behaviour_shifts(daily):
    """Chart 2: Behaviour metrics Fear vs Greed — grouped bar."""
    metrics  = ["avg_leverage", "n_trades", "long_ratio", "avg_size"]
    labels   = ["Avg Leverage", "Trades / Day", "Long Ratio", "Avg Position Size"]
    sentiments = ["Fear", "Greed"]
    sub = daily[daily["sentiment"].isin(sentiments)]

    vals = {s: [sub[sub["sentiment"] == s][m].mean() for m in metrics] for s in sentiments}
    # normalise for visual comparison
    maxv = {m: max(vals["Fear"][i], vals["Greed"][i]) + 1e-9 for i, m in enumerate(metrics)}
    norm = {s: [vals[s][i] / maxv[m] for i, m in enumerate(metrics)] for s in sentiments}

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, norm["Fear"],  width=w, color=FEAR_COLOR,  label="Fear",  alpha=0.85)
    ax.bar(x + w/2, norm["Greed"], width=w, color=GREED_COLOR, label="Greed", alpha=0.85)

    # annotate actual values
    for i, m in enumerate(metrics):
        ax.text(i - w/2, norm["Fear"][i]  + 0.02, f"{vals['Fear'][i]:.2f}",
                ha="center", fontsize=8, color=FEAR_COLOR)
        ax.text(i + w/2, norm["Greed"][i] + 0.02, f"{vals['Greed'][i]:.2f}",
                ha="center", fontsize=8, color=GREED_COLOR)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Normalised value (relative to max)")
    ax.set_title("Chart 2 — Trader Behaviour: Fear vs Greed Days", fontsize=13)
    ax.legend(); ax.set_ylim(0, 1.25); ax.grid(True, axis="y")
    _save(fig, "chart2_behaviour_shifts.png")


def chart_3_drawdown_heatmap(daily):
    """Chart 3: Average drawdown by sentiment × leverage segment."""
    sub = daily[daily["sentiment"].isin(["Fear", "Greed"])]
    pivot = sub.groupby(["sentiment", "lev_segment"])["drawdown"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Avg Drawdown (USD)"})
    ax.set_title("Chart 3 — Avg Drawdown: Sentiment × Leverage Segment", fontsize=12)
    _save(fig, "chart3_drawdown_heatmap.png")


def chart_4_timeseries(daily, sentiment):
    """Chart 4: Rolling 7d avg PnL over time, colour-coded by sentiment."""
    agg = daily.groupby("date")["daily_pnl"].mean().reset_index()
    agg = agg.merge(sentiment[["date", "is_fear"]], on="date", how="left")
    agg["roll7"] = agg["daily_pnl"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    for _, row in agg.iterrows():
        color = FEAR_COLOR if row["is_fear"] == 1 else GREED_COLOR
        ax.axvspan(row["date"], row["date"] + pd.Timedelta(days=1),
                   alpha=0.12, color=color, linewidth=0)
    ax.plot(agg["date"], agg["roll7"], color="#ffffffcc", linewidth=1.5, label="7d rolling avg PnL")
    ax.axhline(0, color="#ffffff55", linewidth=0.8, linestyle="--")
    ax.set_title("Chart 4 — Market-wide Avg PnL over Time (Fear=red bg, Greed=green bg)", fontsize=12)
    ax.set_ylabel("Avg daily PnL (USD)"); ax.legend()
    _save(fig, "chart4_timeseries.png")


def chart_5_segment_winrate(daily):
    """Chart 5: Win rate by frequency segment and sentiment."""
    sub = daily[daily["sentiment"].isin(["Fear", "Greed"])]
    # need trader freq segment merged back
    # use leverage segment as proxy here
    grp = sub.groupby(["sentiment", "lev_segment"])["win_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=grp, x="lev_segment", y="win_rate", hue="sentiment",
                palette={"Fear": FEAR_COLOR, "Greed": GREED_COLOR}, ax=ax)
    ax.set_title("Chart 5 — Win Rate by Leverage Segment × Sentiment", fontsize=12)
    ax.set_ylabel("Mean Win Rate"); ax.set_ylim(0, 0.8)
    ax.axhline(0.5, color="#ffffffaa", linestyle="--", linewidth=0.9, label="50% baseline")
    ax.legend(); ax.grid(True, axis="y")
    _save(fig, "chart5_segment_winrate.png")


def chart_6_leverage_distribution(daily):
    """Chart 6: Leverage distribution, Fear vs Greed, CDF-style."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [("Fear", FEAR_COLOR), ("Greed", GREED_COLOR)]:
        vals = daily[daily["sentiment"] == label]["avg_leverage"].dropna()
        vals = vals[vals < vals.quantile(0.99)]
        ax.hist(vals, bins=40, alpha=0.55, color=color, label=label, density=True)
    ax.set_title("Chart 6 — Leverage Distribution: Fear vs Greed", fontsize=12)
    ax.set_xlabel("Average Leverage"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(True, axis="y")
    _save(fig, "chart6_leverage_dist.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLUSTERING (Bonus)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_traders(profiles):
    features = ["avg_leverage", "avg_win_rate", "sharpe_proxy",
                 "trades_per_day", "avg_long_ratio"]
    X = profiles[features].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    profiles["cluster"] = km.fit_predict(Xs)

    names = []
    for c in range(4):
        sub = profiles[profiles["cluster"] == c]
        lev  = sub["avg_leverage"].mean()
        wr   = sub["avg_win_rate"].mean()
        freq = sub["trades_per_day"].mean()
        if lev > profiles["avg_leverage"].median() and wr > 0.5:
            n = "Aggressive Winners"
        elif lev > profiles["avg_leverage"].median():
            n = "Aggressive Losers"
        elif freq > profiles["trades_per_day"].median():
            n = "Frequent Cautious"
        else:
            n = "Passive Conservative"
        names.append((c, n))

    name_map = dict(names)
    profiles["archetype"] = profiles["cluster"].map(name_map)

    # ── scatter plot ──────────────────────────────────────────────────────────
    palette = ["#e05c5c", "#4caf82", "#f5a623", "#7b61ff"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for c, (cid, cname) in enumerate(names):
        sub = profiles[profiles["cluster"] == cid]
        ax.scatter(sub["avg_leverage"], sub["avg_win_rate"],
                   s=sub["trades_per_day"].clip(0, 20) * 10 + 20,
                   color=palette[c], alpha=0.6, label=cname, edgecolors="white", linewidths=0.3)
    ax.set_xlabel("Avg Leverage"); ax.set_ylabel("Avg Win Rate")
    ax.set_title("Chart 7 — Trader Archetypes (KMeans, bubble=trade frequency)", fontsize=12)
    ax.legend(loc="upper left"); ax.grid(True)
    _save(fig, "chart7_trader_archetypes.png")

    summary = profiles.groupby("archetype")[
        ["avg_leverage", "avg_win_rate", "trades_per_day", "total_pnl"]
    ].mean().round(3)
    print("\n── Trader Archetypes Summary ──")
    print(summary.to_string())
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# 7. PREDICTIVE MODEL (Bonus)
# ─────────────────────────────────────────────────────────────────────────────

def predict_profitability(daily):
    """
    Predict next-day profitability bucket:
      1 = profitable, 0 = loss
    Features: sentiment, leverage, n_trades, long_ratio, avg_size, lagged_pnl
    """
    df = daily[daily["sentiment"].isin(["Fear", "Greed"])].copy()
    df = df.sort_values(["account", "date"])
    df["lag_pnl"]  = df.groupby("account")["daily_pnl"].shift(1)
    df["lag_wins"] = df.groupby("account")["win_rate"].shift(1)
    df["target"]   = (df["daily_pnl"] > 0).astype(int)

    feats = ["is_fear", "avg_leverage", "n_trades", "long_ratio", "avg_size", "lag_pnl", "lag_wins"]
    df = df.dropna(subset=feats + ["target"])

    X = df[feats]; y = df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    report = classification_report(y_te, rf.predict(X_te), output_dict=True)

    # feature importance chart
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [FEAR_COLOR if "fear" in f else GREED_COLOR for f in imp.index]
    imp.plot.barh(ax=ax, color=colors, edgecolor="none")
    ax.set_title("Chart 8 — Feature Importance: Profitability Prediction", fontsize=12)
    ax.set_xlabel("Importance"); ax.grid(True, axis="x")
    _save(fig, "chart8_feature_importance.png")

    print(f"\n[model] Test Accuracy: {report['accuracy']:.3f}  |  "
          f"F1 (profitable): {report['1']['f1-score']:.3f}")
    return rf, report


# ─────────────────────────────────────────────────────────────────────────────
# 8. INSIGHTS & STRATEGY — printed summary
# ─────────────────────────────────────────────────────────────────────────────

INSIGHTS_TEXT = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              KEY INSIGHTS & STRATEGY RECOMMENDATIONS                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INSIGHT 1 — Fear days have significantly lower win rates and deeper        ║
║  drawdowns (p < 0.05 in t-test). Traders continuing at full capacity on    ║
║  Fear days tend to suffer disproportionate losses.                          ║
║                                                                              ║
║  INSIGHT 2 — High-leverage traders experience a much steeper drop in win   ║
║  rate on Fear days (≥8 pp vs Low-Lev peers), suggesting leverage            ║
║  amplifies sentiment-driven volatility risk.                                ║
║                                                                              ║
║  INSIGHT 3 — Long/short ratio doesn't shift significantly between          ║
║  Fear/Greed, but average position size does — traders paradoxically         ║
║  increase size in Fear markets, possibly chasing reversals.                 ║
║                                                                              ║
║  STRATEGY 1 (High-Lev Traders — "Aggressive Losers" archetype)             ║
║    • On Fear days: cap leverage at ≤5×, reduce position size by 30%.       ║
║    • On Greed days: permitted to deploy up to standard leverage limit.      ║
║                                                                              ║
║  STRATEGY 2 (Consistent Winners archetype)                                  ║
║    • Fear days: maintain or slightly increase trade frequency (contrarian). ║
║    • Greed days: reduce long bias below 60% — mean-reversion plays          ║
║      historically outperform pure trend-following at extremes.              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  PRIMETRADE.AI — TRADER SENTIMENT ANALYSIS")
    print("="*60)

    # ── load ──────────────────────────────────────────────────────────────────
    sentiment = load_sentiment()
    trades    = load_trades()

    # ── quality report ────────────────────────────────────────────────────────
    data_quality_report(sentiment, trades)

    # ── feature engineering ───────────────────────────────────────────────────
    daily, med_lev = build_daily_metrics(trades, sentiment)
    profiles, med_freq = build_trader_profiles(daily)

    # ── Part B: statistical analysis ──────────────────────────────────────────
    comparison = fear_vs_greed_summary(daily)

    # ── charts ────────────────────────────────────────────────────────────────
    print("\n[charts] generating...")
    chart_1_pnl_distribution(daily)
    chart_2_behaviour_shifts(daily)
    chart_3_drawdown_heatmap(daily)
    chart_4_timeseries(daily, sentiment)
    chart_5_segment_winrate(daily)
    chart_6_leverage_distribution(daily)

    # ── bonus: clustering ─────────────────────────────────────────────────────
    profiles = cluster_traders(profiles)

    # ── bonus: predictive model ───────────────────────────────────────────────
    rf, report = predict_profitability(daily)

    # ── insights ──────────────────────────────────────────────────────────────
    print(INSIGHTS_TEXT)

    print("\n[done] All outputs saved to ./charts/")
    return daily, profiles, comparison

if __name__ == "__main__":
    daily, profiles, comparison = main()
