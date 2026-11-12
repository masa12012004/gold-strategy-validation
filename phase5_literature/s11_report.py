"""Final equity curves (small multiples by paper family + hybrids) and the summary table."""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import RES, FIG, metrics, ANN

SURF, INK, INK2, GRID, BENCH = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#9a9992"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DAILY_BAR = {"s01_f2f_long_only", "s01_f2f_long_short", "s00_buy_hold_daily", "h5_vol_controlled_gold"}

PANELS = [
    ("Forecast-to-Fill trend/momentum regime, daily (paper 11)",
     [("s01_f2f_long_only", "F2F long-only"), ("s01_f2f_long_short", "F2F long/short")]),
    ("Cubic trend-reversion law and TSMOM, H1 (papers 46, 22, 39)",
     [("s02_cubic_trend_reversion", "Cubic law"), ("s02_cubic_thresh", "Cubic, strong only"), ("s02_tsmom", "TSMOM")]),
    ("HMM intraday momentum, M5 (paper 18)",
     [("s03_hmm_momentum", "HMM momentum"), ("s03_hmm_sideinfo", "HMM + side info")]),
    ("Drift-burst fade, 5-second statistic (paper 44)",
     [("s04_drift_burst_fade", "Train-selected spec"), ("s04_drift_burst_paper_default", "Paper default")]),
    ("Hurst switch and Q-learning, H1 (paper 30)",
     [("s05_sma_momentum", "SMA cross"), ("s05_bollinger_mr", "Bollinger MR"),
      ("s05_hurst_switch", "Hurst switch"), ("s05_q_learning", "Q-learning")]),
    ("OU optimal stopping, M5 (papers 40, 10)",
     [("s06_ou_optimal_stopping", "OU double stopping"), ("s06_ou_tvar_gated", "+ TVAR monitor")]),
    ("Trend predictors, H1 (papers 45, 6, 12, 32)",
     [("s07_xgb_trend_predictor", "XGBoost"), ("s07_logit_trend_predictor", "Logistic")]),
    ("Deep Momentum Network, Sharpe loss, H1 (paper 39)",
     [("s08_dmn_lstm", "LSTM"), ("s08_dmn_lstm_cpd", "LSTM + changepoint")]),
    ("Policy-gradient agent, M15 intraday (papers 9, 8)",
     [("s09_rl_positional", "With positional state"), ("s09_rl_no_positional", "Without")]),
]
HYB = [("h1_slowmom_x_cubic", "H1 regime x cubic"), ("h2_tats_cubic_xgb", "H2 TATS cubic+XGB"),
       ("h3_equal_risk_all", "H3 equal-risk, all 13"), ("h4_walkforward_ensemble", "H4 walk-forward ensemble")]


def daily(name):
    b = pd.read_parquet(f"{RES}/{name}.parquet")
    r = b["net"] if "tday" not in b.columns else b.groupby("tday")["net"].sum()
    if name in DAILY_BAR:
        r = r.shift(1).fillna(0.0)
    r.index = pd.DatetimeIndex(r.index)
    return r


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))


def draw(ax, title, series, bh):
    style(ax)
    eqb = (1 + bh).cumprod()
    ax.plot(eqb.index, eqb.values, color=BENCH, lw=1.2, label="Buy & hold gold")
    for (name, lab), col in zip(series, PAL):
        r = daily(name)
        eq = (1 + r).cumprod()
        sh = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0
        ax.plot(eq.index, eq.values, color=col, lw=2, label=f"{lab}  ({eq.iloc[-1]-1:+.0%}, Sharpe {sh:.2f})")
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=INK2)


if __name__ == "__main__":
    bh = daily("s00_buy_hold_daily")
    fig, axes = plt.subplots(5, 2, figsize=(15, 21), facecolor=SURF)
    for ax, (title, series) in zip(axes.ravel(), PANELS):
        draw(ax, title, series, bh)
    draw(axes.ravel()[9], "Hybrid models (larger in equity_hybrids.png)", HYB, bh)
    fig.suptitle("XAUUSD literature strategies — out-of-sample equity, net of real spread + 0.2 bp slippage\n"
                 "Jul 2025 – Oct 2026, growth of 1.0; every model fitted only on earlier data",
                 x=0.06, ha="left", fontsize=14, color=INK, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(f"{FIG}/equity_all_strategies.png", dpi=130)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9.5), facecolor=SURF, height_ratios=[3, 1.3], sharex=True)
    draw(a1, "Hybrid models vs buy & hold — out-of-sample, net of costs (growth of 1.0)", HYB, bh)
    a1.legend(fontsize=9.5, frameon=False, loc="upper left", labelcolor=INK2)
    style(a2)
    a2.lines[-1].remove()
    for (name, lab), col in zip(HYB, PAL):
        eq = (1 + daily(name)).cumprod()
        a2.plot(eq.index, (eq / eq.cummax() - 1).values, color=col, lw=1.5)
    eqb = (1 + bh).cumprod()
    a2.plot(eqb.index, (eqb / eqb.cummax() - 1).values, color=BENCH, lw=1.2)
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a2.set_ylim(-0.32, 0.01)
    a2.set_title("Drawdown", loc="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/equity_hybrids.png", dpi=130)

    # ---- summary table with beta to gold
    names = ["s00_buy_hold_daily"] + [n for _, s in PANELS for n, _ in s] + [n for n, _ in HYB] + ["h5_vol_controlled_gold"]
    rows = []
    for n in names:
        r = daily(n)
        m = metrics(r, n)
        x = pd.concat([r, bh], axis=1).fillna(0.0)
        m["beta_gold"] = np.cov(x.iloc[:, 0], x.iloc[:, 1])[0, 1] / x.iloc[:, 1].var()
        m["alpha_ann"] = (x.iloc[:, 0].mean() - m["beta_gold"] * x.iloc[:, 1].mean()) * ANN
        h = len(r) // 2
        for k, part in (("sharpe_h1", r.iloc[:h]), ("sharpe_h2", r.iloc[h:])):
            m[k] = part.mean() / part.std() * np.sqrt(ANN) if part.std() > 0 else 0.0
        rows.append(m)
    t = pd.DataFrame(rows).drop(columns=["turnover_per_day", "avg_abs_pos", "sharpe_gross"])
    t.to_csv(f"{RES}/SUMMARY.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
