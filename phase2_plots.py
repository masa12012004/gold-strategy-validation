"""
Phase 2 — equity-curve plots (in-sample + OOS, per-signal, drawdown).
Saves PNGs into clean_Data/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from phase2_common import (BtCfg, CAPITAL_0, load_features, run_backtest, trade_metrics)
from phase2b_signals import (sig_drift_burst_fade, sig_ou_vwap_fade,
                              sig_london_breakout, sig_cubic_revert)
from phase2c_strategy import ensemble

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


def equity_from(df, sig, cfg):
    _, eq = run_backtest(df, sig, df["atr14"].to_numpy(), cfg=cfg)
    return eq[1:]  # align to bar times


def main() -> None:
    df = load_features().sort_values("dt_et").reset_index(drop=True)
    df = df.dropna(subset=["atr14", "close"]).reset_index(drop=True)
    t = pd.to_datetime(df["dt_et"])
    dates = t.dt.date
    unique_days = np.array(sorted(dates.unique()))
    oos_cut = unique_days[int(len(unique_days) * 0.80)]
    oos_ts  = pd.Timestamp(oos_cut).tz_localize(t.dt.tz)

    with open(CLEAN / "phase2_best_params.json") as f:
        best = json.load(f)
    weights = {k: float(v) for k, v in best["weights"].items()}
    params  = best["params"]
    cfg     = BtCfg(entry_thresh=0.35)

    # ===== plot 1: individual signals equity curves =====
    cfg_b = BtCfg(entry_thresh=0.3)
    sigs = {
        "A drift-burst fade":     sig_drift_burst_fade(df, 2.0),
        "B OU VWAP fade":         sig_ou_vwap_fade(df, 1.5),
        "C London ORB":           sig_london_breakout(df),
        "D cubic reversion":      sig_cubic_revert(df),
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, s in sigs.items():
        eq = equity_from(df, s, cfg_b)
        ax.plot(t, eq, lw=1.0, label=name)
    ax.axhline(CAPITAL_0, color="k", lw=0.5, ls="--")
    ax.set_title("Phase 2b — Individual signal equity (0.5% risk/trade, 5-min bars)")
    ax.set_ylabel("Equity ($)"); ax.set_xlabel("")
    ax.legend(loc="lower left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); fig.tight_layout()
    p1 = CLEAN / "phase2b_individual_equity.png"
    fig.savefig(p1, dpi=130); plt.close(fig)

    # ===== plot 2: ensemble (equal) vs walk-forward-best — with OOS line =====
    s_eq   = ensemble(df, {"A":0.25,"B":0.25,"C":0.25,"D":0.25}, {"T_entry":2.5,"z_entry":2.0})
    s_best = ensemble(df, weights, params)
    eq_eq   = equity_from(df, s_eq, BtCfg(entry_thresh=0.35))
    eq_best = equity_from(df, s_best, cfg)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios":[3, 1]})
    ax1.plot(t, eq_eq,   lw=1.2, color="#4c78a8", label="Equal-weight ensemble (Phase 2c)")
    ax1.plot(t, eq_best, lw=1.4, color="#e45756",
             label=f"Walk-forward best {weights}, T={params['T_entry']}, z={params['z_entry']}")
    ax1.axhline(CAPITAL_0, color="k", lw=0.5, ls="--")
    ax1.axhline(CAPITAL_0 * 1.08, color="green", lw=0.8, ls=":", label="+8% target")
    ax1.axhline(CAPITAL_0 * 0.90, color="red",   lw=0.8, ls=":", label="−10% hard limit")
    ax1.axvline(oos_ts, color="gray", lw=1.0, ls="--", label=f"OOS cut ({oos_cut})")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("Phase 2c/2d/2e — Ensemble equity curves (in-sample + out-of-sample)")
    ax1.legend(loc="upper left", fontsize=9)

    # drawdown panel
    def dd(eq):
        peak = np.maximum.accumulate(eq)
        return (eq - peak) / peak * 100
    ax2.fill_between(t, dd(eq_best), 0, color="#e45756", alpha=0.35,
                     label="Walk-forward best DD")
    ax2.fill_between(t, dd(eq_eq),   0, color="#4c78a8", alpha=0.25,
                     label="Equal-weight DD")
    ax2.axhline(-5,  color="orange", lw=0.6, ls=":", label="−5% daily hard")
    ax2.axhline(-10, color="red",    lw=0.6, ls=":", label="−10% total hard")
    ax2.axvline(oos_ts, color="gray", lw=1.0, ls="--")
    ax2.set_ylabel("Drawdown (%)"); ax2.set_xlabel("")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); fig.tight_layout()
    p2 = CLEAN / "phase2_ensemble_equity_dd.png"
    fig.savefig(p2, dpi=130); plt.close(fig)

    # ===== plot 3: histogram of daily PnL + 20-day rolling return (walk-forward best) =====
    trades, eq_full = run_backtest(df, s_best, df["atr14"].to_numpy(), cfg=cfg)
    if len(trades):
        trades["day"] = pd.to_datetime(trades["entry_time"]).dt.date
        dp = trades.groupby("day")["pnl"].sum()
        full_days = pd.date_range(df["dt_et"].min().date(), df["dt_et"].max().date(), freq="D")
        dp = dp.reindex([d.date() for d in full_days]).fillna(0)
        roll = dp.rolling(20).sum() / CAPITAL_0 * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(dp.values, bins=40, color="#4c78a8", edgecolor="white")
        ax1.axvline(0, color="k", lw=0.5)
        ax1.axvline(-CAPITAL_0 * 0.05, color="red", lw=0.8, ls=":", label="−5% daily limit")
        ax1.set_title(f"Daily PnL distribution ({(dp!=0).sum()} active days)")
        ax1.set_xlabel("$ per day"); ax1.set_ylabel("count")
        ax1.legend()

        ax2.plot(roll.index, roll.values, color="#e45756", lw=1.2)
        ax2.axhline(8, color="green", lw=0.8, ls=":", label="+8% target")
        ax2.axhline(0, color="k", lw=0.5)
        ax2.axhline(-10, color="red", lw=0.8, ls=":", label="−10% total limit")
        ax2.set_title("Rolling 20-trading-day return (%)")
        ax2.legend(); ax2.set_ylabel("%")
        fig.tight_layout()
        p3 = CLEAN / "phase2_dailypnl_rolling20d.png"
        fig.savefig(p3, dpi=130); plt.close(fig)

    print(f"Saved:\n  {p1}\n  {p2}\n  {p3}")

if __name__ == "__main__":
    main()
