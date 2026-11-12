"""
Phase 2c — Ensemble strategy with risk governor.

Combines the four signals via a weight vector w (and per-signal activation
thresholds). Runs through the Phase-2 backtest with the full risk layer
(daily soft stop, total soft stop, max trades/day, spread filter).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase2_common import (BtCfg, load_features, run_backtest, pretty, trade_metrics,
                            daily_pnl)
from phase2b_signals import (sig_ou_vwap_fade, sig_drift_burst_fade,
                              sig_cubic_revert, sig_london_breakout)


def build_signals(df: pd.DataFrame, params: dict) -> dict:
    return {
        "A": sig_drift_burst_fade(df, params.get("T_entry", 2.5)),
        "B": sig_ou_vwap_fade(df,     params.get("z_entry", 2.0)),
        "C": sig_london_breakout(df),
        "D": sig_cubic_revert(df),
    }

def ensemble(df: pd.DataFrame, weights: dict, params: dict) -> np.ndarray:
    sigs = build_signals(df, params)
    s = np.zeros(len(df))
    wsum = sum(abs(v) for v in weights.values())
    if wsum == 0: return s
    for k, w in weights.items():
        s += w * sigs[k]
    s = s / wsum
    # spread filter: zero signal when spread > its rolling p95
    if "spread_close" in df.columns:
        spread = df["spread_close"].to_numpy()
    elif {"ask_close","bid_close"}.issubset(df.columns):
        spread = (df["ask_close"] - df["bid_close"]).to_numpy()
    else:
        spread = np.full(len(df), 0.6)
    sp_q = pd.Series(spread).rolling(2000, min_periods=200).quantile(0.95).to_numpy()
    allow_spread = spread <= np.where(np.isnan(sp_q), np.inf, sp_q)
    s = np.where(allow_spread, s, 0.0)
    # new-session cooldown: suppress the first 3 bars
    gap = pd.to_datetime(df["dt_et"]).diff().dt.total_seconds().div(60).fillna(1).to_numpy()
    new_sess = gap > 5
    cool = np.zeros(len(df), dtype=bool)
    k = 0
    for i, x in enumerate(new_sess):
        if x: k = 3
        if k > 0: cool[i] = True; k -= 1
    s = np.where(cool, 0.0, s)
    return s


def run(df: pd.DataFrame, weights: dict, params: dict, cfg: BtCfg,
        label: str = "ensemble") -> dict:
    s = ensemble(df, weights, params)
    atr = df["atr14"].to_numpy()
    trades, eq = run_backtest(df, s, atr, cfg=cfg)
    m = trade_metrics(trades, eq)
    m["label"] = label
    print(f"[{label}]")
    print(pretty(m))
    if len(trades):
        print(f"  exits: {trades['reason'].value_counts().to_dict()}")
        dp = daily_pnl(trades)
        worst_day = dp.min(); best_day = dp.max()
        print(f"  worst day: ${worst_day:+,.1f}  best day: ${best_day:+,.1f}  "
              f"days traded: {len(dp)}")
    return m


def main() -> None:
    df = load_features().sort_values("dt_et").reset_index(drop=True)
    df = df.dropna(subset=["atr14","close"]).reset_index(drop=True)
    print(f"Rows: {len(df):,}  period: {df['dt_et'].min()} -> {df['dt_et'].max()}\n")

    cfg = BtCfg(entry_thresh=0.35, risk_per_trade=0.005,
                stop_k=1.8, target_k=2.8, max_trades_day=6,
                daily_soft=0.03, total_soft=0.07)

    # a few hand-picked weights to try
    runs = [
        ("equal",        {"A":0.25, "B":0.25, "C":0.25, "D":0.25}),
        ("B-heavy",      {"A":0.10, "B":0.55, "C":0.15, "D":0.20}),
        ("A-heavy",      {"A":0.55, "B":0.20, "C":0.10, "D":0.15}),
        ("B+D only",     {"A":0.0,  "B":0.60, "C":0.0,  "D":0.40}),
    ]
    params = {"T_entry": 2.5, "z_entry": 2.0}
    results = []
    for name, w in runs:
        m = run(df, w, params, cfg, label=name)
        results.append(m)
    print("\n--- summary ---")
    for m in results:
        print(f"{m.get('label','?'):10s} {pretty(m)}")


if __name__ == "__main__":
    main()
