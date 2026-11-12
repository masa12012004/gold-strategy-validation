"""
Phase 2b — Single-signal backtests on the 5-min grid.

Each signal returns a numpy array `s[t] in [-1, +1]` and the full pipeline runs
via phase2_common.run_backtest with the shared risk layer DISABLED (we want raw
edge). We print per-signal metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase2_common import (BtCfg, load_features, run_backtest, pretty, trade_metrics)


def sig_ou_vwap_fade(df: pd.DataFrame, z_entry: float = 1.5) -> np.ndarray:
    z = df["vwap_dev_z"].to_numpy()
    s = -np.tanh(z / 2.0)
    # gate to mean-reverting-ish regime (relaxed, since gold trends in-sample)
    hurst = df["hurst"].to_numpy()
    allow = (hurst < 0.60) & (np.abs(z) >= z_entry)
    out = np.where(allow, s, 0.0)
    # kill signal in off / asia-thin session? keep all for now; cost model prevents abuse
    return out

def sig_drift_burst_fade(df: pd.DataFrame, T_entry: float = 2.0) -> np.ndarray:
    T = df["db_T"].to_numpy()
    s = -np.sign(T) * np.minimum(np.abs(T) / 4.0, 1.0)
    allow = np.abs(T) >= T_entry
    return np.where(allow, s, 0.0)

def sig_cubic_revert(df: pd.DataFrame) -> np.ndarray:
    """Cubic trend/reversion signal: a*T - b*T^3, a,b > 0 fitted on first half."""
    T = df["trend_strength"].to_numpy()
    ret = df["ret"].shift(-1).to_numpy()                 # next-bar return (target)
    split = len(df) // 2
    mask = np.isfinite(T[:split]) & np.isfinite(ret[:split])
    X = np.column_stack([T[:split][mask], (T[:split][mask])**3])
    y = ret[:split][mask]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    pred = a * T - b * (T**3)
    # scale to ~[-1,+1]
    sd = np.nanstd(pred)
    s = np.clip(pred / (3 * sd), -1, 1) if sd > 0 else np.zeros_like(pred)
    return s

def sig_london_breakout(df: pd.DataFrame) -> np.ndarray:
    """First 15 min (03 bars) of London session (02:00 ET) range, breakout in next 45 min."""
    s = np.zeros(len(df))
    g = df.copy()
    g["mod"] = g["minute_of_day"]
    g["date"] = pd.to_datetime(g["dt_et"]).dt.date
    for d, grp in g.groupby("date"):
        idx = grp.index.to_numpy()
        mod = grp["mod"].to_numpy()
        hi = grp["high"].to_numpy(); lo = grp["low"].to_numpy()
        cl = grp["close"].to_numpy()
        # London open at 02:00 ET → minute_of_day = 120. First 3 bars = 120..134
        in_range = (mod >= 120) & (mod <= 134)
        if in_range.sum() < 3: continue
        rng_hi = hi[in_range].max()
        rng_lo = lo[in_range].min()
        # Breakout window: 02:15..03:00 ET
        in_win = (mod > 134) & (mod <= 180)
        for k, b in enumerate(in_win):
            if not b: continue
            if cl[k] > rng_hi: s[idx[k]] = +1.0
            if cl[k] < rng_lo: s[idx[k]] = -1.0
    # gate trend regime
    hurst = df["hurst"].to_numpy()
    return np.where(hurst > 0.55, s, 0.0)


def run_one(df: pd.DataFrame, name: str, sig: np.ndarray, cfg: BtCfg) -> dict:
    atr = df["atr14"].to_numpy()
    trades, eq = run_backtest(df, sig, atr, cfg=cfg)
    m = trade_metrics(trades, eq)
    print(f"[{name:28s}]"); print(pretty(m))
    if len(trades):
        by_reason = trades["reason"].value_counts().to_dict()
        print(f"  exits: {by_reason}")
    return m


def main() -> None:
    df = load_features().sort_values("dt_et").reset_index(drop=True)
    df = df.dropna(subset=["atr14","close"]).reset_index(drop=True)
    cfg = BtCfg(entry_thresh=0.3)

    print(f"Rows: {len(df):,}  period: {df['dt_et'].min()} -> {df['dt_et'].max()}\n")

    results = {}
    results["A_drift_burst_fade"]  = run_one(df, "A_drift_burst_fade",  sig_drift_burst_fade(df, 2.0),  cfg)
    results["B_ou_vwap_fade"]      = run_one(df, "B_ou_vwap_fade",      sig_ou_vwap_fade(df, 1.5),      cfg)
    results["C_london_breakout"]   = run_one(df, "C_london_breakout",   sig_london_breakout(df),        cfg)
    results["D_cubic_revert"]      = run_one(df, "D_cubic_revert",      sig_cubic_revert(df),           cfg)

    print("\n--- summary ---")
    for k, m in results.items():
        print(f"{k:28s} {pretty(m)}")


if __name__ == "__main__":
    main()
