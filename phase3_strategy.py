"""
Phase 3 — Argos v2: lit-upgraded signals + friction-adjusted Kelly sizing + CPD gating.

Upgrades vs Phase 2c:
  - Signal B uses diurnal-deflated vwap_dev_z (G5) + Kalman-OU innovation (B1)
  - Signal A uses diurnal-deflated drift-burst T (G5)
  - New Signal E: CPD-gated momentum (G3)   — momentum only on recent changepoint
  - New Signal F: month-of-year prior      (F1)
  - 3-state HMM gating via posteriors      (G1)
  - Friction-adjusted Kelly sizing (R1/R2) replaces fixed 0.5% risk
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phase2_common import (BtCfg, run_backtest, trade_metrics, pretty, daily_pnl)

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


# ------------------------------------------------------- signals (upgraded)

def sig_A_drift_fade(df, T_entry=2.5):
    # Use raw db_T (not deflated) — deflation pushes signals into quiet hours
    T = df["db_T"].to_numpy()
    s = -np.sign(T) * np.minimum(np.abs(T) / 4.0, 1.0)
    return np.where(np.abs(T) >= T_entry, s, 0.0)

def sig_B_ou_fade(df, z_entry=1.5):
    # Primary: raw z-score (worked in Phase 2). Kalman innov used as boost, not gate.
    z = df["vwap_dev_z"].to_numpy()
    k = df["ou_kalman_z"].to_numpy()
    base = -np.tanh(z / 2.0)
    # Hurst gate (same as Phase 2b)
    hurst = df["hurst"].to_numpy()
    allow = (hurst < 0.60) & (np.abs(z) >= z_entry)
    out = np.where(allow, base, 0.0)
    # Boost magnitude when Kalman innovation aligns
    boost = np.where(np.sign(z) == np.sign(k), 1.2, 1.0)
    return np.clip(out * boost, -1, 1)

def sig_C_london_breakout(df):
    s = np.zeros(len(df))
    g = df.copy()
    g["date"] = pd.to_datetime(g["dt_et"]).dt.date
    mod_arr = g["minute_of_day"].to_numpy()
    hi_arr  = g["high"].to_numpy()
    lo_arr  = g["low"].to_numpy()
    cl_arr  = g["close"].to_numpy()
    for d, grp in g.groupby("date"):
        idx = grp.index.to_numpy()
        mod = mod_arr[idx]; hi = hi_arr[idx]; lo = lo_arr[idx]; cl = cl_arr[idx]
        in_rng = (mod >= 120) & (mod <= 134)
        if in_rng.sum() < 3: continue
        rng_hi = hi[in_rng].max(); rng_lo = lo[in_rng].min()
        in_win = (mod > 134) & (mod <= 180)
        for k, b in enumerate(in_win):
            if not b: continue
            if cl[k] > rng_hi: s[idx[k]] = +1.0
            if cl[k] < rng_lo: s[idx[k]] = -1.0
    return s * (df["hmm3_bull"].to_numpy() > 0.5)   # trade only in bull state

def sig_D_cubic(df):
    T = df["trend_strength"].to_numpy()
    ret = df["ret"].shift(-1).to_numpy()
    split = len(df) // 2
    mask = np.isfinite(T[:split]) & np.isfinite(ret[:split])
    X = np.column_stack([T[:split][mask], (T[:split][mask])**3])
    y = ret[:split][mask]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    pred = a * T - b * (T**3)
    sd = np.nanstd(pred)
    return np.clip(pred / (3 * sd), -1, 1) if sd > 0 else np.zeros_like(pred)

def sig_E_cpd_momentum(df, cpd_min=2.5, bars_in=20):
    """Momentum trade right after a changepoint, in the direction of post-cp drift."""
    cpd = df["cpd_score"].to_numpy()
    loc = df["cpd_loc"].to_numpy()
    r1  = df["ret"].to_numpy()
    s = np.zeros(len(df))
    # recent CP: loc <= bars_in and historical cpd_score at loc>= cpd_min
    recent = (loc > 0) & (loc <= bars_in)
    # short-term drift = sum of last 5 log-returns
    drift = pd.Series(r1).rolling(5).sum().to_numpy()
    sgn = np.sign(drift)
    strength = np.minimum(np.abs(drift) * 500, 1.0)       # scale
    s = np.where(recent, sgn * strength, 0.0)
    # only in calm / bull regime (avoid bear-churn)
    bull = df["hmm3_bull"].to_numpy()
    s = np.where(bull > 0.4, s, 0.0)
    return s

def sig_F_month(df):
    m = df["month_prior"].to_numpy()
    scale = np.nanstd(m)
    return np.sign(m) * np.minimum(np.abs(m) / (3 * scale + 1e-18), 1.0) if scale > 0 else np.zeros_like(m)


# ------------------------------------------------------- ensemble + risk

def ensemble3(df, weights, params):
    sigs = {
        "A": sig_A_drift_fade(df,     params.get("T_entry", 2.5)),
        "B": sig_B_ou_fade(df,        params.get("z_entry", 1.8)),
        "C": sig_C_london_breakout(df),
        "D": sig_D_cubic(df),
        "E": sig_E_cpd_momentum(df,   params.get("cpd_min", 2.5)),
        "F": sig_F_month(df),
    }
    s = np.zeros(len(df))
    wsum = sum(abs(v) for v in weights.values())
    if wsum == 0: return s
    for k, w in weights.items():
        s += w * sigs.get(k, 0.0)
    s = s / wsum

    # 3-state HMM gate: block trades when hmm3_bear > 0.5 (regime risk-off)
    bear = df["hmm3_bear"].to_numpy()
    s = np.where(bear > 0.5, 0.0, s)

    # Spread filter
    if {"ask_close","bid_close"}.issubset(df.columns):
        spread = (df["ask_close"] - df["bid_close"]).to_numpy()
    else:
        spread = df["spread_median"].to_numpy()
    sp_q = pd.Series(spread).rolling(2000, min_periods=200).quantile(0.95).to_numpy()
    s = np.where(spread <= np.where(np.isnan(sp_q), np.inf, sp_q), s, 0.0)
    # New-session cooldown
    gap = pd.to_datetime(df["dt_et"]).diff().dt.total_seconds().div(60).fillna(1).to_numpy()
    new_sess = gap > 5
    cool = np.zeros(len(df), dtype=bool); k = 0
    for i, x in enumerate(new_sess):
        if x: k = 3
        if k > 0: cool[i] = True; k -= 1
    s = np.where(cool, 0.0, s)
    return s


def friction_kelly_sizing(df: pd.DataFrame, signal: np.ndarray,
                           sigma_star_daily: float = 0.12,
                           bars_per_day: float = 78.0,
                           w_max: float = 1.5,
                           kelly_frac: float = 1.0) -> np.ndarray:
    """
    Per-bar vol-target multiplier applied to the fixed risk_per_trade.
    Returns a scalar in [0.3, 2.0] mapping to size scaling.
      sigma*_bar = sigma_star_daily / sqrt(252 * bars_per_day)
      w_vol      = min(W_max, sigma* / sigma_hat)     — scale up in quiet, down in loud
    """
    sigma_star = sigma_star_daily / np.sqrt(252 * bars_per_day)
    sigma_hat  = df["sigma_hat"].to_numpy()
    med = np.nanmedian(sigma_hat[sigma_hat > 0])
    sigma_hat  = np.where(sigma_hat <= 0, med, sigma_hat)
    w_vol = np.minimum(w_max, sigma_star / sigma_hat)
    # Normalize around 1.0 so mean size ≈ unchanged vs. Phase 2
    w_vol = w_vol / np.nanmedian(w_vol)
    return kelly_frac * np.clip(w_vol, 0.3, 2.0)


# ------------------------------------------------------- backtest driver

def run_phase3(df: pd.DataFrame, weights: dict, params: dict, cfg: BtCfg,
               label: str = "argos_v2") -> dict:
    s = ensemble3(df, weights, params)
    atr = df["atr14"].to_numpy()

    # Vol-targeted size overlay: scale ATR (bigger ATR -> smaller size for same risk)
    size_scale = friction_kelly_sizing(df, s)
    atr_eff = atr / size_scale    # inflating ATR shrinks size; shrinking ATR grows it
    trades, eq = run_backtest(df, s, atr_eff, cfg=cfg)
    m = trade_metrics(trades, eq)
    m["label"] = label
    print(f"[{label}]"); print(pretty(m))
    if len(trades):
        print(f"  exits: {trades['reason'].value_counts().to_dict()}")
        dp = daily_pnl(trades)
        print(f"  worst day: ${dp.min():+,.1f}  best day: ${dp.max():+,.1f}  days traded: {len(dp)}")
    m["_trades"] = trades; m["_eq"] = eq
    return m


def main() -> None:
    df = pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet")
    df = df.sort_values("dt_et").reset_index(drop=True)
    df = df.dropna(subset=["atr14","close"]).reset_index(drop=True)
    print(f"Rows: {len(df):,}  period: {df['dt_et'].min()} -> {df['dt_et'].max()}\n")

    cfg = BtCfg(entry_thresh=0.25, risk_per_trade=0.005,
                stop_k=1.8, target_k=2.8, max_trades_day=8,
                daily_soft=0.03, total_soft=0.07)

    runs = [
        ("equal_6",   {"A":0.17,"B":0.17,"C":0.17,"D":0.17,"E":0.16,"F":0.16}),
        ("B+E_heavy", {"A":0.10,"B":0.35,"C":0.05,"D":0.10,"E":0.30,"F":0.10}),
        ("A+B+E",     {"A":0.30,"B":0.35,"C":0.00,"D":0.10,"E":0.25,"F":0.00}),
        ("no_cpd",    {"A":0.30,"B":0.35,"C":0.10,"D":0.25,"E":0.00,"F":0.00}),
    ]
    params = {"T_entry": 2.5, "z_entry": 1.8, "cpd_min": 2.5}
    results = []
    for name, w in runs:
        results.append(run_phase3(df, w, params, cfg, label=name))

    print("\n--- Phase 3 summary ---")
    for m in results:
        print(f"{m['label']:12s} {pretty(m)}")


if __name__ == "__main__":
    main()
