"""
Phase 4g / Step 1 — Diagnose and fix the Phase-3 long bias.

Phase 3's candidate pool was 37,167 long vs 683 short. This module finds which
signal causes it and fixes it, because every later step inherits the pool.

Hypothesis: sig_F (month_prior) is CONSTANT within a calendar month. With Phase-3
weights {A:0.25, B:0.10, F:0.10}, F contributes 0.10/0.45 = 0.22 of the blended
signal, which on its own exceeds the 0.10 candidate threshold. So in every month
whose prior is positive (7 of 12), F fires a long on EVERY BAR of that month.
It is not a signal, it is a constant offset.

Fix: a seasonal prior is a *tilt*, not a standalone entry. Two changes:
  1. demean month_prior across the calendar year -> zero average tilt
  2. apply it multiplicatively to the signal that other components produce,
     so it can never originate a trade by itself
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_common import BtCfg, run_backtest, trade_metrics, pretty
from phase3_strategy import (sig_A_drift_fade, sig_B_ou_fade, sig_C_london_breakout,
                             sig_D_cubic, sig_E_cpd_momentum, sig_F_month,
                             friction_kelly_sizing)
from phase4_features import add_derived

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


def diagnose(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    sigs = {
        "A": sig_A_drift_fade(df, params.get("T_entry", 2.5)),
        "B": sig_B_ou_fade(df, params.get("z_entry", 1.8)),
        "C": sig_C_london_breakout(df),
        "D": sig_D_cubic(df),
        "E": sig_E_cpd_momentum(df, params.get("cpd_min", 2.5)),
        "F": sig_F_month(df),
    }
    rows = []
    for k, s in sigs.items():
        nz = s[np.isfinite(s) & (s != 0)]
        rows.append({
            "signal": k,
            "fire_rate_%": round(100 * len(nz) / len(s), 2),
            "long_%": round(100 * (nz > 0).mean(), 2) if len(nz) else np.nan,
            "short_%": round(100 * (nz < 0).mean(), 2) if len(nz) else np.nan,
            "mean": round(float(np.nanmean(s)), 4),
            "mean_abs": round(float(np.nanmean(np.abs(s))), 4),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- fixed ensemble

def sig_F_month_demeaned(df: pd.DataFrame) -> np.ndarray:
    """Seasonal tilt with the annual mean removed, in [-1, 1]."""
    m = df["month_prior"].to_numpy(dtype=np.float64)
    uniq = pd.Series(m).dropna().unique()
    m = m - np.nanmean(uniq)          # demean across the 12 monthly values
    sd = np.nanstd(uniq)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(m)
    return np.clip(m / (2 * sd), -1.0, 1.0)


def ensemble3_fixed(df: pd.DataFrame, weights: dict, params: dict,
                    season_strength: float = 0.25) -> np.ndarray:
    """
    Phase-3 ensemble with the seasonal term demoted from an additive signal to a
    multiplicative tilt. Everything else is unchanged from phase3_strategy.
    """
    sigs = {
        "A": sig_A_drift_fade(df, params.get("T_entry", 2.5)),
        "B": sig_B_ou_fade(df, params.get("z_entry", 1.8)),
        "C": sig_C_london_breakout(df),
        "D": sig_D_cubic(df),
        "E": sig_E_cpd_momentum(df, params.get("cpd_min", 2.5)),
    }
    s = np.zeros(len(df))
    wsum = sum(abs(weights.get(k, 0.0)) for k in sigs)
    if wsum == 0:
        return s
    for k, sg in sigs.items():
        s += weights.get(k, 0.0) * sg
    s = s / wsum

    # seasonal tilt: scales conviction, cannot create a trade on its own
    tilt = 1.0 + season_strength * sig_F_month_demeaned(df)
    s = np.clip(s * tilt, -1.0, 1.0)

    bear = df["hmm3_bear"].to_numpy()
    s = np.where(bear > 0.5, 0.0, s)

    if {"ask_close", "bid_close"}.issubset(df.columns):
        spread = (df["ask_close"] - df["bid_close"]).to_numpy()
    else:
        spread = df["spread_median"].to_numpy()
    sp_q = pd.Series(spread).rolling(2000, min_periods=200).quantile(0.95).to_numpy()
    s = np.where(spread <= np.where(np.isnan(sp_q), np.inf, sp_q), s, 0.0)

    gap = pd.to_datetime(df["dt_et"]).diff().dt.total_seconds().div(60).fillna(1).to_numpy()
    cool = np.zeros(len(df), dtype=bool)
    k = 0
    for i, x in enumerate(gap > 5):
        if x:
            k = 3
        if k > 0:
            cool[i] = True
            k -= 1
    s = np.where(cool, 0.0, s)
    return s


def balance(s: np.ndarray, thresh: float) -> dict:
    fired = s[np.isfinite(s) & (np.abs(s) >= thresh)]
    n = len(fired)
    return {"candidates": n,
            "long": int((fired > 0).sum()),
            "short": int((fired < 0).sum()),
            "long_pct": round(100 * (fired > 0).mean(), 1) if n else np.nan}


def main() -> None:
    with open(CLEAN / "phase3_best_params.json") as f:
        best3 = json.load(f)
    w3 = {k: float(v) for k, v in best3["weights"].items()}
    p3 = best3["params"]

    df = (pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet")
          .sort_values("dt_et").reset_index(drop=True))
    df = df.dropna(subset=["atr14", "close"]).reset_index(drop=True)
    df = add_derived(df)

    print("=" * 76)
    print("PER-SIGNAL DIAGNOSTIC (before fix)")
    print("=" * 76)
    print(diagnose(df, p3).to_string(index=False))

    mp = df.groupby("month")["month_prior"].first()
    print(f"\nmonth_prior by calendar month (x1e6): "
          f"{(mp * 1e6).round(2).to_dict()}")
    print(f"  positive months: {int((mp > 0).sum())}/12   "
          f"annual mean: {mp.mean()*1e6:+.3f}e-6  -> non-zero mean = constant long drift")

    from phase3_strategy import ensemble3
    s_old = ensemble3(df, w3, p3)
    s_new = ensemble3_fixed(df, w3, p3)

    print("\n" + "=" * 76)
    print("CANDIDATE-POOL BALANCE")
    print("=" * 76)
    for th in (0.10, 0.25):
        b_old = balance(s_old, th)
        b_new = balance(s_new, th)
        print(f"\nthreshold {th}:")
        print(f"  Phase 3 original : {b_old['candidates']:>7,} cands  "
              f"long={b_old['long']:>7,}  short={b_old['short']:>6,}  "
              f"({b_old['long_pct']}% long)")
        print(f"  Phase 3 FIXED    : {b_new['candidates']:>7,} cands  "
              f"long={b_new['long']:>7,}  short={b_new['short']:>6,}  "
              f"({b_new['long_pct']}% long)")

    # does the fix also help the equity curve?
    cfg = BtCfg(entry_thresh=0.25, risk_per_trade=0.005, stop_k=1.8, target_k=2.8,
                max_trades_day=8, daily_soft=0.03, total_soft=0.07)
    print("\n" + "=" * 76)
    print("BACKTEST, FULL SAMPLE")
    print("=" * 76)
    for lbl, s in (("phase3_original", s_old), ("phase3_fixed", s_new)):
        scale = friction_kelly_sizing(df, s)
        atr_eff = df["atr14"].to_numpy() / scale
        tr, eq = run_backtest(df, s, atr_eff, cfg=cfg)
        m = trade_metrics(tr, eq)
        print(f"{lbl:18s}{pretty(m)}")
        if len(tr):
            print(f"{'':18s}  long={int((tr['dir']>0).sum())}  "
                  f"short={int((tr['dir']<0).sum())}")

    np.save(CLEAN / "phase4_signal_fixed.npy", s_new)
    print(f"\nSaved fixed signal -> {CLEAN/'phase4_signal_fixed.npy'}")


if __name__ == "__main__":
    main()
