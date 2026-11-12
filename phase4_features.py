"""
Phase 4b — ML feature set definition and causal derived features.

Two feature sets are defined because the Phase-2a/3 matrix contains a few columns
that were fitted on the whole sample (or on the first half), which leaks into any
backtest fold that sits inside the fitting window:

  LEAKY_COLS  — hmm_*/hmm3_* (Baum-Welch EM fitted on the full series),
                sigma_u / sigma_sv / diurnal_* / *_defl (diurnal profile fitted on
                the first half), month_prior (constant table taken from a paper
                whose sample overlaps ours).

  ML_FEATURES_STRICT  — everything causal: known using only data at or before the
                        bar. This is the honest set; headline numbers use it.
  ML_FEATURES         — strict + leaky, reported alongside so the cost of the
                        leakage is visible rather than hidden.

Derived features added here are all backward-looking rolling transforms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- columns straight out of the Phase-3 matrix that are causal -------------
BASE_CAUSAL = [
    "atr14",
    "vwap_dev", "vwap_dev_z",
    "db_T",
    "trend_strength",
    "hurst",
    "cpd_score", "cpd_loc",
    "sigma_fast", "sigma_slow", "sigma_hat",
    "ou_kalman_state", "ou_kalman_z",
    "spread_median",
    "bid_ticks", "ask_ticks",
    "dow", "minute_of_day",
    "ret",
]

# --- fitted on full sample / first half -> leakage risk ----------------------
LEAKY_COLS = [
    "hmm_calm", "hmm3_bull", "hmm3_bear", "hmm3_churn",
    "sigma_u", "sigma_sv",
    "diurnal_mu", "diurnal_sd",
    "vwap_dev_z_defl", "db_T_defl",
    "month_prior",
]

# --- derived, strictly causal ----------------------------------------------
DERIVED = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_26", "ret_78",
    "absret_6", "absret_26",
    "vol_ratio_fs",            # sigma_fast / sigma_slow  (vol regime)
    "atr_pct",                 # atr14 / close            (scale-free vol)
    "spread_bps",              # spread in bps of price
    "spread_z",                # spread vs its own 1-day rolling mean
    "tick_imb",                # (ask_ticks-bid_ticks)/(sum)  -> order-flow tilt
    "tick_imb_6",              # smoothed imbalance
    "tick_total_z",            # activity vs rolling norm
    "rng_pos",                 # where close sits in the bar's high-low range
    "rng_pos_6",
    "dist_hi_26", "dist_lo_26",   # distance to recent extremes in ATR units
    "tod_sin", "tod_cos",      # cyclical time of day
    "dow_sin", "dow_cos",
    "bars_since_sess",         # bars since session gap
    "vwap_dev_z_chg",          # momentum of the mean-reversion state
    "db_T_chg",
    "hurst_chg",
    "trend_x_hurst",           # interaction: trend only matters if persistent
    "z_x_calmvol",             # interaction: reversion only in low vol
]

ML_FEATURES_STRICT = BASE_CAUSAL + DERIVED
ML_FEATURES = ML_FEATURES_STRICT + LEAKY_COLS


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Append the DERIVED columns. All rolling windows look backward only."""
    d = df.copy()
    c = d["close"].astype("float64")
    logp = np.log(c)
    r1 = logp.diff()

    for h in (1, 3, 6, 12, 26, 78):
        d[f"ret_{h}"] = logp.diff(h)
    d["absret_6"] = r1.abs().rolling(6, min_periods=2).mean()
    d["absret_26"] = r1.abs().rolling(26, min_periods=5).mean()

    ss = d["sigma_slow"].replace(0, np.nan)
    d["vol_ratio_fs"] = d["sigma_fast"] / ss
    d["atr_pct"] = d["atr14"] / c

    sp = d["spread_median"].astype("float64")
    d["spread_bps"] = sp / c * 1e4
    sp_mu = sp.rolling(78, min_periods=20).mean()
    sp_sd = sp.rolling(78, min_periods=20).std()
    d["spread_z"] = (sp - sp_mu) / sp_sd.replace(0, np.nan)

    bt = d["bid_ticks"].astype("float64")
    at = d["ask_ticks"].astype("float64")
    tot = (bt + at).replace(0, np.nan)
    d["tick_imb"] = (at - bt) / tot
    d["tick_imb_6"] = d["tick_imb"].rolling(6, min_periods=2).mean()
    tt_mu = tot.rolling(78, min_periods=20).mean()
    tt_sd = tot.rolling(78, min_periods=20).std()
    d["tick_total_z"] = (tot - tt_mu) / tt_sd.replace(0, np.nan)

    rng = (d["high"] - d["low"]).astype("float64").replace(0, np.nan)
    d["rng_pos"] = (c - d["low"]) / rng
    d["rng_pos_6"] = d["rng_pos"].rolling(6, min_periods=2).mean()

    hi26 = d["high"].rolling(26, min_periods=5).max()
    lo26 = d["low"].rolling(26, min_periods=5).min()
    atr = d["atr14"].replace(0, np.nan)
    d["dist_hi_26"] = (hi26 - c) / atr
    d["dist_lo_26"] = (c - lo26) / atr

    mod = d["minute_of_day"].astype("float64")
    d["tod_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    d["tod_cos"] = np.cos(2 * np.pi * mod / 1440.0)
    dw = d["dow"].astype("float64")
    d["dow_sin"] = np.sin(2 * np.pi * dw / 7.0)
    d["dow_cos"] = np.cos(2 * np.pi * dw / 7.0)

    gap = pd.to_datetime(d["dt_et"]).diff().dt.total_seconds().div(60).fillna(5.0)
    new_sess = (gap > 5).to_numpy()
    since = np.zeros(len(d), dtype=np.int32)
    k = 0
    for i, flag in enumerate(new_sess):
        k = 0 if flag else k + 1
        since[i] = k
    d["bars_since_sess"] = since

    d["vwap_dev_z_chg"] = d["vwap_dev_z"].diff(3)
    d["db_T_chg"] = d["db_T"].diff(3)
    d["hurst_chg"] = d["hurst"].diff(6)

    d["trend_x_hurst"] = d["trend_strength"] * (d["hurst"] - 0.5)
    d["z_x_calmvol"] = d["vwap_dev_z"] / d["vol_ratio_fs"].replace(0, np.nan)

    return d


def clean_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Replace inf, clip extreme tails so tree splits and the net both behave."""
    Z = X.replace([np.inf, -np.inf], np.nan).astype("float64")
    lo = Z.quantile(0.001)
    hi = Z.quantile(0.999)
    Z = Z.clip(lower=lo, upper=hi, axis=1)
    return Z
