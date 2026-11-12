"""
Phase 2a — Build the feature / regime matrix on the 5-min grid.

Reads : clean_Data/xauusd_m5.parquet
Writes: clean_Data/xauusd_features_m5.parquet

Produces (every 5-min bar):
    hurst        rolling Hurst exponent (per-day, R/S on the day's 5-min returns)
    hmm_state    2-state Gaussian HMM state on 5-min returns (0=calm, 1=churn)
    hmm_calm     1 if calm state
    atr14        ATR_14 in $/oz
    vwap_dev     log-price minus 60-min VWAP
    vwap_dev_z   z-score of vwap_dev vs its rolling 288-bar std
    ou_kappa     per-day OU mean-reversion speed (on log-price)
    ou_mu        per-day OU mean
    ou_sigma     per-day OU diffusion
    ou_halflife  ln(2)/kappa in bars
    db_T         drift-burst T-statistic (exponential kernel, h=6 bars)
    trend_strength  (p - MA_slow)/sigma
    diurnal_mu   expected log-return by minute_of_day (fit on first half)
    diurnal_sd   expected sd by minute_of_day
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


# ---------------------------------------------------------- feature functions

def hurst_rs(x: np.ndarray) -> float:
    """Rescaled-range Hurst estimator."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20: return np.nan
    lags = [8, 16, 32, min(64, n//2)]
    rs = []
    for lag in lags:
        if lag >= n: continue
        segs = n // lag
        if segs < 2: continue
        vals = []
        for k in range(segs):
            seg = x[k*lag:(k+1)*lag]
            m = seg.mean()
            z = np.cumsum(seg - m)
            r = z.max() - z.min()
            s = seg.std(ddof=1)
            if s > 0 and r > 0: vals.append(r / s)
        if vals: rs.append((lag, np.mean(vals)))
    if len(rs) < 2: return np.nan
    xs = np.log([r[0] for r in rs])
    ys = np.log([r[1] for r in rs])
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)

def ou_mle(x: np.ndarray, dt: float = 1.0) -> tuple[float, float, float]:
    """MLE of OU parameters on a path x: dX = kappa (mu - X) dt + sigma dW."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 40: return (np.nan, np.nan, np.nan)
    Sx  = x[:-1].sum(); Sy = x[1:].sum()
    Sxx = (x[:-1]**2).sum(); Sxy = (x[:-1]*x[1:]).sum(); Syy = (x[1:]**2).sum()
    N = n - 1
    denom = (N*Sxx - Sx*Sx) - (N*Sxy - Sx*Sy)
    if abs(denom) < 1e-12: return (np.nan, np.nan, np.nan)
    mu = (Sy*Sxx - Sx*Sxy) / (N*(Sxx - Sxy) - (Sx*Sx - Sx*Sy))
    a  = (Sxy - mu*Sx - mu*Sy + N*mu*mu) / (Sxx - 2*mu*Sx + N*mu*mu)
    if a <= 0 or a >= 1: return (np.nan, np.nan, np.nan)
    kappa = -np.log(a) / dt
    sigma2_hat = (1/N) * (Syy - 2*a*Sxy + a*a*Sxx - 2*mu*(1-a)*(Sy - a*Sx) + N*mu*mu*(1-a)**2)
    if sigma2_hat <= 0: return (float(kappa), float(mu), np.nan)
    sigma = np.sqrt(2*kappa*sigma2_hat / (1 - a*a))
    return (float(kappa), float(mu), float(sigma))

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def vwap_dev(df: pd.DataFrame, window_bars: int = 12) -> pd.Series:
    # 12 bars on 5-min = 1h
    rolling_mean = df["close"].rolling(window_bars, min_periods=4).mean()
    return np.log(df["close"]) - np.log(rolling_mean)

def drift_burst_T(logp: pd.Series, h: int = 6) -> pd.Series:
    """Exponential-kernel drift-burst T-statistic with bandwidth h bars."""
    dx = logp.diff()
    # exponential weights (decaying backwards)
    w = np.exp(-np.arange(h, 0, -1) / h)[::-1]
    w = w / w.sum()
    mu_hat    = dx.rolling(h).apply(lambda r: np.nansum(r * w), raw=True)
    sigma_hat = dx.rolling(h).apply(lambda r: np.sqrt(np.nansum((r**2) * w)), raw=True)
    return np.sqrt(h) * mu_hat / sigma_hat.replace(0, np.nan)

def trend_strength(logp: pd.Series, n_slow: int = 288) -> pd.Series:
    ma = logp.rolling(n_slow, min_periods=24).mean()
    sd = logp.rolling(n_slow, min_periods=24).std()
    return (logp - ma) / sd.replace(0, np.nan)


# ---------------------------------------------------------------------- main

def main() -> None:
    print("Loading m5 ...", flush=True)
    df = pd.read_parquet(CLEAN / "xauusd_m5.parquet").sort_values("dt_et").reset_index(drop=True)
    df["dt_et"] = pd.to_datetime(df["dt_et"])
    df["date"] = df["dt_et"].dt.date
    df["minute_of_day"] = df["dt_et"].dt.hour * 60 + df["dt_et"].dt.minute
    df["dow"] = df["dt_et"].dt.dayofweek
    df["logp"] = np.log(df["close"])
    df["ret"]  = df["logp"].diff()

    print("ATR + VWAP deviation + drift burst T + trend strength ...", flush=True)
    df["atr14"] = atr(df, 14)
    df["vwap_dev"] = vwap_dev(df, 12)
    df["vwap_dev_z"] = (df["vwap_dev"] - df["vwap_dev"].rolling(288, min_periods=24).mean()
                      ) / df["vwap_dev"].rolling(288, min_periods=24).std().replace(0, np.nan)
    df["db_T"] = drift_burst_T(df["logp"], h=6)
    df["trend_strength"] = trend_strength(df["logp"], n_slow=288)

    print("Per-day Hurst and OU fit ...", flush=True)
    hurst_map = {}
    ou_map    = {}
    for d, g in df.groupby("date"):
        rets = g["ret"].dropna().to_numpy()
        logp = g["logp"].dropna().to_numpy()
        hurst_map[d] = hurst_rs(rets)
        ou_map[d]    = ou_mle(logp)
    df["hurst"]    = df["date"].map(hurst_map).astype(float)
    df["ou_kappa"] = df["date"].map({k: v[0] for k, v in ou_map.items()}).astype(float)
    df["ou_mu"]    = df["date"].map({k: v[1] for k, v in ou_map.items()}).astype(float)
    df["ou_sigma"] = df["date"].map({k: v[2] for k, v in ou_map.items()}).astype(float)
    df["ou_halflife"] = np.log(2.0) / df["ou_kappa"].replace(0, np.nan)

    print("Diurnal priors (fit on first half) ...", flush=True)
    split = df["dt_et"].quantile(0.5)
    train = df[df["dt_et"] <= split]
    di = train.groupby("minute_of_day")["ret"].agg(["mean", "std"]).rename(
        columns={"mean":"diurnal_mu", "std":"diurnal_sd"}).reset_index()
    df = df.merge(di, on="minute_of_day", how="left")

    print("Fitting 2-state Gaussian HMM on 5-min returns ...", flush=True)
    r = df["ret"].fillna(0).to_numpy().reshape(-1, 1)
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=50, random_state=7)
    hmm.fit(r)
    states = hmm.predict(r)
    var = hmm.covars_.reshape(-1)
    calm_state = int(np.argmin(var))
    df["hmm_state"] = states
    df["hmm_calm"]  = (states == calm_state).astype("int8")
    print(f"  HMM means={hmm.means_.ravel()}  vars={var}  calm_state={calm_state}")

    cols = ["dt_et","dt_utc","open","high","low","close","spread_close","spread_median",
            "spread_close_bps","bid_ticks","ask_ticks","session","dow","minute_of_day",
            "logp","ret","atr14","vwap_dev","vwap_dev_z","db_T","trend_strength",
            "hurst","ou_kappa","ou_mu","ou_sigma","ou_halflife",
            "diurnal_mu","diurnal_sd","hmm_state","hmm_calm"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out.to_parquet(CLEAN / "xauusd_features_m5.parquet", index=False, compression="zstd")
    print(f"\nFeature matrix: {len(out):,} rows x {len(out.columns)} cols")
    print(out.describe().T[["count","mean","std","min","50%","max"]].round(4).to_string())
    print(f"\nWrote {CLEAN/'xauusd_features_m5.parquet'}")


if __name__ == "__main__":
    main()
