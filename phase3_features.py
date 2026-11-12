"""
Phase 3 — Extended feature matrix with literature-recommended regime & volatility layers.

Adds on top of Phase-2a features:
    hmm3_state    3-state Gaussian HMM state (0,1,2) on 5-min returns  [HMM Intraday Momentum]
    hmm3_bull     posterior P(state=bull) from forward algorithm
    hmm3_bear     posterior P(state=bear)
    hmm3_churn    posterior P(state=churn)

    cpd_score     online changepoint-detection severity on log-returns [Slow Momentum + CPD]
    cpd_loc       bars since last detected changepoint

    sigma_fast    2-factor SV: fast component (half-life ~12 bars = 1h)  [Bayesian 24h SV]
    sigma_slow    2-factor SV: slow component (half-life ~7 days)
    sigma_hat     combined vol forecast (fast * slow / ref) in return units

    sigma_u       deterministic diurnal vol component (by minute_of_day) [Diurnal-sufficient]
    sigma_sv      stochastic vol residual = sigma_hat / sigma_u

    vwap_dev_defl   vwap-dev deflated by sigma_u (homoskedastic)
    db_T_defl       drift-burst T deflated by sigma_u

    ou_kalman_resid  Kalman-filter residual of OU(vwap_dev)             [Dynamic MR Spreads]
    ou_kalman_z      standardized Kalman residual

    month_prior   expected daily log-return for this month-of-year      [Turn-of-Year effect]

Writes: clean_Data/xauusd_features3_m5.parquet
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


def hmm3_fit(returns: np.ndarray, seed: int = 11) -> tuple[np.ndarray, np.ndarray, dict]:
    """3-state Gaussian HMM. Returns (states, posteriors, labels dict mapping state_idx->role)."""
    r = returns.reshape(-1, 1)
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=seed)
    model.fit(r)
    states = model.predict(r)
    post = model.predict_proba(r)
    means = model.means_.ravel()
    vars_ = model.covars_.reshape(-1)
    # Label by (mean, variance): bull = highest mean, bear = lowest mean, churn = highest variance
    idx = np.argsort(means)
    bear, mid, bull = idx[0], idx[1], idx[2]
    churn = int(np.argmax(vars_))
    # If churn collides with bull/bear, keep tri-labeling as-is
    labels = {bull: "bull", bear: "bear", mid: "churn"}
    return states, post, {"labels": labels, "means": means, "vars": vars_}


def bocpd_score(logret: np.ndarray, window: int = 60, min_run: int = 5) -> np.ndarray:
    """
    Simplified online changepoint score: rolling two-sample t-statistic on a window
    split at the midpoint. Heavy-tails on scale → a changepoint. O(n) with rolling sums.
    Returns an array of |t-stat| as the CPD severity score.
    """
    n = len(logret)
    s = np.full(n, np.nan)
    if n < window + 2: return s
    half = window // 2
    # precompute cumulative sums
    cs  = np.cumsum(np.insert(logret, 0, 0.0))
    cs2 = np.cumsum(np.insert(logret**2, 0, 0.0))
    for i in range(window, n):
        a = i - window
        b = i - half
        c = i
        mu1 = (cs[b] - cs[a]) / half
        mu2 = (cs[c] - cs[b]) / half
        v1  = max((cs2[b] - cs2[a])/half - mu1*mu1, 1e-18)
        v2  = max((cs2[c] - cs2[b])/half - mu2*mu2, 1e-18)
        sp  = np.sqrt((v1 + v2) / half)
        s[i] = abs(mu1 - mu2) / (sp + 1e-18)
    # smooth lightly
    s = pd.Series(s).ewm(span=3, adjust=False).mean().to_numpy()
    return s


def two_factor_sv(logret: np.ndarray, hl_fast: int = 12, hl_slow: int = 2016) -> tuple[np.ndarray, np.ndarray]:
    """Two-factor EWMA stochastic-vol approximation. hl in bars."""
    alpha_f = 1 - np.exp(-np.log(2) / hl_fast)
    alpha_s = 1 - np.exp(-np.log(2) / hl_slow)
    r2 = logret**2
    sf = pd.Series(r2).ewm(alpha=alpha_f, adjust=False).mean().pow(0.5).to_numpy()
    ss = pd.Series(r2).ewm(alpha=alpha_s, adjust=False).mean().pow(0.5).to_numpy()
    return sf, ss


def kalman_ou(x: np.ndarray, kappa: float = 0.05, sigma_q: float = None,
              sigma_r: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple 1-D Kalman filter on OU state: x_t = (1-kappa) * x_{t-1} + w_t, obs = x_t + v_t.
    Returns (state, innovation). If sigma_q/sigma_r are None, estimate from data.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if sigma_q is None: sigma_q = np.nanstd(np.diff(x)) * np.sqrt(0.1)
    if sigma_r is None: sigma_r = np.nanstd(x) * 0.3
    Q = sigma_q ** 2
    R = sigma_r ** 2
    s = np.full(n, 0.0)
    P = 1.0
    innov = np.full(n, np.nan)
    a = 1 - kappa
    s[0] = x[0] if np.isfinite(x[0]) else 0.0
    for t in range(1, n):
        # predict
        s_pred = a * s[t - 1]
        P = a * a * P + Q
        # update
        if np.isfinite(x[t]):
            y = x[t] - s_pred
            S = P + R
            K = P / S
            s[t] = s_pred + K * y
            P = (1 - K) * P
            innov[t] = y / np.sqrt(max(S, 1e-18))
        else:
            s[t] = s_pred
    return s, innov


def main() -> None:
    print("Loading Phase-2a features ...", flush=True)
    df = pd.read_parquet(CLEAN / "xauusd_features_m5.parquet").sort_values("dt_et").reset_index(drop=True)
    df["dt_et"] = pd.to_datetime(df["dt_et"])
    df["date"] = df["dt_et"].dt.date
    df["month"] = df["dt_et"].dt.month

    r = df["ret"].fillna(0).to_numpy()
    logp = np.log(df["close"].to_numpy())
    logr = np.diff(logp, prepend=logp[0])

    print("3-state Gaussian HMM (bull/bear/churn) ...", flush=True)
    states, post, meta = hmm3_fit(r)
    labels = meta["labels"]
    df["hmm3_state"] = states
    for idx, name in labels.items():
        df[f"hmm3_{name}"] = post[:, idx].astype("float32")
    print(f"  means={meta['means'].round(6)}  vars={meta['vars'].round(8)}")
    print(f"  state->role: {labels}")

    print("Online changepoint-detection score (BOCPD-lite) ...", flush=True)
    df["cpd_score"] = bocpd_score(logr, window=60).astype("float32")
    df["cpd_loc"]   = pd.Series(np.where(df["cpd_score"] > 3.0, np.arange(len(df)), np.nan)).ffill().fillna(0)
    df["cpd_loc"]   = (np.arange(len(df)) - df["cpd_loc"].to_numpy()).astype("int32")

    print("Two-factor stochastic vol (fast=1h, slow=7d) ...", flush=True)
    sf, ss = two_factor_sv(logr, hl_fast=12, hl_slow=78*7)
    df["sigma_fast"] = sf.astype("float32")
    df["sigma_slow"] = ss.astype("float32")
    ref = np.nanmean(ss) if np.nanmean(ss) > 0 else 1e-6
    df["sigma_hat"]  = (sf * (ss / ref)).astype("float32")

    print("Diurnal decomposition sigma_u(minute_of_day), fit on first half ...", flush=True)
    split = df["dt_et"].quantile(0.5)
    train_mask = df["dt_et"] <= split
    diurnal = (df.loc[train_mask].assign(abs_r=lambda d: d["ret"].abs())
                .groupby("minute_of_day")["abs_r"].mean())
    df["sigma_u"] = df["minute_of_day"].map(diurnal.to_dict()).astype("float32")
    df["sigma_sv"] = (df["sigma_hat"] / df["sigma_u"].replace(0, np.nan)).astype("float32")

    print("Deflating VWAP-dev z and drift-burst T by sigma_u ...", flush=True)
    sd_u = df["sigma_u"].replace(0, np.nan)
    ref_u = sd_u.median()
    scale = (ref_u / sd_u).astype("float32")
    df["vwap_dev_z_defl"] = (df["vwap_dev_z"] * scale).astype("float32")
    df["db_T_defl"]       = (df["db_T"]        * scale).astype("float32")

    print("Kalman OU on vwap_dev ...", flush=True)
    vd = df["vwap_dev"].ffill().fillna(0).to_numpy()
    s_k, innov = kalman_ou(vd, kappa=0.05)
    df["ou_kalman_state"] = s_k.astype("float32")
    df["ou_kalman_z"]     = pd.Series(innov).astype("float32")

    print("Month-of-year prior (from turn-of-year paper) ...", flush=True)
    month_prior_usd = {1: 1.65, 2: -0.05, 3: -0.95, 4: 0.60, 5: 0.30, 6: -0.29,
                       7: 0.31, 8: 1.51, 9: 2.54, 10: -0.94, 11: 1.67, 12: -0.04}
    # Convert monthly % to per-5-min expected log-return (approx)
    bars_per_month = 78 * 21
    df["month_prior"] = df["month"].map(
        {k: v / 100.0 / bars_per_month for k, v in month_prior_usd.items()}).astype("float32")

    out = df.drop(columns=["date"])
    out.to_parquet(CLEAN / "xauusd_features3_m5.parquet", index=False, compression="zstd")

    print(f"\nFeature matrix: {len(out):,} rows x {len(out.columns)} cols")
    new_cols = ["hmm3_bull","hmm3_bear","hmm3_churn","cpd_score","cpd_loc",
                "sigma_fast","sigma_slow","sigma_hat","sigma_u","sigma_sv",
                "vwap_dev_z_defl","db_T_defl","ou_kalman_z","month_prior"]
    print(out[new_cols].describe().T[["count","mean","std","min","50%","max"]].round(5).to_string())
    print(f"\nWrote {CLEAN/'xauusd_features3_m5.parquet'}")


if __name__ == "__main__":
    main()
