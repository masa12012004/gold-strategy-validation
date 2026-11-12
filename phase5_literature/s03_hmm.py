"""
Paper 18 — "Hidden Markov Models Applied To Intraday Momentum Trading With Side Information".

  * latent trend states m_t (K=3: up / flat / down — the paper's BIC/bridge-sampling choice),
    Gaussian emissions y_t = mu_{m_t} + eps, parameters by Baum-Welch on a rolling window of
    23 trading days, refitted every day "when the market is shut".
  * Inference by the FORWARD algorithm only (filtering, no smoothing -> no look-ahead).
    Forecast: E[y_{t+1}|y_1:t] = sum_j (A' p_t)_j mu_j.  Trade the sign (paper: only the
    sign of momentum signals has traction).
  * Side information (Section 4): (i) volatility ratio sigma_fast/sigma_slow from EWMA
    (lambda=0.79, fast=50, slow=100 obs), (ii) intraday seasonality (time-of-day index).
    Each is mapped to a forecast by a zero-mean smoothed regression "spline" learned on the
    trailing 66 trading days (we use binned means + smoothing instead of MATLAB SLM splines).
    The IOHMM combination is approximated by adding the standardised side forecasts to the
    HMM forecast (documented approximation).
Bars: M5 (1-min trading would not survive a 1.7 bp round trip; the paper is pre-cost).
Hysteresis: a position is only changed when |forecast| exceeds the half-spread.
"""
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from common import load_m5, backtest, metrics, save_metrics, OOS_START

warnings.filterwarnings("ignore")
K, WIN_DAYS, SIDE_DAYS = 3, 23, 66


def forward_filter(model, y, p0):
    A = model.transmat_
    mu = model.means_.ravel()
    sd = np.sqrt(model.covars_.ravel())
    p = p0.copy()
    preds = np.empty(len(y))
    for t, yt in enumerate(y):
        prior = A.T @ p
        lik = np.exp(-0.5 * ((yt - mu) / sd) ** 2) / sd
        p = prior * lik
        s = p.sum()
        p = p / s if s > 0 else np.full(K, 1 / K)
        preds[t] = (A.T @ p) @ mu  # one-step-ahead predicted return
    return preds, p


def ewma_vol(y, lam, n):
    w = lam ** np.arange(n)[::-1]
    w /= w.sum()
    return np.sqrt(np.convolve(y ** 2, w[::-1], mode="full")[:len(y)])


def binned_spline(x_tr, y_tr, x_te, nb=20):
    q = np.unique(np.quantile(x_tr, np.linspace(0, 1, nb + 1)))
    b_tr = np.clip(np.searchsorted(q, x_tr) - 1, 0, len(q) - 2)
    means = pd.Series(y_tr).groupby(b_tr).mean().reindex(range(len(q) - 1)).fillna(0).values
    means = pd.Series(means).rolling(3, center=True, min_periods=1).mean().values  # smooth
    means -= means.mean()  # zero-mean spline (paper: integral forced to zero)
    b_te = np.clip(np.searchsorted(q, x_te) - 1, 0, len(q) - 2)
    return means[b_te]


if __name__ == "__main__":
    d = load_m5()
    y = d["lret"].values * 1e4  # bps
    days = d["tday"].values
    udays = pd.unique(days)
    oos_days = [u for u in udays if pd.Timestamp(u).tz_localize("US/Eastern") >= OOS_START.normalize()]
    vf = ewma_vol(y, 0.79, 50)
    vs = ewma_vol(y, 0.79, 100)
    vr = np.where(vs > 0, vf / vs, 1.0)
    tod = (d.index.hour * 60 + d.index.minute).values
    hmm_fc = np.full(len(d), np.nan)
    side_fc = np.full(len(d), np.nan)
    day_index = pd.Series(np.arange(len(d))).groupby(days).agg(["first", "last"])
    pos_in_days = {u: i for i, u in enumerate(udays)}
    model = None
    for n, u in enumerate(oos_days):
        k = pos_in_days[u]
        if k < SIDE_DAYS:
            continue
        a0 = day_index.loc[udays[k - WIN_DAYS], "first"]
        a1 = day_index.loc[u, "first"]
        b1 = day_index.loc[u, "last"] + 1
        ytr = y[a0:a1].reshape(-1, 1)
        m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=60, tol=1e-3, random_state=0)
        try:
            m.fit(ytr)
            model = m
        except Exception:
            if model is None:
                continue
        # filter through the training window to get the state at today's open, then today
        _, p_open = forward_filter(model, y[a0:a1], model.startprob_)
        preds, _ = forward_filter(model, y[a1:b1], p_open)
        hmm_fc[a1:b1] = preds
        # side information splines, trailing 66 days
        s0 = day_index.loc[udays[k - SIDE_DAYS], "first"]
        ynext = np.r_[y[1:], 0.0]
        f_vr = binned_spline(vr[s0:a1 - 1], ynext[s0:a1 - 1], vr[a1:b1])
        f_tod = binned_spline(tod[s0:a1 - 1], ynext[s0:a1 - 1], tod[a1:b1], nb=46)
        side_fc[a1:b1] = f_vr + f_tod
        if n % 50 == 0:
            print(u, "means(bps):", np.round(model.means_.ravel(), 3), flush=True)
    d["hmm_fc"], d["side_fc"] = hmm_fc, side_fc
    d[["hmm_fc", "side_fc"]].to_parquet("results/_hmm_forecasts.parquet")

    hs_bps = d["hs"].values * 1e4
    rows = []

    def hysteresis(fc, thr):
        pos = np.zeros(len(fc))
        cur = 0.0
        for t in range(len(fc)):
            if np.isfinite(fc[t]) and abs(fc[t]) > thr[t]:
                cur = np.sign(fc[t])
            elif not np.isfinite(fc[t]):
                cur = 0.0
            pos[t] = cur
        return pos

    p1 = hysteresis(hmm_fc, hs_bps)
    rows.append(metrics(backtest(d, p1, "s03_hmm_momentum"), "s03_hmm_momentum"))
    z_h = hmm_fc / np.nanstd(hmm_fc)
    z_s = side_fc / np.nanstd(side_fc)
    comb = (z_h + z_s) * np.nanstd(hmm_fc)
    p2 = hysteresis(comb, hs_bps)
    rows.append(metrics(backtest(d, p2, "s03_hmm_sideinfo"), "s03_hmm_sideinfo"))
    print(pd.DataFrame(rows).round(3).to_string())
    save_metrics(rows, "m_s03.csv")
