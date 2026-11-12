"""
Paper 46 — "Trends, Reversion, and Critical Phenomena": E[R_{t+1}] = b*phi + c*phi^3,
  phi_T(t) = sum_n w_T(n) R^(t-n),  w_T(n) = N_T (n+1) exp(-2n/T),
  N_T = (1-e^{-4/T})^2 / sqrt(1-e^{-8/T}) so that sum w^2 = 1,  phi capped at +-2.5.
  Risk premium mu estimated on training data only (as in the paper's cross-validation).
Paper 22 — returns are deflated by the diurnal (hour-of-day) volatility factor before
  computing trend strengths (the paper shows intraday heteroskedasticity is largely, but
  not entirely, a deterministic diurnal pattern; we add an EWMA term for the remainder).
Paper 39 (benchmark section) — Moskowitz TSMOM: X = sgn(r_{t-L,t}), vol-scaled to 15%.

Bars: H1. Horizons T (in hours) = 2^k, k=2..9  (4h ... 512h ~ 1 month).
Refit monthly, expanding window. Position = forecast scaled by its training std, clipped
to [-1,1], times 15% vol targeting.
"""
import numpy as np
import pandas as pd
from common import (load_h1, backtest, metrics, monthly_refit_points, save_metrics,
                    vol_target, bars_per_year)

HORIZONS = [2 ** k for k in range(2, 10)]


def trend_strength(R: np.ndarray, T: int) -> np.ndarray:
    """Recursive computation of eq. (6)-(7)."""
    a = np.exp(-2.0 / T)
    M = np.sqrt(1 - np.exp(-4.0 / T))          # weight of the exponential variant (eq. 4)
    N = (1 - np.exp(-4.0 / T)) ** 2 / np.sqrt(1 - np.exp(-8.0 / T))
    psi = np.zeros_like(R)
    phi = np.zeros_like(R)
    p_prev, f_prev = 0.0, 0.0
    for t in range(len(R)):
        # psi(t) = sum_n e^{-2n/T} R(t-n);  phi(t) = sum_n (n+1) e^{-2n/T} R(t-n)
        p = a * p_prev + R[t]
        f = a * f_prev + p
        psi[t], phi[t] = p, f
        p_prev, f_prev = p, f
    return np.clip(N * phi, -2.5, 2.5)


def deflated_returns(d: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    r = d["lret"]
    hour = d.index.hour
    tr = r[train_mask]
    diurnal = tr.groupby(hour[train_mask]).std()
    diurnal = diurnal / diurnal.mean()
    f = pd.Series(hour, index=d.index).map(diurnal).fillna(1.0).values
    rd = r.values / f
    s = pd.Series(rd, index=d.index).ewm(span=24 * 20, min_periods=50).std().shift(1).bfill().values
    mu = (rd[train_mask] / s[train_mask]).mean()
    return rd / s - mu


def cubic_positions(d: pd.DataFrame):
    refits = monthly_refit_points(d.index)
    fc_all = pd.Series(np.nan, index=d.index)
    coefs = []
    for i, r0 in enumerate(refits):
        r1 = refits[i + 1] if i + 1 < len(refits) else d.index[-1] + pd.Timedelta(days=1)
        train = (d.index < r0)
        R = deflated_returns(d, np.asarray(train))
        Rn = np.r_[R[1:], 0.0]  # next-bar target
        fc = np.zeros(len(d))
        bc = []
        for T in HORIZONS:
            ph = trend_strength(R, T)
            X = np.c_[ph, ph ** 3][train][T:-1]
            y = Rn[train][T:-1]
            b, c = np.linalg.lstsq(X, y, rcond=None)[0]
            fc += b * ph + c * ph ** 3
            bc.append((T, b, c))
        fc /= len(HORIZONS)
        scale = np.std(fc[train][600:])
        m = (d.index >= r0) & (d.index < r1)
        fc_all[m] = fc[m] / scale
        coefs.append((str(r0)[:10], bc))
    return fc_all.fillna(0.0), coefs


def tsmom_positions(d: pd.DataFrame, bpy: float):
    refits = monthly_refit_points(d.index)
    out = pd.Series(0.0, index=d.index)
    chosen = []
    for i, r0 in enumerate(refits):
        r1 = refits[i + 1] if i + 1 < len(refits) else d.index[-1] + pd.Timedelta(days=1)
        tr = d[d.index < r0]
        best = None
        for L in (24, 72, 120, 252, 504):
            sig = np.sign(np.log(tr["mid"]).diff(L))
            w = vol_target(sig, tr["ret"], bpy, 0.15, span=24 * 20)
            pnl = (w * tr["ret"].shift(-1)).dropna() - w.diff().abs() * tr["hs"]
            sh = pnl.mean() / pnl.std()
            if best is None or sh > best[0]:
                best = (sh, L)
        L = best[1]
        sig = np.sign(np.log(d["mid"]).diff(L))
        w = vol_target(sig, d["ret"], bpy, 0.15, span=24 * 20)
        m = (d.index >= r0) & (d.index < r1)
        out[m] = w[m]
        chosen.append(L)
    return out, chosen


if __name__ == "__main__":
    d = load_h1()
    bpy = bars_per_year(d)
    rows = []
    fc, coefs = cubic_positions(d)
    for (dt, bc) in coefs[:1] + coefs[-1:]:
        print(dt, " ".join(f"T{T}:b={b:+.4f},c={c:+.5f}" for T, b, c in bc))
    sig = fc.clip(-1, 1)
    pos = vol_target(sig, d["ret"], bpy, 0.15, span=24 * 20)
    bt = backtest(d, pos, "s02_cubic_trend_reversion")
    rows.append(metrics(bt, "s02_cubic_trend_reversion"))
    # the same signal traded only when strong (reduces turnover from H1 noise)
    sig2 = sig.where(sig.abs() > 0.5, 0.0)
    bt2 = backtest(d, vol_target(sig2, d["ret"], bpy, 0.15, span=24 * 20), "s02_cubic_thresh")
    rows.append(metrics(bt2, "s02_cubic_thresh"))
    w, Ls = tsmom_positions(d, bpy)
    print("TSMOM lookbacks chosen:", Ls)
    bt3 = backtest(d, w, "s02_tsmom")
    rows.append(metrics(bt3, "s02_tsmom"))
    print(pd.DataFrame(rows).round(3).to_string())
    save_metrics(rows, "m_s02.csv")
