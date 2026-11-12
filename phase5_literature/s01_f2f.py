"""
Paper 11 — "Forecast-to-Fill: Benchmark-Neutral Alpha ... in Gold" (daily bars).

Faithful pipeline:
  EMA(log P) slope -> z (train-window mean/std) -> p_trend=(clip(z,-3,3)+3)/6
  m = 1{P_t/P_{t-50} > 1};  p_bull = 0.6 p_trend + 0.4 m
  entry long: p_bull >= 0.52 and slope > 0      (symmetric short in the L/S variant)
  exits: hard stop 2*ATR14, trailing 1.5*ATR14 from peak, max age 30 days,
         p_bear > 0.5 -> halve (de-risk)
  size: vol target 15% (EWMA lambda 0.94), cap 2.0, confidence (p_bull-0.5)/0.5,
        friction-adjusted fractional Kelly (0.40) with 0.25 baseline.
Adaptation: the paper trains on 10 years; we only have 6 months before OOS, so the
window is expanding from 2025-01 and refitted monthly. EMA lambda is chosen on the
training window from {0.80,0.85,0.90,0.95} by training Sharpe (as the paper does).
"""
import numpy as np
import pandas as pd
from common import (load_daily, backtest, metrics, atr, monthly_refit_points, save_metrics, ANN)

K_MOM, W_SLOPE, THR = 50, 0.6, 0.52
SIG_TGT, WMAX, KELLY_FRAC, LAM_V = 0.15, 2.0, 0.40, 0.94
COST_RT = 1.7e-4  # round-trip cost used inside the Kelly formula (our real spread)


def signals(d, lam, mu, sd):
    y = np.log(d["close"])
    ys = y.ewm(alpha=1 - lam, adjust=False).mean()
    slope = ys.diff()
    z = (slope - mu) / sd
    p_trend = (z.clip(-3, 3) + 3) / 6
    m = (d["close"] / d["close"].shift(K_MOM) > 1).astype(float)
    p_bull = W_SLOPE * p_trend + (1 - W_SLOPE) * m
    return slope, p_bull


def kelly_mult(unit_ret: pd.Series) -> float:
    mu, s2 = unit_ret.mean(), unit_ret.var()
    n, k, eta = 1.0, COST_RT, 0.02 * 1e-3  # impact negligible at our size
    if not np.isfinite(mu) or s2 <= 0 or mu - n * k <= 0:
        return 0.25
    a = 2 * s2 ** 2
    x = (-3 * eta * n ** 1.5 + np.sqrt(9 * eta ** 2 * n ** 3 + 16 * s2 ** 2 * (mu - n * k))) / (2 * a)
    f = KELLY_FRAC * x ** 2
    return float(np.clip(f, 0.25, 1.0))


def run_rules(d, slope, p_bull, kmult, allow_short):
    a = atr(d, 14).values
    c = d["close"].values
    sig_hat = np.sqrt((d["ret"] ** 2).ewm(alpha=1 - LAM_V, adjust=False).mean()).values * np.sqrt(ANN)
    pb, sl = p_bull.values, slope.values
    pos = np.zeros(len(d))
    side, entry, peak, age, half = 0, 0.0, 0.0, 0, False
    for t in range(len(d)):
        if not np.isfinite(a[t]) or not np.isfinite(pb[t]) or not np.isfinite(sig_hat[t]):
            continue
        if side != 0:
            age += 1
            peak = max(peak, c[t]) if side > 0 else min(peak, c[t])
            hit_hard = (c[t] < entry - 2 * a[t]) if side > 0 else (c[t] > entry + 2 * a[t])
            hit_trail = (c[t] < peak - 1.5 * a[t]) if side > 0 else (c[t] > peak + 1.5 * a[t])
            if hit_hard or hit_trail or age >= 30:
                side, half = 0, False
            elif (side > 0 and 1 - pb[t] > 0.5) or (side < 0 and pb[t] > 0.5):
                half = True
        if side == 0:
            if pb[t] >= THR and sl[t] > 0:
                side, entry, peak, age, half = 1, c[t], c[t], 0, False
            elif allow_short and (1 - pb[t]) >= THR and sl[t] < 0:
                side, entry, peak, age, half = -1, c[t], c[t], 0, False
        if side != 0:
            w_vol = min(SIG_TGT / max(sig_hat[t], 1e-6), WMAX)
            conf = abs(pb[t] - 0.5) / 0.5
            w = min(kmult[t] * w_vol * conf, WMAX) * (0.5 if half else 1.0)
            pos[t] = side * w
    return pos


def build(d, allow_short):
    refits = monthly_refit_points(d.index)
    lam_t = pd.Series(np.nan, index=d.index)
    mu_t, sd_t, km_t = lam_t.copy(), lam_t.copy(), lam_t.copy()
    log = []
    for i, r0 in enumerate(refits):
        r1 = refits[i + 1] if i + 1 < len(refits) else d.index[-1] + pd.Timedelta(days=1)
        tr = d[d.index < r0]
        best = None
        for lam in (0.80, 0.85, 0.90, 0.95):
            ys = np.log(tr["close"]).ewm(alpha=1 - lam, adjust=False).mean().diff()
            mu, sd = ys.mean(), ys.std()
            s, pbull = signals(tr, lam, mu, sd)
            p = run_rules(tr, s, pbull, np.full(len(tr), 1.0), allow_short)
            unit = pd.Series(np.sign(p), index=tr.index) * tr["ret"].shift(-1).fillna(0)
            sh = unit.mean() / unit.std() if unit.std() > 0 else -9
            if best is None or sh > best[0]:
                best = (sh, lam, mu, sd, unit)
        _, lam, mu, sd, unit = best
        km = kelly_mult(unit[unit != 0])
        m = (d.index >= r0) & (d.index < r1)
        lam_t[m], mu_t[m], sd_t[m], km_t[m] = lam, mu, sd, km
        log.append((str(r0)[:10], lam, round(km, 3)))
    # compute signal with frozen per-month params
    pos = np.zeros(len(d))
    for (lam, mu, sd), grp in pd.DataFrame({"l": lam_t, "m": mu_t, "s": sd_t}).dropna().groupby(["l", "m", "s"]):
        s, pbull = signals(d, lam, mu, sd)
        p = run_rules(d, s, pbull, km_t.fillna(0.25).values, allow_short)
        idx = d.index.get_indexer(grp.index)
        pos[idx] = p[idx]
    return pos, log


if __name__ == "__main__":
    d = load_daily()
    rows = []
    bh = backtest(d, np.ones(len(d)), "s00_buy_hold_daily")
    rows.append(metrics(bh, "Buy & hold (benchmark)"))
    for short, nm in ((False, "s01_f2f_long_only"), (True, "s01_f2f_long_short")):
        pos, log = build(d, short)
        bt = backtest(d, pos, nm)
        rows.append(metrics(bt, nm))
        print(nm, "refits:", log[:3], "...")
    # beta / alpha of F2F vs buy & hold (paper reports beta ~0.03)
    for nm in ("s01_f2f_long_only", "s01_f2f_long_short"):
        b = pd.read_parquet(f"results/{nm}.parquet").groupby("tday")["net"].sum()
        m = bh.groupby("tday")["net"].sum()
        beta = np.cov(b, m)[0, 1] / m.var()
        print(nm, "beta vs gold:", round(beta, 3),
              "alpha ann:", round((b.mean() - beta * m.mean()) * ANN, 4))
    df = pd.DataFrame(rows)
    print(df.round(3).to_string())
    save_metrics(rows, "m_s01.csv")
