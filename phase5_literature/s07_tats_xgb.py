"""
Paper 45 — TATS: a binary TREND PREDICTOR (XGBoost was the best on gold: 58.7% test accuracy
           on daily data) used to correct a value forecaster.
Papers 6, 12, 32, 9 — feature sets: RSI, MACD, Bollinger %b, ATR, ADX, Williams %R,
           multi-horizon returns, GARCH/EWMA volatility, swing-high/low distance, time of day.

Bars: H1.  Label: sign of the next HZN-bar return (purged HZN bars at every split).
Walk-forward: expanding window, refit monthly. Probability threshold chosen on the last 20%
of each training window (never on the test month). Logistic regression is the benchmark
(as in paper 21). Positions are vol-targeted to 15%.
Saves the out-of-sample probabilities for the TATS hybrid in s10.
"""
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import (load_h1, backtest, metrics, monthly_refit_points, save_metrics, atr,
                    vol_target, bars_per_year)

warnings.filterwarnings("ignore")
HZN = 4


def features(d):
    c, h, l = d["close"], d["high"], d["low"]
    r = d["lret"]
    vol = r.ewm(span=48, min_periods=24).std()
    f = pd.DataFrame(index=d.index)
    for k in (1, 3, 6, 12, 24, 72, 168):
        f[f"ret{k}"] = np.log(c).diff(k) / (vol * np.sqrt(k))
    up, dn = r.clip(lower=0), (-r).clip(lower=0)
    f["rsi14"] = 100 - 100 / (1 + up.ewm(alpha=1 / 14).mean() / dn.ewm(alpha=1 / 14).mean())
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    a = atr(d, 14)
    f["macd"] = macd / a
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / a
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["boll_b"] = (c - ma) / (2 * sd)
    f["atr_pct"] = a / c * 1e4
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    f["willr"] = (hh - c) / (hh - ll)
    pdm = (h.diff()).clip(lower=0).where(h.diff() > -l.diff(), 0)
    ndm = (-l.diff()).clip(lower=0).where(-l.diff() > h.diff(), 0)
    pdi, ndi = pdm.ewm(alpha=1 / 14).mean() / a, ndm.ewm(alpha=1 / 14).mean() / a
    f["adx"] = ((pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14).mean()
    f["di_diff"] = pdi - ndi
    f["vol_ratio"] = r.ewm(span=12).std() / r.ewm(span=96).std()
    f["dist_hi48"] = (h.rolling(48).max() - c) / a      # distance to swing high / low (paper 32)
    f["dist_lo48"] = (c - l.rolling(48).min()) / a
    f["range_bps"] = d["range_bps"] / d["range_bps"].rolling(240).mean()
    f["ticks"] = (d["bid_ticks"] + d["ask_ticks"]) / (d["bid_ticks"] + d["ask_ticks"]).rolling(240).mean()
    hr = d.index.hour + d.index.minute / 60
    f["h_sin"], f["h_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    f["dow"] = d.index.dayofweek
    return f


def choose_thr(p, fwd, cost):
    best = (-9, 0.0)
    for thr in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
        pos = np.where(np.abs(p - 0.5) > thr, np.sign(p - 0.5), 0.0)
        pn = pos * fwd - np.abs(np.diff(np.r_[0, pos])) * cost
        s = pn.mean() / (pn.std() + 1e-12)
        if s > best[0]:
            best = (s, thr)
    return best[1]


if __name__ == "__main__":
    d = load_h1()
    bpy = bars_per_year(d)
    X = features(d)
    fwdH = np.log(d["mid"]).shift(-HZN) - np.log(d["mid"])
    y = (fwdH > 0).astype(int)
    fwd1 = d["ret"].shift(-1).fillna(0).values
    cost = (d["hs"] + 0.2e-4).values
    ok = np.asarray(X.notna().all(1))
    refits = monthly_refit_points(d.index)
    P = {m: np.full(len(d), np.nan) for m in ("xgb", "logit")}
    pos = {m: np.zeros(len(d)) for m in ("xgb", "logit")}
    acc = {m: [] for m in P}
    imp = []
    for i, r0 in enumerate(refits):
        r1 = refits[i + 1] if i + 1 < len(refits) else d.index[-1] + pd.Timedelta(days=1)
        tr_idx = np.where(np.asarray(d.index < r0) & ok)[0][:-HZN]      # purge label overlap
        te_idx = np.where(np.asarray((d.index >= r0) & (d.index < r1)) & ok)[0]
        cut = int(len(tr_idx) * 0.8)
        fit_idx, val_idx = tr_idx[:cut - HZN], tr_idx[cut:]
        models = {}
        for name in P:
            if name == "xgb":
                mk = lambda: xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
                                               subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
                                               reg_lambda=5.0, n_jobs=4, random_state=0, verbosity=0)
                m = mk().fit(X.values[fit_idx], y.values[fit_idx])
                pv = m.predict_proba(X.values[val_idx])[:, 1]
                thr = choose_thr(pv, fwd1[val_idx], cost[val_idx])
                m = mk().fit(X.values[tr_idx], y.values[tr_idx])
                pt = m.predict_proba(X.values[te_idx])[:, 1]
                imp.append(m.feature_importances_)
            else:
                sc = StandardScaler().fit(X.values[fit_idx])
                m = LogisticRegression(C=0.05, max_iter=500).fit(sc.transform(X.values[fit_idx]), y.values[fit_idx])
                pv = m.predict_proba(sc.transform(X.values[val_idx]))[:, 1]
                thr = choose_thr(pv, fwd1[val_idx], cost[val_idx])
                sc = StandardScaler().fit(X.values[tr_idx])
                m = LogisticRegression(C=0.05, max_iter=500).fit(sc.transform(X.values[tr_idx]), y.values[tr_idx])
                pt = m.predict_proba(sc.transform(X.values[te_idx]))[:, 1]
            P[name][te_idx] = pt
            pos[name][te_idx] = np.where(np.abs(pt - 0.5) > thr, np.sign(pt - 0.5), 0.0)
            yt = y.values[te_idx]
            valid = np.isfinite(fwdH.values[te_idx])
            acc[name].append(((pt > 0.5).astype(int) == yt)[valid].mean())
    rows = []
    for name in P:
        w = vol_target(pd.Series(pos[name], index=d.index), d["ret"], bpy, 0.15, span=24 * 20)
        nm = f"s07_{name}_trend_predictor"
        rows.append(metrics(backtest(d, w, nm), nm))
        print(name, "monthly OOS directional accuracy: mean %.4f  min %.4f  max %.4f" %
              (np.mean(acc[name]), np.min(acc[name]), np.max(acc[name])))
    pd.DataFrame({"p_xgb": P["xgb"], "p_logit": P["logit"]}, index=d.index).to_parquet("results/_tats_probs.parquet")
    fi = pd.Series(np.mean(imp, 0), index=X.columns).sort_values(ascending=False)
    print("XGB mean gain share, top 8:\n", fi.head(8).round(3).to_string())
    print(pd.DataFrame(rows).round(3).to_string())
    save_metrics(rows, "m_s07.csv")
