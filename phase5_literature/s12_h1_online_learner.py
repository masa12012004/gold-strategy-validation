"""
H1 (daily F2F regime x hourly cubic) with a trade-by-trade ONLINE learner, 60:30:10 split.

A "trade" is one contiguous run of non-zero H1 position with a constant sign.
At each entry the learner sees only trades that have already CLOSED. It stores, for every
closed trade, the situation at entry (context) and what happened (outcome R, and whether the
loss was an immediate failure), and uses that memory to size the next trade:

  recency memory  : last R, mean R / win rate of the last 5 and 10 trades, streak
  context memory  : k-NN over standardised entry context - "the K most similar past
                    situations, how did they end?"
  bucket memory   : mean R by (session, direction)
  model           : ridge regression on context + memory features, refitted after every trade

Protocol
  TRAIN (first 60% of days): starts with an empty memory (size 1.0 until 40 trades have
        closed), learns as it trades. The learner CONFIGURATION (model x sizing rule) is
        chosen on this split only.
  TEST  (next 30%): configuration frozen; memory keeps growing with train + test trades.
  VAL   (last 10%): same frozen configuration; memory = train + test + val-so-far.
Control = raw H1 with size 1.0 on every trade, same engine and costs.
"""
import itertools
import numpy as np
import pandas as pd
from common import load_h1, load_daily, backtest, metrics, daily_returns, atr, RES, ANN

WARM, K = 40, 25
CONTEXT = ["dir", "strength", "regime", "h_sin", "h_cos", "atr_rel", "vol_ratio",
           "mom6", "mom24", "mom120", "dist_ext", "gap_bars"]


def base_position(h1):
    return pd.read_parquet(f"{RES}/h1_slowmom_x_cubic.parquet")["pos"].reindex(h1.index).fillna(0.0)


def regime_series(h1):
    dly = load_daily()
    f = pd.read_parquet(f"{RES}/s01_f2f_long_short.parquet")["pos"].reindex(dly.index).fillna(0.0)
    key = pd.DataFrame({"t": (h1.index + pd.Timedelta(hours=1)).tz_convert("UTC")}, index=h1.index).reset_index()
    dk = pd.DataFrame({"t": (dly.index + pd.Timedelta(minutes=5)).tz_convert("UTC"), "f": f.values}).sort_values("t")
    return pd.merge_asof(key.sort_values("t"), dk, on="t").set_index("dt")["f"].reindex(h1.index).fillna(0.0)


def build_trades(h1, pos):
    s = np.sign(pos.values)
    rid = np.cumsum(np.r_[True, s[1:] != s[:-1]])
    fwd = h1["ret"].shift(-1).fillna(0).values
    cost = (h1["hs"] + 0.2e-4).values
    c = h1["close"]
    a = atr(h1, 14)
    r = h1["lret"]
    vol = r.ewm(span=48, min_periods=24).std()
    reg = regime_series(h1)
    hr = h1.index.hour.values
    rows = []
    last_exit = None
    for _, ix in pd.Series(np.arange(len(s))).groupby(rid):
        ix = ix.values
        if s[ix[0]] == 0:
            continue
        e, x = ix[0], ix[-1]
        d = s[e]
        p = pos.values[ix]
        # pnl of the trade at base size: bar pnl + entry and exit half-spread
        pnl = (p * fwd[ix]).sum() - abs(p[0]) * cost[e] - abs(p[-1]) * cost[min(x + 1, len(s) - 1)] \
              - (np.abs(np.diff(p)) * cost[ix[1:]]).sum()
        risk = np.mean(np.abs(p)) * vol.values[e] * np.sqrt(max(len(ix), 1))
        mae = np.min(np.cumsum(p * fwd[ix]))
        rows.append({
            "e": e, "x": x, "entry": h1.index[e], "tday": h1["tday"].values[e], "n": len(ix),
            "pnl": pnl, "R": pnl / max(risk, 1e-9), "fast_fail": float(len(ix) <= 2 and pnl < 0), "mae": mae,
            "dir": d, "strength": abs(p[0]), "regime": abs(reg.values[e]),
            "h_sin": np.sin(2 * np.pi * hr[e] / 24), "h_cos": np.cos(2 * np.pi * hr[e] / 24),
            "session": 0 if (hr[e] >= 18 or hr[e] < 3) else (1 if hr[e] < 8 else 2),
            "atr_rel": a.values[e] / a.rolling(480).mean().values[e],
            "vol_ratio": (r.ewm(span=12).std() / r.ewm(span=96).std()).values[e],
            "mom6": d * np.log(c).diff(6).values[e] / (vol.values[e] * np.sqrt(6)),
            "mom24": d * np.log(c).diff(24).values[e] / (vol.values[e] * np.sqrt(24)),
            "mom120": d * np.log(c).diff(120).values[e] / (vol.values[e] * np.sqrt(120)),
            "dist_ext": ((c.rolling(48).max().values[e] - c.values[e]) if d > 0 else
                         (c.values[e] - c.rolling(48).min().values[e])) / a.values[e],
            "gap_bars": np.log1p(e - last_exit) if last_exit is not None else 5.0,
        })
        last_exit = x
    t = pd.DataFrame(rows).reset_index(drop=True)
    t[CONTEXT] = t[CONTEXT].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return t


def memory_features(t, i):
    """Features for trade i from trades 0..i-1 only (all closed before entry i)."""
    h = t.iloc[:i]
    R = h["R"].values
    f = {}
    f["last_R"] = R[-1] if i else 0.0
    f["mean_R5"] = R[-5:].mean() if i else 0.0
    f["mean_R10"] = R[-10:].mean() if i else 0.0
    f["win10"] = (R[-10:] > 0).mean() if i else 0.5
    st = 0
    for v in R[::-1]:
        if (v > 0) == (R[-1] > 0):
            st += 1
        else:
            break
    f["streak"] = (st if i and R[-1] > 0 else -st) if i else 0.0
    f["fastfail5"] = h["fast_fail"].values[-5:].mean() if i else 0.0
    if i >= 10:
        X = h[CONTEXT].values
        mu, sd = X.mean(0), X.std(0) + 1e-9
        dist = (((X - mu) / sd - (t.loc[i, CONTEXT].values.astype(float) - mu) / sd) ** 2).sum(1)
        nn = np.argsort(dist)[:min(K, i)]
        f["knn_R"] = R[nn].mean()
        f["knn_win"] = (R[nn] > 0).mean()
        b = h[(h["session"] == t.loc[i, "session"]) & (h["dir"] == t.loc[i, "dir"])]["R"]
        f["bucket_R"] = b.mean() if len(b) >= 5 else R.mean()
    else:
        f["knn_R"], f["knn_win"], f["bucket_R"] = 0.0, 0.5, 0.0
    return f


def run_learner(t, model, rule):
    """Sequential pass over all trades. Returns size multiplier per trade and the prediction."""
    n = len(t)
    mem = pd.DataFrame([memory_features(t, i) for i in range(n)])
    F = pd.concat([t[CONTEXT], mem], axis=1).values.astype(float)
    R = t["R"].clip(-4, 4).values
    mult, pred = np.ones(n), np.zeros(n)
    for i in range(n):
        if i < WARM:
            continue
        X, y = F[:i], R[:i]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z, z = (X - mu) / sd, (F[i] - mu) / sd
        A = Z.T @ Z + 50.0 * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (y - y.mean()))
        ridge_in = Z @ w
        p_ridge = (z @ w) / (ridge_in.std() + 1e-9)
        knn_hist = mem["knn_R"].values[WARM // 2:i]
        p_knn = (mem["knn_R"].values[i] - knn_hist.mean()) / (knn_hist.std() + 1e-9)
        p = {"ridge": p_ridge, "knn": p_knn, "blend": 0.5 * (p_ridge + p_knn)}[model]
        pred[i] = p
        if rule == "veto":
            mult[i] = 0.0 if p < -0.5 else 1.0
        elif rule == "tilt":
            mult[i] = np.clip(1 + 0.5 * p, 0.5, 1.5)
        else:  # scale
            mult[i] = np.clip(1 + 0.75 * p, 0.0, 2.0)
    return mult, pred


def apply(h1, pos, t, mult):
    p = pos.values.copy()
    for (e, x), m in zip(t[["e", "x"]].values, mult):
        p[e:x + 1] *= m
    return p


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0


if __name__ == "__main__":
    h1 = load_h1()
    pos = base_position(h1)
    t = build_trades(h1, pos)
    ctrl = backtest(h1, pos.values, "h1_control", save=False)
    days = np.sort(ctrl["tday"].unique())
    n = len(days)
    d_tr, d_te = days[int(n * 0.6)], days[int(n * 0.9)]
    seg = lambda r: {"TRAIN": r[r.index < d_tr], "TEST": r[(r.index >= d_tr) & (r.index < d_te)], "VAL": r[r.index >= d_te]}
    t["split"] = np.where(t["tday"] < d_tr, "TRAIN", np.where(t["tday"] < d_te, "TEST", "VAL"))
    print("split boundaries:", str(days[0])[:10], "|", str(d_tr)[:10], "|", str(d_te)[:10], "|", str(days[-1])[:10])
    print("trades per split:", t["split"].value_counts().to_dict(), " total", len(t))
    rc = daily_returns(ctrl)

    # ---- choose the learner configuration on TRAIN only
    res = []
    runs = {}
    for model, rule in itertools.product(("ridge", "knn", "blend"), ("veto", "tilt", "scale")):
        mult, pred = run_learner(t, model, rule)
        bt = backtest(h1, apply(h1, pos, t, mult), "tmp", save=False)
        r = daily_returns(bt)
        runs[(model, rule)] = (mult, pred, r)
        s = seg(r)
        res.append({"model": model, "rule": rule, **{k: round(sharpe(v), 2) for k, v in s.items()},
                    **{k + "_ret": round((1 + v).prod() - 1, 4) for k, v in s.items()}})
    grid = pd.DataFrame(res)
    best = grid.sort_values("TRAIN", ascending=False).iloc[0]
    print("\nAll configurations (chosen on TRAIN Sharpe only; TEST/VAL shown for transparency):")
    print(grid.to_string(index=False))
    print(f"\nChosen on TRAIN: model={best.model}, rule={best.rule}")
    mult, pred, rl = runs[(best.model, best.rule)]
    t["mult"], t["pred"] = mult, pred

    # ---- did the learner's forecast carry information about the NEXT trade?
    rows = []
    for name, r in (("H1 control", rc), ("H1 + online learner", rl)):
        for k, v in list(seg(r).items()) + [("OVERALL", r)]:
            eq = (1 + v).cumprod()
            rows.append({"strategy": name, "split": k, "days": len(v), "return": eq.iloc[-1] - 1, "sharpe": sharpe(v),
                         "max_dd": (eq / eq.cummax() - 1).min(), "ann_vol": v.std() * np.sqrt(ANN)})
    out = pd.DataFrame(rows)
    print("\n", out.round(4).to_string(index=False))
    live = t[t.index >= WARM]
    print("\nPrediction vs realised R of the next trade (rank correlation):")
    for k in ("TRAIN", "TEST", "VAL"):
        g = live[live["split"] == k]
        ic = g["pred"].corr(g["R"], method="spearman") if len(g) > 5 else np.nan
        print(f"  {k:5s} trades {len(g):3d}  IC {ic:+.3f}  mean size {g['mult'].mean():.2f}  "
              f"skipped/cut(<0.75) {int((g['mult'] < 0.75).sum())}  boosted(>1.25) {int((g['mult'] > 1.25).sum())}")
    print("\nTrade statistics by split (control):")
    print(t.groupby("split").agg(trades=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()),
                                 mean_bps=("pnl", lambda x: x.mean() * 1e4), fast_fail=("fast_fail", "mean"),
                                 avg_bars=("n", "mean")).round(3).reindex(["TRAIN", "TEST", "VAL"]).to_string())
    pd.DataFrame({"control": rc, "learner": rl}).to_parquet(f"{RES}/s12_daily.parquet")
    t.drop(columns=["entry"]).assign(entry=t["entry"].astype(str)).to_csv(f"{RES}/s12_trades.csv", index=False)
    out.to_csv(f"{RES}/s12_summary.csv", index=False)
    pd.Series({"d_tr": str(d_tr), "d_te": str(d_te)}).to_json(f"{RES}/s12_splits.json")
