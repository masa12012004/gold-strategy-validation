"""
Phase 4i — "The brain": a memory-conditioned trade selector, 60/30/10.

What makes this different from phase4_ml.py:

  * Trades are NOT independent. Two couplings are explicit:
      (a) DECISION coupling — memory features (recency / k-NN context / bucket)
          mean the model's view of this trade depends on what earlier trades did.
      (b) SIZE coupling — a sequential engine scales risk by realised performance
          of the trades it actually took: it presses after a good run and shrinks
          after a bad one, and equity-proportional sizing compounds both ways.

  * Labels are MAGNITUDE-WEIGHTED (step 2). A +3R win must matter more than a
    +0.1R win, so each sample is weighted by |R| rather than counted as 1.

  * Split is sequential 60 / 30 / 10 with EXPANDING memory, as specified:
      TRAIN 60%  — model fitted here; memory accumulates
      TEST  30%  — traded using the TRAIN-fitted model; memory keeps accumulating
      VAL   10%  — traded using a model refitted on TRAIN+TEST, with memory from
                   everything before it
    No fold ever sees its own future. The memory bank at any instant contains
    only trades that CLOSED before the current entry bar.

Writes: clean_Data/phase4_brain_results.json
        clean_Data/phase4_brain_equity.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from phase2_common import CAPITAL_0, _max_drawdown
from phase4_features import (ML_FEATURES_STRICT, add_derived, clean_matrix)
from phase4_memory import (MEMORY_FEATURES, STRUCTURE_FEATURES, add_structure)

CLEAN = (Path(__file__).resolve().parent / "clean_Data")

RISK_PER_TRADE = 0.005
STOP_K, TARGET_K, MAX_HOLD = 1.5, 2.5, 12
MAX_TRADES_DAY = 6
DAILY_SOFT, TOTAL_SOFT = 0.03, 0.07
MAX_LEV = 10.0

# step-(a) sequential risk overlay
FEEDBACK_ON = True
FB_LOOKBACK = 10          # taken trades
FB_MIN, FB_MAX = 0.5, 1.5

# The wide 0.10 pool exists to give the model enough TRAINING data. For actual
# trading we require the base classical signal to be strong as well, because the
# wide pool is negative-EV (mean R -0.13) and a 0.53-AUC filter cannot rescue it.
STRONG_THRESH = 0.35
# Threshold is calibrated on the last CALIB_FRAC of the fitting window, which the
# model did not fit on. Choosing it on fit data gives a meaningless cut.
CALIB_FRAC = 0.25


# ------------------------------------------------------- sequential engine

def run_brain_backtest(df: pd.DataFrame, approve: np.ndarray, direction: np.ndarray,
                       atr: np.ndarray, equity0: float = CAPITAL_0,
                       feedback: bool = FEEDBACK_ON) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Bar-walking engine with explicit trade-to-trade feedback.

    approve[i]   : True if the brain approves an entry at bar i
    direction[i] : +1 / -1 for that entry
    Risk per trade = RISK_PER_TRADE * equity * feedback_multiplier, where the
    multiplier is a function of the realised R of the last FB_LOOKBACK trades
    THIS ENGINE ACTUALLY TOOK -> each trade's P&L changes the next trade's size.
    """
    n = len(df)
    hi = df["high"].to_numpy(dtype=np.float64)
    lo = df["low"].to_numpy(dtype=np.float64)
    cl = df["close"].to_numpy(dtype=np.float64)
    if {"ask_close", "bid_close"}.issubset(df.columns):
        spread = (df["ask_close"] - df["bid_close"]).to_numpy(dtype=np.float64)
    else:
        spread = df["spread_median"].to_numpy(dtype=np.float64)
    med = np.nanmedian(spread)
    spread = np.where(~np.isfinite(spread) | (spread <= 0), med, spread)
    dates = pd.to_datetime(df["dt_et"]).dt.date.to_numpy()

    equity = np.full(n + 1, equity0, dtype=np.float64)
    eq = equity0
    trades = []
    taken_R: list[float] = []

    i = 0
    cur_date = None
    day_open_eq = equity0
    day_trades = 0
    halted_day = halted_total = False

    while i < n - 1:
        d = dates[i]
        if d != cur_date:
            cur_date, day_open_eq, day_trades, halted_day = d, eq, 0, False

        ok = True
        if halted_total or halted_day:
            ok = False
        elif eq <= equity0 * (1 - TOTAL_SOFT):
            halted_total = True; ok = False
        elif eq <= day_open_eq * (1 - DAILY_SOFT):
            halted_day = True; ok = False
        elif day_trades >= MAX_TRADES_DAY:
            ok = False

        a = atr[i]
        if not (ok and approve[i] and np.isfinite(a) and a > 0):
            equity[i + 1] = eq
            i += 1
            continue

        # ---- sequential feedback multiplier from trades already taken
        mult = 1.0
        if feedback and len(taken_R) >= 5:
            recent = np.array(taken_R[-FB_LOOKBACK:])
            mult = float(np.clip(1.0 + 0.35 * recent.mean(), FB_MIN, FB_MAX))

        dirn = float(direction[i])
        entry = cl[i]
        stop = entry - dirn * STOP_K * a
        tgt = entry + dirn * TARGET_K * a
        risk_oz = STOP_K * a
        risk_cash = RISK_PER_TRADE * eq * mult

        size_oz = risk_cash / risk_oz
        notional = size_oz * entry
        if notional > MAX_LEV * eq:
            size_oz *= (MAX_LEV * eq) / notional

        exit_price, exit_bar, reason = None, i, "hold_end"
        for k in range(1, MAX_HOLD + 1):
            if i + k >= n:
                break
            h, l = hi[i + k], lo[i + k]
            if dirn > 0:
                if l <= stop: exit_price, reason, exit_bar = stop, "stop", i + k; break
                if h >= tgt:  exit_price, reason, exit_bar = tgt, "target", i + k; break
            else:
                if h >= stop: exit_price, reason, exit_bar = stop, "stop", i + k; break
                if l <= tgt:  exit_price, reason, exit_bar = tgt, "target", i + k; break
        if exit_price is None:
            exit_bar = min(i + MAX_HOLD, n - 1)
            exit_price = cl[exit_bar]

        gross = size_oz * dirn * (exit_price - entry)
        cost = size_oz * 2 * spread[i]
        pnl = gross - cost
        r = pnl / max(risk_cash, 1e-9)

        trades.append({"entry_time": df["dt_et"].iloc[i],
                       "exit_time": df["dt_et"].iloc[exit_bar],
                       "dir": dirn, "entry": entry, "exit": exit_price,
                       "size_oz": size_oz, "mult": mult, "pnl": pnl, "r": r,
                       "reason": reason, "bars_held": exit_bar - i})
        taken_R.append(r)

        if exit_bar > i:
            for j in range(i, exit_bar):
                equity[j + 1] = eq + pnl * (j - i + 1) / (exit_bar - i)
        eq += pnl
        equity[min(exit_bar + 1, n)] = eq
        day_trades += 1
        i = exit_bar + 1

    for k in range(i + 1, n + 1):
        equity[k] = eq
    return pd.DataFrame(trades), equity


# ------------------------------------------------------- model

def fit_brain(X, y, w, seed=7):
    pos = max(int(y.sum()), 1)
    spw = float((len(y) - pos) / pos)
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=25,
        reg_lambda=3.0, reg_alpha=0.5, gamma=0.5,
        scale_pos_weight=spw, objective="binary:logistic",
        eval_metric="auc", tree_method="hist", n_jobs=-1, random_state=seed)
    m.fit(X, y, sample_weight=w)
    return m


def fit_brain_rf(X, y, w, seed=7):
    m = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("clf", RandomForestClassifier(
                      n_estimators=500, max_depth=7, min_samples_leaf=40,
                      max_features="sqrt", class_weight="balanced_subsample",
                      n_jobs=-1, random_state=seed))])
    m.fit(X, y, clf__sample_weight=w)
    return m


def pick_threshold(p, r, lo=0.30, hi=0.80, min_frac=0.12):
    """Cut maximising total R on the fitting data, with a support floor."""
    best, cut = -1e18, 0.5
    for c in np.arange(lo, hi, 0.01):
        sel = p >= c
        if sel.sum() < max(25, min_frac * len(p)):
            continue
        v = r[sel].sum()
        if v > best:
            best, cut = v, float(c)
    return cut


def seg_stats(trades, eq, label):
    if len(trades) == 0:
        return {"segment": label, "trades": 0, "total_ret": 0.0,
                "max_dd": 0.0, "hit_rate": np.nan, "avg_R": np.nan,
                "final_equity": float(eq[-1])}
    rets = eq[1:] / eq[:-1] - 1
    return {"segment": label,
            "trades": int(len(trades)),
            "hit_rate": float((trades["pnl"] > 0).mean()),
            "avg_R": float(trades["r"].mean()),
            "total_ret": float(eq[-1] / eq[0] - 1),
            "max_dd": float(_max_drawdown(eq)),
            "sharpe_ann": float(np.nanmean(rets) / (np.nanstd(rets) + 1e-12)
                                * np.sqrt(252 * 78)),
            "final_equity": float(eq[-1]),
            "long": int((trades["dir"] > 0).sum()),
            "short": int((trades["dir"] < 0).sum()),
            "mean_mult": float(trades["mult"].mean())}


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "p2"
    suffix = "" if base == "p3" else f"_{base}"
    model_kind = sys.argv[2] if len(sys.argv) > 2 else "xgb"

    df = (pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet")
          .sort_values("dt_et").reset_index(drop=True))
    df = df.dropna(subset=["atr14", "close"]).reset_index(drop=True)
    df = add_derived(df)
    df = add_structure(df)

    ev = pd.read_parquet(CLEAN / f"phase4_memory_dataset{suffix}.parquet")
    ev = ev.sort_values("i").reset_index(drop=True)

    feats = [c for c in (ML_FEATURES_STRICT + STRUCTURE_FEATURES + MEMORY_FEATURES)
             if c in ev.columns]
    X = clean_matrix(ev[feats])
    y = (ev["r_net"] > 0).astype(int).to_numpy()
    r = ev["r_net"].to_numpy()
    w = np.clip(np.abs(r), 0.05, 5.0)          # magnitude weighting (step 2)

    n = len(ev)
    a = int(n * 0.60)
    b = int(n * 0.90)
    seg = {"train": slice(0, a), "test": slice(a, b), "val": slice(b, n)}
    print(f"events={n:,}  features={len(feats)}  model={model_kind}")
    for k, s in seg.items():
        lo_bar, hi_bar = int(ev['i'].iloc[s.start]), int(ev['i'].iloc[s.stop - 1])
        print(f"  {k:5s}: events {s.start:,}..{s.stop:,}  "
              f"bars {lo_bar:,}..{hi_bar:,}  "
              f"{df['dt_et'].iloc[lo_bar]} -> {df['dt_et'].iloc[hi_bar]}")

    fit = fit_brain if model_kind == "xgb" else fit_brain_rf

    strong = ev["argos_abs"].to_numpy() >= STRONG_THRESH
    print(f"  strong candidates (|signal|>={STRONG_THRESH}): {strong.sum():,}"
          f" of {n:,} ({100*strong.mean():.1f}%)   "
          f"mean R wide={r.mean():+.3f}  strong={r[strong].mean():+.3f}")

    # ---- fit on TRAIN (minus calibration tail), calibrate cut on that tail
    ca = int(a * (1 - CALIB_FRAC))
    m1 = fit(X.iloc[0:ca], y[0:ca], w[0:ca])
    p_cal = m1.predict_proba(X.iloc[ca:a])[:, 1]
    cut1 = pick_threshold(p_cal[strong[ca:a]], r[ca:a][strong[ca:a]])
    p_tr = m1.predict_proba(X.iloc[seg["train"]])[:, 1]
    p_te = m1.predict_proba(X.iloc[seg["test"]])[:, 1]

    # ---- refit on TRAIN+TEST (minus calibration tail), decide on VAL
    cb = int(b * (1 - CALIB_FRAC))
    m2 = fit(X.iloc[0:cb], y[0:cb], w[0:cb])
    p_cal2 = m2.predict_proba(X.iloc[cb:b])[:, 1]
    cut2 = pick_threshold(p_cal2[strong[cb:b]], r[cb:b][strong[cb:b]])
    p_va = m2.predict_proba(X.iloc[seg["val"]])[:, 1]

    print(f"\n  cut1={cut1:.2f} calibrated on held-out train tail "
          f"(events {ca:,}..{a:,})")
    print(f"  cut2={cut2:.2f} calibrated on held-out tail (events {cb:,}..{b:,})")

    probs = {"train": p_tr, "test": p_te, "val": p_va}
    cuts = {"train": cut1, "test": cut1, "val": cut2}

    # ---- run each segment through the sequential engine
    atr_all = df["atr14"].to_numpy()
    results, curves = [], {}

    for name, s in seg.items():
        lo_bar = int(ev["i"].iloc[s.start])
        hi_bar = min(int(ev["i"].iloc[s.stop - 1]) + MAX_HOLD + 2, len(df))
        sub = df.iloc[lo_bar:hi_bar].reset_index(drop=True)
        nb = len(sub)

        pos = ev["i"].to_numpy()[s] - lo_bar
        seg_strong = strong[s]
        seg_dir = ev["dir"].to_numpy()[s]
        in_range = (pos >= 0) & (pos < nb)

        # brain: strong base signal AND model approves
        ok = in_range & seg_strong & (probs[name] >= cuts[name])
        approve = np.zeros(nb, dtype=bool)
        dirv = np.zeros(nb)
        approve[pos[ok]] = True
        dirv[pos[ok]] = seg_dir[ok]
        tr, eq = run_brain_backtest(sub, approve, dirv, atr_all[lo_bar:hi_bar])
        st = seg_stats(tr, eq, name)
        st["approved"] = int(ok.sum())
        st["candidates"] = int(seg_strong.sum())
        results.append(st)
        curves[name] = {"t": pd.to_datetime(sub["dt_et"]).to_numpy(),
                        "eq": eq[1:], "trades": tr}

        # control: same strong candidates, same engine, NO brain filter
        okc = in_range & seg_strong
        approve_c = np.zeros(nb, dtype=bool)
        dirv_c = np.zeros(nb)
        approve_c[pos[okc]] = True
        dirv_c[pos[okc]] = seg_dir[okc]
        trc, eqc = run_brain_backtest(sub, approve_c, dirv_c, atr_all[lo_bar:hi_bar])
        stc = seg_stats(trc, eqc, name + "_nofilter")
        stc["approved"] = int(okc.sum())
        stc["candidates"] = int(seg_strong.sum())
        results.append(stc)
        curves[name + "_nofilter"] = {"t": pd.to_datetime(sub["dt_et"]).to_numpy(),
                                      "eq": eqc[1:], "trades": trc}

    res = pd.DataFrame(results)
    print(f"\n{'='*94}\nBRAIN RESULTS — 60/30/10, each segment starts at ${CAPITAL_0:,.0f}\n{'='*94}")
    cols = ["segment", "candidates", "approved", "trades", "hit_rate", "avg_R",
            "total_ret", "max_dd", "sharpe_ann", "final_equity", "long", "short",
            "mean_mult"]
    show = res[[c for c in cols if c in res.columns]].copy()
    for c in ("hit_rate", "total_ret", "max_dd"):
        show[c] = (show[c].astype(float) * 100).round(2)
    for c in ("avg_R", "sharpe_ann", "mean_mult"):
        show[c] = show[c].astype(float).round(3)
    show["final_equity"] = show["final_equity"].astype(float).round(0)
    print(show.to_string(index=False))

    out = {"base": base, "model": model_kind, "n_features": len(feats),
           "cuts": {"train": cut1, "val": cut2},
           "results": results}
    with open(CLEAN / f"phase4_brain_results{suffix}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    rows = []
    for k, v in curves.items():
        rows.append(pd.DataFrame({"segment": k, "dt_et": v["t"], "equity": v["eq"]}))
    pd.concat(rows).to_parquet(CLEAN / f"phase4_brain_equity{suffix}.parquet",
                               index=False, compression="zstd")
    print(f"\nWrote {CLEAN/f'phase4_brain_results{suffix}.json'}")
    print(f"Wrote {CLEAN/f'phase4_brain_equity{suffix}.parquet'}")

    if model_kind == "xgb":
        bst = m2.get_booster()
        bst.feature_names = list(feats)
        g = pd.Series(bst.get_score(importance_type="gain"))
        g = (g / g.sum() * 100).sort_values(ascending=False)
        print(f"\n--- top 20 features by gain (model refit on train+test) ---")
        print(g.head(20).round(2).to_string())
        mem_share = g[[i for i in g.index if i.startswith(("mem_", "knn_", "buck_"))]].sum()
        str_share = g[[i for i in g.index if i in STRUCTURE_FEATURES]].sum()
        print(f"\nmemory features = {mem_share:.1f}% of total gain")
        print(f"structure features = {str_share:.1f}% of total gain")


if __name__ == "__main__":
    main()
