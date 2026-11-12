"""
Phase 3 — Walk-forward optimization with the extended signal set.
Searches over weights for A/B/D/E/F (C = London breakout held out, low value).
Plus T_entry, z_entry, and CPD min thresholds.
Compares to Phase 2d winner.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_common import (BtCfg, run_backtest, trade_metrics, daily_pnl)
from phase3_strategy import ensemble3, friction_kelly_sizing
from phase2d_walkforward import mc_pass_probability

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


def score(m: dict, lam: float = 2.0) -> float:
    if m.get("trades", 0) < 15: return -1e9
    return m["sharpe_ann"] + lam * m["max_dd"]


def eval_cfg(df, weights, params, cfg):
    s = ensemble3(df, weights, params)
    atr = df["atr14"].to_numpy()
    size_scale = friction_kelly_sizing(df, s)
    atr_eff = atr / size_scale
    trades, eq = run_backtest(df, s, atr_eff, cfg=cfg)
    m = trade_metrics(trades, eq)
    m["_trades"] = trades; m["_eq"] = eq
    return m


def grid_search(df, cfg):
    best = (-1e18, None, None, None)
    grid = {
        "wA": [0.10, 0.25, 0.40],
        "wB": [0.10, 0.25, 0.40],
        "wD": [0.00, 0.15, 0.30],
        "wE": [0.00, 0.15, 0.30],
        "wF": [0.00, 0.10],
        "T":  [2.5, 3.5],
        "z":  [1.5, 2.0],
    }
    for wA, wB, wD, wE, wF, T, z in itertools.product(
        grid["wA"], grid["wB"], grid["wD"], grid["wE"], grid["wF"], grid["T"], grid["z"]):
        tot = wA + wB + wD + wE + wF
        if tot == 0: continue
        w = {"A":wA, "B":wB, "C":0.0, "D":wD, "E":wE, "F":wF}
        p = {"T_entry":T, "z_entry":z, "cpd_min":2.5}
        m = eval_cfg(df, w, p, cfg)
        sc = score(m)
        if sc > best[0]:
            best = (sc, w, p, m)
    return best[1], best[2], best[3]


def main() -> None:
    df = pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet").sort_values("dt_et").reset_index(drop=True)
    df = df.dropna(subset=["atr14","close"]).reset_index(drop=True)

    cfg = BtCfg(entry_thresh=0.25, risk_per_trade=0.005, stop_k=1.8, target_k=2.8,
                max_trades_day=8, daily_soft=0.03, total_soft=0.07)

    dates = pd.to_datetime(df["dt_et"]).dt.date
    unique_days = np.array(sorted(dates.unique()))
    oos_cut = unique_days[int(len(unique_days) * 0.80)]
    in_sample = unique_days[:int(len(unique_days) * 0.80)]
    print(f"In-sample days: {len(in_sample)}  OOS cut: {oos_cut}")

    folds = np.array_split(in_sample, 3)
    fold_results = []
    for i in range(1, 3):
        train_days = np.concatenate(folds[:i])
        val_days   = folds[i]
        m_train = df[dates.isin(train_days)].reset_index(drop=True)
        m_val   = df[dates.isin(val_days)].reset_index(drop=True)
        print(f"\n--- fold {i}: train {train_days[0]}..{train_days[-1]}  "
              f"validate {val_days[0]}..{val_days[-1]} ---")
        w, p, _ = grid_search(m_train, cfg)
        print(f"  best w: {w}  p: {p}")
        m_v = eval_cfg(m_val, w, p, cfg)
        print(f"  VALIDATE trades={m_v['trades']} hit={m_v['hit_rate']*100:.1f}% "
              f"avgR={m_v['avg_R']:+.2f} ret={m_v['total_ret']*100:+.2f}% "
              f"maxDD={m_v['max_dd']*100:+.2f}% Sharpe={m_v['sharpe_ann']:+.2f}")
        fold_results.append({"weights": w, "params": p, "val": m_v})

    best = max(fold_results, key=lambda r: score(r["val"]))
    print(f"\n=== Walk-forward best ===\nWeights: {best['weights']}  Params: {best['params']}")

    full = df[dates.isin(in_sample)].reset_index(drop=True)
    m_full = eval_cfg(full, best["weights"], best["params"], cfg)
    mc = mc_pass_probability(m_full["_trades"], days_per_sim=20, n_sims=10000)
    print(f"\nFull in-sample: trades={m_full['trades']} ret={m_full['total_ret']*100:+.2f}% "
          f"maxDD={m_full['max_dd']*100:+.2f}% Sharpe={m_full['sharpe_ann']:+.2f}")
    print(f"\nMC (10k × 20-day): P(hit 8%)={mc['hit_target']*100:.1f}%  "
          f"P(bust daily)={mc['bust_daily']*100:.1f}%  "
          f"P(bust total)={mc['bust_total']*100:.1f}%  "
          f"P(PASS)={mc['pass']*100:.1f}%")

    # OOS
    oos = df[dates >= oos_cut].reset_index(drop=True)
    m_oos = eval_cfg(oos, best["weights"], best["params"], cfg)
    print(f"\nOOS: trades={m_oos['trades']} hit={m_oos['hit_rate']*100:.1f}% "
          f"ret={m_oos['total_ret']*100:+.2f}% maxDD={m_oos['max_dd']*100:+.2f}% "
          f"Sharpe={m_oos['sharpe_ann']:+.2f}")

    with open(CLEAN / "phase3_best_params.json", "w") as f:
        json.dump({"weights": best["weights"], "params": best["params"],
                   "val": {k: v for k, v in best["val"].items() if not k.startswith("_")},
                   "in_sample": {k: v for k, v in m_full.items() if not k.startswith("_")},
                   "oos":      {k: v for k, v in m_oos.items()  if not k.startswith("_")},
                   "mc": mc}, f, indent=2, default=str)
    print(f"\nSaved {CLEAN/'phase3_best_params.json'}")


if __name__ == "__main__":
    main()
