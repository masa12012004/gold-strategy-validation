"""
Phase 2d — Walk-forward hyperparameter search + Monte Carlo pass-probability.

Splits the sample into (train, validate) folds that slide forward through time.
On each train fold, searches a small parameter grid for the best score
(risk-penalized Sharpe). Evaluates the selected params on the validate fold.

Monte-Carlo: bootstraps completed trades to estimate P(hit +8% within 30 days
while respecting the -5% daily / -10% total DD hard limits), under the
challenge rules. Writes `clean_Data/phase2_best_params.json` with the
best-on-validate weights/thresholds.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_common import (BtCfg, CAPITAL_0, DAILY_DD_HARD, TOTAL_DD_HARD,
                            TARGET_RET, load_features, run_backtest,
                            trade_metrics, daily_pnl)
from phase2c_strategy import ensemble

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


# ------------------------------------------------- walk-forward orchestration

def score(m: dict, lam: float = 2.0) -> float:
    """Risk-penalized Sharpe; zero trades => -inf."""
    if m.get("trades", 0) < 10: return -1e9
    return m["sharpe_ann"] + lam * m["max_dd"]   # max_dd is negative

def eval_params(df: pd.DataFrame, weights: dict, params: dict, cfg: BtCfg) -> dict:
    s = ensemble(df, weights, params)
    atr = df["atr14"].to_numpy()
    trades, eq = run_backtest(df, s, atr, cfg=cfg)
    m = trade_metrics(trades, eq)
    m["_trades"] = trades
    m["_eq"]     = eq
    return m

def grid_search(df: pd.DataFrame, cfg: BtCfg) -> tuple[dict, dict, dict]:
    best = (-1e18, None, None, None)
    # tight grid, keep runtime reasonable
    wA_grid   = [0.0, 0.25, 0.5]
    wB_grid   = [0.0, 0.25, 0.5]
    wD_grid   = [0.0, 0.25, 0.5]
    T_grid    = [2.5, 3.5]
    z_grid    = [1.5, 2.0, 2.5]
    for wA, wB, wD, T, z in itertools.product(wA_grid, wB_grid, wD_grid, T_grid, z_grid):
        if wA + wB + wD == 0: continue
        weights = {"A": wA, "B": wB, "C": 0.0, "D": wD}
        params  = {"T_entry": T, "z_entry": z}
        m = eval_params(df, weights, params, cfg)
        sc = score(m)
        if sc > best[0]:
            best = (sc, weights, params, m)
    return best[1], best[2], best[3]


# ------------------------------------------------- Monte Carlo pass simulator

def mc_pass_probability(trades: pd.DataFrame, days_per_sim: int = 20,
                         n_sims: int = 5000, capital0: float = CAPITAL_0,
                         target: float = TARGET_RET,
                         daily_dd_hard: float = DAILY_DD_HARD,
                         total_dd_hard: float = TOTAL_DD_HARD,
                         rng_seed: int = 7) -> dict:
    if len(trades) == 0:
        return {"p_pass":0.0, "p_hit_target":0.0, "p_bust_total":0.0, "p_bust_daily":0.0}
    rng = np.random.default_rng(rng_seed)
    trades = trades.copy()
    trades["day"] = pd.to_datetime(trades["entry_time"]).dt.date
    day_pnls = trades.groupby("day")["pnl"].sum().to_numpy()
    if len(day_pnls) < 2:
        return {"p_pass":0.0, "p_hit_target":0.0, "p_bust_total":0.0, "p_bust_daily":0.0}

    outcomes = {"hit_target":0, "bust_total":0, "bust_daily":0, "pass":0}
    for _ in range(n_sims):
        eq = capital0
        peak = capital0
        bust_t = False; bust_d = False; hit = False
        for _d in range(days_per_sim):
            dp = day_pnls[rng.integers(0, len(day_pnls))]
            day_open = eq
            eq += dp
            if eq - day_open <= -daily_dd_hard * day_open:
                bust_d = True; break
            if (eq - capital0) / capital0 <= -total_dd_hard:
                bust_t = True; break
            if (eq - capital0) / capital0 >= target:
                hit = True; break
        if hit: outcomes["hit_target"] += 1
        if bust_t: outcomes["bust_total"] += 1
        if bust_d: outcomes["bust_daily"] += 1
        if hit and not (bust_t or bust_d): outcomes["pass"] += 1
    return {k: v / n_sims for k, v in outcomes.items()}


# ------------------------------------------------- main

def main() -> None:
    df = load_features().sort_values("dt_et").reset_index(drop=True)
    df = df.dropna(subset=["atr14","close"]).reset_index(drop=True)
    cfg = BtCfg(entry_thresh=0.35)

    dates = pd.to_datetime(df["dt_et"]).dt.date
    unique_days = np.array(sorted(dates.unique()))
    print(f"Days in sample: {len(unique_days)}  ({unique_days[0]} .. {unique_days[-1]})")

    # 3 folds; last 20% is held out for phase 2e (OOS)
    oos_cut = unique_days[int(len(unique_days) * 0.80)]
    print(f"Held-out OOS cut: trading days >= {oos_cut} reserved for Phase 2e")
    use = unique_days[:int(len(unique_days) * 0.80)]
    fold_edges = np.array_split(use, 3)

    fold_results = []
    for i, fold in enumerate(fold_edges):
        if i == 0: continue  # need prior fold for train
        train_days = np.concatenate(fold_edges[:i])
        val_days   = fold
        m_train = df[dates.isin(train_days)].reset_index(drop=True)
        m_val   = df[dates.isin(val_days)].reset_index(drop=True)
        print(f"\n--- fold {i}:  train {train_days[0]}..{train_days[-1]}"
              f"  validate {val_days[0]}..{val_days[-1]} ---")
        w, p, _ = grid_search(m_train, cfg)
        print(f"  best weights: {w}  params: {p}")
        m_v = eval_params(m_val, w, p, cfg)
        print(f"  VALIDATE: trades={m_v['trades']} hit={m_v['hit_rate']*100:.1f}% "
              f"avgR={m_v['avg_R']:+.2f} ret={m_v['total_ret']*100:+.2f}% "
              f"maxDD={m_v['max_dd']*100:+.2f}% Sharpe={m_v['sharpe_ann']:+.2f}")
        fold_results.append({"weights": w, "params": p, "val": m_v})

    # Pick best-on-validate
    best = max(fold_results, key=lambda r: score(r["val"]))
    print("\n=== Walk-forward best ===")
    print(f"Weights: {best['weights']}  params: {best['params']}")

    # Monte Carlo on COMBINED train+validate trades of the winning config
    full_train = df[dates.isin(use)].reset_index(drop=True)
    m_full = eval_params(full_train, best["weights"], best["params"], cfg)
    mc = mc_pass_probability(m_full["_trades"], days_per_sim=20, n_sims=10000)
    print(f"\nOn full in-sample (train+val) with best params:")
    print(f"  trades={m_full['trades']} ret={m_full['total_ret']*100:+.2f}% "
          f"maxDD={m_full['max_dd']*100:+.2f}% Sharpe={m_full['sharpe_ann']:+.2f}")
    print(f"\nMonte Carlo over 10,000 20-day paths:")
    print(f"  P(hit +8%)                      = {mc['hit_target']*100:5.1f}%")
    print(f"  P(bust -5% daily, hard)         = {mc['bust_daily']*100:5.1f}%")
    print(f"  P(bust -10% total, hard)        = {mc['bust_total']*100:5.1f}%")
    print(f"  P(PASS: hit target AND no bust) = {mc['pass']*100:5.1f}%")

    out = {"weights": best["weights"], "params": best["params"],
           "val_metrics": {k: v for k, v in best["val"].items() if not k.startswith("_")},
           "mc":  mc}
    with open(CLEAN / "phase2_best_params.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote best params: {CLEAN/'phase2_best_params.json'}")


if __name__ == "__main__":
    main()
