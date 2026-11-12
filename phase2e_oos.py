"""
Phase 2e — Out-of-sample (OOS) test on the held-out last-20% of data.

Loads the best params from Phase 2d and runs the ensemble strategy, untouched,
on the held-out period. Reports: total return, DD, Sharpe, worst day, days with
hard-stop breach, and Monte Carlo pass probability on OOS trades.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_common import (BtCfg, CAPITAL_0, DAILY_DD_HARD, TOTAL_DD_HARD,
                            load_features, run_backtest, trade_metrics,
                            daily_pnl, pretty)
from phase2c_strategy import ensemble
from phase2d_walkforward import mc_pass_probability

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


def main() -> None:
    with open(CLEAN / "phase2_best_params.json") as f:
        best = json.load(f)
    weights = best["weights"]
    params  = best["params"]
    print(f"Loaded best weights: {weights}  params: {params}")

    df = load_features().sort_values("dt_et").reset_index(drop=True)
    df = df.dropna(subset=["atr14","close"]).reset_index(drop=True)
    dates = pd.to_datetime(df["dt_et"]).dt.date
    unique_days = np.array(sorted(dates.unique()))
    oos_cut = unique_days[int(len(unique_days) * 0.80)]
    oos = df[dates >= oos_cut].reset_index(drop=True)
    print(f"OOS period: {oos['dt_et'].min()} -> {oos['dt_et'].max()}  "
          f"({pd.to_datetime(oos['dt_et']).dt.date.nunique()} trading days)")

    cfg = BtCfg(entry_thresh=0.35)
    s = ensemble(oos, {k: float(v) for k, v in weights.items()}, params)
    trades, eq = run_backtest(oos, s, oos["atr14"].to_numpy(), cfg=cfg)
    m = trade_metrics(trades, eq)
    print(f"\n[OOS backtest] {pretty(m)}")
    if len(trades):
        print(f"  exits: {trades['reason'].value_counts().to_dict()}")
        dp = daily_pnl(trades)
        hard_days = (dp / CAPITAL_0 <= -DAILY_DD_HARD).sum()
        print(f"  trading days: {len(dp)}  worst day: ${dp.min():+,.1f}  "
              f"best day: ${dp.max():+,.1f}  days with >=5% daily loss: {int(hard_days)}")

    # OOS Monte Carlo
    mc = mc_pass_probability(trades, days_per_sim=20, n_sims=10000)
    print(f"\nOOS Monte Carlo (10k paths, 20-day windows):")
    print(f"  P(hit +8%)                       = {mc['hit_target']*100:5.1f}%")
    print(f"  P(bust -5% daily, hard)          = {mc['bust_daily']*100:5.1f}%")
    print(f"  P(bust -10% total, hard)         = {mc['bust_total']*100:5.1f}%")
    print(f"  P(PASS: hit target AND no bust)  = {mc['pass']*100:5.1f}%")

    # Diagnostic: rolling 20-day windows actually realized in OOS
    if len(trades):
        dp_all = dp.reindex(sorted(dp.index)).fillna(0)
        if len(dp_all) >= 20:
            wins = dp_all.rolling(20).sum().dropna() / CAPITAL_0
            print(f"\n  Realized 20-day windows in OOS: "
                  f"max={wins.max()*100:+.2f}%  min={wins.min()*100:+.2f}%  "
                  f"P(>=+8%)={100*(wins>=0.08).mean():.1f}%")


if __name__ == "__main__":
    main()
