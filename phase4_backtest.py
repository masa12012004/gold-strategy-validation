"""
Phase 4d — Does the ML layer actually improve the strategy?

The AUC numbers in phase4_ml are only a proxy. What matters is the equity curve
through the same engine Phases 2 and 3 used. So: take the out-of-fold
probabilities (never in-sample, purged and embargoed), use them to gate or size
the Argos signal, and run the identical backtester.

Everything is compared on the *same bar span* — the span covered by OOF
predictions — because the ML variants cannot trade before their first fold. A
comparison against Phase 3's full-sample run would flatter whichever strategy
happened to own the better period.

Variants:
  argos_raw       Phase 3 signal, engine threshold 0.25, no ML            (control)
  argos_wide      Phase 3 signal, threshold 0.10 (the ML candidate pool)  (control)
  ml_gate_<m>     wide pool, keep only events with p >= training-fold cut
  ml_size_<m>     wide pool, keep p >= cut AND scale size by p
  ml_strict_<m>   wide pool, keep only high-confidence p >= cut + 0.05
  argos_and_ml    engine threshold 0.25 AND p >= cut (intersection)

Writes: clean_Data/phase4_backtest_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_common import BtCfg, run_backtest, trade_metrics, pretty, CAPITAL_0
from phase3_strategy import ensemble3, friction_kelly_sizing
from phase4_features import add_derived

CLEAN = (Path(__file__).resolve().parent / "clean_Data")
MODELS = ["logit", "rf", "xgb", "lgbm"]


def load(base: str = "p3") -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    pfile = "phase2_best_params.json" if base == "p2" else "phase3_best_params.json"
    with open(CLEAN / pfile) as f:
        bestp = json.load(f)
    weights = {k: float(v) for k, v in bestp["weights"].items()}
    params = bestp["params"]

    df = (pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet")
          .sort_values("dt_et").reset_index(drop=True))
    df = df.dropna(subset=["atr14", "close"]).reset_index(drop=True)
    df = add_derived(df)

    suffix = "" if base == "p3" else f"_{base}"
    oof = pd.read_parquet(CLEAN / f"phase4_oof_predictions{suffix}.parquet")
    oof = oof[oof["fold"] >= 0].reset_index(drop=True)
    return df, oof, weights, params


def main(base: str = "p3") -> None:
    df, oof, weights, params = load(base)
    print(f"BASE = {base}  weights={weights}\n")

    if base == "p2":
        from phase2c_strategy import ensemble as ensemble2
        cfg = BtCfg(entry_thresh=0.35, risk_per_trade=0.005)
        s_full = ensemble2(df, weights, params)
        atr_eff_full = df["atr14"].to_numpy()
    else:
        cfg = BtCfg(entry_thresh=0.25, risk_per_trade=0.005, stop_k=1.8,
                    target_k=2.8, max_trades_day=8,
                    daily_soft=0.03, total_soft=0.07)
        s_full = ensemble3(df, weights, params)
        size_scale = friction_kelly_sizing(df, s_full)
        atr_eff_full = df["atr14"].to_numpy() / size_scale

    # restrict every variant to the OOF-covered span
    lo = int(oof["i"].min())
    hi = int(oof["i"].max()) + 1
    sl = slice(lo, hi)
    sub = df.iloc[sl].reset_index(drop=True)
    s_sub = s_full[sl]
    atr_sub = atr_eff_full[sl]
    n = len(sub)
    print(f"OOF-covered span: bars {lo:,}..{hi:,}  ({n:,} bars)  "
          f"{sub['dt_et'].iloc[0]} -> {sub['dt_et'].iloc[-1]}")
    print(f"OOF events in span: {len(oof):,}\n")

    # map each OOF event onto the sub-array position
    pos = oof["i"].to_numpy() - lo
    valid = (pos >= 0) & (pos < n)
    pos = pos[valid]
    oof = oof.loc[valid].reset_index(drop=True)

    results = {}

    def record(label: str, sig: np.ndarray, atr: np.ndarray, cfg_use: BtCfg):
        tr, eq = run_backtest(sub, sig, atr, cfg=cfg_use)
        m = trade_metrics(tr, eq)
        m["label"] = label
        results[label] = {k: v for k, v in m.items() if not k.startswith("_")}
        print(f"{label:20s}{pretty(m)}")
        return m, tr, eq

    cfg_wide = BtCfg(**{**cfg.__dict__, "entry_thresh": 0.10})

    print("--- controls (no ML) ---")
    record("argos_raw", s_sub, atr_sub, cfg)
    record("argos_wide", s_sub, atr_sub, cfg_wide)

    print("\n--- ML gate: keep only events the model likes ---")
    for mdl in MODELS:
        p = oof[f"p_{mdl}"].to_numpy()
        cut = oof[f"cut_{mdl}"].to_numpy()
        keep = np.zeros(n, dtype=bool)
        keep[pos[p >= cut]] = True
        record(f"ml_gate_{mdl}", np.where(keep, s_sub, 0.0), atr_sub, cfg_wide)

    print("\n--- ML gate + probability sizing ---")
    for mdl in MODELS:
        p = oof[f"p_{mdl}"].to_numpy()
        cut = oof[f"cut_{mdl}"].to_numpy()
        keep_m = p >= cut
        keep = np.zeros(n, dtype=bool)
        mult = np.ones(n)
        keep[pos[keep_m]] = True
        # scale size by confidence above the cut, 0.5x..1.5x
        conf = np.clip((p - cut) / 0.15, 0.0, 1.0)
        mult[pos[keep_m]] = 0.5 + conf[keep_m]
        record(f"ml_size_{mdl}", np.where(keep, s_sub, 0.0),
               atr_sub / mult, cfg_wide)

    print("\n--- high-confidence only (cut + 0.05) ---")
    for mdl in MODELS:
        p = oof[f"p_{mdl}"].to_numpy()
        cut = oof[f"cut_{mdl}"].to_numpy() + 0.05
        keep = np.zeros(n, dtype=bool)
        keep[pos[p >= cut]] = True
        record(f"ml_strict_{mdl}", np.where(keep, s_sub, 0.0), atr_sub, cfg_wide)

    print("\n--- intersection: Argos 0.25 AND model approves ---")
    for mdl in MODELS:
        p = oof[f"p_{mdl}"].to_numpy()
        cut = oof[f"cut_{mdl}"].to_numpy()
        keep = np.zeros(n, dtype=bool)
        keep[pos[p >= cut]] = True
        sig = np.where(keep, s_sub, 0.0)
        record(f"argos_and_{mdl}", sig, atr_sub, cfg)

    # ---- ranking
    print(f"\n{'='*86}\nRanked by return (same bar span, OOF predictions only)\n{'='*86}")
    tab = (pd.DataFrame(results).T
           .sort_values("total_ret", ascending=False)
           [["trades", "hit_rate", "avg_R", "total_ret", "max_dd",
             "sharpe_ann", "final_equity"]])
    tab["hit_rate"] = (tab["hit_rate"].astype(float) * 100).round(1)
    tab["total_ret"] = (tab["total_ret"].astype(float) * 100).round(2)
    tab["max_dd"] = (tab["max_dd"].astype(float) * 100).round(2)
    tab["avg_R"] = tab["avg_R"].astype(float).round(3)
    tab["sharpe_ann"] = tab["sharpe_ann"].astype(float).round(2)
    tab["final_equity"] = tab["final_equity"].astype(float).round(0)
    print(tab.to_string())

    ctrl = results["argos_raw"]["total_ret"]
    print(f"\nControl (argos_raw, no ML) = {ctrl*100:+.2f}%")
    best_lbl = tab.index[0]
    print(f"Best variant = {best_lbl} at {tab.iloc[0]['total_ret']:+.2f}%  "
          f"=> ML delta {tab.iloc[0]['total_ret'] - ctrl*100:+.2f} pp")
    ml_only = tab.drop(index="argos_raw", errors="ignore")
    best_ml = ml_only.index[0]
    print(f"Best ML variant = {best_ml} at {ml_only.iloc[0]['total_ret']:+.2f}%  "
          f"=> vs control {ml_only.iloc[0]['total_ret'] - ctrl*100:+.2f} pp")

    suffix = "" if base == "p3" else f"_{base}"
    with open(CLEAN / f"phase4_backtest_results{suffix}.json", "w") as f:
        json.dump({"base": base,
                   "span": {"lo": lo, "hi": hi,
                            "start": str(sub["dt_et"].iloc[0]),
                            "end": str(sub["dt_et"].iloc[-1])},
                   "results": results}, f, indent=2, default=str)
    print(f"\nWrote {CLEAN/f'phase4_backtest_results{suffix}.json'}")


if __name__ == "__main__":
    import sys
    main(base=sys.argv[1] if len(sys.argv) > 1 else "p3")
