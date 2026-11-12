"""
Phase 4f — Equity curves: Phase 2 (classical) vs Phase 3 (lit-upgraded)
vs Phase 4 (ML meta-label gate), all on the OOF-covered span.

Also plots the ML diagnostics worth looking at: the probability distribution
split by realised outcome (does the model separate winners at all?) and the
XGBoost gain importance profile (is any feature actually carrying signal?).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase2_common import BtCfg, CAPITAL_0, run_backtest
from phase2c_strategy import ensemble as ensemble2
from phase3_strategy import ensemble3, friction_kelly_sizing
from phase4_features import add_derived

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


def dd(eq):
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / peak * 100


def main() -> None:
    with open(CLEAN / "phase2_best_params.json") as f:
        best2 = json.load(f)
    with open(CLEAN / "phase3_best_params.json") as f:
        best3 = json.load(f)

    df = (pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet")
          .sort_values("dt_et").reset_index(drop=True))
    df = df.dropna(subset=["atr14", "close"]).reset_index(drop=True)
    df = add_derived(df)

    oof = pd.read_parquet(CLEAN / "phase4_oof_predictions.parquet")
    oof = oof[oof["fold"] >= 0].reset_index(drop=True)
    lo, hi = int(oof["i"].min()), int(oof["i"].max()) + 1
    sub = df.iloc[lo:hi].reset_index(drop=True)
    t = pd.to_datetime(sub["dt_et"])
    n = len(sub)

    cfg3 = BtCfg(entry_thresh=0.25, risk_per_trade=0.005, stop_k=1.8, target_k=2.8,
                 max_trades_day=8, daily_soft=0.03, total_soft=0.07)

    # Phase 2
    w2 = {k: float(v) for k, v in best2["weights"].items()}
    s2 = ensemble2(df, w2, best2["params"])[lo:hi]
    _, eq2 = run_backtest(sub, s2, sub["atr14"].to_numpy(),
                          cfg=BtCfg(entry_thresh=0.35, risk_per_trade=0.005))
    eq2 = eq2[1:]

    # Phase 3
    w3 = {k: float(v) for k, v in best3["weights"].items()}
    s3_full = ensemble3(df, w3, best3["params"])
    scale = friction_kelly_sizing(df, s3_full)
    atr_eff = (df["atr14"].to_numpy() / scale)[lo:hi]
    s3 = s3_full[lo:hi]
    _, eq3 = run_backtest(sub, s3, atr_eff, cfg=cfg3)
    eq3 = eq3[1:]

    # Phase 4 — pick the model that won the backtest comparison
    with open(CLEAN / "phase4_backtest_results.json") as f:
        bt = json.load(f)["results"]
    inter = {k: v for k, v in bt.items() if k.startswith("argos_and_")}
    best_mdl = max(inter, key=lambda k: inter[k]["total_ret"]).replace("argos_and_", "")
    print(f"Best intersection model: {best_mdl}")

    pos = oof["i"].to_numpy() - lo
    p = oof[f"p_{best_mdl}"].to_numpy()
    cut = oof[f"cut_{best_mdl}"].to_numpy()
    keep = np.zeros(n, dtype=bool)
    ok = (pos >= 0) & (pos < n) & (p >= cut)
    keep[pos[ok]] = True
    s4 = np.where(keep, s3, 0.0)
    _, eq4 = run_backtest(sub, s4, atr_eff, cfg=cfg3)
    eq4 = eq4[1:]

    # ---------------------------------------------------- figure 1: equity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(t, eq2, lw=1.2, color="#4c78a8", label="Phase 2 — classical")
    ax1.plot(t, eq3, lw=1.2, color="#54a24b", label="Phase 3 — lit-upgraded")
    ax1.plot(t, eq4, lw=1.6, color="#e45756",
             label=f"Phase 4 — Argos ∩ ML gate ({best_mdl})")
    ax1.axhline(CAPITAL_0, color="k", lw=0.5, ls="--")
    ax1.axhline(CAPITAL_0 * 1.08, color="green", lw=0.8, ls=":", label="+8% target")
    ax1.axhline(CAPITAL_0 * 0.90, color="red", lw=0.8, ls=":", label="-10% hard")
    ax1.set_title("Phase 2 vs 3 vs 4 — equity on the out-of-fold span "
                  "(ML predictions are purged + embargoed, never in-sample)")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.fill_between(t, dd(eq2), 0, color="#4c78a8", alpha=0.3, label="P2 DD")
    ax2.fill_between(t, dd(eq3), 0, color="#54a24b", alpha=0.3, label="P3 DD")
    ax2.fill_between(t, dd(eq4), 0, color="#e45756", alpha=0.35, label="P4 DD")
    ax2.axhline(-5, color="orange", lw=0.6, ls=":", label="-5% daily")
    ax2.axhline(-10, color="red", lw=0.6, ls=":", label="-10% total")
    ax2.set_ylabel("DD (%)")
    ax2.legend(loc="lower left", fontsize=8, ncol=3)
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    out1 = CLEAN / "phase4_equity_comparison.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"Saved {out1}")

    # ------------------------------------------- figure 2: ML diagnostics
    with open(CLEAN / "phase4_ml_report.json") as f:
        rep = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    y = oof["y"].to_numpy().astype(bool)
    ax.hist(p[y], bins=40, alpha=0.6, color="#54a24b", density=True, label="winners")
    ax.hist(p[~y], bins=40, alpha=0.6, color="#e45756", density=True, label="losers")
    ax.axvline(np.nanmedian(cut), color="k", ls="--", lw=1,
               label=f"cut ≈ {np.nanmedian(cut):.2f}")
    ax.set_title(f"OOF probability by outcome ({best_mdl})\n"
                 "near-total overlap = weak separation")
    ax.set_xlabel("P(win)")
    ax.legend(fontsize=8)

    ax = axes[1]
    g = pd.Series(rep["meta_strict"]["xgb_gain_pct"]).sort_values(ascending=False).head(18)
    ax.barh(range(len(g)), g.values[::-1], color="#4c78a8")
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index[::-1], fontsize=7)
    ax.set_xlabel("% of total gain")
    ax.set_title("XGBoost gain importance (strict)\nflat profile = no dominant edge")

    ax = axes[2]
    folds = pd.DataFrame(rep["meta_strict"]["folds"])
    for m, grp in folds.groupby("model"):
        ax.plot(grp["fold"], grp["auc"], marker="o", lw=1.3, label=m)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="random")
    ax.set_xlabel("fold")
    ax.set_ylabel("AUC")
    ax.set_title("Out-of-fold AUC by model\n(0.52–0.54 = thin but real)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out2 = CLEAN / "phase4_ml_diagnostics.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"Saved {out2}")

    # ---------------------------------------------------- summary
    def summ(eq, label):
        r = eq[-1] / eq[0] - 1
        d = dd(eq)
        return f"{label:34s} final=${eq[-1]:,.0f}  ret={r*100:+.2f}%  maxDD={d.min():+.2f}%"
    print()
    print(summ(eq2, "Phase 2 classical"))
    print(summ(eq3, "Phase 3 lit-upgraded"))
    print(summ(eq4, f"Phase 4 Argos AND ML ({best_mdl})"))


if __name__ == "__main__":
    main()
