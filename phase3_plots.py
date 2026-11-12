"""Phase 3 equity curve vs Phase 2 best."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from phase2_common import BtCfg, CAPITAL_0, run_backtest, trade_metrics
from phase2c_strategy import ensemble as ensemble2
from phase3_strategy import ensemble3, friction_kelly_sizing

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


def main() -> None:
    with open(CLEAN / "phase2_best_params.json") as f:
        best2 = json.load(f)
    with open(CLEAN / "phase3_best_params.json") as f:
        best3 = json.load(f)

    # Phase 2 ran on features2; Phase 3 on features3. We need both on the same plot grid,
    # so use the features3 parquet for timestamps and features2 columns are a subset.
    df3 = pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet").sort_values("dt_et").reset_index(drop=True)
    df3 = df3.dropna(subset=["atr14","close"]).reset_index(drop=True)
    t = pd.to_datetime(df3["dt_et"])
    oos_ts = t.quantile(0.80)

    cfg = BtCfg(entry_thresh=0.25, risk_per_trade=0.005, stop_k=1.8, target_k=2.8,
                max_trades_day=8, daily_soft=0.03, total_soft=0.07)

    # Phase 2 (reconstruct on same bars)
    w2 = {k: float(v) for k, v in best2["weights"].items()}
    s2 = ensemble2(df3, w2, best2["params"])
    _, eq2 = run_backtest(df3, s2, df3["atr14"].to_numpy(),
                          cfg=BtCfg(entry_thresh=0.35, risk_per_trade=0.005))
    eq2 = eq2[1:]

    # Phase 3
    w3 = {k: float(v) for k, v in best3["weights"].items()}
    s3 = ensemble3(df3, w3, best3["params"])
    size_scale = friction_kelly_sizing(df3, s3)
    atr_eff = df3["atr14"].to_numpy() / size_scale
    _, eq3 = run_backtest(df3, s3, atr_eff, cfg=cfg)
    eq3 = eq3[1:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios":[3, 1]})
    ax1.plot(t, eq2, lw=1.3, color="#4c78a8", label=f"Phase 2 best {w2}")
    ax1.plot(t, eq3, lw=1.3, color="#54a24b", label=f"Phase 3 best {w3}")
    ax1.axhline(CAPITAL_0, color="k", lw=0.5, ls="--")
    ax1.axhline(CAPITAL_0 * 1.08, color="green", lw=0.8, ls=":", label="+8% target")
    ax1.axhline(CAPITAL_0 * 0.90, color="red",   lw=0.8, ls=":", label="-10% hard")
    ax1.axvline(oos_ts, color="gray", lw=1.0, ls="--", label="OOS cut")
    ax1.set_title("Phase 2 vs Phase 3 — equity curves (Argos classical vs lit-upgraded)")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=9)

    def dd(eq):
        peak = np.maximum.accumulate(eq)
        return (eq - peak) / peak * 100
    ax2.fill_between(t, dd(eq2), 0, color="#4c78a8", alpha=0.35, label="Phase 2 DD")
    ax2.fill_between(t, dd(eq3), 0, color="#54a24b", alpha=0.35, label="Phase 3 DD")
    ax2.axhline(-5,  color="orange", lw=0.6, ls=":", label="-5% daily")
    ax2.axhline(-10, color="red",    lw=0.6, ls=":", label="-10% total")
    ax2.axvline(oos_ts, color="gray", lw=1.0, ls="--")
    ax2.set_ylabel("DD (%)"); ax2.legend(loc="lower left", fontsize=8)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); fig.tight_layout()
    out = CLEAN / "phase3_vs_phase2_equity.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"Saved {out}")

    # Summary table
    def summ(eq, label):
        r = eq[-1] / eq[0] - 1
        d = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
        return f"{label:18s}  final=${eq[-1]:,.0f}  ret={r*100:+.2f}%  maxDD={d.min()*100:+.2f}%"
    print(summ(eq2, "Phase 2 best"))
    print(summ(eq3, "Phase 3 best"))


if __name__ == "__main__":
    main()
