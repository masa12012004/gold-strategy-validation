"""
Phase 4j — Equity curves for the 60/30/10 brain, all segments in one image.

Top row: TRAIN / TEST / VAL side by side, each starting at $5,000, brain vs the
same strong-signal pool with no brain filter.
Bottom row: the continuous run (memory and equity carried across all three), and
the feature-gain split showing how much the model leans on memory.
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

from phase2_common import CAPITAL_0

CLEAN = (Path(__file__).resolve().parent / "clean_Data")
SEGS = ["train", "test", "val"]
TITLES = {"train": "TRAIN (60%) — in-sample",
          "test": "TEST (30%) — out-of-sample",
          "val": "VALIDATION (10%) — fully held out"}


def dd(eq):
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / peak * 100


def main() -> None:
    suffix = "_p2"
    eqdf = pd.read_parquet(CLEAN / f"phase4_brain_equity{suffix}.parquet")
    with open(CLEAN / f"phase4_brain_results{suffix}.json") as f:
        rep = json.load(f)
    res = {r["segment"]: r for r in rep["results"]}

    fig = plt.figure(figsize=(16.5, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.32, wspace=0.22)

    # ---------------- top row: one panel per segment
    for j, seg in enumerate(SEGS):
        ax = fig.add_subplot(gs[0, j])
        a = eqdf[eqdf["segment"] == seg]
        b = eqdf[eqdf["segment"] == seg + "_nofilter"]
        ta = pd.to_datetime(a["dt_et"])
        tb = pd.to_datetime(b["dt_et"])

        ax.plot(tb, b["equity"], lw=1.1, color="#999999",
                label=f"no brain  {res[seg+'_nofilter']['total_ret']*100:+.2f}%")
        ax.plot(ta, a["equity"], lw=1.6, color="#e45756",
                label=f"brain  {res[seg]['total_ret']*100:+.2f}%")
        ax.axhline(CAPITAL_0, color="k", lw=0.6, ls="--")
        ax.axhline(CAPITAL_0 * 1.08, color="green", lw=0.9, ls=":",
                   label="+8% target")
        ax.axhline(CAPITAL_0 * 0.90, color="red", lw=0.9, ls=":")

        r = res[seg]
        ax.set_title(f"{TITLES[seg]}\n"
                     f"{r['trades']} trades · hit {r['hit_rate']*100:.1f}% · "
                     f"avgR {r['avg_R']:+.2f} · DD {r['max_dd']*100:.2f}%",
                     fontsize=10)
        ax.set_ylabel("Equity ($)" if j == 0 else "")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2 if seg != "val" else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)

    # ---------------- bottom left+mid: continuous run
    ax = fig.add_subplot(gs[1, :2])
    eq_c, t_c, marks = [], [], []
    run = CAPITAL_0
    for seg in SEGS:
        a = eqdf[eqdf["segment"] == seg]
        e = a["equity"].to_numpy()
        scaled = run * (e / e[0])
        eq_c.append(scaled)
        t_c.append(pd.to_datetime(a["dt_et"]))
        run = scaled[-1]
        marks.append(pd.to_datetime(a["dt_et"]).iloc[0])

    eq_nf, t_nf = [], []
    run_nf = CAPITAL_0
    for seg in SEGS:
        b = eqdf[eqdf["segment"] == seg + "_nofilter"]
        e = b["equity"].to_numpy()
        scaled = run_nf * (e / e[0])
        eq_nf.append(scaled)
        t_nf.append(pd.to_datetime(b["dt_et"]))
        run_nf = scaled[-1]

    T = pd.concat(t_c)
    E = np.concatenate(eq_c)
    Tn = pd.concat(t_nf)
    En = np.concatenate(eq_nf)
    ax.plot(Tn, En, lw=1.1, color="#999999",
            label=f"no brain, compounded   final ${En[-1]:,.0f}")
    ax.plot(T, E, lw=1.6, color="#e45756",
            label=f"brain, compounded   final ${E[-1]:,.0f}")
    ax.axhline(CAPITAL_0, color="k", lw=0.6, ls="--")
    ax.axhline(CAPITAL_0 * 1.08, color="green", lw=0.9, ls=":", label="+8% target")
    for m, lbl in zip(marks[1:], ["TEST begins", "VAL begins"]):
        ax.axvline(m, color="navy", lw=1.0, ls="--", alpha=0.7)
        ax.text(m, ax.get_ylim()[1], f" {lbl}", fontsize=8,
                color="navy", va="top")
    ax.set_title("Continuous run — memory and equity carried across all three segments",
                 fontsize=10)
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)

    # ---------------- bottom right: segment summary table as a bar chart
    ax = fig.add_subplot(gs[1, 2])
    labels = ["TRAIN", "TEST", "VAL"]
    brain = [res[s]["total_ret"] * 100 for s in SEGS]
    nof = [res[s + "_nofilter"]["total_ret"] * 100 for s in SEGS]
    x = np.arange(3)
    ax.bar(x - 0.2, nof, 0.4, color="#999999", label="no brain")
    ax.bar(x + 0.2, brain, 0.4, color="#e45756", label="brain")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(8, color="green", lw=0.9, ls=":", label="+8% target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Return (%)")
    ax.set_title("Return by segment", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    for xi, v in zip(x - 0.2, nof):
        ax.text(xi, v, f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    for xi, v in zip(x + 0.2, brain):
        ax.text(xi, v, f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)

    fig.suptitle("Phase 4 — memory-conditioned brain, 60/30/10 sequential split "
                 "(memory expands forward; no segment sees its own future)",
                 fontsize=12.5, y=0.985)
    out = CLEAN / "phase4_brain_equity.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    print(f"\nContinuous: no-brain ${En[-1]:,.0f}  brain ${E[-1]:,.0f}")
    for s in SEGS:
        r, rn = res[s], res[s + "_nofilter"]
        print(f"  {s:5s} brain {r['total_ret']*100:+6.2f}%  "
              f"(DD {r['max_dd']*100:+.2f}%, {r['trades']} trades)   "
              f"| no-brain {rn['total_ret']*100:+6.2f}%  "
              f"(DD {rn['max_dd']*100:+.2f}%, {rn['trades']} trades)")


if __name__ == "__main__":
    main()
