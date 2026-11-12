"""Phase 4b — trade-frequency sweep visualisation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLEAN = (Path(__file__).resolve().parent / "clean_Data")
ORDER = ["A_baseline", "B_2x", "C_4x", "D_8x", "E_max"]


def main() -> None:
    with open(CLEAN / "phase4b_results.json") as f:
        rows = json.load(f)["configs"]
    res = pd.DataFrame(rows).set_index("config")
    mon = pd.read_csv(CLEAN / "phase4b_monthly.csv")

    r = res.loc[ORDER]
    x = r["trades_per_month"].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))

    # 1. frequency vs mean monthly return
    ax = axes[0, 0]
    ax.plot(x, r["mean_month_%"], "o-", color="#e45756", lw=2, ms=8)
    for xi, yi, n in zip(x, r["mean_month_%"], ORDER):
        ax.annotate(n.split("_")[0], (xi, yi), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("trades per month"); ax.set_ylabel("mean monthly return (%)")
    ax.set_title("More trades → worse returns", fontsize=11)
    ax.grid(alpha=0.25)

    # 2. frequency vs edge quality
    ax = axes[0, 1]
    ax.plot(x, r["hit_rate"], "o-", color="#4c78a8", lw=2, ms=7, label="hit rate %")
    ax.set_xlabel("trades per month"); ax.set_ylabel("hit rate (%)", color="#4c78a8")
    ax.axhline(50, color="#4c78a8", ls=":", lw=1)
    ax2 = ax.twinx()
    ax2.plot(x, r["avg_R"], "s--", color="#54a24b", lw=2, ms=7, label="avg R")
    ax2.set_ylabel("avg R per trade", color="#54a24b")
    ax2.axhline(0, color="#54a24b", ls=":", lw=1)
    ax.set_title("Edge per trade decays with frequency", fontsize=11)
    ax.grid(alpha=0.25)

    # 3. drawdown breaches
    ax = axes[0, 2]
    ax.bar(range(len(r)), r["months_dd_breach"], color="#e45756", alpha=0.85)
    ax.plot(range(len(r)), r["months"], "k--", lw=1, label="months tested")
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels([n.split("_")[0] for n in ORDER])
    ax.set_ylabel("months breaching -5% daily DD")
    ax.set_title("Challenge-fatal drawdowns", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate(r["months_dd_breach"]):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)

    # 4. monthly return distribution
    ax = axes[1, 0]
    data = [mon[mon["config"] == c]["ret_%"].dropna().to_numpy() for c in ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for p in bp["boxes"]:
        p.set_facecolor("#9ecae1")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(8, color="green", ls=":", lw=1.2, label="+8% target")
    ax.axhline(-5, color="red", ls=":", lw=1.2, label="-5% daily limit")
    ax.set_xticklabels([n.split("_")[0] for n in ORDER])
    ax.set_ylabel("monthly return (%)")
    ax.set_title("Monthly return distribution", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

    # 5. months reaching target
    ax = axes[1, 1]
    w = 0.38
    idx = np.arange(len(r))
    ax.bar(idx - w / 2, r["pos_months"], w, color="#54a24b", label="positive months")
    ax.bar(idx + w / 2, r["months_ge_8pct"], w, color="#f58518", label="months ≥ +8%")
    ax.set_xticks(idx)
    ax.set_xticklabels([n.split("_")[0] for n in ORDER])
    ax.set_ylabel("count (of 18)")
    ax.set_title("Months that actually worked", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    for i, (a, b) in enumerate(zip(r["pos_months"], r["months_ge_8pct"])):
        ax.text(i - w / 2, a, str(int(a)), ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, b, str(int(b)), ha="center", va="bottom", fontsize=9)

    # 6. does the brain help at high frequency?
    ax = axes[1, 2]
    c4 = mon[mon["config"] == "C_4x"].set_index("month")["ret_%"]
    cn = mon[mon["config"] == "C_nofilter"].set_index("month")["ret_%"]
    both = pd.concat([c4.rename("brain"), cn.rename("no brain")], axis=1).dropna()
    idx = np.arange(len(both))
    ax.bar(idx - 0.2, both["no brain"], 0.4, color="#999999", label="no brain")
    ax.bar(idx + 0.2, both["brain"], 0.4, color="#e45756", label="brain")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(idx[::3])
    ax.set_xticklabels([str(m)[2:] for m in both.index[::3]], fontsize=7, rotation=45)
    ax.set_ylabel("monthly return (%)")
    ax.set_title(f"Brain vs no brain at same frequency (C_4x)\n"
                 f"mean {both['brain'].mean():+.2f}% vs "
                 f"{both['no brain'].mean():+.2f}%", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Phase 4b — online brain, concurrent entries, trade-frequency sweep "
                 "(each month an independent $5,000 attempt; brain learns across all)",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = CLEAN / "phase4b_frequency.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
