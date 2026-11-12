"""
Switch-case study: when should we trade H1, the policy-gradient agent, the H4 ensemble or
HMM + side information?

Sleeves (daily net returns, each already net of its own trading costs), every one scaled to
the same 15% volatility target with a trailing 20-day estimate (leverage cap 3), so that the
switch is a choice of MODEL, not a choice of leverage:
    H1   daily F2F regime x hourly cubic         RL   policy gradient, positional state
    H4   walk-forward ensemble of the 13 bases   HMM  HMM momentum + side information
Market state for day D uses daily gold bars up to D-1 only:
    trend : daily Forecast-to-Fill regime  (up / flat / down)
    vol   : 20-day realised vol vs its expanding median  (high / low)
    er    : 10-day efficiency ratio |net move| / sum|moves| vs expanding median (trending / choppy)
Switch table: in each state cell pick the sleeve with the best Sharpe among past days in that
cell (needs >= 12 days and a positive mean; otherwise the default H1).
    FROZEN  : partition and table learned on TRAIN (first 60%), then fixed for TEST and VAL.
    ONLINE  : same partition, table re-estimated every day from all days so far
              (train -> test -> val, the same sequential learning protocol as s12).
A switch day is charged 3 bp.  Splits are the same dates as s12.
"""
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import load_daily, RES, FIG, ANN

SURF, INK, INK2, GRID, BENCH = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#9a9992"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SLEEVES = {"H1": "h1_slowmom_x_cubic", "RL": "s09_rl_positional", "H4": "h4_walkforward_ensemble", "HMM": "s03_hmm_sideinfo"}
TGT, SWITCH_COST, MIN_N = 0.15, 3e-4, 12


def daily(name):
    b = pd.read_parquet(f"{RES}/{name}.parquet")
    r = b["net"] if "tday" not in b.columns else b.groupby("tday")["net"].sum()
    r.index = pd.DatetimeIndex(r.index)
    return r


def vol_control(r, tgt=TGT, win=20, cap=3.0):
    s = r.rolling(win, min_periods=10).std().shift(1) * np.sqrt(ANN)
    return r * (tgt / s).clip(upper=cap).fillna(0.0)


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(ANN) if len(r) > 2 and r.std() > 0 else 0.0


def states(index):
    d = load_daily()
    d.index = pd.DatetimeIndex(d["tday"].values)
    f2f = pd.read_parquet(f"{RES}/s01_f2f_long_short.parquet")
    f2f = pd.Series(np.sign(f2f["pos"].values), index=pd.DatetimeIndex(f2f["tday"].values)).reindex(d.index).fillna(0.0)
    rv = d["lret"].rolling(20).std()
    er = d["close"].diff(10).abs() / d["close"].diff().abs().rolling(10).sum()
    s = pd.DataFrame({
        "trend": f2f.map({1.0: "up", 0.0: "flat", -1.0: "down"}),
        "vol": np.where(rv > rv.expanding(40).median(), "hiVol", "loVol"),
        "er": np.where(er > er.expanding(40).median(), "trending", "choppy"),
    }, index=d.index)
    return s.shift(1).reindex(index).ffill().bfill()   # state for day D known at close of D-1


def table(R, key, mask):
    """state cell -> (best sleeve, stats) from the days in mask."""
    out = {}
    for cell, g in R[mask].groupby(key[mask]):
        st = {c: (sharpe(g[c]), g[c].mean() * 1e4) for c in R.columns}
        best = max(st, key=lambda c: st[c][0])
        pick = best if (len(g) >= MIN_N and st[best][1] > 0) else "H1"
        out[cell] = (pick, len(g), st)
    return out


def run_switch(R, key, choice):
    ch = pd.Series(choice, index=R.index)
    r = pd.Series(R.values[np.arange(len(R)), [R.columns.get_loc(c) for c in ch]], index=R.index)
    sw = (ch != ch.shift(1)).astype(float)
    sw.iloc[0] = 0.0
    return r - sw * SWITCH_COST, ch


if __name__ == "__main__":
    R = pd.DataFrame({k: vol_control(daily(v)) for k, v in SLEEVES.items()}).dropna()
    sp = json.load(open(f"{RES}/s12_splits.json"))
    d_tr, d_te = pd.Timestamp(sp["d_tr"]), pd.Timestamp(sp["d_te"])
    tr, te, va = R.index < d_tr, (R.index >= d_tr) & (R.index < d_te), R.index >= d_te
    S = states(R.index)
    splits = {"TRAIN": tr, "TEST": te, "VAL": va, "OVERALL": np.ones(len(R), bool)}

    print("Each sleeve at 15% target vol - Sharpe / return by split")
    print(pd.DataFrame({k: {f"{c} sharpe": round(sharpe(R[c][m]), 2) for c in R.columns} |
                           {f"{c} ret": round((1 + R[c][m]).prod() - 1, 3) for c in R.columns}
                        for k, m in splits.items()}).to_string())
    print("\nCorrelation of sleeves (train):\n", R[tr].corr().round(2).to_string())

    # ---- learn partition + table on TRAIN
    parts = [("trend",), ("vol",), ("er",), ("trend", "vol"), ("trend", "er"), ("vol", "er")]
    best = None
    for p in parts:
        key = S[list(p)].agg("|".join, axis=1)
        tb = table(R, key, tr)
        r, _ = run_switch(R, key, [tb.get(k, ("H1",))[0] for k in key])
        s = sharpe(r[tr])
        print(f"partition {'x'.join(p):12s} train Sharpe {s:5.2f}   test {sharpe(r[te]):5.2f}   val {sharpe(r[va]):5.2f}")
        if best is None or s > best[0]:
            best = (s, p, key, tb)
    _, p, key, tb = best
    print(f"\nChosen on TRAIN: partition = {' x '.join(p)}")
    print("\nSWITCH TABLE learned on TRAIN, and whether each case held up afterwards (Sharpe of each sleeve inside the cell):")
    after = table(R, key, te | va)
    rows = []
    for cell, (pick, n, st) in sorted(tb.items()):
        a = after.get(cell)
        rows.append({"state": cell, "trade": pick, "train_days": n,
                     **{f"{c}_train": round(st[c][0], 2) for c in R.columns},
                     "later_days": a[1] if a else 0,
                     **{f"{c}_later": (round(a[2][c][0], 2) if a else np.nan) for c in R.columns}})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(f"{RES}/s13_switch_table.csv", index=False)

    r_frozen, ch_frozen = run_switch(R, key, [tb.get(k, ("H1",))[0] for k in key])
    # ---- online: table re-estimated every day from all past days
    choice = []
    for i in range(len(R)):
        if i < 60:
            choice.append("H1")
            continue
        past = np.zeros(len(R), bool)
        past[:i] = True
        g = R[past & (key == key.iloc[i]).values]
        if len(g) < MIN_N:
            choice.append("H1")
            continue
        st = {c: sharpe(g[c]) for c in R.columns}
        b = max(st, key=st.get)
        choice.append(b if g[b].mean() > 0 else "H1")
    r_online, ch_online = run_switch(R, key, choice)
    # ---- no-switch references
    r_eq = vol_control(R.mean(1))
    trail = R.rolling(20).apply(lambda x: x.mean() / (x.std() + 1e-12)).shift(1)
    r_mom, ch_mom = run_switch(R, key, trail.idxmax(1).fillna("H1").tolist())

    series = {"H1 at 15% vol": R["H1"], "Frozen switch table": r_frozen, "Online switch table": r_online,
              "Equal blend of the 4": r_eq}
    extra = {"Trailing-20d-best switch": r_mom}
    rows = []
    for nm, r in {**series, **extra}.items():
        for k, m in splits.items():
            v = r[m]
            eq = (1 + v).cumprod()
            rows.append({"strategy": nm, "split": k, "return": eq.iloc[-1] - 1, "sharpe": sharpe(v),
                         "max_dd": (eq / eq.cummax() - 1).min(), "ann_vol": v.std() * np.sqrt(ANN)})
    out = pd.DataFrame(rows)
    print("\n", out.round(3).to_string(index=False))
    out.to_csv(f"{RES}/s13_summary.csv", index=False)
    for nm, ch in (("frozen", ch_frozen), ("online", ch_online), ("trailing", ch_mom)):
        print(f"{nm:8s} share of days per sleeve:", ch.value_counts(normalize=True).round(2).to_dict(),
              " switches:", int((ch != ch.shift()).sum() - 1))
    pd.DataFrame({**series, **extra}).to_parquet(f"{RES}/s13_daily.parquet")

    # ------------------------------------------------------------------ figures
    def style(ax):
        ax.set_facecolor(SURF)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(GRID)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    def four_panel(ser, title, fname, bench=None):
        fig = plt.figure(figsize=(15, 11), facecolor=SURF)
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, :])]
        some = next(iter(ser.values()))
        masks = {"TRAIN (first 60%)": some.index < d_tr, "TEST (next 30%)": (some.index >= d_tr) & (some.index < d_te),
                 "VALIDATION (last 10%)": some.index >= d_te, "OVERALL": np.ones(len(some), bool)}
        for ax, (nm, m) in zip(axes, masks.items()):
            style(ax)
            if bench is not None:
                b = bench.reindex(some.index).fillna(0)[m]
                ax.plot(b.index, (1 + b).cumprod().values, color=BENCH, lw=1.2, label="Buy & hold gold")
            for (lab, r), col in zip(ser.items(), PAL):
                v = r[m]
                eq = (1 + v).cumprod()
                ax.plot(eq.index, eq.values, color=col, lw=2,
                        label=f"{lab}  ({eq.iloc[-1]-1:+.1%}, Sharpe {sharpe(v):.2f}, DD {(eq/eq.cummax()-1).min():.1%})")
            if nm == "OVERALL":
                for x in (d_tr, d_te):
                    ax.axvline(x, color=INK2, lw=0.8, ls=(0, (4, 3)))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            else:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1 if m.sum() < 60 else 2))
            ax.set_title(f"{nm} — {int(m.sum())} days, starts at 1.0", loc="left", fontsize=10.5, color=INK, fontweight="bold")
            ax.legend(fontsize=8 if nm != "OVERALL" else 9.5, frameon=False, loc="upper left", labelcolor=INK2)
        fig.suptitle(title, x=0.045, ha="left", fontsize=13.5, color=INK, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        fig.savefig(f"{FIG}/{fname}", dpi=130)

    bh = daily("s00_buy_hold_daily").shift(1).fillna(0)
    l = pd.read_parquet(f"{RES}/s12_daily.parquet")
    l.index = pd.DatetimeIndex(l.index)
    four_panel({"H1 control (every trade, size 1)": l["control"], "H1 + trade-by-trade learner": l["learner"]},
               "Hybrid H1 with online trade learning — equity by split, net of costs", "h1_learner_splits.png", bh)
    four_panel(series, "Switch-case study: H1 vs switching between H1 / RL / H4 / HMM (all sleeves at 15% vol), net of costs",
               "switch_splits.png")
