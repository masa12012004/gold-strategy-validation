"""
What H1 + learner looks like on a small MT5 account: 0.01-lot (1 oz) minimum size,
overnight swap on longs, and the count of trades per week / month.

  lots_t = quantise( pos_t * mult * LEV * equity_t / (price_t * 100) ), step 0.01,
           with a dead-band so the size only changes when the target is > 0.6 step away.
"""
import sys
import numpy as np
import pandas as pd
from common import load_h1, RES, ANN, EXTRA_SLIP_BPS
from s12_h1_online_learner import apply

EQ0, STEP, OZ, BAND = 5577.70, 0.01, 100.0, 0.6
SWAP_LONG_BP = 498.9 * 0.001 / 4036 * 1e4   # Exness XAUUSD long swap, bp of notional per night


def quantise(raw, cur):
    if raw == 0:
        return 0
    return int(np.round(raw)) if abs(raw - cur) > BAND else cur


def simulate(h1, p, lev, eq0=EQ0, quant=True):
    c, hs = h1["close"].values, h1["hs"].values
    hr, wd = h1.index.hour.values, h1.index.weekday.values
    eq, cur = eq0, 0
    eqs, lots = np.zeros(len(p)), np.zeros(len(p))
    for t in range(len(p) - 1):
        raw = p[t] * lev * eq / (c[t] * OZ) / STEP
        new = quantise(raw, cur) if quant else raw
        eq -= abs(new - cur) * STEP * OZ * c[t] * (hs[t] + EXTRA_SLIP_BPS * 1e-4)
        cur = new
        eq += cur * STEP * OZ * (c[t + 1] - c[t])
        if hr[t] == 16 and cur > 0:                       # position carried through 17:00 ET
            eq -= cur * STEP * OZ * c[t] * SWAP_LONG_BP * 1e-4 * (3 if wd[t] == 2 else 1)
        eqs[t], lots[t] = eq, cur * STEP if quant else cur * STEP
    eqs[-1] = eq
    return pd.Series(eqs, index=h1.index), pd.Series(lots, index=h1.index)


if __name__ == "__main__":
    h1 = load_h1()
    base = pd.read_parquet(f"{RES}/h1_slowmom_x_cubic.parquet")["pos"].reindex(h1.index).fillna(0.0)
    t = pd.read_csv(f"{RES}/s12_trades.csv")
    learner = pd.Series(apply(h1, base, t, t["mult"].values), index=h1.index)
    oos = base.index >= base[base != 0].index[0].normalize()
    oos = h1.index >= pd.Timestamp("2025-07-01", tz="US/Eastern")
    h, pl, pc = h1[oos], learner[oos].values, base[oos].values

    # ---- trade frequency of the model itself (no quantisation)
    t["entry"] = pd.to_datetime(t["entry"], utc=True).dt.tz_convert("US/Eastern")
    for nm, tt in (("all H1 signals (control)", t), ("learner, size > 0", t[t["mult"] > 0.05])):
        wk = tt.groupby(tt["entry"].dt.tz_localize(None).dt.to_period("W")).size()
        mo = tt.groupby(tt["entry"].dt.tz_localize(None).dt.to_period("M")).size()
        allw = pd.period_range(h.index[0].tz_localize(None), h.index[-1].tz_localize(None), freq="W")
        allm = pd.period_range(h.index[0].tz_localize(None), h.index[-1].tz_localize(None), freq="M")
        wk, mo = wk.reindex(allw, fill_value=0), mo.reindex(allm, fill_value=0)
        print(f"{nm}: {len(tt)} trades | per week mean {wk.mean():.2f} median {wk.median():.0f} "
              f"min {wk.min()} max {wk.max()} zero-weeks {int((wk == 0).sum())}/{len(wk)} | "
              f"per month mean {mo.mean():.1f} median {mo.median():.0f} min {mo.min()} max {mo.max()}")
        if nm.startswith("all"):
            print("   per month:", mo.to_dict())
    print(f"hold time (bars=hours): median {t['n'].median():.0f}, mean {t['n'].mean():.1f}, "
          f"<=2h {(t['n'] <= 2).mean():.0%}, >24h {(t['n'] > 24).mean():.0%}; "
          f"time in market {(pc != 0).mean():.0%}; long share {(t['dir'] > 0).mean():.0%}")

    rows = []
    for name, p in (("learner", pl), ("control", pc)):
        for lev in (1, 2, 3):
            for quant in (False, True):
                eq, lots = simulate(h, p, lev, quant=quant)
                d = eq.groupby(h["tday"]).last().pct_change().fillna(eq.groupby(h["tday"]).last().iloc[0] / EQ0 - 1)
                s = np.sign(lots.values)
                n_tr = int(((s != 0) & (np.r_[0, s[:-1]] != s)).sum())
                n_ord = int((lots.diff().abs() > 1e-9).sum())
                weeks = len(d) / 5
                rows.append({"book": name, "lev": lev, "0.01-lot": quant, "ret": eq.iloc[-1] / EQ0 - 1,
                             "sharpe": d.mean() / d.std() * np.sqrt(ANN),
                             "max_dd": (eq / eq.cummax() - 1).min(), "vol": d.std() * np.sqrt(ANN),
                             "trades": n_tr, "trades/wk": n_tr / weeks, "orders/wk": n_ord / weeks,
                             "max_lots": lots.abs().max(), "in_mkt": (lots != 0).mean()})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    out.to_csv(f"{RES}/s14_live_sizing.csv", index=False)
