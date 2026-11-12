"""
Paper 44 — "The drift burst hypothesis" (Christensen, Oomen, Reno).

  T_t = sqrt(h/K2) * mu_hat_t / sigma_hat_t     with left-sided exponential kernel
  mu_hat: drift bandwidth h = 5 min;  sigma_hat: vol bandwidth 5h = 25 min, Parzen HAC
  pre-averaging kn = 3 with g(x)=min(x,1-x)   (paper section 5.2)
  A burst is flagged where |T| > crit at a local extreme (max 1 per 5-minute window).
  Paper finding (Table 4 / Fig 7): post-burst 5-min return = b * pre-burst 5-min return,
  b ~ -0.23, reversal in ~71% of events; stronger for negative bursts on high volume.
Strategy: at the close of the detection minute, take the opposite side of the 5-min
pre-burst move and hold H minutes. crit in {4.0,4.5,5.0}, H in {5,10,15}, and the
"negative-only/high-activity" variant are chosen on the TRAINING half-year only.
Bars: M1 mid with real spread.
"""
import numpy as np
import pandas as pd
from common import load_m1, backtest, metrics, save_metrics, OOS_START

H_DRIFT, H_VOL, KN, LAGS = 5.0, 25.0, 3, 6


def preaverage(y, kn):
    # g(j/kn) weights for j=1..kn-1; backward-looking so value at t uses y[t-kn..t]
    g = np.minimum(np.arange(1, kn) / kn, 1 - np.arange(1, kn) / kn)
    dy = np.r_[0.0, np.diff(y)]
    out = np.zeros_like(y)
    for j, gj in enumerate(g, start=1):
        out += gj * np.r_[np.zeros(kn - j), dy[:len(dy) - (kn - j)]]
    return out


def ew(x, h):
    a = np.exp(-1.0 / h)
    return pd.Series(x).ewm(alpha=1 - a, adjust=True).mean().values  # normalised kernel mean


def drift_burst_stat_5s(m1_index):
    """Paper-faithful: statistic on the 5-second mid grid (h = 60 obs = 5 min, vol 300 obs),
    then mapped to M1 bars as the signed value with the largest |T| inside each minute."""
    g = pd.read_parquet("results/_mid5s.parquet")
    T5 = drift_burst_stat(g, bar_sec=5)
    s = pd.Series(T5, index=g.index)
    mins = s.index.floor("1min")
    key = pd.Series(np.abs(T5), index=s.index).groupby(mins).idxmax()
    tm = s.loc[key.values]
    tm.index = key.index
    tm.index = tm.index.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
    tm = tm[tm.index.notna()]
    tm = tm[~tm.index.duplicated()]
    return tm.reindex(m1_index).fillna(0.0).values


def drift_burst_stat(d, bar_sec=60):
    global H_DRIFT, H_VOL
    hd, hv = H_DRIFT * 60 / bar_sec, H_VOL * 60 / bar_sec
    y = np.log(d["mid"].values.astype(float))
    idx = d.index.values
    gap = np.r_[False, np.diff(idx).astype("timedelta64[s]").astype(np.int64) > max(60, 2 * bar_sec)]
    H_DRIFT_, H_VOL_ = H_DRIFT, H_VOL
    H_DRIFT, H_VOL = hd, hv
    try:
        return _stat(y, gap, burn=int(hv))
    finally:
        H_DRIFT, H_VOL = H_DRIFT_, H_VOL_


def _stat(y, gap, burn):
    dy = preaverage(y, KN)
    dy[gap] = 0.0
    a = np.exp(-1.0 / H_DRIFT)
    num = pd.Series(dy).ewm(alpha=1 - a, adjust=False).mean().values  # ~ (1-a) sum a^j dY
    # HAC variance of increments with Parzen weights, vol bandwidth 25
    gam = ew(dy * dy, H_VOL)
    L = LAGS + 2 * (KN - 1)
    for l in range(1, L + 1):
        x = l / (L + 1)
        w = 1 - 6 * x ** 2 + 6 * x ** 3 if x <= 0.5 else 2 * (1 - x) ** 3
        prod = dy * np.r_[np.zeros(l), dy[:-l]]
        gam = gam + 2 * w * ew(prod, H_VOL)
    gam = np.maximum(gam, 1e-14)
    # Var(num) under null = s2 * (1-a)^2 * sum a^{2j} = s2 * (1-a)^2/(1-a^2)
    T = num / np.sqrt(gam * (1 - a) ** 2 / (1 - a ** 2))
    # burn-in of one vol bandwidth after each gap
    since_gap = pd.Series(np.cumsum(gap)).groupby(np.cumsum(gap)).cumcount().values
    T[since_gap < burn] = 0.0
    return T


def make_positions(d, T, crit, hold, neg_only, act_q):
    tick = (d["bid_ticks"] + d["ask_ticks"]).values
    act = pd.Series(tick).rolling(5).sum().values
    act_ref = pd.Series(tick).rolling(60 * 24 * 5, min_periods=500).median().values * 5
    pre = np.log(d["mid"]).diff(5).values
    absT = np.abs(T)
    loc_max = absT >= pd.Series(absT).rolling(5, min_periods=1).max().values
    events = np.where((absT > crit) & loc_max)[0]
    pos = np.zeros(len(d))
    last = -10
    for e in events:
        if e - last < 5:
            continue
        if neg_only and pre[e] >= 0:
            continue
        if act_q and not (act[e] > 1.5 * act_ref[e]):
            continue
        side = -np.sign(pre[e])
        pos[e:e + hold] = side
        last = e
    return pos, len(events)


if __name__ == "__main__":
    d = load_m1()
    T = drift_burst_stat_5s(d.index)
    print("T stat: sd=%.3f  q(1%%,99%%)=%s  kurt=%.2f" % (np.nanstd(T), np.round(np.nanpercentile(T, [1, 99]), 2),
                                                        pd.Series(T).kurt()))
    tr = d.index < OOS_START
    grid = []
    for crit in (4.0, 4.5, 5.0):
        for hold in (5, 10, 15):
            for neg_only in (False, True):
                for act_q in (False, True):
                    pos, _ = make_positions(d, T, crit, hold, neg_only, act_q)
                    fwd = d["ret"].shift(-1).fillna(0).values
                    pnl = pos * fwd - np.abs(np.diff(np.r_[0, pos])) * (d["hs"].values + 0.2e-4)
                    p = pd.Series(pnl[tr]).groupby(d["tday"].values[tr]).sum()
                    ntr = (np.diff(np.r_[0, pos[tr]]) != 0).sum() / 2
                    grid.append((crit, hold, neg_only, act_q, p.sum(), p.mean() / (p.std() + 1e-12) * np.sqrt(252), ntr))
    g = pd.DataFrame(grid, columns=["crit", "hold", "neg_only", "act", "train_ret", "train_sharpe", "trades"])
    print(g.sort_values("train_sharpe", ascending=False).head(8).round(4).to_string())
    best = g[g.trades >= 40].sort_values("train_sharpe", ascending=False).iloc[0]
    print("chosen on training:", best.to_dict())
    rows = []
    pos, n_ev = make_positions(d, T, best.crit, int(best.hold), bool(best.neg_only), bool(best.act))
    bt = backtest(d, pos, "s04_drift_burst_fade")
    rows.append(metrics(bt, "s04_drift_burst_fade"))
    # paper-default spec for transparency (crit 4.5, 5-min hold, both sides)
    pos2, _ = make_positions(d, T, 4.5, 5, False, False)
    bt2 = backtest(d, pos2, "s04_drift_burst_paper_default")
    rows.append(metrics(bt2, "s04_drift_burst_paper_default"))
    # event study OOS: reversal regression R+ = a + b R-
    pre = np.log(d["mid"]).diff(5).values
    post = np.log(d["mid"]).shift(-5).values - np.log(d["mid"]).values
    absT = np.abs(T)
    ev = np.where((absT > 4.5) & (absT >= pd.Series(absT).rolling(5, min_periods=1).max().values))[0]
    ev = ev[np.r_[True, np.diff(ev) >= 5]]
    ok = np.isfinite(post[ev]) & np.isfinite(pre[ev])
    b = np.polyfit(pre[ev][ok], post[ev][ok], 1)[0]
    rev = np.mean(np.sign(pre[ev][ok]) != np.sign(post[ev][ok]))
    print(f"events |T|>4.5: {len(ev)}  slope b={b:.3f}  reversal share={rev:.1%}  "
          f"median |pre| = {np.median(np.abs(pre[ev][ok]))*1e4:.1f} bp")
    print(pd.DataFrame(rows).round(3).to_string())
    save_metrics(rows, "m_s04.csv")
