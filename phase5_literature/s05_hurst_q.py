"""
Paper 30 — "Optimizing Returns Using the Hurst Exponent and Q Learning on Momentum and
Mean Reversion Strategies".

  Momentum: SMA(5) / SMA(10) crossover (golden cross long, death cross short).
  Mean reversion: Bollinger(20, 2): long below lower band, short above upper, exit at the MA.
  Switch: Hurst > boundary -> momentum, else mean reversion.  The paper's Section 4 slides
          the boundary away from 0.5; we pick it on the training window (expanding, monthly).
  Q-learning (Section 6): state = discretised (SMA5-SMA10)_t and _(t-1) (ATR-normalised),
          plus the Hurst regime; actions {short, flat, long} held for one bar; the agent acts
          ONLY when the learned Q-value is statistically significant (t > 2), as the paper
          reports that forcing it to act in every state performed poorly.
Bars: H1 (paper uses daily; with 22 months of data daily gives too few decisions).
Hurst: rolling rescaled-range (R/S) estimate over 256 bars on scales 8..128.
"""
import numpy as np
import pandas as pd
from common import load_h1, backtest, metrics, monthly_refit_points, save_metrics, atr

HWIN = 256


def hurst_rs(x: np.ndarray) -> float:
    scales = np.array([8, 16, 32, 64, 128])
    rs = []
    for s in scales:
        n = len(x) // s
        seg = x[: n * s].reshape(n, s)
        z = np.cumsum(seg - seg.mean(1, keepdims=True), axis=1)
        r = z.max(1) - z.min(1)
        sd = seg.std(1)
        ok = sd > 0
        rs.append(np.mean(r[ok] / sd[ok]))
    return np.polyfit(np.log(scales), np.log(rs), 1)[0]


def rolling_hurst(r: np.ndarray, step: int = 6) -> np.ndarray:
    h = np.full(len(r), np.nan)
    for t in range(HWIN, len(r), step):
        h[t] = hurst_rs(r[t - HWIN:t])
    return pd.Series(h).ffill().values


def base_signals(d):
    c = d["mid"]
    s5, s10 = c.rolling(5).mean(), c.rolling(10).mean()
    mom = np.sign(s5 - s10).fillna(0).values
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    up, lo = ma + 2 * sd, ma - 2 * sd
    mr = np.zeros(len(d))
    cur = 0
    cv, upv, lov, mav = c.values, up.values, lo.values, ma.values
    for t in range(len(d)):
        if not np.isfinite(mav[t]):
            continue
        if cur == 0:
            if cv[t] < lov[t]:
                cur = 1
            elif cv[t] > upv[t]:
                cur = -1
        elif (cur == 1 and cv[t] >= mav[t]) or (cur == -1 and cv[t] <= mav[t]):
            cur = 0
        mr[t] = cur
    return mom, mr, (s5 - s10).values


def pnl(d, pos, mask):
    fwd = d["ret"].shift(-1).fillna(0).values
    p = pos * fwd - np.abs(np.diff(np.r_[0, pos])) * (d["hs"].values + 0.2e-4)
    p = pd.Series(p[mask]).groupby(d["tday"].values[mask]).sum()
    return p.mean() / (p.std() + 1e-12) * np.sqrt(252)


if __name__ == "__main__":
    d = load_h1()
    H = rolling_hurst(d["lret"].values)
    mom, mr, diff = base_signals(d)
    a = atr(d, 14).values
    nd = diff / a
    refits = monthly_refit_points(d.index)
    pos_sw = np.zeros(len(d))
    pos_q = np.zeros(len(d))
    log = []
    fwd = d["ret"].shift(-1).fillna(0).values
    cost = d["hs"].values + 0.2e-4
    for i, r0 in enumerate(refits):
        r1 = refits[i + 1] if i + 1 < len(refits) else d.index[-1] + pd.Timedelta(days=1)
        tr = np.asarray((d.index < r0) & np.isfinite(H))
        te = np.asarray((d.index >= r0) & (d.index < r1))
        # --- Hurst boundary sweep (paper sec. 4) on training
        best = None
        for b in np.round(np.arange(0.40, 0.71, 0.025), 3):
            p = np.where(H > b, mom, mr)
            s = pnl(d, p, tr)
            if best is None or s > best[0]:
                best = (s, b)
        b = best[1]
        pos_sw[te] = np.where(H[te] > b, mom[te], mr[te])
        # --- tabular Q-learning (one-step reward, gamma=0), significance-gated
        q1 = np.quantile(nd[tr & np.isfinite(nd)], [0.2, 0.4, 0.6, 0.8])
        s1 = np.searchsorted(q1, nd)
        s0 = np.r_[2, s1[:-1]]
        hb = (H > 0.5).astype(int)
        state = s1 * 10 + s0 * 2 + hb
        R = {}
        for act in (-1, 1):
            rew = act * fwd - cost  # enter + hold one bar (conservative: pay half spread each bar)
            g = pd.DataFrame({"s": state[tr], "r": rew[tr]}).groupby("s")["r"]
            R[act] = g.agg(["mean", "std", "count"])
        act_map = {}
        for s in np.unique(state[tr]):
            best_a, best_t = 0, 2.0
            for act in (-1, 1):
                if s in R[act].index:
                    m, sd, n = R[act].loc[s]
                    t = m / (sd / np.sqrt(n)) if sd > 0 and n > 30 else 0
                    if t > best_t:
                        best_a, best_t = act, t
            act_map[s] = best_a
        pos_q[te] = pd.Series(state[te]).map(act_map).fillna(0).values
        log.append((str(r0)[:10], b, sum(v != 0 for v in act_map.values())))
    print("refits (date, hurst boundary, #active Q states):", log)
    print("Hurst OOS mean %.3f, share>0.5 %.2f" % (np.nanmean(H[d.index >= refits[0]]),
                                                  np.nanmean(H[d.index >= refits[0]] > 0.5)))
    rows = []
    rows.append(metrics(backtest(d, mom, "s05_sma_momentum"), "s05_sma_momentum"))
    rows.append(metrics(backtest(d, mr, "s05_bollinger_mr"), "s05_bollinger_mr"))
    rows.append(metrics(backtest(d, pos_sw, "s05_hurst_switch"), "s05_hurst_switch"))
    rows.append(metrics(backtest(d, pos_q, "s05_q_learning"), "s05_q_learning"))
    print(pd.DataFrame(rows).round(3).to_string())
    save_metrics(rows, "m_s05.csv")
