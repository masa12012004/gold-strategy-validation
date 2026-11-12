"""
Paper 40 — "Speculative Futures Trading under Mean Reversion" (Leung, Li, Li, Zheng)
  Optimal double stopping for an OU process with transaction cost c and discount r:
    F(x)=int u^{r/mu-1} exp(sqrt(2mu/s^2)(x-theta)u - u^2/2) du,  G(x)= same with (theta-x)
    exit level  b*: F(b) = (b-c) F'(b);           V(x) = (b*-c)F(x)/F(b*)  (x<b*), x-c else
    entry level d*: G(d)(V'(d)-1) = G'(d)(V(d)-d-c);  enter long when X <= d*
  Chooser option (long or short): short side by reflecting X -> -X. The paper's finding
  that the chooser delays entry is reproduced by requiring X to cross the side-specific d*.
Paper 10 — "Dynamic modeling of mean-reverting spreads" (Triantafyllopoulos & Montana)
  Time-varying AR(1) X_t = phi_t X_{t-1} + e_t, phi_t random walk, on-line Kalman with
  discount factor delta (forgetting), used as a real-time MONITOR: trade only while the
  filtered half-life ln2/-ln(phi_t) is short enough.
Single-asset adaptation (no silver / second leg in the data): the mean-reverting "spread"
is X = 1e4 * (log mid - EMA_L(log mid)) on M5 bars, L chosen on training.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
warnings.filterwarnings('ignore')
from common import load_m5, backtest, metrics, monthly_refit_points, save_metrics


def fit_ou(x, dt=1.0):
    x0, x1 = x[:-1], x[1:]
    b, a = np.polyfit(x0, x1, 1)
    b = min(max(b, 1e-6), 0.999999)
    mu = -np.log(b) / dt
    theta = a / (1 - b)
    res = x1 - (a + b * x0)
    sig = np.sqrt(res.var() * 2 * mu / (1 - b ** 2))
    return theta, mu, sig


def ou_levels(theta, mu, sig, c, r):
    k = np.sqrt(2 * mu / sig ** 2)
    p = r / mu - 1

    def F(x, der=0):
        f = lambda u: u ** p * np.exp(k * (x - theta) * u - u * u / 2) * ((k * u) ** der)
        return quad(f, 0, np.inf, limit=200)[0]

    def G(x, der=0):
        f = lambda u: u ** p * np.exp(k * (theta - x) * u - u * u / 2) * ((-k * u) ** der)
        return quad(f, 0, np.inf, limit=200)[0]

    sd_eq = sig / np.sqrt(2 * mu)
    lo, hi = theta - 4 * sd_eq, theta + 6 * sd_eq
    hb = lambda b: F(b) - (b - c) * F(b, 1)
    b_star = brentq(hb, theta - 2 * sd_eq + c, hi)
    V = lambda x: (b_star - c) * F(x) / F(b_star) if x < b_star else x - c
    Vp = lambda x: (b_star - c) * F(x, 1) / F(b_star) if x < b_star else 1.0
    hd = lambda d: G(d) * (Vp(d) - 1) - G(d, 1) * (V(d) - d - c)
    grid = np.linspace(lo, b_star - 1e-6, 120)
    vals = np.array([hd(g) for g in grid])
    sgn = np.where(np.diff(np.sign(vals)) != 0)[0]
    d_star = brentq(hd, grid[sgn[-1]], grid[sgn[-1] + 1]) if len(sgn) else theta - 2 * sd_eq
    return d_star, b_star


def tvar_kalman(x, delta=0.995, n_var=None):
    """Discount-factor Kalman for phi_t in x_t = phi_t x_{t-1} + e_t."""
    phi, P = 0.9, 1.0
    out = np.empty(len(x))
    out[0] = phi
    v = n_var if n_var else np.var(np.diff(x))
    for t in range(1, len(x)):
        P = P / delta
        h = x[t - 1]
        S = h * P * h + v
        K = P * h / S
        phi = phi + K * (x[t] - phi * h)
        P = (1 - K * h) * P
        out[t] = phi
    return out


def simulate(X, d_long, b_long, d_short, b_short, gate, max_hold):
    pos = np.zeros(len(X))
    cur, age = 0, 0
    for t in range(len(X)):
        if cur == 0 and gate[t]:
            if X[t] <= d_long[t]:
                cur, age = 1, 0
            elif -X[t] <= d_short[t]:
                cur, age = -1, 0
        elif cur == 1 and (X[t] >= b_long[t] or age >= max_hold):
            cur = 0
        elif cur == -1 and (-X[t] >= b_short[t] or age >= max_hold):
            cur = 0
        age += 1
        pos[t] = cur
    return pos


def spread(d, L):
    lp = np.log(d["mid"])
    return (1e4 * (lp - lp.ewm(span=L, adjust=False).mean())).values


if __name__ == "__main__":
    d = load_m5()
    c_bps = float(2 * (d["hs"].median() * 1e4 + 0.2))  # round-trip cost in bps of the spread
    r = 1e-4  # per-bar discount (time value / impatience)
    refits = monthly_refit_points(d.index)
    phis = {}
    rows, logs = [], []
    MAXH = 288
    pos_ou = np.zeros(len(d))
    pos_ou_g = np.zeros(len(d))
    for i, r0 in enumerate(refits):
        r1 = refits[i + 1] if i + 1 < len(refits) else d.index[-1] + pd.Timedelta(days=1)
        tr = np.asarray(d.index < r0)
        te = np.asarray((d.index >= r0) & (d.index < r1))
        best = None
        for L, r in [(L, r) for L in (48, 144, 288) for r in (1e-4, 1e-3, 5e-3)]:
            X = spread(d, L)
            xt = X[tr][L:]
            th, mu, sg = fit_ou(xt)
            try:
                dl, bl = ou_levels(th, mu, sg, c_bps, r)
                ds, bs = ou_levels(-th, mu, sg, c_bps, r)
            except Exception as e:
                continue
            ones = np.ones(len(d))
            p = simulate(X, dl * ones, bl * ones, ds * ones, bs * ones, np.ones(len(d), bool), MAXH)
            fwd = d["ret"].shift(-1).fillna(0).values
            pn = p * fwd - np.abs(np.diff(np.r_[0, p])) * (d["hs"].values + 0.2e-4)
            s = pd.Series(pn[tr]).groupby(d["tday"].values[tr]).sum()
            sh = s.mean() / (s.std() + 1e-12) * np.sqrt(252)
            if best is None or sh > best[0]:
                best = (sh, L, th, mu, sg, dl, bl, ds, bs, p, r)
        sh, L, th, mu, sg, dl, bl, ds, bs, p, r = best
        pos_ou[te] = p[te]
        # TVAR monitor gate, half-life threshold chosen on training
        if L not in phis:
            phis[L] = tvar_kalman(spread(d, L))
        hl = np.log(2) / -np.log(np.clip(phis[L], 1e-6, 0.999999))
        gbest = None
        X = spread(d, L)
        ones = np.ones(len(d))
        for hmax in (24, 72, 144, 288, 1e9):
            pg = simulate(X, dl * ones, bl * ones, ds * ones, bs * ones, hl < hmax, MAXH)
            fwd = d["ret"].shift(-1).fillna(0).values
            pn = pg * fwd - np.abs(np.diff(np.r_[0, pg])) * (d["hs"].values + 0.2e-4)
            s = pd.Series(pn[tr]).groupby(d["tday"].values[tr]).sum()
            shg = s.mean() / (s.std() + 1e-12) * np.sqrt(252)
            if gbest is None or shg > gbest[0]:
                gbest = (shg, hmax, pg)
        pos_ou_g[te] = gbest[2][te]
        logs.append((str(r0)[:10], L, round(th, 2), round(mu, 4), round(sg, 2),
                     round(dl, 1), round(bl, 1), round(sh, 2), gbest[1], r))
    print("cost(bps rt)=%.2f" % c_bps)
    print("date, L, theta, mu, sigma, d*, b*, trainSharpe, hl_max")
    for l in logs:
        print(l)
    rows.append(metrics(backtest(d, pos_ou, "s06_ou_optimal_stopping"), "s06_ou_optimal_stopping"))
    rows.append(metrics(backtest(d, pos_ou_g, "s06_ou_tvar_gated"), "s06_ou_tvar_gated"))
    print(pd.DataFrame(rows).round(3).to_string())
    save_metrics(rows, "m_s06.csv")
