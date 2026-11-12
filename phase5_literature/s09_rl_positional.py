"""
Paper 9 — "Deep Reinforcement Learning with Positional Context for Intraday Trading"
  state s_t in R^13 = 9 price features (returns over 5 lookbacks, RSI, ADX, ULTOSC, WILLR)
                    + 4 POSITIONAL features (time left, current position, position return,
                      daily return so far); intraday only, flat at the end of the day.
Paper 8 — "Commodities Trading through Deep Policy Gradient Methods"
  continuous action a_t in [-1,1] = position weight; reward r_t = r_net - lambda*var(r_net)
  (eq. 8-10) with transaction costs on |a_t - a_{t-1}|; the ACTOR-ONLY direct policy
  gradient beat actor-critic in that paper, so that is the optimiser used here (paper 9
  uses PPO; the state design, action semantics, costs and episode structure are paper 9's).
  The risk-sensitivity lambda in {0, 0.1} is picked on the validation tail.

Bars: M15, one episode = one trading day (17:00 ET to 17:00 ET), all days batched.
Expanding window, refit every 3 months, 3 seeds averaged. An ablation without the
positional features is reported, as in paper 9 Table 9.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import _finish, DATA, backtest, metrics, save_metrics, atr, OOS_START

torch.set_num_threads(2)
SIG_TGT = 0.15


def price_features(d):
    c, h, l = d["close"], d["high"], d["low"]
    r = d["lret"]
    vol = r.ewm(span=96, min_periods=48).std()
    f = pd.DataFrame(index=d.index)
    for k in (1, 2, 4, 8, 16):
        f[f"r{k}"] = np.log(c).diff(k) / (vol * np.sqrt(k))
    up, dn = r.clip(lower=0), (-r).clip(lower=0)
    f["rsi"] = (100 - 100 / (1 + up.ewm(alpha=1 / 14).mean() / dn.ewm(alpha=1 / 14).mean())) / 50 - 1
    a = atr(d, 14)
    pdm = (h.diff()).clip(lower=0).where(h.diff() > -l.diff(), 0)
    ndm = (-l.diff()).clip(lower=0).where(-l.diff() > h.diff(), 0)
    pdi, ndi = pdm.ewm(alpha=1 / 14).mean() / a, ndm.ewm(alpha=1 / 14).mean() / a
    f["adx"] = ((pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14).mean() * 2 - 1
    pc = c.shift(1)
    bp = c - np.minimum(l, pc)
    tr = np.maximum(h, pc) - np.minimum(l, pc)
    avg = lambda n: bp.rolling(n).sum() / tr.rolling(n).sum()
    f["ultosc"] = (4 * avg(7) + 2 * avg(14) + avg(28)) / 7 * 2 - 1
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    f["willr"] = (hh - c) / (hh - ll) * 2 - 1
    return f.clip(-5, 5), vol


class Policy(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x):
        return torch.tanh(self.net(x)).squeeze(-1)


def rollout(net, Z, R, C, M, positional=True):
    """Z:[D,L,9] features, R:[D,L] next-bar scaled return, C:[D,L] scaled cost, M:[D,L] mask.
    Returns positions [D,L] and per-step net pnl [D,L]. Last valid bar of each day is flat."""
    D, L, _ = Z.shape
    last = (M.cumsum(1) == M.sum(1, keepdim=True)) & (M > 0)
    nvalid = M.sum(1, keepdim=True).clamp(min=1)
    prev = torch.zeros(D)
    pr = torch.zeros(D)
    dr = torch.zeros(D)
    P, PN = [], []
    for t in range(L):
        tl = 1 - (M[:, :t].sum(1) / nvalid.squeeze(1))
        if positional:
            x = torch.cat([Z[:, t], tl[:, None], prev.detach()[:, None], pr[:, None] * 20, dr[:, None] * 20], 1)
        else:
            x = Z[:, t]
        a = net(x) * M[:, t]
        a = torch.where(last[:, t], torch.zeros_like(a), a)      # close before the session ends
        pnl = a * R[:, t] - (a - prev).abs() * C[:, t]
        flip = (torch.sign(a.detach()) != torch.sign(prev.detach()))
        pr = torch.where(flip, torch.zeros_like(pr), pr) + pnl.detach()
        dr = dr + pnl.detach()
        prev = a
        P.append(a); PN.append(pnl)
    return torch.stack(P, 1), torch.stack(PN, 1)


def objective(pn, M, lam):
    x = pn[M > 0]
    return -(x.mean() - lam * x.var()) / (x.std().detach() + 1e-9)


def train(Z, R, C, M, lam, seed, positional):
    torch.manual_seed(seed)
    D = Z.shape[0]
    cut = int(D * 0.9)
    net = Policy(Z.shape[2] + (4 if positional else 0))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    best, state, bad = -np.inf, None, 0
    for ep in range(150):
        net.train()
        perm = torch.randperm(cut)
        for b in range(0, cut, 64):
            ii = perm[b:b + 64]
            opt.zero_grad()
            _, pn = rollout(net, Z[ii], R[ii], C[ii], M[ii], positional)
            objective(pn, M[ii], lam).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            _, pn = rollout(net, Z[cut:], R[cut:], C[cut:], M[cut:], positional)
            day = (pn * M[cut:]).sum(1)
            v = (day.mean() / (day.std() + 1e-9)).item() * np.sqrt(252)
        if v > best + 1e-3:
            best, bad = v, 0
            state = {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    net.load_state_dict(state)
    net.eval()
    return net, best


def run(d, feats, scale, positional, name):
    days = pd.unique(d["tday"])
    grp = d.groupby("tday").indices
    L = max(len(v) for v in grp.values())
    nd = len(days)
    Zr = np.zeros((nd, L, feats.shape[1]), np.float32)
    R = np.zeros((nd, L), np.float32); C = np.zeros((nd, L), np.float32); M = np.zeros((nd, L), np.float32)
    IDX = np.full((nd, L), -1)
    fv = feats.values
    rnext = (scale * d["ret"].shift(-1).fillna(0)).values
    cost = (scale * (d["hs"] + 0.2e-4)).values
    okrow = np.isfinite(fv).all(1) & np.isfinite(rnext) & np.isfinite(cost)
    for i, u in enumerate(days):
        ix = grp[u]
        ix = ix[okrow[ix]]
        n = len(ix)
        Zr[i, :n], R[i, :n], C[i, :n], M[i, :n], IDX[i, :n] = fv[ix], rnext[ix], cost[ix], 1, ix
    dts = pd.DatetimeIndex(days).tz_localize("US/Eastern")
    refits = pd.date_range(OOS_START.normalize(), dts[-1], freq="3MS", tz="US/Eastern")
    pos = np.zeros(len(d))
    log = []
    for i, r0 in enumerate(refits):
        r1 = refits[i + 1] if i + 1 < len(refits) else dts[-1] + pd.Timedelta(days=1)
        tr = np.where((dts < r0) & (M.sum(1) > 20))[0]
        te = np.where((dts >= r0) & (dts < r1))[0]
        mu = Zr[tr][M[tr] > 0].mean(0); sd = Zr[tr][M[tr] > 0].std(0) + 1e-9
        Z = torch.tensor((Zr - mu) / sd)
        Rt, Ct, Mt = torch.tensor(R), torch.tensor(C), torch.tensor(M)
        # pick lambda on validation
        cands = []
        for lam in (0.0, 0.1):
            nets = [train(Z[tr], Rt[tr], Ct[tr], Mt[tr], lam, s, positional) for s in range(3)]
            cands.append((np.mean([v for _, v in nets]), lam, [n for n, _ in nets]))
        v, lam, nets = max(cands, key=lambda x: x[0])
        with torch.no_grad():
            P = np.mean([rollout(n, Z[te], Rt[te], Ct[te], Mt[te], positional)[0].numpy() for n in nets], 0)
        m = IDX[te] >= 0
        pos[IDX[te][m]] = P[m]
        log.append((str(r0)[:10], len(tr), lam, round(float(v), 2), round(float(np.abs(P[m]).mean()), 3)))
    print(name, "(refit, train_days, lambda, val_sharpe, mean|pos|):")
    for l in log:
        print("  ", l)
    w = pd.Series(pos, index=d.index) * scale.fillna(0)
    return metrics(backtest(d, w.clip(-2, 2), name), name)


if __name__ == "__main__":
    d = _finish(pd.read_parquet(os.path.join(DATA, "xauusd_m15.parquet")))
    bpy = len(d) / d["tday"].nunique() * 252
    feats, vol = price_features(d)
    scale = (SIG_TGT / (vol * np.sqrt(bpy))).clip(upper=3.0)
    rows = [run(d, feats, scale, True, "s09_rl_positional"),
            run(d, feats, scale, False, "s09_rl_no_positional")]
    print(pd.DataFrame(rows).round(3).to_string())
    save_metrics(rows, "m_s09.csv")
