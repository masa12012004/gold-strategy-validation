"""
Paper 39 — "Slow Momentum with Fast Reversion: Deep Learning and Changepoint Detection".

  Inputs (per bar): vol-normalised returns over {1, 24, 72, 120, 252} bars, MACD (8,24),
  (16,48), (32,96), and the changepoint SEVERITY nu_t and LOCATION gamma_t for a lookback l.
  Network: LSTM -> time-distributed dense -> tanh position in (-1,1).
  Loss: negative annualised Sharpe of vol-scaled strategy returns (eq. 14), with turnover cost.
  Train/val split 90/10 of each expanding training window, early stopping on validation.

Changepoint module: the paper fits a Matern-3/2 GP and a changepoint-kernel GP per window and
uses the reduction in negative log marginal likelihood. gpflow is not installed and 10k GP
fits per lookback are impractical here, so nu/gamma come from the same idea with a cheaper
likelihood: single-Gaussian window vs best two-segment Gaussian (mean and variance change),
nu = sigmoid(delta log-lik - BIC penalty), gamma = normalised location of the best split.
Bars: H1. Refit every 3 months (expanding). 3 seeds averaged. Both variants (with / without
CPD) are reported, as in the paper.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import load_h1, backtest, metrics, save_metrics, bars_per_year, OOS_START

torch.set_num_threads(2)
SEQ, HID, LBW = 63, 16, 48
SIG_TGT = 0.15


def cpd_features(r: np.ndarray, l: int = LBW, m: int = 6):
    n = len(r)
    c1 = np.r_[0, np.cumsum(r)]
    c2 = np.r_[0, np.cumsum(r * r)]
    nu = np.zeros(n)
    gam = np.zeros(n)
    t = np.arange(l, n + 1)                       # window = r[t-l:t]
    S1, S2 = c1[t] - c1[t - l], c2[t] - c2[t - l]
    var0 = np.maximum(S2 / l - (S1 / l) ** 2, 1e-18)
    ll0 = -0.5 * l * np.log(var0)
    best = np.full(len(t), -np.inf)
    arg = np.zeros(len(t))
    for k in range(m, l - m + 1):
        a1, a2 = c1[t - l + k] - c1[t - l], c2[t - l + k] - c2[t - l]
        b1, b2 = S1 - a1, S2 - a2
        va = np.maximum(a2 / k - (a1 / k) ** 2, 1e-18)
        vb = np.maximum(b2 / (l - k) - (b1 / (l - k)) ** 2, 1e-18)
        ll = -0.5 * k * np.log(va) - 0.5 * (l - k) * np.log(vb)
        upd = ll > best
        best[upd] = ll[upd]
        arg[upd] = k
    gain = best - ll0 - 1.5 * np.log(l)           # 3 extra parameters, BIC-style penalty
    nu[l - 1:] = 1 / (1 + np.exp(-gain))
    gam[l - 1:] = arg / l
    return nu, gam


def make_inputs(d, bpy):
    r = d["lret"]
    sig = r.ewm(span=60, min_periods=30).std()
    f = pd.DataFrame(index=d.index)
    for k in (1, 24, 72, 120, 252):
        f[f"r{k}"] = np.log(d["mid"]).diff(k) / (sig * np.sqrt(k))
    p = d["mid"]
    for s, L in ((8, 24), (16, 48), (32, 96)):
        m = p.ewm(halflife=np.log(0.5) / np.log(1 - 1 / s)).mean() - p.ewm(halflife=np.log(0.5) / np.log(1 - 1 / L)).mean()
        q = m / p.rolling(63).std()
        f[f"macd{s}"] = q / q.rolling(252).std()
    nu, gam = cpd_features(r.values)
    f["cpd_nu"], f["cpd_gamma"] = nu, gam
    scale = (SIG_TGT / (sig * np.sqrt(bpy))).clip(upper=3.0)   # vol-scaling of the traded position
    return f.clip(-5, 5), scale


class DMN(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.lstm = nn.LSTM(nin, HID, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(HID, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return torch.tanh(self.out(self.drop(h))).squeeze(-1)


def windows(X, R, C, idx):
    """Non-overlapping-ish sequences ending inside idx (stride SEQ//3)."""
    xs, rs, cs = [], [], []
    for e in range(idx[0] + SEQ, idx[-1] + 1, SEQ // 3):
        xs.append(X[e - SEQ:e]); rs.append(R[e - SEQ:e]); cs.append(C[e - SEQ:e])
    return (torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(rs), dtype=torch.float32),
            torch.tensor(np.array(cs), dtype=torch.float32))


def sharpe_loss(pos, r, c, bpy):
    turn = torch.cat([pos[:, :1].abs(), (pos[:, 1:] - pos[:, :-1]).abs()], 1)
    pnl = pos * r - turn * c
    return -(pnl.mean() / (pnl.std() + 1e-9)) * np.sqrt(bpy)


def train_fold(X, R, C, tr_idx, bpy, seed):
    torch.manual_seed(seed)
    cut = int(len(tr_idx) * 0.9)
    xt, rt, ct = windows(X, R, C, tr_idx[:cut])
    xv, rv, cv = windows(X, R, C, tr_idx[cut:])
    net = DMN(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    for ep in range(80):
        net.train()
        perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 64):
            ii = perm[b:b + 64]
            opt.zero_grad()
            loss = sharpe_loss(net(xt[ii]), rt[ii], ct[ii], bpy)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = sharpe_loss(net(xv), rv, cv, bpy).item()
        if vl < best - 1e-3:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, -best


def run(d, feats, scale, bpy, cols, name):
    Xall = feats[cols]
    ok = np.asarray(Xall.notna().all(1) & scale.notna())
    first = np.where(ok)[0][0]
    # scaled next-bar return and cost in the same units
    R = (scale * d["ret"].shift(-1).fillna(0)).fillna(0).values
    C = (scale * (d["hs"] + 0.2e-4)).fillna(0).values
    refits = pd.date_range(OOS_START, d.index[-1], freq="3MS", tz="US/Eastern")
    pos = np.zeros(len(d))
    log = []
    for i, r0 in enumerate(refits):
        r1 = refits[i + 1] if i + 1 < len(refits) else d.index[-1] + pd.Timedelta(days=1)
        tr_idx = np.where(np.asarray(d.index < r0))[0]
        tr_idx = tr_idx[tr_idx >= first][:-1]
        te_idx = np.where(np.asarray((d.index >= r0) & (d.index < r1)))[0]
        mu, sd = Xall.values[tr_idx].mean(0), Xall.values[tr_idx].std(0) + 1e-9
        X = np.nan_to_num((Xall.values - mu) / sd)
        preds, vs = [], []
        for seed in range(3):
            net, v = train_fold(X, R, C, tr_idx, bpy, seed)
            a = te_idx[0] - SEQ
            with torch.no_grad():
                # causal online inference: LSTM run forward from SEQ bars before the test block
                out = net(torch.tensor(X[a:te_idx[-1] + 1][None], dtype=torch.float32))[0].numpy()
            preds.append(out[SEQ:]); vs.append(v)
        pos[te_idx] = np.mean(preds, 0)
        log.append((str(r0)[:10], len(tr_idx), round(float(np.mean(vs)), 2), round(float(np.mean(pos[te_idx])), 3),
                    round(float(np.mean(np.abs(pos[te_idx]))), 3)))
    print(name, "(refit, n_train, val_sharpe, mean_pos, mean_abs_pos):")
    for l in log:
        print("  ", l)
    w = pd.Series(pos, index=d.index) * scale.fillna(0)
    return metrics(backtest(d, w.clip(-2, 2), name), name)


if __name__ == "__main__":
    d = load_h1()
    bpy = bars_per_year(d)
    feats, scale = make_inputs(d, bpy)
    base = [c for c in feats.columns if not c.startswith("cpd")]
    rows = [run(d, feats, scale, bpy, base, "s08_dmn_lstm"),
            run(d, feats, scale, bpy, list(feats.columns), "s08_dmn_lstm_cpd")]
    print(pd.DataFrame(rows).round(3).to_string())
    save_metrics(rows, "m_s08.csv")
