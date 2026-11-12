"""
Phase 4e — Deep learning: Deep Momentum Network (LSTM trained on Sharpe ratio).

Why this architecture and not price regression:

The corpus mentions LSTM 94 times, far more than any other learner, but the
gold-specific results are blunt about naive use of it. The TATS paper applied
LSTM and Bi-LSTM to daily gold prices and found "very poor performance in trend
detection (TD accuracy is close to 50%) ... about half of the forecasts produced
by these models are in the wrong trend", concluding MSE/MAE are not sufficient
metrics. The NIFTY volatility comparison likewise found "the LSTM model
underperformed" EGARCH. And one FX paper warns outright that "the inconsistency
of the LSTM model can lead to poor results".

The failure in all three cases is the objective, not the architecture: minimising
MSE on price has almost nothing to do with making money. So this module follows
Lim, Zohren & Roberts' Deep Momentum Networks instead, which the corpus
describes as "a Long Short-Term Memory deep learning architecture [that] achieves
this by directly optimising on the Sharpe ratio of the signal". The net emits a
position in [-1, 1] via tanh and the loss is the negative Sharpe of
position * next-period return, net of a turnover cost. No labels, no barriers —
the objective *is* the thing we care about.

  loss = -(mean(pnl) / std(pnl)) * sqrt(ann),
  pnl_t = w_t * r_{t+1} - cost * |w_t - w_{t-1}|

Protocol: same expanding-window splits as phase4_ml, sequences of 32 bars,
features standardised on the training block only, early stopping on a validation
tail carved out of the training block (never the test block).

Writes: clean_Data/phase4_dmn_positions.parquet
        clean_Data/phase4_dmn_report.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from phase4_features import ML_FEATURES_STRICT, add_derived, clean_matrix

warnings.filterwarnings("ignore")
CLEAN = (Path(__file__).resolve().parent / "clean_Data")

SEQ_LEN = 32
HIDDEN = 32
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 60
PATIENCE = 8
BATCH = 256
COST_BPS = 1.5          # round-trip cost charged on position change, in bps
BARS_PER_YEAR = 252 * 78
N_FOLDS = 5
SEED = 17


class DMN(nn.Module):
    """LSTM -> dropout -> linear -> tanh position in [-1, 1]."""

    def __init__(self, n_feat: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                  # x: (B, T, F)
        h, _ = self.lstm(x)
        z = self.drop(h[:, -1, :])         # last timestep
        return torch.tanh(self.head(z)).squeeze(-1)


def sharpe_loss(w: torch.Tensor, r_next: torch.Tensor,
                cost: float = COST_BPS * 1e-4) -> torch.Tensor:
    """Negative annualised Sharpe of the position's PnL, with turnover cost."""
    pnl = w * r_next
    turn = torch.abs(w[1:] - w[:-1])
    pnl = torch.cat([pnl[:1], pnl[1:] - cost * turn])
    mu = pnl.mean()
    sd = pnl.std() + 1e-8
    return -(mu / sd) * np.sqrt(BARS_PER_YEAR)


def make_sequences(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Strided view -> (n - seq_len + 1, seq_len, n_feat). Index k ends at bar k+seq_len-1."""
    n, f = X.shape
    s0, s1 = X.strides
    return np.lib.stride_tricks.as_strided(
        X, shape=(n - seq_len + 1, seq_len, f), strides=(s0, s0, s1), writeable=False)


def train_fold(Xtr, rtr, Xva, rva, n_feat, device, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DMN(n_feat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    rtr_t = torch.tensor(rtr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    rva_t = torch.tensor(rva, dtype=torch.float32, device=device)

    best_va, best_state, bad = np.inf, None, 0
    n = len(Xtr_t)
    for ep in range(EPOCHS):
        model.train()
        # contiguous blocks, not shuffled samples: the Sharpe loss needs the
        # time ordering inside a batch for the turnover term to mean anything
        starts = np.arange(0, n - BATCH, BATCH)
        np.random.shuffle(starts)
        for st in starts:
            sl = slice(st, st + BATCH)
            opt.zero_grad()
            w = model(Xtr_t[sl])
            loss = sharpe_loss(w, rtr_t[sl])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            va = float(sharpe_loss(model(Xva_t), rva_t).item())
        if va < best_va - 1e-4:
            best_va, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, -best_va


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}  device={device}")

    df = (pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet")
          .sort_values("dt_et").reset_index(drop=True))
    df = df.dropna(subset=["atr14", "close"]).reset_index(drop=True)
    df = add_derived(df)

    feats = ML_FEATURES_STRICT
    X = clean_matrix(df[feats])
    X = X.fillna(X.median()).to_numpy(dtype=np.float64)
    logp = np.log(df["close"].to_numpy(dtype=np.float64))
    r_next = np.append(np.diff(logp), 0.0)          # return from t to t+1

    n_feat = X.shape[1]
    print(f"bars={len(df):,}  features={n_feat}  seq_len={SEQ_LEN}")

    # sequence k ends at bar k+SEQ_LEN-1, so it predicts r_next[k+SEQ_LEN-1]
    end_bar = np.arange(SEQ_LEN - 1, len(df))
    y_seq = r_next[end_bar]

    pos_all = np.full(len(df), np.nan)
    rows = []
    bounds = np.linspace(len(end_bar) // (N_FOLDS + 1), len(end_bar),
                         N_FOLDS + 1).astype(int)

    for f in range(N_FOLDS):
        a, b = bounds[f], bounds[f + 1]
        # embargo one day of bars between train and test
        tr_hi = max(a - 78, 0)
        if tr_hi < 3000:
            continue
        va_lo = int(tr_hi * 0.85)

        # standardise on the training block only
        tr_bars = end_bar[:tr_hi]
        mu = X[tr_bars].mean(0)
        sd = X[tr_bars].std(0) + 1e-8
        Xs = ((X - mu) / sd).astype(np.float32)
        Xs = np.clip(Xs, -8, 8)
        seqs = make_sequences(Xs, SEQ_LEN)

        Xtr, rtr = seqs[:va_lo], y_seq[:va_lo]
        Xva, rva = seqs[va_lo:tr_hi], y_seq[va_lo:tr_hi]
        Xte, rte = seqs[a:b], y_seq[a:b]

        t0 = time.time()
        model, va_sharpe = train_fold(Xtr, rtr, Xva, rva, n_feat, device)
        model.eval()
        with torch.no_grad():
            w_te = model(torch.tensor(Xte, dtype=torch.float32,
                                      device=device)).cpu().numpy()

        pnl = w_te * rte
        turn = np.abs(np.diff(w_te, prepend=w_te[0]))
        pnl = pnl - COST_BPS * 1e-4 * turn
        sh = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(BARS_PER_YEAR)
        # long-only buy-and-hold over the same block, for reference
        bh = rte.mean() / (rte.std() + 1e-12) * np.sqrt(BARS_PER_YEAR)

        pos_all[end_bar[a:b]] = w_te
        rows.append({"fold": f, "n_train": int(va_lo), "n_test": int(b - a),
                     "val_sharpe": float(va_sharpe),
                     "test_sharpe": float(sh),
                     "buyhold_sharpe": float(bh),
                     "mean_pos": float(w_te.mean()),
                     "abs_pos": float(np.abs(w_te).mean()),
                     "turnover": float(turn.mean()),
                     "secs": round(time.time() - t0, 1)})
        rr = rows[-1]
        print(f" fold {f}: train={rr['n_train']:,} test={rr['n_test']:,} "
              f"valSharpe={rr['val_sharpe']:+.2f} testSharpe={rr['test_sharpe']:+.2f} "
              f"(buy&hold {rr['buyhold_sharpe']:+.2f}) "
              f"meanPos={rr['mean_pos']:+.3f} |pos|={rr['abs_pos']:.3f} "
              f"turn={rr['turnover']:.4f} ({rr['secs']}s)")

    res = pd.DataFrame(rows)
    print(f"\n--- DMN fold summary ---")
    print(res.round(4).to_string(index=False))
    print(f"\nmean test Sharpe = {res['test_sharpe'].mean():+.3f}   "
          f"mean buy&hold = {res['buyhold_sharpe'].mean():+.3f}   "
          f"folds positive = {int((res['test_sharpe'] > 0).sum())}/{len(res)}")

    out = pd.DataFrame({"dt_et": df["dt_et"], "dmn_pos": pos_all,
                        "close": df["close"]})
    out.to_parquet(CLEAN / "phase4_dmn_positions.parquet",
                   index=False, compression="zstd")
    with open(CLEAN / "phase4_dmn_report.json", "w") as f:
        json.dump({"config": {"seq_len": SEQ_LEN, "hidden": HIDDEN,
                              "dropout": DROPOUT, "lr": LR, "epochs": EPOCHS,
                              "cost_bps": COST_BPS, "n_features": n_feat},
                   "folds": res.to_dict(orient="records"),
                   "mean_test_sharpe": float(res["test_sharpe"].mean()),
                   "mean_buyhold_sharpe": float(res["buyhold_sharpe"].mean())},
                  f, indent=2, default=str)
    print(f"\nWrote {CLEAN/'phase4_dmn_positions.parquet'}")
    print(f"Wrote {CLEAN/'phase4_dmn_report.json'}")


if __name__ == "__main__":
    main()
