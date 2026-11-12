"""
Phase 4b — Online brain, concurrent entries, trade-frequency sweep.

Three changes versus phase4_brain.py:

1. THE BRAIN UPDATES AFTER EVERY TRADE.
   In Phase 4 the memory features were precomputed from the candidate sequence,
   which made them deterministic but meant the brain never learned from what it
   actually did. Here the brain is a live object. The instant a position closes,
   its PnL, R, direction, hold time, exit reason AND the full situation vector it
   was opened in (regime, trend, structure, vol, session, memory state at the
   time) are absorbed. The very next decision sees them.

   "Completely after every trade" is implemented at three speeds:
     * memory bank  -> updated on every close, O(1)
     * online model -> SGD logistic `partial_fit` on every close
     * batch model  -> XGBoost refit every REFIT_EVERY closes (an incremental
                       tree model does not exist; this is the practical bound)
   Final score blends the online and batch models.

   NOTE this introduces genuine policy feedback: memory now depends on which
   trades the brain chose to take, so the features are no longer independent of
   the model. That is what was asked for and it is what live trading does, but it
   means results are policy-dependent and cannot be compared to Phase 4's numbers
   feature-for-feature.

2. MULTIPLE CONCURRENT ENTRIES.
   The engine holds up to `max_concurrent` open positions, with a cap on total
   open risk. Equity is marked to market every bar so drawdown accounts for
   unrealised losses on all open positions, not just closed ones.

3. TRADE-FREQUENCY SWEEP.
   Configurations from conservative to aggressive, reported per month, so the
   effect of trading more is visible rather than assumed.

Writes: clean_Data/phase4b_results.json
        clean_Data/phase4b_monthly.csv
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree
from sklearn.linear_model import SGDClassifier

from phase2_common import CAPITAL_0, _max_drawdown
from phase4_features import ML_FEATURES_STRICT, add_derived, clean_matrix
from phase4_memory import CONTEXT_COLS, STRUCTURE_FEATURES, add_structure

warnings.filterwarnings("ignore")
CLEAN = (Path(__file__).resolve().parent / "clean_Data")

STOP_K, TARGET_K, MAX_HOLD = 1.5, 2.5, 12
MAX_LEV = 10.0
DAILY_SOFT, TOTAL_SOFT = 0.03, 0.07

K_NEIGHBOURS = 50
REBUILD_EVERY = 25          # k-NN tree rebuild, in closed trades
REFIT_EVERY = 50            # XGBoost refit, in closed trades
MIN_BANK = 60               # trades before the brain is allowed an opinion
RECENCY_WINDOWS = (3, 5, 10, 20)

# The brain cannot start untrained: in the first run every aggressive config
# flooded January with unfiltered trades and hit the total-loss halt in month 1.
# So it first PAPER-TRADES a warm-up block -- full learning, no capital at risk --
# and only then begins trading for real.
WARMUP_FRAC = 0.20
# Total-loss halt is measured from the running PEAK, not from day-one equity, and
# it releases once equity recovers. A permanent halt made months 2-22 untradeable
# and silently turned the sweep into a one-month experiment.
TRAIL_HALT = True

LIVE_MEM_FEATURES = (
    [f"mem_winrate_{k}" for k in RECENCY_WINDOWS]
    + [f"mem_meanR_{k}" for k in RECENCY_WINDOWS]
    + ["mem_streak", "mem_last_R", "mem_bars_since_trade", "mem_open_n",
       "mem_cum_R", "knn_meanR", "knn_winrate", "knn_dist", "knn_n",
       "knn_meanR_samedir", "buck_state_meanR", "buck_state_winrate",
       "buck_sess_meanR", "buck_vol_meanR", "buck_dir_meanR"]
)


def sess_of(m: float) -> int:
    if m < 180:  return 0
    if m < 480:  return 1
    if m < 720:  return 2
    return 3


# ----------------------------------------------------------------- the brain

class OnlineBrain:
    """Live memory + online classifier. Everything updates on trade close."""

    def __init__(self, n_static: int, seed: int = 7):
        self.ctx: list[np.ndarray] = []      # context vector per closed trade
        self.feat: list[np.ndarray] = []     # full feature vector at entry
        self.R: list[float] = []
        self.dirs: list[float] = []
        self.state: list[int] = []
        self.sess: list[int] = []
        self.volq: list[int] = []

        self.recent = deque(maxlen=max(RECENCY_WINDOWS))
        self.streak = 0
        self.cum_R = 0.0
        self.last_exit_bar = np.nan
        self.last_R = np.nan

        self.acc = {"state": {}, "sess": {}, "vol": {}, "dir": {}}

        self.tree = None
        self.mu = None
        self.sd = None
        self.n_since_rebuild = 0
        self.n_since_refit = 0

        self.sgd = SGDClassifier(loss="log_loss", alpha=1e-4,
                                 learning_rate="adaptive", eta0=0.01,
                                 random_state=seed, warm_start=True)
        self.sgd_ready = False
        self.xgbm = None
        self.n_static = n_static
        self.seed = seed

    # -------------------------------------------------- feature construction
    def memory_features(self, ctx_vec, direction, state, sess, volq,
                        bar, n_open) -> np.ndarray:
        f = {}
        rec = np.array(self.recent, dtype=float)
        for k in RECENCY_WINDOWS:
            if len(rec) >= max(3, k // 2):
                tail = rec[-k:]
                f[f"mem_winrate_{k}"] = float((tail > 0).mean())
                f[f"mem_meanR_{k}"] = float(tail.mean())
            else:
                f[f"mem_winrate_{k}"] = np.nan
                f[f"mem_meanR_{k}"] = np.nan
        f["mem_streak"] = self.streak
        f["mem_last_R"] = self.last_R
        f["mem_bars_since_trade"] = (bar - self.last_exit_bar
                                     if np.isfinite(self.last_exit_bar) else np.nan)
        f["mem_open_n"] = n_open
        f["mem_cum_R"] = self.cum_R

        for name, key in (("state", state), ("sess", sess),
                          ("vol", volq), ("dir", int(direction > 0))):
            v = self.acc[name].get(key)
            if v and v[1] >= 5:
                f[f"buck_{name}_meanR"] = v[0] / v[1]
                if name == "state":
                    f["buck_state_winrate"] = v[2] / v[1]
            else:
                f[f"buck_{name}_meanR"] = np.nan
                if name == "state":
                    f["buck_state_winrate"] = np.nan

        f["knn_meanR"] = f["knn_winrate"] = f["knn_dist"] = np.nan
        f["knn_n"] = 0.0
        f["knn_meanR_samedir"] = np.nan
        if self.tree is not None and len(self.ctx) >= MIN_BANK:
            q = ((ctx_vec - self.mu) / self.sd).reshape(1, -1)
            kk = min(K_NEIGHBOURS, len(self.ctx))
            dist, nb = self.tree.query(q, k=kk)
            nb = np.atleast_1d(nb.ravel()).astype(int)
            dist = np.atleast_1d(dist.ravel())
            Rn = np.array(self.R)[nb]
            f["knn_meanR"] = float(Rn.mean())
            f["knn_winrate"] = float((Rn > 0).mean())
            f["knn_dist"] = float(dist.mean())
            f["knn_n"] = float(len(nb))
            dm = np.array(self.dirs)[nb] == direction
            if dm.sum() >= 5:
                f["knn_meanR_samedir"] = float(Rn[dm].mean())
        return np.array([f[k] for k in LIVE_MEM_FEATURES], dtype=np.float64)

    # -------------------------------------------------- prediction
    def predict(self, x: np.ndarray) -> float:
        xi = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
        ps = []
        if self.xgbm is not None:
            ps.append(float(self.xgbm.predict_proba(xi)[0, 1]))
        if self.sgd_ready:
            ps.append(float(self.sgd.predict_proba(xi)[0, 1]))
        return float(np.mean(ps)) if ps else 0.5

    # -------------------------------------------------- learning
    def update(self, feat_vec, ctx_vec, R, direction, state, sess, volq, exit_bar):
        """Absorb one closed trade. Called the instant a position closes."""
        self.ctx.append(ctx_vec)
        self.feat.append(np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0))
        self.R.append(float(R))
        self.dirs.append(float(direction))
        self.state.append(int(state)); self.sess.append(int(sess))
        self.volq.append(int(volq))

        self.recent.append(float(R))
        self.cum_R += float(R)
        self.last_R = float(R)
        self.last_exit_bar = exit_bar
        if R > 0:
            self.streak = self.streak + 1 if self.streak >= 0 else 1
        else:
            self.streak = self.streak - 1 if self.streak <= 0 else -1

        w = (R > 0)
        for name, key in (("state", state), ("sess", sess),
                          ("vol", volq), ("dir", int(direction > 0))):
            v = self.acc[name].setdefault(key, [0.0, 0, 0.0])
            v[0] += R; v[1] += 1; v[2] += w

        # --- online SGD step, every single trade
        y = np.array([int(R > 0)])
        X = self.feat[-1].reshape(1, -1)
        sw = np.array([min(abs(R), 5.0) + 0.05])
        try:
            if not self.sgd_ready:
                if len(set(int(r > 0) for r in self.R)) > 1 and len(self.R) >= 20:
                    Xa = np.vstack(self.feat)
                    ya = np.array([int(r > 0) for r in self.R])
                    swa = np.clip(np.abs(self.R), 0.05, 5.0)
                    self.sgd.fit(Xa, ya, sample_weight=swa)
                    self.sgd_ready = True
            else:
                self.sgd.partial_fit(X, y, sample_weight=sw)
        except Exception:
            pass

        # --- k-NN bank rebuild
        self.n_since_rebuild += 1
        if (self.n_since_rebuild >= REBUILD_EVERY and len(self.ctx) >= MIN_BANK):
            Xb = np.vstack(self.ctx)
            self.mu = Xb.mean(0)
            self.sd = Xb.std(0) + 1e-9
            self.tree = cKDTree((Xb - self.mu) / self.sd)
            self.n_since_rebuild = 0

        # --- batch refit
        self.n_since_refit += 1
        if self.n_since_refit >= REFIT_EVERY and len(self.R) >= MIN_BANK:
            ya = np.array([int(r > 0) for r in self.R])
            if len(np.unique(ya)) > 1:
                Xa = np.vstack(self.feat)
                swa = np.clip(np.abs(self.R), 0.05, 5.0)
                pos = max(int(ya.sum()), 1)
                m = xgb.XGBClassifier(
                    n_estimators=250, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                    reg_lambda=3.0, gamma=0.5,
                    scale_pos_weight=float((len(ya) - pos) / pos),
                    objective="binary:logistic", tree_method="hist",
                    n_jobs=-1, random_state=self.seed, verbosity=0)
                m.fit(Xa, ya, sample_weight=swa)
                self.xgbm = m
            self.n_since_refit = 0


# ----------------------------------------------------------------- engine

@dataclass
class Cfg:
    name: str
    entry_thresh: float = 0.35
    max_concurrent: int = 1
    max_trades_day: int = 6
    risk_per_trade: float = 0.005
    max_open_risk: float = 0.02
    accept_pct: float = 50.0        # accept trades above this pct of recent p
    allow_same_dir: bool = True
    feedback: bool = True


@dataclass
class Pos:
    i: int
    direction: float
    entry: float
    stop: float
    tgt: float
    size_oz: float
    risk_cash: float
    feat: np.ndarray = field(default=None)
    ctx: np.ndarray = field(default=None)
    state: int = 0
    sess: int = 0
    volq: int = 0
    paper: bool = False


def run_online(df, sig, static_X, ctx_X, state_v, volq_v, cfg: Cfg,
               brain: OnlineBrain, equity0=CAPITAL_0, warmup_bars: int = 0,
               monthly_reset: bool = True):
    """
    monthly_reset: each calendar month is an independent $5,000 challenge attempt
    (equity, peak and both halts reset; open positions are closed at the
    boundary). The BRAIN's memory is NOT reset -- it keeps learning across every
    month, which is the point. Without this, one bad week puts a config below the
    trailing halt and it can never trade again, since equity only moves by
    trading; that deadlock is what made the aggressive configs trade LESS.
    """
    n = len(df)
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    mod = df["minute_of_day"].to_numpy()
    atr = df["atr14"].to_numpy(np.float64)
    if {"ask_close", "bid_close"}.issubset(df.columns):
        spread = (df["ask_close"] - df["bid_close"]).to_numpy(np.float64)
    else:
        spread = df["spread_median"].to_numpy(np.float64)
    med = np.nanmedian(spread)
    spread = np.where(~np.isfinite(spread) | (spread <= 0), med, spread)
    dates = pd.to_datetime(df["dt_et"]).dt.date.to_numpy()

    eq = equity0
    peak = equity0
    equity = np.full(n, equity0, dtype=np.float64)
    open_pos: list[Pos] = []
    trades = []
    p_hist = deque(maxlen=400)

    cur_date, day_open_eq, day_trades, halted_day = None, equity0, 0, False
    months = pd.to_datetime(df["dt_et"]).dt.to_period("M").to_numpy()
    cur_month = None

    for i in range(n - 1):
        paper = i < warmup_bars           # learning only, no capital at risk
        d = dates[i]

        if monthly_reset and months[i] != cur_month:
            if cur_month is not None:
                # close everything at the boundary, letting the brain learn first
                for p in open_pos:
                    gross = p.size_oz * p.direction * (cl[i] - p.entry)
                    pnl = gross - p.size_oz * 2 * spread[p.i]
                    R = pnl / max(p.risk_cash, 1e-9)
                    if not p.paper:
                        eq += pnl
                        trades.append({"entry_time": df["dt_et"].iloc[p.i],
                                       "exit_time": df["dt_et"].iloc[i],
                                       "dir": p.direction, "pnl": pnl, "r": R,
                                       "reason": "month_end",
                                       "bars_held": i - p.i,
                                       "size_oz": p.size_oz})
                    brain.update(p.feat, p.ctx, R, p.direction, p.state,
                                 p.sess, p.volq, i)
                open_pos = []
            cur_month = months[i]
            eq = equity0
            peak = equity0
            halted_day = False

        if d != cur_date:
            cur_date, day_open_eq, day_trades, halted_day = d, eq, 0, False

        # ---------- 1. manage open positions, learn the instant one closes
        still: list[Pos] = []
        for p in open_pos:
            h, l = hi[i], lo[i]
            exit_price, reason = None, None
            if p.direction > 0:
                if l <= p.stop:   exit_price, reason = p.stop, "stop"
                elif h >= p.tgt:  exit_price, reason = p.tgt, "target"
            else:
                if h >= p.stop:   exit_price, reason = p.stop, "stop"
                elif l <= p.tgt:  exit_price, reason = p.tgt, "target"
            if exit_price is None and (i - p.i) >= MAX_HOLD:
                exit_price, reason = cl[i], "hold_end"

            if exit_price is None:
                still.append(p)
                continue

            gross = p.size_oz * p.direction * (exit_price - p.entry)
            cost = p.size_oz * 2 * spread[p.i]
            pnl = gross - cost
            R = pnl / max(p.risk_cash, 1e-9)
            if not p.paper:
                eq += pnl
                peak = max(peak, eq)
                trades.append({"entry_time": df["dt_et"].iloc[p.i],
                               "exit_time": df["dt_et"].iloc[i],
                               "dir": p.direction, "pnl": pnl, "r": R,
                               "reason": reason, "bars_held": i - p.i,
                               "size_oz": p.size_oz})
            # >>> the brain learns, immediately, before any further decision <<<
            brain.update(p.feat, p.ctx, R, p.direction, p.state, p.sess,
                         p.volq, i)
        open_pos = still

        # ---------- 2. risk gates (paper mode ignores capital limits)
        ok = True
        ref = peak if TRAIL_HALT else equity0
        if not paper:
            if halted_day:
                ok = False
            elif eq <= ref * (1 - TOTAL_SOFT):
                ok = False                      # releases when equity recovers
            elif eq <= day_open_eq * (1 - DAILY_SOFT):
                halted_day = True; ok = False
            elif day_trades >= cfg.max_trades_day:
                ok = False
        if len(open_pos) >= cfg.max_concurrent:
            ok = False

        s = sig[i]
        a = atr[i]
        if ok and np.isfinite(s) and abs(s) >= cfg.entry_thresh and np.isfinite(a) and a > 0:
            direction = 1.0 if s > 0 else -1.0
            if not cfg.allow_same_dir and any(p.direction == direction for p in open_pos):
                ok = False
            open_risk = sum(p.risk_cash for p in open_pos)
            if ok and (open_risk + cfg.risk_per_trade * eq) <= cfg.max_open_risk * eq:
                st, sv = int(state_v[i]), sess_of(mod[i])
                vq = int(volq_v[i])
                memf = brain.memory_features(ctx_X[i], direction, st, sv, vq,
                                             i, len(open_pos))
                feat = np.concatenate([static_X[i], memf])

                p_win = brain.predict(feat)
                p_hist.append(p_win)
                if len(brain.R) < MIN_BANK:
                    accept = True                      # warm-up: take everything
                else:
                    cut = np.percentile(p_hist, cfg.accept_pct) if len(p_hist) >= 30 else 0.5
                    accept = p_win >= cut

                if accept:
                    mult = 1.0
                    if cfg.feedback and len(brain.recent) >= 5:
                        mult = float(np.clip(1.0 + 0.35 * np.mean(brain.recent),
                                             0.5, 1.5))
                    risk_cash = cfg.risk_per_trade * eq * mult
                    entry = cl[i]
                    size_oz = risk_cash / (STOP_K * a)
                    notional = size_oz * entry
                    cap = MAX_LEV * eq / max(len(open_pos) + 1, 1)
                    if notional > cap:
                        size_oz *= cap / notional
                        risk_cash *= cap / notional
                    open_pos.append(Pos(
                        i=i, direction=direction, entry=entry,
                        stop=entry - direction * STOP_K * a,
                        tgt=entry + direction * TARGET_K * a,
                        size_oz=size_oz, risk_cash=risk_cash,
                        feat=feat, ctx=ctx_X[i], state=st, sess=sv, volq=vq,
                        paper=paper))
                    if not paper:
                        day_trades += 1

        # ---------- 3. mark to market
        unreal = sum(p.size_oz * p.direction * (cl[i] - p.entry)
                     for p in open_pos if not p.paper)
        equity[i] = eq + unreal

    equity[-1] = eq
    return pd.DataFrame(trades), equity


# ----------------------------------------------------------------- reporting

def monthly_table(trades: pd.DataFrame, equity: np.ndarray,
                  df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["dt_et"])
    m = t.dt.to_period("M")
    eq = pd.Series(equity, index=m.to_numpy())
    first = eq.groupby(level=0).first()
    last = eq.groupby(level=0).last()
    ret = (last / first - 1) * 100

    if len(trades):
        tt = trades.copy()
        tt["month"] = pd.to_datetime(tt["entry_time"]).dt.to_period("M")
        cnt = tt.groupby("month").size()
        win = tt.groupby("month")["pnl"].apply(lambda s: (s > 0).mean() * 100)
        pnl = tt.groupby("month")["pnl"].sum()
    else:
        cnt = win = pnl = pd.Series(dtype=float)

    out = pd.DataFrame({"ret_%": ret.round(2), "trades": cnt,
                        "hit_%": win.round(1), "pnl_$": pnl.round(0)})
    out["trades"] = out["trades"].fillna(0).astype(int)
    return out


def summarize(trades, equity, df, label):
    """Equity resets each month, so aggregate stats are built from monthly blocks."""
    per = pd.to_datetime(df["dt_et"]).dt.to_period("M")
    months = per.nunique()
    mt = monthly_table(trades, equity, df)

    # worst intra-month drawdown, and the compounded path across attempts
    dds = []
    for _, idx in pd.Series(np.arange(len(equity)), index=per.to_numpy()).groupby(level=0):
        e = equity[idx.to_numpy()]
        if len(e) > 1:
            dds.append(_max_drawdown(e))
    mr = mt["ret_%"].to_numpy() / 100.0
    compounded = float(np.prod(1 + mr) - 1)

    pos_m = int((mt["ret_%"] > 0).sum())
    hit8 = int((mt["ret_%"] >= 8).sum())
    hit5dd = int((pd.Series(dds) <= -0.05).sum()) if dds else 0
    return {
        "config": label,
        "trades": int(len(trades)),
        "trades_per_month": round(len(trades) / max(months, 1), 1),
        "hit_rate": round(float((trades["pnl"] > 0).mean()) * 100, 1) if len(trades) else np.nan,
        "avg_R": round(float(trades["r"].mean()), 3) if len(trades) else np.nan,
        "compounded_%": round(compounded * 100, 2),
        "worst_month_dd_%": round(float(np.min(dds)) * 100, 2) if dds else np.nan,
        "months": int(months),
        "pos_months": pos_m,
        "months_ge_8pct": hit8,
        "months_dd_breach": hit5dd,
        "best_month_%": round(float(mt["ret_%"].max()), 2),
        "worst_month_%": round(float(mt["ret_%"].min()), 2),
        "mean_month_%": round(float(mt["ret_%"].mean()), 2),
        "median_month_%": round(float(mt["ret_%"].median()), 2),
    }, mt


def main() -> None:
    df = (pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet")
          .sort_values("dt_et").reset_index(drop=True))
    df = df.dropna(subset=["atr14", "close"]).reset_index(drop=True)
    df = add_derived(df)
    df = add_structure(df)

    with open(CLEAN / "phase2_best_params.json") as f:
        best2 = json.load(f)
    from phase2c_strategy import ensemble as ensemble2
    sig = ensemble2(df, {k: float(v) for k, v in best2["weights"].items()},
                    best2["params"])

    static_cols = [c for c in (ML_FEATURES_STRICT + STRUCTURE_FEATURES)
                   if c in df.columns]
    SX = clean_matrix(df[static_cols])
    SX = SX.fillna(SX.median()).to_numpy(np.float64)
    CX = clean_matrix(df[CONTEXT_COLS])
    CX = CX.fillna(CX.median()).to_numpy(np.float64)
    state_v = (df["hmm3_state"].to_numpy() if "hmm3_state" in df.columns
               else np.zeros(len(df), int))
    volq_v = pd.qcut(df["atr_pct"].rank(method="first"), 3,
                     labels=False).fillna(1).to_numpy()

    n_feat = SX.shape[1] + len(LIVE_MEM_FEATURES)
    print(f"bars={len(df):,}  static={SX.shape[1]}  memory={len(LIVE_MEM_FEATURES)}"
          f"  total={n_feat}")
    print(f"period {df['dt_et'].iloc[0]} -> {df['dt_et'].iloc[-1]}\n")

    configs = [
        Cfg("A_baseline",   entry_thresh=0.35, max_concurrent=1, max_trades_day=6,  accept_pct=50),
        Cfg("B_2x",         entry_thresh=0.28, max_concurrent=2, max_trades_day=10, accept_pct=45),
        Cfg("C_4x",         entry_thresh=0.20, max_concurrent=3, max_trades_day=16, accept_pct=40),
        Cfg("D_8x",         entry_thresh=0.14, max_concurrent=5, max_trades_day=26, accept_pct=35),
        Cfg("E_max",        entry_thresh=0.10, max_concurrent=8, max_trades_day=40, accept_pct=30),
        Cfg("C_nofilter",   entry_thresh=0.20, max_concurrent=3, max_trades_day=16, accept_pct=0),
    ]

    warmup_bars = int(len(df) * WARMUP_FRAC)
    live = slice(warmup_bars, len(df))
    df_live = df.iloc[live].reset_index(drop=True)
    print(f"warm-up (paper, learning only): bars 0..{warmup_bars:,} "
          f"-> {df['dt_et'].iloc[warmup_bars]}")
    print(f"live trading: {df_live['dt_et'].iloc[0]} -> "
          f"{df_live['dt_et'].iloc[-1]}  ({len(df_live):,} bars)\n")

    rows, monthlies = [], {}
    for cfg in configs:
        brain = OnlineBrain(n_static=SX.shape[1])
        tr, eq = run_online(df, sig, SX, CX, state_v, volq_v, cfg, brain,
                            warmup_bars=warmup_bars)
        eq = eq[live]
        s, mt = summarize(tr, eq, df_live, cfg.name)
        s["learned_from"] = len(brain.R)
        rows.append(s)
        monthlies[cfg.name] = mt
        print(f"{cfg.name:12s} trades={s['trades']:>4d} "
              f"({s['trades_per_month']:>5.1f}/mo)  hit={s['hit_rate']}%  "
              f"avgR={s['avg_R']}  mean/mo={s['mean_month_%']:+6.2f}%  "
              f"best={s['best_month_%']:+6.2f}%  worst={s['worst_month_%']:+6.2f}%  "
              f"months>=8%: {s['months_ge_8pct']}/{s['months']}  "
              f"DDbreach: {s['months_dd_breach']}  (brain learned from "
              f"{s['learned_from']} trades)")

    res = pd.DataFrame(rows)
    print(f"\n{'='*104}\nTRADE-FREQUENCY SWEEP\n{'='*104}")
    print(res[["config", "trades", "trades_per_month", "hit_rate", "avg_R",
               "mean_month_%", "median_month_%", "best_month_%",
               "worst_month_%", "worst_month_dd_%", "pos_months",
               "months_ge_8pct", "months_dd_breach", "months",
               "compounded_%"]].to_string(index=False))

    print(f"\n{'='*104}\nPER-MONTH DETAIL\n{'='*104}")
    allm = []
    for name, mt in monthlies.items():
        mm = mt.copy()
        mm.insert(0, "config", name)
        mm.insert(1, "month", mm.index.astype(str))
        allm.append(mm.reset_index(drop=True))
    allm = pd.concat(allm, ignore_index=True)

    piv_r = allm.pivot(index="month", columns="config", values="ret_%")
    piv_t = allm.pivot(index="month", columns="config", values="trades")
    print("\nMonthly return %:")
    print(piv_r.round(2).to_string())
    print("\nTrades per month:")
    print(piv_t.fillna(0).astype(int).to_string())

    allm.to_csv(CLEAN / "phase4b_monthly.csv", index=False)
    with open(CLEAN / "phase4b_results.json", "w") as f:
        json.dump({"configs": rows}, f, indent=2, default=str)
    print(f"\nWrote {CLEAN/'phase4b_results.json'}")
    print(f"Wrote {CLEAN/'phase4b_monthly.csv'}")


if __name__ == "__main__":
    main()
