"""
Phase 4h / Steps 2-4 — The memory engine ("the brain").

The point: trades are NOT independent. Every candidate trade is scored partly by
what happened the last time the market looked like this, and partly by how the
strategy has been performing lately. Three memory systems, all strictly causal
(a trade can only ever see trades that CLOSED before it opened):

  1. RECENCY MEMORY   — rolling outcome of the last k closed trades: win rate,
                        mean R, current streak, time since last win/loss.
                        "Am I in sync with the market right now?"

  2. CONTEXT MEMORY   — the brain proper. A k-NN bank over a standardised context
                        vector (regime, vol, Hurst, stretch, trend, session). For
                        each candidate we retrieve the K most similar past
                        situations and summarise what actually happened in them.
                        "The last 50 times it looked like this, what happened?"

  3. BUCKET MEMORY    — coarse conditional stats: mean R of past trades sharing
                        the same HMM state / session / vol tercile / direction.
                        Lower resolution than k-NN but far more stable.

Plus STRUCTURE features (swing points, break-of-structure, higher-high/lower-low
counts), since "what structure is the market in" was part of the ask.

Causality guarantee: the k-NN bank is rebuilt every REBUILD_EVERY events using
only events whose exit bar precedes the current block's first entry bar. Stale by
up to one block, never forward-looking.

Note on circularity: memory is built from the base strategy's CANDIDATE sequence
and their realised outcomes, not from the subset the ML chose to take. That keeps
the feature deterministic and identical between train and test — otherwise the
features would depend on the model that consumes them. Feedback from *taken*
trades is applied separately, as a sizing overlay in the engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CLEAN = (Path(__file__).resolve().parent / "clean_Data")

# context vector used by the k-NN brain
CONTEXT_COLS = ["hurst", "vol_ratio_fs", "atr_pct", "vwap_dev_z",
                "trend_strength", "db_T", "cpd_score", "tod_sin", "tod_cos"]

K_NEIGHBOURS = 50
REBUILD_EVERY = 250
MIN_BANK = 300

RECENCY_WINDOWS = (3, 5, 10, 20)

MEMORY_FEATURES = (
    [f"mem_winrate_{k}" for k in RECENCY_WINDOWS]
    + [f"mem_meanR_{k}" for k in RECENCY_WINDOWS]
    + ["mem_streak", "mem_last_R", "mem_bars_since_trade",
       "mem_bars_since_win", "mem_bars_since_loss", "mem_equity_drift",
       "knn_meanR", "knn_winrate", "knn_medR", "knn_dist", "knn_n",
       "knn_meanR_samedir", "knn_winrate_samedir",
       "buck_state_meanR", "buck_state_winrate",
       "buck_sess_meanR", "buck_sess_winrate",
       "buck_vol_meanR", "buck_vol_winrate",
       "buck_dir_meanR", "buck_dir_winrate"]
)

STRUCTURE_FEATURES = ["swing_hi_dist", "swing_lo_dist", "bars_since_swing_hi",
                      "bars_since_swing_lo", "bos_up", "bos_down",
                      "hh_count", "ll_count", "struct_trend", "struct_range"]


# --------------------------------------------------------------- structure

def add_structure(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Fractal swing points: a swing high is a bar whose high exceeds the k bars on
    each side. Confirmed k bars late, so we shift by k to stay causal.
    """
    d = df.copy()
    hi = d["high"].to_numpy(dtype=np.float64)
    lo = d["low"].to_numpy(dtype=np.float64)
    c = d["close"].to_numpy(dtype=np.float64)
    n = len(d)

    is_sh = np.zeros(n, dtype=bool)
    is_sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        w_hi = hi[i - k:i + k + 1]
        w_lo = lo[i - k:i + k + 1]
        if hi[i] == w_hi.max() and (w_hi.argmax() == k):
            is_sh[i] = True
        if lo[i] == w_lo.min() and (w_lo.argmin() == k):
            is_sl[i] = True
    # confirmation lag: only known k bars later
    is_sh = np.roll(is_sh, k); is_sh[:k] = False
    is_sl = np.roll(is_sl, k); is_sl[:k] = False

    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    sh_bar = np.full(n, np.nan)
    sl_bar = np.full(n, np.nan)
    cur_sh, cur_sl, cur_shb, cur_slb = np.nan, np.nan, np.nan, np.nan
    hh = np.zeros(n, dtype=np.int32)
    ll = np.zeros(n, dtype=np.int32)
    hh_c, ll_c = 0, 0
    prev_sh, prev_sl = np.nan, np.nan
    for i in range(n):
        if is_sh[i]:
            if np.isfinite(cur_sh):
                prev_sh = cur_sh
                hh_c = hh_c + 1 if hi[i - k] > prev_sh else 0
            cur_sh, cur_shb = hi[i - k] if i >= k else hi[i], i
        if is_sl[i]:
            if np.isfinite(cur_sl):
                prev_sl = cur_sl
                ll_c = ll_c + 1 if lo[i - k] < prev_sl else 0
            cur_sl, cur_slb = lo[i - k] if i >= k else lo[i], i
        last_sh[i], last_sl[i] = cur_sh, cur_sl
        sh_bar[i], sl_bar[i] = cur_shb, cur_slb
        hh[i], ll[i] = hh_c, ll_c

    atr = d["atr14"].replace(0, np.nan).to_numpy()
    d["swing_hi_dist"] = (last_sh - c) / atr
    d["swing_lo_dist"] = (c - last_sl) / atr
    d["bars_since_swing_hi"] = np.arange(n) - sh_bar
    d["bars_since_swing_lo"] = np.arange(n) - sl_bar
    d["bos_up"] = (c > last_sh).astype(np.int8)
    d["bos_down"] = (c < last_sl).astype(np.int8)
    d["hh_count"] = hh
    d["ll_count"] = ll
    rng = (last_sh - last_sl) / atr
    d["struct_trend"] = (d["hh_count"] - d["ll_count"]).astype(np.float64)
    d["struct_range"] = rng
    return d


# --------------------------------------------------------------- memory build

def _sess(minute_of_day: np.ndarray) -> np.ndarray:
    """0=Asia, 1=London, 2=NY-overlap, 3=NY-late (ET minutes)."""
    m = minute_of_day
    out = np.zeros(len(m), dtype=np.int8)
    out[(m >= 180) & (m < 480)] = 1
    out[(m >= 480) & (m < 720)] = 2
    out[(m >= 720)] = 3
    return out


def build_memory(events: pd.DataFrame, df: pd.DataFrame,
                 k_neighbours: int = K_NEIGHBOURS) -> pd.DataFrame:
    """
    events: candidate table from phase4_labels (must have i, exit_i, dir, r_net,
            plus the CONTEXT_COLS already attached), sorted by entry bar.
    Returns the same table with MEMORY_FEATURES appended.
    """
    ev = events.sort_values("i").reset_index(drop=True).copy()
    n = len(ev)
    entry_i = ev["i"].to_numpy()
    exit_i = ev["exit_i"].to_numpy()
    r = ev["r_net"].to_numpy()
    win = (r > 0).astype(np.float64)
    direction = ev["dir"].to_numpy()

    # ---- context matrix, standardised on an expanding basis (causal)
    C = ev[CONTEXT_COLS].replace([np.inf, -np.inf], np.nan).astype(np.float64)
    C = C.fillna(C.median()).to_numpy()

    # ---- bucket keys
    state = (df["hmm3_state"].to_numpy()[entry_i]
             if "hmm3_state" in df.columns else np.zeros(n, dtype=int))
    sess = _sess(df["minute_of_day"].to_numpy()[entry_i])
    volq = pd.qcut(pd.Series(df["atr_pct"].to_numpy()[entry_i]).rank(method="first"),
                   3, labels=False).to_numpy()
    dirk = (direction > 0).astype(int)

    out = {k: np.full(n, np.nan) for k in MEMORY_FEATURES}

    # ---- running recency state
    closed_order = np.argsort(exit_i, kind="stable")
    closed_ptr = 0
    hist_r: list[float] = []
    hist_exit: list[int] = []
    last_win_bar, last_loss_bar, last_trade_bar = np.nan, np.nan, np.nan
    streak = 0
    cum_r = 0.0

    # bucket accumulators: dict -> [sum_r, count, sum_win]
    acc_state: dict = {}
    acc_sess: dict = {}
    acc_vol: dict = {}
    acc_dir: dict = {}

    def bstat(acc, key):
        v = acc.get(key)
        if not v or v[1] < 5:
            return np.nan, np.nan
        return v[0] / v[1], v[2] / v[1]

    # ---- k-NN bank, rebuilt periodically
    tree = None
    bank_idx = np.array([], dtype=int)
    mu = np.zeros(C.shape[1])
    sd = np.ones(C.shape[1])
    next_rebuild = 0

    for t in range(n):
        cur_bar = entry_i[t]

        # absorb every trade that closed strictly before this entry
        while closed_ptr < n:
            j = closed_order[closed_ptr]
            if exit_i[j] >= cur_bar:
                break
            hist_r.append(r[j])
            hist_exit.append(exit_i[j])
            cum_r += r[j]
            last_trade_bar = exit_i[j]
            if r[j] > 0:
                last_win_bar = exit_i[j]
                streak = streak + 1 if streak >= 0 else 1
            else:
                last_loss_bar = exit_i[j]
                streak = streak - 1 if streak <= 0 else -1
            for acc, key in ((acc_state, state[j]), (acc_sess, sess[j]),
                             (acc_vol, volq[j]), (acc_dir, dirk[j])):
                v = acc.setdefault(key, [0.0, 0, 0.0])
                v[0] += r[j]; v[1] += 1; v[2] += win[j]
            closed_ptr += 1

        # ---- recency features
        for k in RECENCY_WINDOWS:
            if len(hist_r) >= max(3, k // 2):
                tail = np.array(hist_r[-k:])
                out[f"mem_winrate_{k}"][t] = float((tail > 0).mean())
                out[f"mem_meanR_{k}"][t] = float(tail.mean())
        out["mem_streak"][t] = streak
        out["mem_last_R"][t] = hist_r[-1] if hist_r else np.nan
        out["mem_bars_since_trade"][t] = (cur_bar - last_trade_bar
                                          if np.isfinite(last_trade_bar) else np.nan)
        out["mem_bars_since_win"][t] = (cur_bar - last_win_bar
                                        if np.isfinite(last_win_bar) else np.nan)
        out["mem_bars_since_loss"][t] = (cur_bar - last_loss_bar
                                         if np.isfinite(last_loss_bar) else np.nan)
        out["mem_equity_drift"][t] = cum_r

        # ---- bucket memory
        for name, acc, key in (("state", acc_state, state[t]),
                               ("sess", acc_sess, sess[t]),
                               ("vol", acc_vol, volq[t]),
                               ("dir", acc_dir, dirk[t])):
            mr, wr = bstat(acc, key)
            out[f"buck_{name}_meanR"][t] = mr
            out[f"buck_{name}_winrate"][t] = wr

        # ---- k-NN context brain
        if t >= next_rebuild and closed_ptr >= MIN_BANK:
            bank_idx = closed_order[:closed_ptr]
            Xb = C[bank_idx]
            mu, sd = Xb.mean(0), Xb.std(0) + 1e-9
            tree = cKDTree((Xb - mu) / sd)
            next_rebuild = t + REBUILD_EVERY

        if tree is not None and len(bank_idx) >= MIN_BANK:
            q = ((C[t] - mu) / sd).reshape(1, -1)
            kk = min(k_neighbours, len(bank_idx))
            dist, nb = tree.query(q, k=kk)
            nb = np.atleast_1d(nb.ravel())
            dist = np.atleast_1d(dist.ravel())
            gi = bank_idx[nb]
            # only neighbours that closed before this entry
            ok = exit_i[gi] < cur_bar
            gi, dist = gi[ok], dist[ok]
            if len(gi) >= 5:
                out["knn_meanR"][t] = r[gi].mean()
                out["knn_medR"][t] = np.median(r[gi])
                out["knn_winrate"][t] = win[gi].mean()
                out["knn_dist"][t] = dist.mean()
                out["knn_n"][t] = len(gi)
                sd_mask = direction[gi] == direction[t]
                if sd_mask.sum() >= 5:
                    out["knn_meanR_samedir"][t] = r[gi][sd_mask].mean()
                    out["knn_winrate_samedir"][t] = win[gi][sd_mask].mean()

    for k, v in out.items():
        ev[k] = v
    return ev


def main() -> None:
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "p2"
    suffix = "" if base == "p3" else f"_{base}"

    from phase4_features import add_derived

    df = (pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet")
          .sort_values("dt_et").reset_index(drop=True))
    df = df.dropna(subset=["atr14", "close"]).reset_index(drop=True)
    df = add_derived(df)
    df = add_structure(df)

    ev = pd.read_parquet(CLEAN / f"phase4_meta_dataset{suffix}.parquet")
    ev = ev.sort_values("i").reset_index(drop=True)

    # attach structure + context columns at the entry bar
    need = [c for c in (STRUCTURE_FEATURES + CONTEXT_COLS) if c not in ev.columns]
    if need:
        ev = pd.concat([ev.reset_index(drop=True),
                        df.iloc[ev["i"].to_numpy()][need].reset_index(drop=True)],
                       axis=1)

    print(f"Building memory over {len(ev):,} candidate events (base={base}) ...")
    ev = build_memory(ev, df)

    cov = {c: f"{ev[c].notna().mean()*100:.0f}%" for c in
           ["mem_meanR_10", "knn_meanR", "buck_state_meanR", "knn_n"]}
    print(f"  coverage: {cov}")
    print(f"  knn_meanR    mean={ev['knn_meanR'].mean():+.4f} "
          f"std={ev['knn_meanR'].std():.4f}")
    print(f"  mem_meanR_10 mean={ev['mem_meanR_10'].mean():+.4f} "
          f"std={ev['mem_meanR_10'].std():.4f}")

    # does memory actually separate outcomes?
    print("\n--- does memory correlate with the NEXT trade's R? ---")
    for c in ["knn_meanR", "knn_meanR_samedir", "knn_winrate", "mem_meanR_10",
              "mem_winrate_10", "mem_streak", "buck_state_meanR",
              "buck_sess_meanR", "buck_vol_meanR", "struct_trend", "bos_up"]:
        if c in ev.columns:
            s = ev[[c, "r_net"]].dropna()
            if len(s) > 200:
                print(f"  {c:22s} corr={s[c].corr(s['r_net']):+.4f}  n={len(s):,}")

    ev.to_parquet(CLEAN / f"phase4_memory_dataset{suffix}.parquet",
                  index=False, compression="zstd")
    print(f"\nWrote {CLEAN/f'phase4_memory_dataset{suffix}.parquet'}  "
          f"({len(ev):,} x {ev.shape[1]})")


if __name__ == "__main__":
    main()
