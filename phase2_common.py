"""
Shared utilities for Phase 2: data loading, metrics, and the trade-level backtest engine.
Called from phase2a..phase2e. No ML/DL anywhere — pure math + stochastic-process signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = (Path(__file__).resolve().parent / "clean_Data")

# Challenge constants
CAPITAL_0        = 5000.0
TARGET_RET       = 0.08
DAILY_DD_HARD    = 0.05
TOTAL_DD_HARD    = 0.10
DAILY_DD_SOFT    = 0.03
TOTAL_DD_SOFT    = 0.07
MAX_LEVERAGE     = 75.0
MAX_LEV_USED     = 10.0
RISK_PER_TRADE   = 0.005        # 0.5% of equity per trade
MAX_TRADES_DAY   = 6
MAX_HOLD_BARS    = 12           # 60 min on 5-min bars
STOP_K           = 1.5          # stop = 1.5 * ATR
TARGET_K         = 2.5          # target = 2.5 * ATR
SPREAD_FLOOR_BPS = 0.5          # minimum cost if spread missing

def load_m5() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN / "xauusd_m5.parquet")
    df = df.sort_values("dt_et").reset_index(drop=True)
    return df

def load_features() -> pd.DataFrame:
    return pd.read_parquet(CLEAN / "xauusd_features_m5.parquet")

# ------------------------------------------------------------------ metrics

def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())

def trade_metrics(trades: pd.DataFrame, equity_curve: np.ndarray) -> dict:
    if len(trades) == 0:
        return {"trades": 0}
    pnl = trades["pnl"].values
    rmult = trades["r"].values
    wins = pnl > 0
    returns = equity_curve[1:] / equity_curve[:-1] - 1
    sharpe = (np.nanmean(returns) / (np.nanstd(returns) + 1e-12)) * np.sqrt(252 * 78)  # 78 bars/day
    return {
        "trades":      int(len(trades)),
        "hit_rate":    float(wins.mean()),
        "avg_R":       float(rmult.mean()),
        "median_R":    float(np.median(rmult)),
        "total_pnl":   float(pnl.sum()),
        "total_ret":   float(equity_curve[-1] / equity_curve[0] - 1),
        "max_dd":      float(_max_drawdown(equity_curve)),
        "sharpe_ann":  float(sharpe),
        "final_equity": float(equity_curve[-1]),
    }

def pretty(m: dict) -> str:
    if m.get("trades", 0) == 0:
        return "  no trades"
    return (f"  trades={m['trades']:>4d}  hit={m['hit_rate']*100:5.1f}%  avgR={m['avg_R']:+.2f}  "
            f"ret={m['total_ret']*100:+6.2f}%  maxDD={m['max_dd']*100:+5.2f}%  Sharpe={m['sharpe_ann']:+.2f}  "
            f"endE=${m['final_equity']:,.0f}")

# --------------------------------------------------------------- backtester

@dataclass
class BtCfg:
    risk_per_trade: float = RISK_PER_TRADE
    stop_k:         float = STOP_K
    target_k:       float = TARGET_K
    max_hold:       int   = MAX_HOLD_BARS
    max_lev:        float = MAX_LEV_USED
    max_trades_day: int   = MAX_TRADES_DAY
    daily_soft:     float = DAILY_DD_SOFT
    total_soft:     float = TOTAL_DD_SOFT
    spread_mult:    float = 1.0          # multiplier on spread_close cost
    entry_thresh:   float = 0.5          # |signal| > this to enter

def run_backtest(df: pd.DataFrame, signal: np.ndarray, atr: np.ndarray,
                 cfg: BtCfg = BtCfg(), equity0: float = CAPITAL_0) -> tuple[pd.DataFrame, np.ndarray]:
    """
    df must have columns: dt_et, high, low, close, spread_close (or mid_*). We use mid.
    signal[t] : desired direction/strength at bar t, entry at bar t+1 open (= mid_close[t]).
    atr[t]    : stop distance proxy in $/oz at bar t.
    Returns (trades_df, equity_curve per bar).
    """
    n = len(df)
    mid_h = df["high"].to_numpy(dtype=np.float64)
    mid_l = df["low"].to_numpy(dtype=np.float64)
    mid_c = df["close"].to_numpy(dtype=np.float64)
    if "spread_close" in df.columns:
        spread = df["spread_close"].to_numpy(dtype=np.float64)
    elif {"ask_close","bid_close"}.issubset(df.columns):
        spread = (df["ask_close"] - df["bid_close"]).to_numpy(dtype=np.float64)
    elif "spread_median" in df.columns:
        spread = df["spread_median"].to_numpy(dtype=np.float64)
    else:
        spread = np.full(n, 0.6)
    med = np.nanmedian(spread)
    spread = np.where(np.isnan(spread), med, spread)
    spread = np.where(spread <= 0, med, spread)
    dates  = pd.to_datetime(df["dt_et"]).dt.date.to_numpy()

    equity = np.full(n + 1, equity0, dtype=np.float64)
    eq = equity0
    peak = equity0
    trades = []

    i = 0
    cur_date = None
    day_equity_open = equity0
    day_trades = 0
    halted_day = False
    halted_total = False

    while i < n - 1:
        d = dates[i]
        if d != cur_date:
            cur_date = d
            day_equity_open = eq
            day_trades = 0
            halted_day = False

        can_trade = True
        if halted_total: can_trade = False
        elif halted_day: can_trade = False
        elif eq <= equity0 * (1 - cfg.total_soft):
            halted_total = True; can_trade = False
        elif eq <= day_equity_open * (1 - cfg.daily_soft):
            halted_day = True; can_trade = False
        elif day_trades >= cfg.max_trades_day:
            can_trade = False

        s = signal[i] if can_trade else 0.0
        a = atr[i]
        take_trade = (can_trade and np.isfinite(s) and np.isfinite(a)
                      and abs(s) >= cfg.entry_thresh and a > 0)

        if not take_trade:
            equity[i + 1] = eq
            i += 1
            continue

        direction = 1.0 if s > 0 else -1.0
        entry = mid_c[i]
        stop  = entry - direction * cfg.stop_k   * a
        tgt   = entry + direction * cfg.target_k * a
        risk_per_oz = cfg.stop_k * a

        size_oz = (cfg.risk_per_trade * eq) / risk_per_oz
        notional = size_oz * entry
        max_notional = cfg.max_lev * eq
        if notional > max_notional:
            size_oz *= (max_notional / notional)

        exit_price = None
        exit_bar   = i
        reason = "hold_end"
        for k in range(1, cfg.max_hold + 1):
            if i + k >= n: break
            hi = mid_h[i + k]; lo = mid_l[i + k]
            if direction > 0:
                if lo <= stop: exit_price = stop; reason = "stop";   exit_bar = i + k; break
                if hi >= tgt:  exit_price = tgt;  reason = "target"; exit_bar = i + k; break
            else:
                if hi >= stop: exit_price = stop; reason = "stop";   exit_bar = i + k; break
                if lo <= tgt:  exit_price = tgt;  reason = "target"; exit_bar = i + k; break
        if exit_price is None:
            exit_bar = min(i + cfg.max_hold, n - 1)
            exit_price = mid_c[exit_bar]

        cost_per_oz = cfg.spread_mult * spread[i]
        gross = size_oz * direction * (exit_price - entry)
        cost  = size_oz * 2 * cost_per_oz
        pnl   = gross - cost
        r     = pnl / (cfg.risk_per_trade * eq)

        trades.append({
            "entry_time": df["dt_et"].iloc[i],
            "exit_time":  df["dt_et"].iloc[exit_bar],
            "dir":        direction,
            "entry":      entry,
            "exit":       exit_price,
            "size_oz":    size_oz,
            "atr":        a,
            "signal":     s,
            "spread":     spread[i],
            "pnl":        pnl,
            "r":          r,
            "reason":     reason,
            "bars_held":  exit_bar - i,
        })

        # linear fill of equity across the held bars, then commit
        if exit_bar > i:
            for j in range(i, exit_bar):
                frac = (j - i + 1) / (exit_bar - i)
                equity[j + 1] = eq + pnl * frac
        eq += pnl
        equity[exit_bar + 1 if exit_bar + 1 <= n else n] = eq
        if eq > peak: peak = eq
        day_trades += 1
        i = exit_bar + 1

    # carry forward any trailing bars
    for k in range(i + 1, n + 1):
        equity[k] = eq
    return pd.DataFrame(trades), equity

# ------------------------------------------------------------- daily aggregation

def daily_pnl(trades: pd.DataFrame) -> pd.Series:
    if len(trades) == 0: return pd.Series(dtype=float)
    t = trades.copy()
    t["day"] = pd.to_datetime(t["entry_time"]).dt.date
    return t.groupby("day")["pnl"].sum()
