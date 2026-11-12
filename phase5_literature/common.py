"""
Phase 5 — shared engine for the literature replication study.

Conventions (identical for every strategy so the equity curves are comparable):
  * pos[t] is decided at the CLOSE of bar t using information <= t only.
  * pos[t] earns ret[t+1] = mid_close[t+1] / mid_close[t] - 1.
  * Cost: |pos[t] - pos[t-1]| * half_spread_frac[t]   (crossing half the real
    bid/ask spread on every unit of turnover, i.e. a full spread per round trip)
    plus EXTRA_SLIP_BPS per unit turnover.
  * Walk-forward: everything is fitted on data strictly before the OOS bar it
    is used on. OOS evaluation window is fixed for every strategy.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(ROOT), "clean_Data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

OOS_START = pd.Timestamp("2025-07-01", tz="US/Eastern")
OOS_END = pd.Timestamp("2026-10-31", tz="US/Eastern")
EXTRA_SLIP_BPS = 0.2
ANN = 252


# --------------------------------------------------------------------------- data
def _finish(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["dt"] = pd.to_datetime(d["dt_et"])
    d = d.set_index("dt").sort_index()
    d["mid"] = d["close"]
    d["ret"] = d["mid"].pct_change().fillna(0.0)
    d["lret"] = np.log(d["mid"]).diff().fillna(0.0)
    sp = d["spread_median"] if "spread_median" in d else (d["ask_close"] - d["bid_close"])
    d["hs"] = (sp / d["mid"] / 2.0).clip(upper=10e-4)  # half spread as fraction, cap 10bp
    # trading day ends 17:00 ET -> shift by 7h so the day label is the session date
    d["tday"] = (d.index + pd.Timedelta(hours=7)).normalize().tz_localize(None)
    return d


def load_m5() -> pd.DataFrame:
    return _finish(pd.read_parquet(os.path.join(DATA, "xauusd_m5.parquet")))


def load_h1() -> pd.DataFrame:
    return _finish(pd.read_parquet(os.path.join(DATA, "xauusd_h1.parquet")))


def load_m1() -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(DATA, "xauusd_m1.parquet"),
                        columns=["dt_et", "open", "high", "low", "close", "spread_median",
                                 "bid_ticks", "ask_ticks"])
    return _finish(d)


def load_daily() -> pd.DataFrame:
    m5 = load_m5()
    g = m5.groupby("tday")
    d = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "hs": g["hs"].median(), "nbars": g.size(),
    })
    d = d[d["nbars"] > 100]  # drop stub sessions
    d.index = pd.DatetimeIndex(d.index, name="dt").tz_localize("US/Eastern") + pd.Timedelta(hours=16, minutes=55)
    d["mid"] = d["close"]
    d["ret"] = d["mid"].pct_change().fillna(0.0)
    d["lret"] = np.log(d["mid"]).diff().fillna(0.0)
    d["tday"] = d.index.tz_localize(None).normalize()
    return d


# --------------------------------------------------------------------------- backtest
def backtest(df: pd.DataFrame, pos, name: str, save: bool = True) -> pd.DataFrame:
    """Vectorised next-bar backtest on any bar frequency. Returns bar-level frame."""
    pos = pd.Series(np.asarray(pos, dtype=float), index=df.index).fillna(0.0)
    fwd = df["ret"].shift(-1).fillna(0.0)
    gross = pos * fwd
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * (df["hs"] + EXTRA_SLIP_BPS * 1e-4)
    out = pd.DataFrame({"pos": pos, "gross": gross, "cost": cost, "net": gross - cost,
                        "turn": turn, "tday": df["tday"]}, index=df.index)
    out = out[(out.index >= OOS_START) & (out.index < OOS_END)]
    if save:
        out.to_parquet(os.path.join(RES, f"{name}.parquet"))
    return out


def daily_returns(bt: pd.DataFrame, col: str = "net") -> pd.Series:
    return bt.groupby("tday")[col].sum()


def metrics(bt: pd.DataFrame | pd.Series, name: str = "") -> dict:
    if isinstance(bt, pd.DataFrame):
        r = daily_returns(bt)
        g = daily_returns(bt, "gross")
        turn_day = bt.groupby("tday")["turn"].sum().mean()
        expo = bt["pos"].abs().mean()
    else:
        r, g, turn_day, expo = bt, bt, np.nan, np.nan
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(r) / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    sd = r.std()
    sh = r.mean() / sd * np.sqrt(ANN) if sd > 0 else 0.0
    shg = g.mean() / g.std() * np.sqrt(ANN) if g.std() > 0 else 0.0
    lo, hi = block_bootstrap_sharpe(r.values)
    return {"strategy": name, "total_ret": eq.iloc[-1] - 1, "cagr": cagr, "ann_vol": sd * np.sqrt(ANN),
            "sharpe": sh, "sharpe_ci_lo": lo, "sharpe_ci_hi": hi, "sharpe_gross": shg,
            "max_dd": dd, "turnover_per_day": turn_day, "avg_abs_pos": expo, "days": len(r)}


def block_bootstrap_sharpe(r: np.ndarray, block: int = 10, n: int = 2000, seed: int = 0):
    """Moving-block bootstrap 90% CI for the annualised Sharpe (daily returns)."""
    r = np.asarray(r, float)
    T = len(r)
    if T < 3 * block or r.std() == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    s = r[idx]
    sh = s.mean(1) / s.std(1) * np.sqrt(ANN)
    return float(np.percentile(sh, 5)), float(np.percentile(sh, 95))


# --------------------------------------------------------------------------- helpers
def ewm_vol(r: pd.Series, span: int) -> pd.Series:
    return r.ewm(span=span, min_periods=max(5, span // 3)).std()


def vol_target(signal: pd.Series, r: pd.Series, bars_per_year: float, target: float = 0.15,
               span: int = 60, cap: float = 2.0) -> pd.Series:
    """Paper 11 / 38 / 39 volatility targeting: w = signal * target / sigma_hat, capped."""
    sig_ann = ewm_vol(r, span) * np.sqrt(bars_per_year)
    w = signal * (target / sig_ann)
    return w.clip(-cap, cap).fillna(0.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(1)
    return tr.rolling(n, min_periods=n).mean()


def monthly_refit_points(index: pd.DatetimeIndex, start=OOS_START, end=OOS_END):
    """First bar of each calendar month inside the OOS window (refit dates)."""
    idx = index[(index >= start) & (index < end)]
    months = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first()
    return [pd.Timestamp(x) for x in months.tolist()]


def bars_per_year(df: pd.DataFrame) -> float:
    days = df["tday"].nunique()
    return len(df) / days * ANN


def save_metrics(rows: list[dict], fname: str):
    pd.DataFrame(rows).to_csv(os.path.join(RES, fname), index=False)
