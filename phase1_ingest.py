"""
Phase 1 — mega ingest for XAUUSD research.

Combines three HistData.com NT feeds into a single enriched 1-minute master:
    - DAT_NT_XAUUSD_M1_*.csv                (1-min OHLC, bid-based)
    - HISTDATA_COM_NT_XAUUSD_T_BID*.zip    (bid ticks, ~second precision)
    - HISTDATA_COM_NT_XAUUSD_T_ASK*.zip    (ask ticks, ~second precision)

Note: there is no separate "last" feed for XAUUSD (it is OTC, no exchange tape).
Mid and spread are derived from BID/ASK ticks.

Input : <repository>/input_Data/
Output: <repository>/clean_Data/
          xauusd_m1.parquet       mega 1-min master
          xauusd_m5.parquet       5-min  resample (mid-based)
          xauusd_m15.parquet      15-min resample
          xauusd_h1.parquet       1-hour resample
          ingest_report.txt

Conventions
-----------
- Source timezone is US/Eastern (HistData NT).
- M1 duplicate minutes are aggregated open=first, high=max, low=min, close=last.
- Per-minute bid/ask OHLC come from the tick feeds; spread stats come from
  per-tick bid/ask alignment via merge_asof (latest known ask at each bid tick).
- Resamples use label='left', closed='left'.
"""
from __future__ import annotations

import glob
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR  = (Path(__file__).resolve().parent / "input_Data")
OUTPUT_DIR = (Path(__file__).resolve().parent / "clean_Data")
REPORT     = OUTPUT_DIR / "ingest_report.txt"

SRC_TZ    = "US/Eastern"
M1_COLS   = ["dt", "open", "high", "low", "close", "vol"]
TICK_COLS = ["dt", "price", "vol"]


# --------------------------------------------------------------------------- M1

def load_m1() -> pd.DataFrame:
    files = sorted(glob.glob(str(INPUT_DIR / "DAT_NT_XAUUSD_M1_*.csv")))
    if not files:
        raise FileNotFoundError(f"No M1 CSVs in {INPUT_DIR}")
    frames = [pd.read_csv(f, sep=";", header=None, names=M1_COLS) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
    df = df[df[["open", "high", "low", "close"]].notna().all(axis=1)]
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[df["high"] >= df["low"]]
    agg = (df.groupby("dt", sort=True)
             .agg(open=("open",  "first"),
                  high=("high",  "max"),
                  low =("low",   "min"),
                  close=("close","last"),
                  n   =("open",  "size"))
             .reset_index())
    agg["m1_dup"] = (agg["n"] - 1).astype("int32")
    agg = agg.drop(columns="n").set_index("dt")
    return agg


# ------------------------------------------------------------------------- ticks

def read_tick_zip(path: Path, value_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as fh:
            df = pd.read_csv(fh, sep=";", header=None, names=TICK_COLS, usecols=[0, 1])
    df["dt"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
    df = df.rename(columns={"price": value_name})[["dt", value_name]]
    df[value_name] = df[value_name].astype("float32")
    return df.sort_values("dt", kind="mergesort").reset_index(drop=True)


def month_tick_features(ym: str) -> pd.DataFrame | None:
    bid_zip = INPUT_DIR / f"HISTDATA_COM_NT_XAUUSD_T_BID{ym}.zip"
    ask_zip = INPUT_DIR / f"HISTDATA_COM_NT_XAUUSD_T_ASK{ym}.zip"
    if not (bid_zip.exists() and ask_zip.exists()):
        return None

    bid = read_tick_zip(bid_zip, "bid")
    ask = read_tick_zip(ask_zip, "ask")

    bid_m = bid.set_index("dt")["bid"].resample("1min").ohlc()
    bid_m.columns = [f"bid_{c}" for c in bid_m.columns]
    bid_m["bid_ticks"] = bid.set_index("dt")["bid"].resample("1min").count().astype("int32")

    ask_m = ask.set_index("dt")["ask"].resample("1min").ohlc()
    ask_m.columns = [f"ask_{c}" for c in ask_m.columns]
    ask_m["ask_ticks"] = ask.set_index("dt")["ask"].resample("1min").count().astype("int32")

    aligned = pd.merge_asof(bid, ask, on="dt", direction="backward").dropna(subset=["ask"])
    aligned["spread"] = (aligned["ask"] - aligned["bid"]).astype("float32")
    sp = (aligned.set_index("dt")["spread"].resample("1min")
                 .agg(spread_median="median", spread_mean="mean",
                      spread_min   ="min",    spread_max ="max"))

    out = bid_m.join(ask_m, how="outer").join(sp, how="outer")
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    del bid, ask, aligned
    return out


def load_tick_features() -> pd.DataFrame:
    bid_zips = sorted(glob.glob(str(INPUT_DIR / "HISTDATA_COM_NT_XAUUSD_T_BID*.zip")))
    yms = [re.search(r"BID(\d{6})", p).group(1) for p in bid_zips]
    parts = []
    for ym in yms:
        print(f"  ticks {ym} ...", flush=True)
        f = month_tick_features(ym)
        if f is not None:
            parts.append(f)
    if not parts:
        raise FileNotFoundError("No tick zips found")
    return pd.concat(parts).sort_index()


# ------------------------------------------------------------------------ merge

def enrich(m1: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
    df = m1.join(ticks, how="left")
    df = df.sort_index()

    df["mid_open"]  = (df["bid_open"]  + df["ask_open"])  / 2
    df["mid_high"]  = (df["bid_high"]  + df["ask_high"])  / 2
    df["mid_low"]   = (df["bid_low"]   + df["ask_low"])   / 2
    df["mid_close"] = (df["bid_close"] + df["ask_close"]) / 2

    df["spread_open"]  = (df["ask_open"]  - df["bid_open"]) .astype("float32")
    df["spread_close"] = (df["ask_close"] - df["bid_close"]).astype("float32")
    df["spread_close_bps"] = (df["spread_close"] / df["mid_close"] * 1e4).astype("float32")

    df["dt_et"]  = df.index.tz_localize(SRC_TZ, ambiguous="infer", nonexistent="shift_forward")
    df["dt_utc"] = df["dt_et"].dt.tz_convert("UTC")

    et_hour = df["dt_et"].dt.hour
    session = np.where((et_hour >= 18) | (et_hour < 2), "asia",
              np.where(et_hour < 8,  "london",
              np.where(et_hour < 17, "ny", "off")))
    df["session"]       = pd.Categorical(session, categories=["asia", "london", "ny", "off"])
    df["dow"]           = df["dt_et"].dt.dayofweek.astype("int8")
    df["minute_of_day"] = (df["dt_et"].dt.hour * 60 + df["dt_et"].dt.minute).astype("int16")

    mid_c = df["mid_close"].where(df["mid_close"].notna(), df["close"])
    df["ret"]       = mid_c.pct_change()
    df["log_ret"]   = np.log(mid_c).diff()
    df["range_bps"] = ((df["high"] - df["low"]) / df["close"] * 1e4).astype("float32")

    gap = df.index.to_series().diff().dt.total_seconds().div(60)
    df["gap_min"]     = gap.fillna(0).astype("int32")
    df["new_session"] = (gap > 5).astype("int8")

    df = df.reset_index(names="dt_naive").drop(columns="dt_naive")
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    s = df.set_index("dt_et")
    out = s.resample(rule, label="left", closed="left").agg(
        open =("mid_open",  "first"),
        high =("mid_high",  "max"),
        low  =("mid_low",   "min"),
        close=("mid_close", "last"),
        bid_close    =("bid_close",    "last"),
        ask_close    =("ask_close",    "last"),
        spread_median=("spread_median","mean"),
        spread_max   =("spread_max",   "max"),
        bid_ticks    =("bid_ticks",    "sum"),
        ask_ticks    =("ask_ticks",    "sum"),
        bars         =("mid_open",     "size"),
    ).dropna(subset=["open"]).reset_index()
    out["dt_utc"]    = out["dt_et"].dt.tz_convert("UTC")
    out["ret"]       = out["close"].pct_change()
    out["log_ret"]   = np.log(out["close"]).diff()
    out["range_bps"] = ((out["high"] - out["low"]) / out["close"] * 1e4).astype("float32")
    return out


# ----------------------------------------------------------------------- report

def summarize(df: pd.DataFrame) -> str:
    L = []
    w = L.append
    w("=" * 64)
    w("Phase 1 mega ingest — XAUUSD (M1 + BID ticks + ASK ticks)")
    w("=" * 64)
    w(f"Source timezone  : {SRC_TZ}")
    w(f"Date range       : {df['dt_et'].min()}  ->  {df['dt_et'].max()}")
    w(f"Rows (minutes)   : {len(df):,}")
    w(f"Trading days     : {df['dt_et'].dt.date.nunique()}")
    w("")
    w("--- Coverage of tick features on the M1 grid ---")
    have_bid = df["bid_close"].notna().sum()
    have_ask = df["ask_close"].notna().sum()
    have_sp  = df["spread_median"].notna().sum()
    w(f"  minutes with bid OHLC  : {have_bid:>9,}  ({100*have_bid/len(df):.2f}%)")
    w(f"  minutes with ask OHLC  : {have_ask:>9,}  ({100*have_ask/len(df):.2f}%)")
    w(f"  minutes with spread    : {have_sp:>9,}  ({100*have_sp /len(df):.2f}%)")
    w("")
    w("--- Spread (USD per oz, on bars that have ticks on both sides) ---")
    w(df["spread_median"].dropna().describe().round(4).to_string())
    w(f"\nspread_close_bps (p50 / p95 / p99) : "
      f"{df['spread_close_bps'].quantile(0.50):.2f} / "
      f"{df['spread_close_bps'].quantile(0.95):.2f} / "
      f"{df['spread_close_bps'].quantile(0.99):.2f}  bps")
    w("")
    w("--- Mid close ---")
    w(df["mid_close"].dropna().describe().round(2).to_string())
    w("")
    w("--- 1-min log return (mid) ---")
    w(df["log_ret"].dropna().describe().round(6).to_string())
    w(f"  |ret| > 1%: {(df['ret'].abs() > 0.01).sum()}")
    w(f"  |ret| > 2%: {(df['ret'].abs() > 0.02).sum()}")
    w("")
    w("--- Bars per session ---")
    w(df["session"].value_counts().to_string())
    w("")
    w("--- Tick counts per minute (summary) ---")
    w(pd.DataFrame({"bid_ticks": df["bid_ticks"].describe(),
                    "ask_ticks": df["ask_ticks"].describe()}).round(1).to_string())
    return "\n".join(L)


# -------------------------------------------------------------------------- main

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading M1 ...", flush=True)
    m1 = load_m1()
    print(f"  M1 minutes: {len(m1):,}")

    print("Building per-minute tick features (BID + ASK)", flush=True)
    ticks = load_tick_features()
    print(f"  tick-feature minutes: {len(ticks):,}")

    print("Merging + enriching ...", flush=True)
    mega = enrich(m1, ticks)

    mega.to_parquet(OUTPUT_DIR / "xauusd_m1.parquet",  index=False, compression="zstd")
    resample(mega, "5min" ).to_parquet(OUTPUT_DIR / "xauusd_m5.parquet",  index=False, compression="zstd")
    resample(mega, "15min").to_parquet(OUTPUT_DIR / "xauusd_m15.parquet", index=False, compression="zstd")
    resample(mega, "1h"   ).to_parquet(OUTPUT_DIR / "xauusd_h1.parquet",  index=False, compression="zstd")

    report = summarize(mega)
    REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
