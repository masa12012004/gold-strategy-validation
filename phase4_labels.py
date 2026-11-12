"""
Phase 4a — Supervised labels for the ML layer.

Design follows the TATS paper (Mehrabian et al., gold-specific): a *classical value
forecaster* is paired with a *learned trend/quality predictor*. Our classical
forecaster is the Argos ensemble (Phase 3); the learned part is a classifier that
decides whether an Argos entry is worth taking. That is meta-labelling in the
Lopez de Prado sense, and it is the structure the gold paper found actually works
(XGBoost trend predictor: 58.66% test accuracy vs ~50% for LSTM/Bi-LSTM).

Two label sets are produced:

  META  (primary use) — only at bars where Argos fires. y=1 if the trade that the
        backtester WOULD have taken ends profitable (target before stop, net of
        spread), y=0 otherwise. Direction comes from Argos; the model only learns
        "is this a good trade", never the direction. This is the lower-variance
        problem and it cannot flip a correct signal into a wrong one.

  DIR   (diagnostic) — triple-barrier direction label on *every* bar, so we can
        reproduce the papers' "directional accuracy" numbers and check whether a
        model can predict gold direction at all (the gold literature says barely).

Barrier logic is copied from phase2_common.run_backtest bar-for-bar (same stop_k,
target_k, max_hold, same intrabar high/low touch order, same 2x spread cost) so a
label of 1 means exactly "this trade made money in the live engine".

Writes: clean_Data/phase4_meta_dataset.parquet
        clean_Data/phase4_dir_dataset.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_common import BtCfg
from phase3_strategy import ensemble3, friction_kelly_sizing

CLEAN = (Path(__file__).resolve().parent / "clean_Data")


# --------------------------------------------------------------- barrier engine

def _spread_vec(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    if "spread_close" in df.columns:
        sp = df["spread_close"].to_numpy(dtype=np.float64)
    elif {"ask_close", "bid_close"}.issubset(df.columns):
        sp = (df["ask_close"] - df["bid_close"]).to_numpy(dtype=np.float64)
    elif "spread_median" in df.columns:
        sp = df["spread_median"].to_numpy(dtype=np.float64)
    else:
        sp = np.full(n, 0.6)
    med = np.nanmedian(sp)
    sp = np.where(~np.isfinite(sp) | (sp <= 0), med, sp)
    return sp


def triple_barrier(df: pd.DataFrame, idx: np.ndarray, dirs: np.ndarray,
                   atr: np.ndarray, cfg: BtCfg) -> pd.DataFrame:
    """
    Replicates run_backtest's exit logic for a set of candidate entry bars.
    Entry at close of bar i (same as the engine). Returns one row per event with
    net R-multiple, win flag, exit reason and bars held.
    """
    hi = df["high"].to_numpy(dtype=np.float64)
    lo = df["low"].to_numpy(dtype=np.float64)
    cl = df["close"].to_numpy(dtype=np.float64)
    spread = _spread_vec(df)
    n = len(df)

    out = {k: [] for k in ("i", "dir", "r_net", "y", "reason", "bars_held",
                           "exit_i", "mfe_r", "mae_r")}

    for i, direction in zip(idx, dirs):
        a = atr[i]
        if not (np.isfinite(a) and a > 0):
            continue
        entry = cl[i]
        stop = entry - direction * cfg.stop_k * a
        tgt = entry + direction * cfg.target_k * a
        risk_per_oz = cfg.stop_k * a

        exit_price, exit_bar, reason = None, i, "hold_end"
        run_max_fav, run_max_adv = 0.0, 0.0
        for k in range(1, cfg.max_hold + 1):
            if i + k >= n:
                break
            h, l = hi[i + k], lo[i + k]
            # track excursions in R units before resolving barriers
            fav = direction * ((h if direction > 0 else l) - entry) / risk_per_oz
            adv = direction * ((l if direction > 0 else h) - entry) / risk_per_oz
            run_max_fav = max(run_max_fav, fav)
            run_max_adv = min(run_max_adv, adv)
            if direction > 0:
                if l <= stop:
                    exit_price, reason, exit_bar = stop, "stop", i + k; break
                if h >= tgt:
                    exit_price, reason, exit_bar = tgt, "target", i + k; break
            else:
                if h >= stop:
                    exit_price, reason, exit_bar = stop, "stop", i + k; break
                if l <= tgt:
                    exit_price, reason, exit_bar = tgt, "target", i + k; break
        if exit_price is None:
            exit_bar = min(i + cfg.max_hold, n - 1)
            exit_price = cl[exit_bar]

        # net R: gross move minus round-trip spread, both in units of risk
        gross_oz = direction * (exit_price - entry)
        cost_oz = 2.0 * cfg.spread_mult * spread[i]
        r_net = (gross_oz - cost_oz) / risk_per_oz

        out["i"].append(int(i))
        out["dir"].append(float(direction))
        out["r_net"].append(float(r_net))
        out["y"].append(int(r_net > 0))
        out["reason"].append(reason)
        out["bars_held"].append(int(exit_bar - i))
        out["exit_i"].append(int(exit_bar))
        out["mfe_r"].append(float(run_max_fav))
        out["mae_r"].append(float(run_max_adv))

    return pd.DataFrame(out)


# --------------------------------------------------------------- dataset build

def build_meta(df: pd.DataFrame, weights: dict, params: dict, cfg: BtCfg,
               base: str = "p3") -> pd.DataFrame:
    """
    Meta-label dataset: one row per candidate entry of the base classical strategy.
    base="p3" uses the Phase-3 ensemble with friction-Kelly sizing; base="p2" uses
    the Phase-2 ensemble with plain ATR sizing, matching how each was backtested.
    """
    if base == "p2":
        from phase2c_strategy import ensemble as ensemble2
        s = ensemble2(df, weights, params)
        size_scale = np.ones(len(df))
        atr_eff = df["atr14"].to_numpy()
    else:
        s = ensemble3(df, weights, params)
        size_scale = friction_kelly_sizing(df, s)
        atr_eff = df["atr14"].to_numpy() / size_scale

    fired = np.where(np.isfinite(s) & (np.abs(s) >= cfg.entry_thresh))[0]
    # the engine never enters on the last bar
    fired = fired[fired < len(df) - 1]
    dirs = np.sign(s[fired])

    lab = triple_barrier(df, fired, dirs, atr_eff, cfg)
    lab["argos_signal"] = s[lab["i"].to_numpy()]
    lab["argos_abs"] = np.abs(lab["argos_signal"])
    lab["size_scale"] = size_scale[lab["i"].to_numpy()]
    lab["atr_eff"] = atr_eff[lab["i"].to_numpy()]
    return lab


def build_dir(df: pd.DataFrame, cfg: BtCfg, stride: int = 1) -> pd.DataFrame:
    """
    Diagnostic direction dataset on every stride-th bar. Label = which barrier the
    *long* side hits first: +1 target, -1 stop, 0 timeout. Used only to measure
    whether direction is learnable at all.
    """
    atr = df["atr14"].to_numpy()
    idx = np.arange(0, len(df) - 1, stride)
    lab = triple_barrier(df, idx, np.ones(len(idx)), atr, cfg)
    m = {"target": 1, "stop": -1, "hold_end": 0}
    lab["y_dir"] = lab["reason"].map(m).astype(int)
    return lab


def attach_features(df: pd.DataFrame, lab: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Join feature values as of the entry bar (known at bar close -> no leakage)."""
    feat = df.iloc[lab["i"].to_numpy()][cols].reset_index(drop=True)
    meta = lab.reset_index(drop=True)
    out = pd.concat([meta, feat], axis=1)
    out["dt_et"] = df["dt_et"].to_numpy()[lab["i"].to_numpy()]
    return out


def main(base: str = "p3") -> None:
    pfile = "phase2_best_params.json" if base == "p2" else "phase3_best_params.json"
    with open(CLEAN / pfile) as f:
        bestp = json.load(f)
    weights = {k: float(v) for k, v in bestp["weights"].items()}
    params = bestp["params"]
    print(f"BASE = {base}   weights={weights}   params={params}")

    df = (pd.read_parquet(CLEAN / "xauusd_features3_m5.parquet")
          .sort_values("dt_et").reset_index(drop=True))
    df = df.dropna(subset=["atr14", "close"]).reset_index(drop=True)
    print(f"Bars: {len(df):,}   {df['dt_et'].min()} -> {df['dt_et'].max()}")

    print("Adding causal derived features ...")
    from phase4_features import add_derived
    df = add_derived(df)

    cfg = BtCfg(entry_thresh=0.25, risk_per_trade=0.005, stop_k=1.8, target_k=2.8,
                max_trades_day=8, daily_soft=0.03, total_soft=0.07)

    if base == "p2":
        # Phase 2 was backtested at entry_thresh 0.35 with plain risk sizing
        cfg = BtCfg(entry_thresh=0.35, risk_per_trade=0.005)
    # relax entry_thresh so the classifier sees a wider candidate pool than the
    # engine's own cut (it can always re-tighten).
    cfg_wide = BtCfg(**{**cfg.__dict__, "entry_thresh": 0.10})

    from phase4_features import ML_FEATURES

    suffix = "" if base == "p3" else f"_{base}"
    print("\nBuilding META dataset (candidates at relaxed threshold) ...")
    meta = build_meta(df, weights, params, cfg_wide, base=base)
    meta = attach_features(df, meta, ML_FEATURES)
    meta.to_parquet(CLEAN / f"phase4_meta_dataset{suffix}.parquet",
                    index=False, compression="zstd")
    print(f"  events={len(meta):,}  base win-rate={meta['y'].mean()*100:.2f}%  "
          f"mean R={meta['r_net'].mean():+.3f}")
    print(f"  exits: {meta['reason'].value_counts().to_dict()}")
    print(f"  long={int((meta['dir']>0).sum()):,}  short={int((meta['dir']<0).sum()):,}")

    if base == "p3":
        print("\nBuilding DIR dataset (every 3rd bar, diagnostic) ...")
        dirs = build_dir(df, cfg, stride=3)
        dirs = attach_features(df, dirs, ML_FEATURES)
        dirs.to_parquet(CLEAN / "phase4_dir_dataset.parquet",
                        index=False, compression="zstd")
        vc = dirs["y_dir"].value_counts().to_dict()
        print(f"  rows={len(dirs):,}  label mix (+1/-1/0)={vc}")

    print(f"\nWrote {CLEAN/f'phase4_meta_dataset{suffix}.parquet'}")


if __name__ == "__main__":
    import sys
    main(base=sys.argv[1] if len(sys.argv) > 1 else "p3")
