"""
Hybrids. Every rule below is fixed a priori from the papers or uses only past data.

  fix   s03_hmm_sideinfo : recombined with EXPANDING (past-only) forecast std (the first
                           version scaled by the full-sample std — a small look-ahead).
  H1  slow momentum x fast signal (papers 11 + 46 + 39): the H1 cubic trend/reversion
      position is taken only when it agrees with the daily Forecast-to-Fill regime.
  H2  TATS (paper 45): value forecaster = cubic law, trend predictor = XGBoost.
      Agree -> full size in that direction; conflict -> follow the trend predictor at
      delta = 0.25 size.
  H3  equal-risk ensemble of ALL thirteen base strategies (no selection whatsoever).
  H4  walk-forward ensemble (paper 38 volatility control): each month, weight_i is
      proportional to max(trailing Sharpe_i, 0)/vol_i using only realised OOS returns up to
      the previous day (equal-risk for the first 60 days), then scaled to 10% target vol
      with a trailing 20-day estimate, leverage cap 2.
  H5  volatility-controlled long gold (paper 38's simplest improvement on buy & hold).
"""
import numpy as np
import pandas as pd
from common import (load_h1, load_m5, load_daily, backtest, metrics, save_metrics, vol_target,
                    bars_per_year, RES, ANN)

BASE = ["s01_f2f_long_only", "s01_f2f_long_short", "s02_cubic_trend_reversion", "s02_tsmom",
        "s03_hmm_momentum", "s03_hmm_sideinfo", "s04_drift_burst_fade", "s05_hurst_switch",
        "s05_q_learning", "s06_ou_optimal_stopping", "s07_xgb_trend_predictor",
        "s08_dmn_lstm_cpd", "s09_rl_positional"]
DAILY_BAR = {"s01_f2f_long_only", "s01_f2f_long_short", "s00_buy_hold_daily", "h5_vol_controlled_gold"}


def daily(name):
    b = pd.read_parquet(f"{RES}/{name}.parquet")
    r = b.groupby("tday")["net"].sum()
    if name in DAILY_BAR:          # pnl of a daily bar is earned on the following session
        r = r.shift(1).fillna(0.0)
    return r


def full_pos(name, index):
    return pd.read_parquet(f"{RES}/{name}.parquet")["pos"].reindex(index).fillna(0.0)


if __name__ == "__main__":
    rows = []
    # ---- fix: HMM + side information with past-only scaling
    m5 = load_m5()
    fc = pd.read_parquet(f"{RES}/_hmm_forecasts.parquet").reindex(m5.index)
    sh = fc["hmm_fc"].expanding(2000).std().shift(1)
    ss = fc["side_fc"].expanding(2000).std().shift(1)
    comb = (fc["hmm_fc"] / sh + fc["side_fc"] / ss) * sh
    thr = (m5["hs"] * 1e4).values
    cv = comb.values
    pos = np.zeros(len(m5)); cur = 0.0
    for t in range(len(m5)):
        if not np.isfinite(cv[t]):
            cur = 0.0
        elif abs(cv[t]) > thr[t]:
            cur = np.sign(cv[t])
        pos[t] = cur
    rows.append(metrics(backtest(m5, pos, "s03_hmm_sideinfo"), "s03_hmm_sideinfo (past-only scaling)"))

    # ---- H1 / H2 on H1 bars
    h1 = load_h1()
    bpy = bars_per_year(h1)
    cubic = full_pos("s02_cubic_trend_reversion", h1.index)
    dly = load_daily()
    f2f = full_pos("s01_f2f_long_short", dly.index)
    key = pd.DataFrame({"t": (h1.index + pd.Timedelta(hours=1)).tz_convert("UTC")}, index=h1.index).reset_index()
    dk = pd.DataFrame({"t": (dly.index + pd.Timedelta(minutes=5)).tz_convert("UTC"), "f2f": f2f.values}).sort_values("t")
    regime = pd.merge_asof(key.sort_values("t"), dk, on="t").set_index("dt")["f2f"].reindex(h1.index).fillna(0.0)
    agree = np.sign(cubic) == np.sign(regime)
    p_h1 = cubic.where(agree & (regime != 0), 0.0)
    rows.append(metrics(backtest(h1, p_h1, "h1_slowmom_x_cubic"), "H1 slow-momentum regime x cubic"))

    probs = pd.read_parquet(f"{RES}/_tats_probs.parquet").reindex(h1.index)
    tp = np.sign(probs["p_xgb"] - 0.5).fillna(0.0)
    vf = np.sign(cubic)
    sig = pd.Series(np.where(tp == vf, tp, 0.25 * tp), index=h1.index)
    p_h2 = vol_target(sig, h1["ret"], bpy, 0.15, span=24 * 20)
    rows.append(metrics(backtest(h1, p_h2, "h2_tats_cubic_xgb"), "H2 TATS (cubic VF + XGB TP)"))

    # ---- H5 vol-controlled long gold (daily)
    p_h5 = vol_target(pd.Series(1.0, index=dly.index), dly["ret"], ANN, 0.15, span=20, cap=1.5)
    rows.append(metrics(backtest(dly, p_h5, "h5_vol_controlled_gold"), "H5 vol-controlled long gold"))

    # ---- ensembles on daily net returns
    R = pd.DataFrame({n: daily(n) for n in BASE}).fillna(0.0)
    R.to_parquet(f"{RES}/_daily_matrix.parquet")
    vol = R.rolling(60, min_periods=20).std().shift(1)
    w_eq = (1 / vol).replace([np.inf, -np.inf], np.nan)
    w_eq = w_eq.div(w_eq.sum(1), axis=0).fillna(0.0)

    def vol_control(r, tgt=0.10, win=20, cap=2.0):
        s = r.rolling(win, min_periods=10).std().shift(1) * np.sqrt(ANN)
        lev = (tgt / s).clip(upper=cap).fillna(0.0)
        return r * lev

    r_h3 = vol_control((w_eq * R).sum(1))
    r_h3.to_frame("net").to_parquet(f"{RES}/h3_equal_risk_all.parquet")
    rows.append(metrics(r_h3, "H3 equal-risk ensemble, all 13"))

    month = pd.Series(R.index.to_period("M"), index=R.index)
    W = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    wlog = {}
    for m in month.unique():
        days = R.index[month == m]
        hist = R[R.index < days[0]]
        if len(hist) < 60:
            w = w_eq.loc[days[0]]
        else:
            sr = hist.mean() / hist.std().replace(0, np.nan)
            w = (sr.clip(lower=0) / hist.std()).fillna(0.0)
            w = w / w.sum() if w.sum() > 0 else w_eq.loc[days[0]]
        W.loc[days] = w.values
        wlog[str(m)] = w.round(2)
    r_h4 = vol_control((W * R).sum(1))
    r_h4.to_frame("net").to_parquet(f"{RES}/h4_walkforward_ensemble.parquet")
    rows.append(metrics(r_h4, "H4 walk-forward Sharpe-weighted ensemble"))
    wl = pd.DataFrame(wlog).T
    wl.to_csv(f"{RES}/h4_weights.csv")
    print("H4 monthly weights (non-zero columns):")
    print(wl.loc[:, (wl > 0.02).any()].iloc[::3].to_string())
    print(pd.DataFrame(rows).round(3).to_string())
    save_metrics(rows, "m_s10.csv")
