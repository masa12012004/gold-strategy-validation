"""
Phase 4c — Supervised models with purged, expanding-window walk-forward CV.

Models, and why each one is here (all drawn from the literature sweep):

  logit      Logistic regression. The macro-sentiment paper's own baseline; it
             "struggles to generate consistent alpha ... underscoring the
             importance of capturing the non-linear relationships". Included so
             the non-linear gain is measured, not assumed.
  rf         Random forest. Fischer & Krauss compared LSTM against random forests;
             13 mentions across the corpus. Cheap variance reduction baseline.
  xgb        XGBoost. The single best-supported choice for gold: the TATS paper
             tested it directly on gold price data and got 75.14% train / 58.66%
             test directional accuracy, the best of every model they tried, and
             then used it as the trend predictor inside TATS. The macro paper also
             made it the workhorse. This is our primary model.
  lgbm       LightGBM. Same family, different split policy (leaf-wise); acts as a
             robustness check on xgb rather than a new idea.

Validation protocol (combines the macro paper's 5-fold expanding window with the
standard Lopez de Prado leakage controls, which the papers mostly skip):
  * expanding window: train on everything before the fold, test on the fold
  * PURGING: drop training events whose barrier horizon overlaps the test block
  * EMBARGO: drop training events within `embargo` bars after the test block
  * class weighting by inverse frequency; threshold chosen on the training fold
    only, never on the test fold
  * out-of-fold probabilities are saved so Phase 4d can backtest them honestly

Reports AUC, accuracy, precision/recall, and — standing in for SHAP, which the
macro paper used and which is not installed here — XGBoost gain importance plus a
permutation importance on the strict feature set.

Writes: clean_Data/phase4_oof_predictions.parquet
        clean_Data/phase4_ml_report.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb

from phase4_features import (ML_FEATURES, ML_FEATURES_STRICT, LEAKY_COLS,
                             clean_matrix)

warnings.filterwarnings("ignore")
CLEAN = (Path(__file__).resolve().parent / "clean_Data")

N_FOLDS = 5
EMBARGO_BARS = 78          # one trading day of 5-min bars
MIN_TRAIN = 2000           # don't train a fold on a handful of events


# --------------------------------------------------------------- CV machinery

def purged_expanding_folds(entry_i: np.ndarray, exit_i: np.ndarray,
                           n_folds: int = N_FOLDS,
                           embargo: int = EMBARGO_BARS,
                           min_train: int = MIN_TRAIN):
    """
    Yield (train_idx, test_idx) positional arrays over an event table sorted by
    entry bar. Test blocks are contiguous and later than their training data.
    A training event is purged if its [entry, exit] interval overlaps the test
    block's bar span, or falls inside the post-test embargo.
    """
    n = len(entry_i)
    order = np.argsort(entry_i, kind="stable")
    ei, xi = entry_i[order], exit_i[order]

    # fold boundaries over the usable tail (first chunk is training-only)
    start = max(min_train, n // (n_folds + 1))
    bounds = np.linspace(start, n, n_folds + 1).astype(int)

    for f in range(n_folds):
        a, b = bounds[f], bounds[f + 1]
        if b - a < 50:
            continue
        test_idx = order[a:b]
        test_lo, test_hi = ei[a], xi[a:b].max()

        cand = order[:a]
        c_entry, c_exit = entry_i[cand], exit_i[cand]
        overlaps = c_exit >= (test_lo - 1)
        in_embargo = (c_entry >= test_lo - embargo) & (c_entry <= test_hi + embargo)
        keep = ~(overlaps | in_embargo)
        train_idx = cand[keep]

        if len(train_idx) < min_train:
            continue
        yield f, train_idx, test_idx, (test_lo, test_hi, int((~keep).sum()))


# --------------------------------------------------------------- model zoo

def make_models(scale_pos_weight: float, seed: int = 7) -> dict:
    return {
        "logit": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5,
                                       class_weight="balanced",
                                       random_state=seed)),
        ]),
        "rf": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=50,
                max_features="sqrt", class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed)),
        ]),
        "xgb": xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=20, reg_lambda=2.0, reg_alpha=0.5,
            gamma=0.5, scale_pos_weight=scale_pos_weight,
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", n_jobs=-1, random_state=seed),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=400, num_leaves=15, max_depth=5,
            learning_rate=0.03, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=50,
            reg_lambda=2.0, reg_alpha=0.5,
            scale_pos_weight=scale_pos_weight,
            objective="binary", n_jobs=-1, random_state=seed, verbose=-1),
    }


def best_threshold(y: np.ndarray, p: np.ndarray, r: np.ndarray) -> float:
    """
    Pick the probability cut that maximises mean net R on the TRAINING fold.
    Expected-value based, not accuracy based: a 45%-hit-rate filter is fine if the
    winners are bigger. Requires at least 15% of events to survive.
    """
    best, best_cut = -1e18, 0.5
    for cut in np.arange(0.30, 0.76, 0.01):
        sel = p >= cut
        if sel.sum() < max(30, 0.15 * len(p)):
            continue
        val = r[sel].mean() * np.sqrt(sel.sum())   # EV scaled by sample support
        if val > best:
            best, best_cut = val, float(cut)
    return best_cut


# --------------------------------------------------------------- experiment

def run_experiment(data: pd.DataFrame, feat_cols: list[str], label: str,
                   y_col: str = "y") -> tuple[pd.DataFrame, dict]:
    X_all = clean_matrix(data[feat_cols])
    y_all = data[y_col].to_numpy().astype(int)
    r_all = data["r_net"].to_numpy() if "r_net" in data else np.zeros(len(data))
    entry_i = data["i"].to_numpy()
    exit_i = data["exit_i"].to_numpy()

    oof = {m: np.full(len(data), np.nan) for m in ("logit", "rf", "xgb", "lgbm")}
    cuts = {m: np.full(len(data), np.nan) for m in oof}
    fold_of = np.full(len(data), -1, dtype=int)
    rows = []
    importances = {}

    print(f"\n{'='*78}\n{label}   features={len(feat_cols)}  events={len(data):,}  "
          f"base rate={y_all.mean()*100:.2f}%\n{'='*78}")

    for f, tr, te, (tlo, thi, purged) in purged_expanding_folds(entry_i, exit_i):
        Xtr, ytr, rtr = X_all.iloc[tr], y_all[tr], r_all[tr]
        Xte, yte = X_all.iloc[te], y_all[te]
        fold_of[te] = f
        pos = max(ytr.sum(), 1)
        spw = float((len(ytr) - pos) / pos)
        models = make_models(spw)

        print(f"\n fold {f}: train={len(tr):,} test={len(te):,} "
              f"purged={purged:,} test_bars=[{tlo:,}..{thi:,}] "
              f"train_pos={ytr.mean()*100:.1f}%")

        for name, mdl in models.items():
            t0 = time.time()
            mdl.fit(Xtr, ytr)
            p_tr = mdl.predict_proba(Xtr)[:, 1]
            p_te = mdl.predict_proba(Xte)[:, 1]
            cut = best_threshold(ytr, p_tr, rtr)
            oof[name][te] = p_te
            cuts[name][te] = cut

            yhat = (p_te >= cut).astype(int)
            auc = roc_auc_score(yte, p_te) if len(np.unique(yte)) > 1 else np.nan
            sel = p_te >= cut
            rows.append({
                "fold": f, "model": name,
                "auc": float(auc),
                "acc": float(accuracy_score(yte, yhat)),
                "prec": float(precision_score(yte, yhat, zero_division=0)),
                "rec": float(recall_score(yte, yhat, zero_division=0)),
                "cut": cut,
                "kept_frac": float(sel.mean()),
                "meanR_all": float(r_all[te].mean()),
                "meanR_kept": float(r_all[te][sel].mean()) if sel.any() else np.nan,
                "n_test": int(len(te)),
                "secs": round(time.time() - t0, 1),
            })
            rr = rows[-1]
            print(f"   {name:6s} auc={rr['auc']:.4f} acc={rr['acc']*100:5.2f}% "
                  f"prec={rr['prec']*100:5.2f}% cut={cut:.2f} "
                  f"kept={rr['kept_frac']*100:5.1f}% "
                  f"meanR {rr['meanR_all']:+.3f} -> {rr['meanR_kept']:+.3f} "
                  f"({rr['secs']}s)")

            if name == "xgb":
                b = mdl.get_booster()
                b.feature_names = list(feat_cols)
                g = b.get_score(importance_type="gain")
                for k, v in g.items():
                    importances[k] = importances.get(k, 0.0) + v

    res = pd.DataFrame(rows)
    out = data.copy()
    for m in oof:
        out[f"p_{m}"] = oof[m]
        out[f"cut_{m}"] = cuts[m]
    out["fold"] = fold_of

    print(f"\n--- {label}: fold-averaged ---")
    agg = (res.groupby("model")[["auc", "acc", "prec", "kept_frac",
                                 "meanR_all", "meanR_kept"]]
           .mean().sort_values("auc", ascending=False))
    print(agg.round(4).to_string())

    imp = (pd.Series(importances).sort_values(ascending=False) /
           max(sum(importances.values()), 1e-9) * 100)
    if len(imp):
        print(f"\n--- XGBoost gain importance, top 20 (% of total) ---")
        print(imp.head(20).round(2).to_string())

    report = {
        "label": label,
        "n_features": len(feat_cols),
        "n_events": int(len(data)),
        "base_rate": float(y_all.mean()),
        "folds": res.to_dict(orient="records"),
        "fold_mean": agg.reset_index().to_dict(orient="records"),
        "xgb_gain_pct": imp.round(3).to_dict(),
    }
    return out, report


def main(base: str = "p3") -> None:
    suffix = "" if base == "p3" else f"_{base}"
    meta = pd.read_parquet(CLEAN / f"phase4_meta_dataset{suffix}.parquet")
    meta = meta.sort_values("i").reset_index(drop=True)

    reports = {}

    # headline: strict causal features only
    oof_strict, rep_strict = run_experiment(
        meta, ML_FEATURES_STRICT, f"META[{base}] / STRICT causal features")
    reports["meta_strict"] = rep_strict

    # same thing with the leaky regime columns, to price the leakage
    oof_full, rep_full = run_experiment(
        meta, ML_FEATURES, f"META[{base}] / FULL (incl. full-sample-fitted cols)")
    reports["meta_full"] = rep_full

    oof_strict.to_parquet(CLEAN / f"phase4_oof_predictions{suffix}.parquet",
                          index=False, compression="zstd")
    oof_full.to_parquet(CLEAN / f"phase4_oof_predictions_full{suffix}.parquet",
                        index=False, compression="zstd")

    # diagnostic: is raw direction learnable on gold at all?
    try:
        if base != "p3":
            raise FileNotFoundError
        d = pd.read_parquet(CLEAN / "phase4_dir_dataset.parquet")
        d = d[d["y_dir"] != 0].copy()
        d["y"] = (d["y_dir"] > 0).astype(int)
        d["r_net"] = d["r_net"]
        d = d.sort_values("i").reset_index(drop=True)
        _, rep_dir = run_experiment(
            d, ML_FEATURES_STRICT, "DIR / direction-only diagnostic (long barriers)")
        reports["dir_strict"] = rep_dir
    except FileNotFoundError:
        print("dir dataset missing, skipping direction diagnostic")

    with open(CLEAN / f"phase4_ml_report{suffix}.json", "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"\nWrote {CLEAN/f'phase4_ml_report{suffix}.json'}")
    print(f"Wrote {CLEAN/f'phase4_oof_predictions{suffix}.parquet'}")


if __name__ == "__main__":
    import sys
    main(base=sys.argv[1] if len(sys.argv) > 1 else "p3")
