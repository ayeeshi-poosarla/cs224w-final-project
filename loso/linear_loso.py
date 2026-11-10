#!/usr/bin/env python3
"""Linear (Ridge) baseline with leave-one-subject-out (LOSO) evaluation on mcPHASES.

Predicts same-day urinary LH and E3G from the 7-day windows built by the shared
preprocessing (processing/preprocess_cs224w.py). Reads its outputs only:
    processing/output/cs224w_2022_only/graph_inputs.npz
    processing/output/cs224w_2022_only/examples_index.csv
Trains in log(1+y) space (the targets stored in graph_inputs.npz) and reports
participant-averaged MAE / RMSE in original units. Requires numpy, pandas, scikit-learn.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORMONES = (("lh", "mIU/mL"), ("e3g", "ng/mL"))   # column order of y in graph_inputs.npz
ALPHAS = np.logspace(-2, 4, 13)                   # Ridge L2 penalties searched by inner CV


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path,
                   default=here.parent / "processing" / "output" / "cs224w_2022_only")
    p.add_argument("--output-dir", type=Path, default=here / "outputs")
    p.add_argument("--inner-folds", type=int, default=5,
                   help="participant-grouped folds (within the training participants) used to choose alpha")
    p.add_argument("--min-train", type=int, default=10, help="min observed training labels per hormone")
    p.add_argument("--clip", action="store_true",
                   help="optionally clip log-predictions to the observed training target range (off by default)")
    return p.parse_args()


def load_inputs(folder: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Flatten each [days, 8 modalities, 13 slots] window into one tabular row.

    Preprocessing pads each modality to 13 feature slots (feature_names == "" for padding);
    those slots are always 0, so they are dropped. Missing features were stored as 0 with
    feature_present = 0, so the presence masks are appended as missingness indicators and
    no further imputation is needed. Only wearable/CGM/diary features are used: the npz holds
    no hormone values, cycle phase or other target-derived inputs.
    """
    with np.load(folder / "graph_inputs.npz", allow_pickle=False) as data:
        values = data["node_features"].astype(np.float64)   # [n, days, 8, 13]
        present = data["feature_present"].astype(bool)      # same shape
        y = data["y"].astype(np.float64)                    # [n, 2] = log1p(LH), log1p(E3G); NaN = not measured
        names = data["feature_names"].astype(str)           # [8, 13]
    index = pd.read_csv(folder / "examples_index.csv", dtype={"id": str})
    n, days = values.shape[:2]
    if len(index) != n or y.shape != (n, 2) or present.shape != values.shape:
        raise ValueError("graph_inputs.npz and examples_index.csv do not describe the same examples")

    real = names != ""                                  # real feature slots
    v = np.where(present, values, 0.0)[:, :, real]      # [n, days, n_features]
    m = present[:, :, real].astype(np.float64)
    X = np.concatenate([v.reshape(n, -1), m.reshape(n, -1)], axis=1)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite feature values found; expected preprocessing to zero-fill missing values")

    offsets = range(-(days - 1), 1)                     # d-6 ... d0 (d0 = target day)
    base = [f"d{t}_{f}" for t in offsets for f in names[real]]
    return X, y, index, base + [f"{c}_present" for c in base]


def fit_predict(X_tr, y_tr, groups_tr, X_te, inner_folds, clip):
    """Standardize + Ridge, alpha chosen by participant-grouped CV inside the training set only."""
    pipe = make_pipeline(StandardScaler(), Ridge())
    k = min(inner_folds, len(np.unique(groups_tr)))
    if k >= 2:
        splits = list(GroupKFold(n_splits=k).split(X_tr, y_tr, groups_tr))
        search = GridSearchCV(pipe, {"ridge__alpha": ALPHAS}, cv=splits,
                              scoring="neg_mean_squared_error")
        search.fit(X_tr, y_tr)
        model, alpha = search.best_estimator_, search.best_params_["ridge__alpha"]
    else:
        model, alpha = pipe.set_params(ridge__alpha=1.0).fit(X_tr, y_tr), 1.0
    z = model.predict(X_te)
    if clip:   # optional sensitivity analysis; disabled for the primary baseline
        z = np.clip(z, y_tr.min(), y_tr.max())
    return np.maximum(np.expm1(z), 0.0), float(alpha)


def run_loso(X, y, ids, inner_folds, min_train, clip) -> pd.DataFrame:
    subjects = sorted(np.unique(ids), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    rows = []
    for pid in subjects:
        test = ids == pid
        train = ~test   # the held-out participant never enters scaling, alpha selection or fitting
        row = {"id": pid, "n_test_windows": int(test.sum())}
        for j, (hormone, _) in enumerate(HORMONES):
            tr = train & np.isfinite(y[:, j])   # drop windows whose target day lacks this hormone
            te = test & np.isfinite(y[:, j])
            row.update({f"n_train_{hormone}": int(tr.sum()), f"n_test_{hormone}": int(te.sum()),
                        f"alpha_{hormone}": np.nan,
                        f"linear_{hormone}_mae": np.nan, f"linear_{hormone}_rmse": np.nan})
            if tr.sum() < min_train or te.sum() == 0:
                continue
            pred, alpha = fit_predict(X[tr], y[tr, j], ids[tr], X[te], inner_folds, clip)
            err = pred - np.expm1(y[te, j])   # errors in original units
            row[f"alpha_{hormone}"] = alpha
            row[f"linear_{hormone}_mae"] = float(np.mean(np.abs(err)))
            row[f"linear_{hormone}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        rows.append(row)
        print(f"LOSO held-out {pid:>3}: LH MAE={row['linear_lh_mae']:.3f}  E3G MAE={row['linear_e3g_mae']:.3f}")
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> None:
    print("\n=== Participant-averaged summary (LOSO, original units) ===")
    for hormone, unit in HORMONES:
        mae, rmse = metrics[f"linear_{hormone}_mae"], metrics[f"linear_{hormone}_rmse"]
        print(f"linear {hormone.upper():3s}  MAE={mae.mean():.3f} {unit}  RMSE={rmse.mean():.3f} {unit}  "
              f"(participants={int(mae.notna().sum())})")


def main() -> None:
    args = parse_args()
    X, y, index, feature_names = load_inputs(args.input_dir)
    ids = index["id"].to_numpy(dtype=str)
    print(f"Examples={len(X)}  features={X.shape[1]}  participants={len(np.unique(ids))}  "
          f"LH obs={np.isfinite(y[:, 0]).sum()}  E3G obs={np.isfinite(y[:, 1]).sum()}")
    metrics = run_loso(X, y, ids, args.inner_folds, args.min_train, args.clip)
    summarize(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "linear_metrics.csv"
    metrics.to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
