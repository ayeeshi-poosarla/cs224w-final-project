#!/usr/bin/env python3
"""Linear (Ridge) baseline with a personalized temporal split on mcPHASES.

For each participant and study interval, fit a Ridge model on windows whose target day
falls in the first --train-days days of that interval, and evaluate on later windows.
Uses the same features and targets as linear_loso.py, read from the shared preprocessing
outputs (processing/output/):
    cs224w_2022_only/graph_inputs.npz, cs224w_2022_only/examples_index.csv
    daily_table_2022_only.csv   (only to find each interval's first day)
Trains in log(1+y) space and reports participant-averaged MAE / RMSE in original units.
Requires numpy, pandas, scikit-learn.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORMONES = (("lh", "mIU/mL"), ("e3g", "ng/mL"))   # column order of y in graph_inputs.npz
ALPHAS = np.logspace(-2, 4, 13)                   # Ridge L2 penalties searched by inner CV
WINDOW_DAYS = 7                                   # preprocessing default (--days 7)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path,
                   default=here.parent / "processing" / "output" / "cs224w_2022_only")
    p.add_argument("--output-dir", type=Path, default=here / "outputs")
    p.add_argument("--train-days", type=int, default=30,
                   help="windows whose target day is in the first N days of the interval are training data")
    p.add_argument("--gap-days", type=int, default=0,
                   help="skip test windows ending within this many days after the training period "
                        "(0 matches the XGBoost baseline; 6 makes train and test windows share no days)")
    p.add_argument("--min-train", type=int, default=5, help="min observed training labels per hormone (as XGBoost)")
    p.add_argument("--min-test", type=int, default=5, help="min observed test labels per hormone (as XGBoost)")
    p.add_argument("--inner-splits", type=int, default=3, help="chronological folds used to choose alpha")
    p.add_argument("--fallback-alpha", type=float, default=10.0,
                   help="alpha used when there are too few training windows for chronological CV")
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


def relative_days(index: pd.DataFrame, input_dir: Path) -> np.ndarray:
    """Day of each example's target day within its interval (1 = first day of the interval).

    The interval starts at the participant's first diary row, the first row of the daily grid
    built by preprocessing. examples_index.csv only lists hormone days, so its own minimum could
    start later; the daily table gives the true start. Do not use the split_by_time column: it
    encodes the preprocessing's separate 60/20/20 split.
    """
    suffix = input_dir.name.removeprefix("cs224w_")     # e.g. "2022_only" or "both_rounds"
    daily_path = input_dir.parent / f"daily_table_{suffix}.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Temporal evaluation needs {daily_path} to find each interval's first day")
    daily = pd.read_csv(daily_path, usecols=["id", "study_interval", "day_in_study"], dtype={"id": str})
    starts = (daily.groupby(["id", "study_interval"], as_index=False)["day_in_study"].min()
              .rename(columns={"day_in_study": "start_day"}))
    merged = index[["id", "study_interval", "day_in_study"]].merge(
        starts, on=["id", "study_interval"], how="left", validate="many_to_one")
    if merged["start_day"].isna().any():
        raise ValueError("Some examples have no interval start in the daily table")
    # Explicitly preserve row alignment with graph_inputs.npz.
    if len(merged) != len(index):
        raise ValueError("Interval-start merge changed the number of examples")
    return (merged["day_in_study"] - merged["start_day"] + 1).to_numpy(dtype=int)


def fit_predict(X_tr, y_tr, train_days, X_te, inner_splits, fallback_alpha, clip):
    """Tune Ridge using chronological folds separated by actual calendar days.

    A seven-day window ending on day d includes days d-6 through d. Each
    validation fold therefore starts at least WINDOW_DAYS days after the last
    training target day. Missing hormone-label days do not shrink this gap.
    """
    pipe = make_pipeline(StandardScaler(), Ridge())
    folds = []
    if inner_splits >= 2 and len(y_tr) >= inner_splits + 3:
        # Candidate validation blocks are chronological; prune each training
        # block using actual day indices, not the count of labeled examples.
        splitter = TimeSeriesSplit(n_splits=inner_splits)
        for candidate_train, validation in splitter.split(X_tr):
            first_val_day = train_days[validation].min()
            safe_train = candidate_train[train_days[candidate_train] <= first_val_day - WINDOW_DAYS]
            if len(safe_train) >= 3 and len(validation) >= 1:
                folds.append((safe_train, validation))
    if len(folds) >= 2:
        search = GridSearchCV(pipe, {"ridge__alpha": ALPHAS}, cv=folds,
                              scoring="neg_mean_squared_error")
        search.fit(X_tr, y_tr)
        model, alpha = search.best_estimator_, search.best_params_["ridge__alpha"]
    else:
        model = pipe.set_params(ridge__alpha=fallback_alpha).fit(X_tr, y_tr)
        alpha = fallback_alpha
    z = model.predict(X_te)
    if clip:   # optional sensitivity analysis; disabled for the primary baseline
        z = np.clip(z, y_tr.min(), y_tr.max())
    return np.maximum(np.expm1(z), 0.0), float(alpha)


def run_temporal(X, y, index, rel_day, args) -> pd.DataFrame:
    rows = []
    keys = index[["id", "study_interval"]].drop_duplicates()
    keys = sorted(map(tuple, keys.to_numpy()), key=lambda k: ((0, int(k[0])) if k[0].isdigit() else (1, k[0]), k[1]))
    ids, intervals = index["id"].to_numpy(dtype=str), index["study_interval"].to_numpy()
    for pid, interval in keys:
        member = (ids == pid) & (intervals == interval)
        train = member & (rel_day <= args.train_days)
        test = member & (rel_day > args.train_days + args.gap_days)
        row = {"id": pid, "study_interval": interval,
               "n_train_windows": int(train.sum()), "n_test_windows": int(test.sum())}
        usable = False
        for j, (hormone, _) in enumerate(HORMONES):
            tr = np.flatnonzero(train & np.isfinite(y[:, j]))
            tr = tr[np.argsort(rel_day[tr], kind="stable")]   # chronological, for TimeSeriesSplit
            te = np.flatnonzero(test & np.isfinite(y[:, j]))
            row.update({f"n_train_{hormone}": len(tr), f"n_test_{hormone}": len(te),
                        f"alpha_{hormone}": np.nan,
                        f"linear_{hormone}_mae": np.nan, f"linear_{hormone}_rmse": np.nan})
            if len(tr) < args.min_train or len(te) < args.min_test:
                continue
            usable = True
            pred, alpha = fit_predict(X[tr], y[tr, j], X[te], args.inner_splits,
                                      args.fallback_alpha, not args.no_clip)
            err = pred - np.expm1(y[te, j])   # errors in original units
            row[f"alpha_{hormone}"] = alpha
            row[f"linear_{hormone}_mae"] = float(np.mean(np.abs(err)))
            row[f"linear_{hormone}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        if usable:
            rows.append(row)
            print(f"Temporal id={pid} interval={interval}: "
                  f"LH MAE={row['linear_lh_mae']:.3f}  E3G MAE={row['linear_e3g_mae']:.3f}")
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        print("No participant-interval met the temporal split criteria.")
        return
    # Average a participant's intervals first, then average participants (as the XGBoost baseline).
    by_id = metrics.groupby("id", as_index=False).mean(numeric_only=True)
    print("\n=== Participant-averaged summary (temporal, original units) ===")
    for hormone, unit in HORMONES:
        mae, rmse = by_id[f"linear_{hormone}_mae"], by_id[f"linear_{hormone}_rmse"]
        print(f"linear {hormone.upper():3s}  MAE={mae.mean():.3f} {unit}  RMSE={rmse.mean():.3f} {unit}  "
              f"(participants={int(mae.notna().sum())})")


def main() -> None:
    args = parse_args()
    X, y, index, feature_names = load_inputs(args.input_dir)
    rel_day = relative_days(index, args.input_dir)
    print(f"Examples={len(X)}  features={X.shape[1]}  "
          f"LH obs={np.isfinite(y[:, 0]).sum()}  E3G obs={np.isfinite(y[:, 1]).sum()}")
    metrics = run_temporal(X, y, index, rel_day, args)
    summarize(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "linear_metrics.csv"
    metrics.to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
