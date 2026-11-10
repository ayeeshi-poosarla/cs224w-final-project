#!/usr/bin/env python3
"""XGBoost baseline with leave-one-subject-out (LOSO) evaluation on mcPHASES.

Predicts same-day urinary LH and E3G (estrogen) from a 7-day window of Fitbit,
CGM, and diary features. Trains in log(1+y) space and reports participant-
averaged MAE / RMSE in original units.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

KEYS = ["id", "day_in_study"]
TARGETS = {"lh": "lh", "e3g": "estrogen"}
DIARY_COLS = [
    "flow_volume",
    "appetite",
    "exerciselevel",
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
]
ORDINAL_MAP = {
    "not at all": 0,
    "none": 0,
    "very low": 1,
    "low": 2,
    "mild": 2,
    "moderate": 3,
    "medium": 3,
    "high": 4,
    "very high": 5,
    "severe": 5,
    "spotting": 1,
    "light": 2,
    "heavy": 4,
    "very heavy": 5,
}
SENSOR_COLS_DEFAULT = [
    "resting_hr",
    "nightly_temperature",
    "nightly_temp_std",
    "rmssd",
    "breathing_rate",
    "sleep_score",
    "sedentary",
    "lightly",
    "moderately",
    "very",
    "glucose_mean",
    "glucose_std",
    "glucose_min",
    "glucose_max",
]
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    verbosity=0,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=here.parent / "data")
    p.add_argument("--output-dir", type=Path, default=here / "outputs")
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-val-subjects", type=int, default=8)
    return p.parse_args()


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[warn] missing {path.name}; skipping")
        return None
    return pd.read_csv(path)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def encode_ordinal(series: pd.Series) -> pd.Series:
    def map_one(v):
        if pd.isna(v):
            return np.nan
        key = str(v).strip().lower()
        if key in ORDINAL_MAP:
            return float(ORDINAL_MAP[key])
        try:
            return float(key)
        except ValueError:
            return np.nan

    return series.map(map_one)


def daily_mean(df: pd.DataFrame, cols: list[str], rename: dict[str, str] | None = None) -> pd.DataFrame:
    out = df.groupby(KEYS, as_index=False)[cols].mean()
    if rename:
        out = out.rename(columns=rename)
    return out


def load_daily_table(data_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    hs = _read_csv(data_dir / "hormones_and_selfreport.csv")
    if hs is None:
        raise FileNotFoundError(f"Required file missing: {data_dir / 'hormones_and_selfreport.csv'}")

    hs = hs.copy()
    hs["id"] = hs["id"].astype(str)
    hs["day_in_study"] = _to_numeric(hs["day_in_study"]).astype(int)
    hs["study_interval"] = _to_numeric(hs["study_interval"]).astype(int)
    hs["lh"] = _to_numeric(hs["lh"])
    hs["estrogen"] = _to_numeric(hs["estrogen"])

    diary_present = [c for c in DIARY_COLS if c in hs.columns]
    for c in diary_present:
        hs[c] = encode_ordinal(hs[c])

    daily = hs[KEYS + ["study_interval", "lh", "estrogen"] + diary_present].copy()

    # Resting heart rate
    rhr = _read_csv(data_dir / "resting_heart_rate.csv")
    if rhr is not None:
        rhr = rhr.copy()
        rhr["id"] = rhr["id"].astype(str)
        rhr["day_in_study"] = _to_numeric(rhr["day_in_study"]).astype(int)
        rhr["value"] = _to_numeric(rhr["value"])
        rhr = rhr[rhr["value"] > 0]
        rhr = daily_mean(rhr, ["value"], {"value": "resting_hr"})
        daily = daily.merge(rhr, on=KEYS, how="left")

    # Nightly computed temperature (assign to wake / sleep-end day)
    temp = _read_csv(data_dir / "computed_temperature.csv")
    if temp is not None:
        temp = temp.copy()
        temp["id"] = temp["id"].astype(str)
        temp["day_in_study"] = _to_numeric(temp["sleep_end_day_in_study"]).astype(int)
        temp["nightly_temperature"] = _to_numeric(temp["nightly_temperature"])
        std_col = "baseline_relative_nightly_standard_deviation"
        keep = ["nightly_temperature"]
        if std_col in temp.columns:
            temp["nightly_temp_std"] = _to_numeric(temp[std_col])
            keep.append("nightly_temp_std")
        temp = daily_mean(temp, keep)
        daily = daily.merge(temp, on=KEYS, how="left")

    # HRV RMSSD
    hrv = _read_csv(data_dir / "heart_rate_variability_details.csv")
    if hrv is not None:
        hrv = hrv.copy()
        hrv["id"] = hrv["id"].astype(str)
        hrv["day_in_study"] = _to_numeric(hrv["day_in_study"]).astype(int)
        hrv["rmssd"] = _to_numeric(hrv["rmssd"])
        hrv = daily_mean(hrv, ["rmssd"])
        daily = daily.merge(hrv, on=KEYS, how="left")

    # Respiratory rate
    resp = _read_csv(data_dir / "respiratory_rate_summary.csv")
    if resp is not None:
        resp = resp.copy()
        resp["id"] = resp["id"].astype(str)
        resp["day_in_study"] = _to_numeric(resp["day_in_study"]).astype(int)
        resp["breathing_rate"] = _to_numeric(resp["full_sleep_breathing_rate"])
        resp = daily_mean(resp, ["breathing_rate"])
        daily = daily.merge(resp, on=KEYS, how="left")

    # Sleep score
    sleep = _read_csv(data_dir / "sleep_score.csv")
    if sleep is not None:
        sleep = sleep.copy()
        sleep["id"] = sleep["id"].astype(str)
        sleep["day_in_study"] = _to_numeric(sleep["day_in_study"]).astype(int)
        sleep["sleep_score"] = _to_numeric(sleep["overall_score"])
        sleep = daily_mean(sleep, ["sleep_score"])
        daily = daily.merge(sleep, on=KEYS, how="left")

    # Active minutes
    act = _read_csv(data_dir / "active_minutes.csv")
    if act is not None:
        act = act.copy()
        act["id"] = act["id"].astype(str)
        act["day_in_study"] = _to_numeric(act["day_in_study"]).astype(int)
        for c in ["sedentary", "lightly", "moderately", "very"]:
            act[c] = _to_numeric(act[c])
        act = daily_mean(act, ["sedentary", "lightly", "moderately", "very"])
        daily = daily.merge(act, on=KEYS, how="left")

    # Glucose daily stats
    glu = _read_csv(data_dir / "glucose.csv")
    if glu is not None:
        glu = glu.copy()
        glu["id"] = glu["id"].astype(str)
        glu["day_in_study"] = _to_numeric(glu["day_in_study"]).astype(int)
        glu["glucose_value"] = _to_numeric(glu["glucose_value"])
        g = (
            glu.groupby(KEYS)["glucose_value"]
            .agg(glucose_mean="mean", glucose_std="std", glucose_min="min", glucose_max="max")
            .reset_index()
        )
        daily = daily.merge(g, on=KEYS, how="left")

    sensor_cols = [c for c in SENSOR_COLS_DEFAULT if c in daily.columns]
    diary_cols = [c for c in diary_present if c in daily.columns]
    daily = daily.sort_values(["id", "study_interval", "day_in_study"]).reset_index(drop=True)
    return daily, sensor_cols, diary_cols


def causal_normalize_and_fill(
    daily: pd.DataFrame,
    sensor_cols: list[str],
    max_ffill: int = 2,
) -> pd.DataFrame:
    """Z-score sensors using only prior days within (id, study_interval); ffill <= max_ffill."""
    out = daily.copy()
    pieces = []
    for _, g in out.groupby(["id", "study_interval"], sort=False):
        g = g.sort_values("day_in_study").copy()
        for col in sensor_cols:
            x = g[col].astype(float)
            # Expanding mean/std excluding current day (shift by 1)
            prior_mean = x.shift(1).expanding(min_periods=1).mean()
            prior_std = x.shift(1).expanding(min_periods=2).std()
            z = (x - prior_mean) / prior_std.replace(0, np.nan)
            # First day(s) with no prior stats stay NaN / use raw centered at 0
            z = z.where(prior_std.notna() & (prior_std > 0), np.nan)
            # Forward-fill normalized values up to max_ffill days
            filled = z.copy()
            # Track gaps: only fill through runs of length <= max_ffill
            is_na = filled.isna()
            # group consecutive NaNs
            groups = is_na.ne(is_na.shift(fill_value=False)).cumsum()
            for _, idx in filled.groupby(groups).groups.items():
                idx = list(idx)
                if not is_na.loc[idx[0]]:
                    continue
                if len(idx) <= max_ffill:
                    filled.loc[idx] = filled.ffill().loc[idx]
                # longer gaps remain NaN beyond first max_ffill? keep all NaN for long gaps
            g[col] = filled
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def build_windows(
    daily: pd.DataFrame,
    feature_cols: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return X, y_lh, y_e3g, ids, rounds, feature_names for contiguous windows."""
    X_list, y_lh, y_e3g, ids, rounds = [], [], [], [], []
    feat_names: list[str] | None = None

    for (pid, round_), g in daily.groupby(["id", "study_interval"], sort=False):
        g = g.sort_values("day_in_study").reset_index(drop=True)
        days = g["day_in_study"].to_numpy()
        feats = g[feature_cols].to_numpy(dtype=float)
        lh = g["lh"].to_numpy(dtype=float)
        e3g = g["estrogen"].to_numpy(dtype=float)

        for i in range(len(g) - window + 1):
            day_slice = days[i : i + window]
            # Contiguous calendar days within the interval
            if day_slice[-1] - day_slice[0] != window - 1:
                continue
            if not np.all(np.diff(day_slice) == 1):
                continue

            end_lh, end_e3g = lh[i + window - 1], e3g[i + window - 1]
            if np.isnan(end_lh) and np.isnan(end_e3g):
                continue

            block = feats[i : i + window]  # (W, F)
            # Require some Fitbit coverage if resting_hr or sleep_score present
            fitbit_idx = [feature_cols.index(c) for c in ("resting_hr", "sleep_score", "nightly_temperature") if c in feature_cols]
            if fitbit_idx:
                if np.all(np.isnan(block[:, fitbit_idx])):
                    continue

            masks = (~np.isnan(block)).astype(float)
            flat = np.concatenate([block.reshape(-1), masks.reshape(-1)])
            if feat_names is None:
                names = []
                for t in range(window):
                    for c in feature_cols:
                        names.append(f"d{t}_{c}")
                for t in range(window):
                    for c in feature_cols:
                        names.append(f"d{t}_{c}_mask")
                feat_names = names

            X_list.append(flat)
            y_lh.append(end_lh)
            y_e3g.append(end_e3g)
            ids.append(str(pid))
            rounds.append(int(round_))

    if not X_list:
        raise RuntimeError("No valid windows constructed; check data and window size.")

    return (
        np.vstack(X_list),
        np.asarray(y_lh, dtype=float),
        np.asarray(y_e3g, dtype=float),
        np.asarray(ids),
        np.asarray(rounds),
        feat_names or [],
    )


def fit_predict_hormone(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, XGBRegressor | None]:
    """Train on log(1+y); return original-unit predictions on X_test."""
    obs_tr = ~np.isnan(y_train)
    if obs_tr.sum() < 10:
        return np.full(len(X_test), np.nan), None

    z_tr = np.log1p(y_train[obs_tr])
    params = dict(XGB_PARAMS)
    fit_kwargs: dict = {"verbose": False}

    obs_va = ~np.isnan(y_val) if len(y_val) else None
    if obs_va is not None and obs_va.sum() >= 5:
        params["early_stopping_rounds"] = 30
        fit_kwargs["eval_set"] = [(X_val[obs_va], np.log1p(y_val[obs_va]))]

    model = XGBRegressor(**params, random_state=seed)
    model.fit(X_train[obs_tr], z_tr, **fit_kwargs)
    z_hat = model.predict(X_test)
    return np.expm1(z_hat), model


def round_median_predict(
    y_train: np.ndarray,
    rounds_train: np.ndarray,
    rounds_test: np.ndarray,
) -> np.ndarray:
    preds = np.full(len(rounds_test), np.nan)
    for r in np.unique(rounds_test):
        tr = (rounds_train == r) & (~np.isnan(y_train))
        te = rounds_test == r
        if tr.sum() == 0:
            tr = ~np.isnan(y_train)
        if tr.sum() == 0:
            continue
        preds[te] = np.median(y_train[tr])
    return preds


def run_loso(
    X: np.ndarray,
    y_lh: np.ndarray,
    y_e3g: np.ndarray,
    ids: np.ndarray,
    rounds: np.ndarray,
    feat_names: list[str],
    seed: int,
    n_val_subjects: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(np.unique(ids), key=lambda s: int(s) if str(s).isdigit() else s))
    rows = []
    importance_accum = {h: np.zeros(len(feat_names), dtype=float) for h in TARGETS}
    importance_counts = {h: 0 for h in TARGETS}

    targets = {"lh": y_lh, "e3g": y_e3g}

    for test_id in subjects:
        test_mask = ids == test_id
        train_pool = subjects[subjects != test_id]
        # Participant-intact validation split
        n_val = min(n_val_subjects, max(1, len(train_pool) // 5))
        val_ids = set(rng.choice(train_pool, size=n_val, replace=False).tolist())
        train_ids = set(train_pool) - val_ids

        tr = np.isin(ids, list(train_ids))
        va = np.isin(ids, list(val_ids))
        te = test_mask

        row = {"test_id": test_id, "n_test": int(te.sum()), "n_train": int(tr.sum()), "n_val": int(va.sum())}

        for hname, y in targets.items():
            med_pred = round_median_predict(y[tr], rounds[tr], rounds[te])
            xgb_pred, model = fit_predict_hormone(X[tr], y[tr], X[va], y[va], X[te], seed)

            for tag, pred in (("median", med_pred), ("xgb", xgb_pred)):
                obs = ~np.isnan(y[te])
                if obs.sum():
                    err = pred[obs] - y[te][obs]
                    mae = float(np.mean(np.abs(err)))
                    rmse = float(np.sqrt(np.mean(err ** 2)))
                else:
                    mae = rmse = float("nan")
                row[f"{tag}_{hname}_mae"] = mae
                row[f"{tag}_{hname}_rmse"] = rmse

            if model is not None:
                importance_accum[hname] += model.feature_importances_
                importance_counts[hname] += 1

        rows.append(row)
        print(
            f"LOSO held-out {test_id:>3}: "
            f"xgb LH MAE={row['xgb_lh_mae']:.3f}  "
            f"xgb E3G MAE={row['xgb_e3g_mae']:.3f}"
        )

    metrics = pd.DataFrame(rows)
    imp_rows = []
    for hname, acc in importance_accum.items():
        if importance_counts[hname] == 0:
            continue
        mean_imp = acc / importance_counts[hname]
        for fname, val in zip(feat_names, mean_imp):
            imp_rows.append({"hormone": hname, "feature": fname, "importance": float(val)})
    importance = pd.DataFrame(imp_rows)
    return metrics, importance


def summarize(metrics: pd.DataFrame) -> None:
    print("\n=== Participant-averaged summary (LOSO) ===")
    for model in ("median", "xgb"):
        for h, unit in (("lh", "mIU/mL"), ("e3g", "ng/mL")):
            mae_col = f"{model}_{h}_mae"
            rmse_col = f"{model}_{h}_rmse"
            print(
                f"{model:7s} {h.upper():3s}  "
                f"MAE={metrics[mae_col].mean():.3f} {unit}  "
                f"RMSE={metrics[rmse_col].mean():.3f} {unit}"
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading daily features from {args.data_dir} ...")
    daily, sensor_cols, diary_cols = load_daily_table(args.data_dir)
    print(f"Daily rows={len(daily)}  sensors={sensor_cols}  diary={diary_cols}")

    daily = causal_normalize_and_fill(daily, sensor_cols)
    feature_cols = sensor_cols + diary_cols

    print(f"Building {args.window}-day windows ...")
    X, y_lh, y_e3g, ids, rounds, feat_names = build_windows(daily, feature_cols, args.window)
    print(
        f"Windows={len(X)}  features={X.shape[1]}  "
        f"LH obs={np.sum(~np.isnan(y_lh))}  E3G obs={np.sum(~np.isnan(y_e3g))}"
    )

    metrics, importance = run_loso(
        X, y_lh, y_e3g, ids, rounds, feat_names, args.seed, args.n_val_subjects
    )
    summarize(metrics)

    metrics_path = args.output_dir / "metrics.csv"
    imp_path = args.output_dir / "feature_importance.csv"
    metrics.to_csv(metrics_path, index=False)
    importance.to_csv(imp_path, index=False)
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {imp_path}")

    if not importance.empty:
        print("\nTop features (LH):")
        top = (
            importance[importance["hormone"] == "lh"]
            .sort_values("importance", ascending=False)
            .head(10)
        )
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
