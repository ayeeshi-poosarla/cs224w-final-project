#!/usr/bin/env python3
"""XGBoost baseline with personalized temporal split on mcPHASES.

For each participant, train on 7-day windows whose final day falls in the first
30 days of that participant's study-interval segment, and evaluate on later
windows (months 2–3). Predicts same-day urinary LH and E3G (estrogen). Trains
in log(1+y) space and reports participant-averaged MAE / RMSE in original units.
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
    p.add_argument(
        "--train-days",
        type=int,
        default=30,
        help="Use windows ending in the first N days of each interval as train",
    )
    p.add_argument("--min-train", type=int, default=5, help="Min observed train labels per hormone")
    p.add_argument("--min-test", type=int, default=5, help="Min observed test labels per hormone")
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

    rhr = _read_csv(data_dir / "resting_heart_rate.csv")
    if rhr is not None:
        rhr = rhr.copy()
        rhr["id"] = rhr["id"].astype(str)
        rhr["day_in_study"] = _to_numeric(rhr["day_in_study"]).astype(int)
        rhr["value"] = _to_numeric(rhr["value"])
        rhr = rhr[rhr["value"] > 0]
        rhr = daily_mean(rhr, ["value"], {"value": "resting_hr"})
        daily = daily.merge(rhr, on=KEYS, how="left")

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

    hrv = _read_csv(data_dir / "heart_rate_variability_details.csv")
    if hrv is not None:
        hrv = hrv.copy()
        hrv["id"] = hrv["id"].astype(str)
        hrv["day_in_study"] = _to_numeric(hrv["day_in_study"]).astype(int)
        hrv["rmssd"] = _to_numeric(hrv["rmssd"])
        hrv = daily_mean(hrv, ["rmssd"])
        daily = daily.merge(hrv, on=KEYS, how="left")

    resp = _read_csv(data_dir / "respiratory_rate_summary.csv")
    if resp is not None:
        resp = resp.copy()
        resp["id"] = resp["id"].astype(str)
        resp["day_in_study"] = _to_numeric(resp["day_in_study"]).astype(int)
        resp["breathing_rate"] = _to_numeric(resp["full_sleep_breathing_rate"])
        resp = daily_mean(resp, ["breathing_rate"])
        daily = daily.merge(resp, on=KEYS, how="left")

    sleep = _read_csv(data_dir / "sleep_score.csv")
    if sleep is not None:
        sleep = sleep.copy()
        sleep["id"] = sleep["id"].astype(str)
        sleep["day_in_study"] = _to_numeric(sleep["day_in_study"]).astype(int)
        sleep["sleep_score"] = _to_numeric(sleep["overall_score"])
        sleep = daily_mean(sleep, ["sleep_score"])
        daily = daily.merge(sleep, on=KEYS, how="left")

    act = _read_csv(data_dir / "active_minutes.csv")
    if act is not None:
        act = act.copy()
        act["id"] = act["id"].astype(str)
        act["day_in_study"] = _to_numeric(act["day_in_study"]).astype(int)
        for c in ["sedentary", "lightly", "moderately", "very"]:
            act[c] = _to_numeric(act[c])
        act = daily_mean(act, ["sedentary", "lightly", "moderately", "very"])
        daily = daily.merge(act, on=KEYS, how="left")

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
    out = daily.copy()
    pieces = []
    for _, g in out.groupby(["id", "study_interval"], sort=False):
        g = g.sort_values("day_in_study").copy()
        for col in sensor_cols:
            x = g[col].astype(float)
            prior_mean = x.shift(1).expanding(min_periods=1).mean()
            prior_std = x.shift(1).expanding(min_periods=2).std()
            z = (x - prior_mean) / prior_std.replace(0, np.nan)
            z = z.where(prior_std.notna() & (prior_std > 0), np.nan)
            filled = z.copy()
            is_na = filled.isna()
            groups = is_na.ne(is_na.shift(fill_value=False)).cumsum()
            for _, idx in filled.groupby(groups).groups.items():
                idx = list(idx)
                if not is_na.loc[idx[0]]:
                    continue
                if len(idx) <= max_ffill:
                    filled.loc[idx] = filled.ffill().loc[idx]
            g[col] = filled
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def build_windows(
    daily: pd.DataFrame,
    feature_cols: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return X, y_lh, y_e3g, ids, rounds, end_days, feature_names."""
    X_list, y_lh, y_e3g, ids, rounds, end_days = [], [], [], [], [], []
    feat_names: list[str] | None = None

    for (pid, round_), g in daily.groupby(["id", "study_interval"], sort=False):
        g = g.sort_values("day_in_study").reset_index(drop=True)
        days = g["day_in_study"].to_numpy()
        feats = g[feature_cols].to_numpy(dtype=float)
        lh = g["lh"].to_numpy(dtype=float)
        e3g = g["estrogen"].to_numpy(dtype=float)
        # Relative day within this interval segment (1-indexed from first observed day)
        rel_start = int(days.min())

        for i in range(len(g) - window + 1):
            day_slice = days[i : i + window]
            if day_slice[-1] - day_slice[0] != window - 1:
                continue
            if not np.all(np.diff(day_slice) == 1):
                continue

            end_lh, end_e3g = lh[i + window - 1], e3g[i + window - 1]
            if np.isnan(end_lh) and np.isnan(end_e3g):
                continue

            block = feats[i : i + window]
            fitbit_idx = [
                feature_cols.index(c)
                for c in ("resting_hr", "sleep_score", "nightly_temperature")
                if c in feature_cols
            ]
            if fitbit_idx and np.all(np.isnan(block[:, fitbit_idx])):
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
            # Days since start of this interval (absolute day - min day + 1)
            end_days.append(int(day_slice[-1] - rel_start + 1))

    if not X_list:
        raise RuntimeError("No valid windows constructed; check data and window size.")

    return (
        np.vstack(X_list),
        np.asarray(y_lh, dtype=float),
        np.asarray(y_e3g, dtype=float),
        np.asarray(ids),
        np.asarray(rounds),
        np.asarray(end_days, dtype=int),
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
    obs_tr = ~np.isnan(y_train)
    if obs_tr.sum() < 5:
        return np.full(len(X_test), np.nan), None

    z_tr = np.log1p(y_train[obs_tr])
    params = dict(XGB_PARAMS)
    fit_kwargs: dict = {"verbose": False}

    obs_va = ~np.isnan(y_val) if len(y_val) else None
    if obs_va is not None and obs_va.sum() >= 3:
        params["early_stopping_rounds"] = 30
        fit_kwargs["eval_set"] = [(X_val[obs_va], np.log1p(y_val[obs_va]))]

    model = XGBRegressor(**params, random_state=seed)
    model.fit(X_train[obs_tr], z_tr, **fit_kwargs)
    return np.expm1(model.predict(X_test)), model


def median_predict(y_train: np.ndarray, n_test: int) -> np.ndarray:
    obs = ~np.isnan(y_train)
    if obs.sum() == 0:
        return np.full(n_test, np.nan)
    return np.full(n_test, np.median(y_train[obs]))


def run_temporal(
    X: np.ndarray,
    y_lh: np.ndarray,
    y_e3g: np.ndarray,
    ids: np.ndarray,
    rounds: np.ndarray,
    end_days: np.ndarray,
    feat_names: list[str],
    seed: int,
    train_days: int,
    min_train: int,
    min_test: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = {"lh": y_lh, "e3g": y_e3g}
    rows = []
    importance_accum = {h: np.zeros(len(feat_names), dtype=float) for h in TARGETS}
    importance_counts = {h: 0 for h in TARGETS}

    # Evaluate per (participant, interval) then average by participant
    pairs = sorted(
        {(str(i), int(r)) for i, r in zip(ids, rounds)},
        key=lambda t: (int(t[0]) if t[0].isdigit() else t[0], t[1]),
    )

    for pid, round_ in pairs:
        mask = (ids == pid) & (rounds == round_)
        if mask.sum() == 0:
            continue

        # Relative end-day already computed within interval
        tr = mask & (end_days <= train_days)
        te = mask & (end_days > train_days)
        if tr.sum() == 0 or te.sum() == 0:
            continue

        # Chronological early-stopping slice from the end of train
        tr_idx = np.where(tr)[0]
        order = np.argsort(end_days[tr_idx])
        tr_idx = tr_idx[order]
        n_val = max(1, len(tr_idx) // 5)
        va_idx = tr_idx[-n_val:]
        tr_fit_idx = tr_idx[:-n_val] if len(tr_idx) > n_val else tr_idx

        row = {
            "id": pid,
            "study_interval": round_,
            "n_train": int(len(tr_fit_idx)),
            "n_val": int(len(va_idx)),
            "n_test": int(te.sum()),
        }
        usable = False

        for hname, y in targets.items():
            n_tr_obs = int(np.sum(~np.isnan(y[tr_fit_idx])))
            n_te_obs = int(np.sum(~np.isnan(y[te])))
            row[f"n_train_{hname}"] = n_tr_obs
            row[f"n_test_{hname}"] = n_te_obs
            if n_tr_obs < min_train or n_te_obs < min_test:
                row[f"median_{hname}_mae"] = np.nan
                row[f"median_{hname}_rmse"] = np.nan
                row[f"xgb_{hname}_mae"] = np.nan
                row[f"xgb_{hname}_rmse"] = np.nan
                continue

            usable = True
            med = median_predict(y[tr_fit_idx], int(te.sum()))
            xgb_pred, model = fit_predict_hormone(
                X[tr_fit_idx], y[tr_fit_idx], X[va_idx], y[va_idx], X[te], seed
            )

            yt = y[te]
            for tag, pred in (("median", med), ("xgb", xgb_pred)):
                obs = ~np.isnan(yt)
                err = pred[obs] - yt[obs]
                row[f"{tag}_{hname}_mae"] = float(np.mean(np.abs(err)))
                row[f"{tag}_{hname}_rmse"] = float(np.sqrt(np.mean(err ** 2)))

            if model is not None:
                importance_accum[hname] += model.feature_importances_
                importance_counts[hname] += 1

        if usable:
            rows.append(row)
            print(
                f"Temporal id={pid} round={round_}: "
                f"xgb LH MAE={row.get('xgb_lh_mae', float('nan')):.3f}  "
                f"xgb E3G MAE={row.get('xgb_e3g_mae', float('nan')):.3f}"
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
    if metrics.empty:
        print("No participants met the temporal split criteria.")
        return
    # Average across participant-interval rows, then optionally by participant
    print("\n=== Participant-interval averaged summary (temporal) ===")
    for model in ("median", "xgb"):
        for h, unit in (("lh", "mIU/mL"), ("e3g", "ng/mL")):
            mae_col = f"{model}_{h}_mae"
            rmse_col = f"{model}_{h}_rmse"
            if mae_col not in metrics.columns:
                continue
            print(
                f"{model:7s} {h.upper():3s}  "
                f"MAE={metrics[mae_col].mean(skipna=True):.3f} {unit}  "
                f"RMSE={metrics[rmse_col].mean(skipna=True):.3f} {unit}"
            )

    # Participant-averaged (mean over intervals within id, then mean over ids)
    print("\n=== Participant-averaged summary (temporal) ===")
    by_id = metrics.groupby("id", as_index=False).mean(numeric_only=True)
    for model in ("median", "xgb"):
        for h, unit in (("lh", "mIU/mL"), ("e3g", "ng/mL")):
            mae_col = f"{model}_{h}_mae"
            rmse_col = f"{model}_{h}_rmse"
            if mae_col not in by_id.columns:
                continue
            print(
                f"{model:7s} {h.upper():3s}  "
                f"MAE={by_id[mae_col].mean(skipna=True):.3f} {unit}  "
                f"RMSE={by_id[rmse_col].mean(skipna=True):.3f} {unit}"
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
    X, y_lh, y_e3g, ids, rounds, end_days, feat_names = build_windows(
        daily, feature_cols, args.window
    )
    print(
        f"Windows={len(X)}  features={X.shape[1]}  "
        f"LH obs={np.sum(~np.isnan(y_lh))}  E3G obs={np.sum(~np.isnan(y_e3g))}"
    )

    metrics, importance = run_temporal(
        X,
        y_lh,
        y_e3g,
        ids,
        rounds,
        end_days,
        feat_names,
        args.seed,
        args.train_days,
        args.min_train,
        args.min_test,
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
