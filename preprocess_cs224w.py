"""CS224W preprocessing: mcPHASES v1.0.0 files to per-day modality graphs for hormone prediction.

    python preprocess_cs224w.py --data-dir /path/to/mcphases-...-1.0.0 [--rounds both] [--days 7]

Writes to ./output/:
  cs224w_2022_only/graph_inputs.npz   node_features [examples, days, 8 modalities, 13], feature_present (same
                                      shape), node_present [examples, days, 8], y [examples, 2] = log(1 + LH),
                                      log(1 + E3G) on the last day (NaN if not measured), modality and feature names
  cs224w_2022_only/examples_index.csv id, study_interval, day_in_study (the target day), split_by_time
  daily_table_2022_only.csv           the cleaned daily table (one row per person-day, 29 features)
  cleaning_summary_2022_only.csv      what each cleaning step changed
  heart_rate_daily_stats.csv          cache of the 2 GB heart_rate.csv (first run ~10 min, later runs ~1 min)

Steps (reasons and sources in preprocessing_decisions.md; numbers in brackets refer to it):
  1. Clean each file and summarise it to one value per person-day: 29 features in 8 modalities [1-3]
  2. Scale: per participant and round, z-score each sensor feature with the mean and SD of earlier days only,
     clipped at +-5 SD; diary items divided by their scale maximum [4.2, 4.4]
  3. Build one example per hormone day: the k most recent days including that day (k = --days), 8 modality
     nodes per day; a modality-day with no value is a missing node, and nothing is imputed [5]
Edges (same-day, same-modality across days, cross-modality across days) are built in the model code from the
node grid. Leave-one-subject-out folds use the id column. Needs numpy and pandas.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["id", "study_interval", "day_in_study"]
PR = ["id", "study_interval"]
OUT = Path(__file__).resolve().parent / "output"   # all outputs go here
CHUNK = 5_000_000

MIN_WEAR_HOURS = 10          # [2.6] All of Us valid-day rule: heart-rate values need >= 10 h of readings
MIN_SLEEP_MINUTES = 180      # [2.7] All of Us: main sleeps under 180 min excluded
GLUCOSE_MIN_SHARE = 0.70     # [2.9] CGM consensus (Battelino 2019): >= 70% of the 288 five-minute readings
MGDL_PER_MMOL = 18.0         # [2.4]
EXCLUDED_IN_2024 = {         # [1.3] secondary version: 2024 rows shared with other participant IDs
    "resting_heart_rate.csv", "sleep.csv", "heart_rate_variability_details.csv", "computed_temperature.csv",
    "respiratory_rate_summary.csv", "active_minutes.csv", "steps.csv"}

LIKERT = {"Not at all": 0, "Very Low/Little": 1, "Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5,
          **{str(i): i for i in range(6)}}
FLOW = {s: i for i, s in enumerate(["Not at all", "Spotting / Very Light", "Light", "Somewhat Light", "Moderate",
                                    "Somewhat Heavy", "Heavy", "Very Heavy"])}
SYMPTOMS = ["appetite", "exerciselevel", "headaches", "cramps", "sorebreasts", "fatigue", "sleepissue", "moodswing",
            "stress", "foodcravings", "indigestion", "bloating"]
# each feature's modality (the CS224W proposal's 8 node types; CS230 groups them as Fitbit / glucose / diary)
MODALITY = {
    "resting_hr": "heart rate", "hr_mean": "heart rate", "hr_sd": "heart rate",
    "nightly_skin_temperature": "nightly temperature",
    "nightly_rmssd": "heart rate variability",
    "respiration": "respiratory rate",
    "sleep_minutes": "sleep", "sleep_efficiency": "sleep",
    "light_minutes": "activity", "moderate_minutes": "activity", "very_minutes": "activity", "steps": "activity",
    "glucose_mean": "glucose", "glucose_sd": "glucose", "glucose_min": "glucose", "glucose_max": "glucose",
    **{f"diary_{s}": "symptoms" for s in SYMPTOMS + ["flow_volume"]},
}
SOURCE = {"heart rate": "Fitbit", "nightly temperature": "Fitbit", "heart rate variability": "Fitbit",
          "respiratory rate": "Fitbit", "sleep": "Fitbit", "activity": "Fitbit", "glucose": "CGM", "symptoms": "diary"}

ROUNDS = "2022"
LOG = []   # what each cleaning step removed, written to cleaning_summary.csv


def log(step, rnd, n, detail=""):
    LOG.append({"step": step, "study_interval": rnd, "n": int(n), "detail": detail})


def minutes(day, clock):
    return day.to_numpy() * 1440 + pd.to_timedelta(clock, errors="coerce").dt.total_seconds().to_numpy() / 60


def read(data, name, cols, key_time=None):
    """Read one file's columns, keep the rounds in use, and apply deduplication [2.1]: collapse exact duplicate
    rows, then drop record keys (participant, round, day, time) whose copies disagree."""
    d = pd.read_csv(data / name, usecols=cols)
    keep_2024 = ROUNDS == "both" and name not in EXCLUDED_IN_2024
    d = d[d.study_interval == 2022] if not keep_2024 else d
    day = next(c for c in cols if "day_in_study" in c)
    key = ["id", "study_interval", day] + ([key_time] if key_time else [])
    for r, g in d.groupby("study_interval"):
        n0 = len(g); g1 = g.drop_duplicates(); n1 = len(g1)
        n2 = (~g1.duplicated(key, keep=False)).sum()
        log("deduplication: exact copies removed", r, n0 - n1, name)
        log("deduplication: rows in disagreeing keys removed", r, n1 - n2, name)
    d = d.drop_duplicates()
    return d[~d.duplicated(key, keep=False)]


# ---------------------------------------------------------------- daily features

def heart_rate(data):
    """hr_mean, hr_sd per day from heart_rate.csv, and hours worn (minutes with a reading / 60) for [2.6].
    heart_rate.csv has one row per participant and timestamp (no deduplication needed). Cached."""
    cache = OUT / "heart_rate_daily_stats.csv"
    if not cache.exists():
        parts = []
        for i, c in enumerate(pd.read_csv(data / "heart_rate.csv", usecols=KEYS + ["timestamp", "bpm"], chunksize=CHUNK)):
            c = c.dropna(subset=["bpm"])
            c["minute"] = c.timestamp.str.slice(0, 2).astype(int) * 60 + c.timestamp.str.slice(3, 5).astype(int)
            c["sq"] = c.bpm.astype(float) ** 2
            parts.append(c.groupby(KEYS + ["minute"]).agg(n=("bpm", "size"), s=("bpm", "sum"), ss=("sq", "sum")))
            print(f"  heart_rate.csv chunk {i + 1}", flush=True)
        m = pd.concat(parts).groupby(level=[0, 1, 2, 3]).sum()
        d = m.groupby(level=[0, 1, 2]).agg(n=("n", "sum"), s=("s", "sum"), ss=("ss", "sum"), minutes=("n", "size"))
        d.reset_index().to_csv(cache, index=False)
    d = pd.read_csv(cache).set_index(KEYS)
    d = d[d.index.get_level_values("study_interval") == 2022] if ROUNDS == "2022" else d
    mean = d.s / d.n
    sd = np.sqrt((d.ss / d.n - mean ** 2).clip(lower=0))
    worn = d.minutes / 60 >= MIN_WEAR_HOURS
    for r in (2022, 2024):
        sel = d.index.get_level_values("study_interval") == r
        if sel.any():
            log("valid day: heart rate < 10 h of readings", r, (~worn[sel]).sum(), "hr_mean, hr_sd set to missing")
    return pd.DataFrame({"hr_mean": mean.where(worn), "hr_sd": sd.where(worn), "hours_with_hr": d.minutes / 60})


def resting_hr(data):
    r = read(data, "resting_heart_rate.csv", KEYS + ["value"])
    for rnd, g in r.groupby("study_interval"):
        log("resting heart rate of 0 set to missing", rnd, (g.value == 0).sum(), "Fitbit's code for no estimate")
    r = r[r.value > 0]                                                      # [2.5]
    return r.groupby(KEYS).value.mean().rename("resting_hr")


def main_sleeps(data):
    s = read(data, "sleep.csv", ["id", "study_interval", "sleep_start_day_in_study", "sleep_start_timestamp",
                                 "sleep_end_day_in_study", "sleep_end_timestamp", "minutesasleep", "efficiency",
                                 "mainsleep"], key_time="sleep_start_timestamp")
    s = s[s.mainsleep.astype(str).str.lower() == "true"].copy()
    s["start"] = minutes(s.sleep_start_day_in_study, s.sleep_start_timestamp)
    s["end"] = minutes(s.sleep_end_day_in_study, s.sleep_end_timestamp)
    for rnd, g in s.groupby("study_interval"):
        log("main sleep < 180 min removed", rnd, (g.minutesasleep < MIN_SLEEP_MINUTES).sum(), "")
    return s[s.minutesasleep >= MIN_SLEEP_MINUTES].rename(columns={"sleep_end_day_in_study": "day_in_study"})


def sleep(sleeps):
    g = sleeps.groupby(KEYS)
    return pd.DataFrame({"sleep_minutes": g.minutesasleep.mean(), "sleep_efficiency": g.efficiency.mean()})


def hrv(data, sleeps):
    """Nightly RMSSD: median of the 5-minute values inside a main sleep, assigned to the day the sleep ends (as in
    the preliminary code and Fitbit's nightly HRV)."""
    h = read(data, "heart_rate_variability_details.csv", KEYS + ["timestamp", "rmssd"], key_time="timestamp")
    h = h[h.rmssd > 0]
    out = []
    for key, g in h.groupby(PR):
        s = sleeps[(sleeps.id == key[0]) & (sleeps.study_interval == key[1])].sort_values("start")
        if s.empty:
            continue
        t = minutes(g.day_in_study, g.timestamp)
        i = np.clip(np.searchsorted(s.start.to_numpy(), t, side="right") - 1, 0, len(s) - 1)
        inside = (t >= s.start.to_numpy()[i]) & (t < s.end.to_numpy()[i])
        out.append(pd.DataFrame({"id": key[0], "study_interval": key[1], "day_in_study": s.day_in_study.to_numpy()[i][inside],
                                 "rmssd": g.rmssd.to_numpy()[inside]}))
    return pd.concat(out).groupby(KEYS).rmssd.median().rename("nightly_rmssd")


def temperature(data):
    c = read(data, "computed_temperature.csv", ["id", "study_interval", "sleep_end_day_in_study", "sleep_start_timestamp",
                                                "type", "nightly_temperature"], key_time="sleep_start_timestamp")
    c = c[c.type == "SKIN"].rename(columns={"sleep_end_day_in_study": "day_in_study"})
    return c.groupby(KEYS).nightly_temperature.mean().rename("nightly_skin_temperature")


def respiration(data):
    r = read(data, "respiratory_rate_summary.csv", KEYS + ["timestamp", "full_sleep_breathing_rate"], key_time="timestamp")
    for rnd, g in r.groupby("study_interval"):
        log("breathing rate of 0 or -1 set to missing", rnd, (g.full_sleep_breathing_rate <= 0).sum(), "Fitbit's code for no estimate")
    r = r[r.full_sleep_breathing_rate > 0]                                  # [2.5]
    return r.groupby(KEYS).full_sleep_breathing_rate.mean().rename("respiration")


def activity(data):
    a = read(data, "active_minutes.csv", KEYS + ["lightly", "moderately", "very"])
    a = a.groupby(KEYS).mean().rename(columns={"lightly": "light_minutes", "moderately": "moderate_minutes", "very": "very_minutes"})
    s = read(data, "steps.csv", KEYS + ["timestamp", "steps"], key_time="timestamp")
    return a.join(s.groupby(KEYS).steps.sum(), how="outer")


def glucose(data):
    g = read(data, "glucose.csv", KEYS + ["timestamp", "glucose_value"], key_time="timestamp")
    med = g.groupby("id").glucose_value.median()
    mgdl = med[med > 30].index                                              # [2.4] participants recorded in mg/dL
    log("glucose converted from mg/dL to mmol/L", 2022, g.id.isin(mgdl).sum(), "participants " + ", ".join(map(str, mgdl)))
    g.loc[g.id.isin(mgdl), "glucose_value"] /= MGDL_PER_MMOL
    s = g.groupby(KEYS).glucose_value
    out = pd.DataFrame({"glucose_mean": s.mean(), "glucose_sd": s.std(), "glucose_min": s.min(), "glucose_max": s.max(), "n": s.size()})
    short = out.n < GLUCOSE_MIN_SHARE * 288
    log("glucose day < 70% of readings removed", 2022, short.sum(), "")
    return out[~short].drop(columns="n")


def diary(data):
    h = pd.read_csv(data / "hormones_and_selfreport.csv")
    h = h[h.study_interval == 2022] if ROUNDS == "2022" else h
    d = h[KEYS + ["phase"]].copy()
    for c in ["lh", "estrogen"]:
        for r, g in h.groupby("study_interval"):
            log(f"{'LH' if c == 'lh' else 'E3G'} of 0 set to missing", r, (g[c] == 0).sum(), "the test's lowest reading is 1.0")
        d[c] = h[c].where(h[c] > 0)                                         # [2.2]
    for c in SYMPTOMS:
        d[f"diary_{c}"] = h[c].astype("string").map(LIKERT).astype(float)
    d["diary_flow_volume"] = h.flow_volume.map(FLOW).astype(float)
    return d


def build(data):
    d = diary(data)
    # each person-round: every day from the first to the last diary row (as in the preliminary code)
    grid = pd.MultiIndex.from_tuples([(p, r, day) for (p, r), g in d.groupby(PR)
                                      for day in range(g.day_in_study.min(), g.day_in_study.max() + 1)], names=KEYS)
    sleeps = main_sleeps(data)
    parts = [heart_rate(data), resting_hr(data), sleep(sleeps), hrv(data, sleeps), temperature(data),
             respiration(data), activity(data), glucose(data)]
    t = d.set_index(KEYS).reindex(grid).join(pd.concat(parts, axis=1), how="left").reset_index()

    return t[KEYS + ["phase", "lh", "estrogen", "hours_with_hr"] + list(MODALITY)]



# ---------------------------------------------------------------- scaling and split

CLIP_SD, FILL_DAYS, MIN_HISTORY = 5, 2, 3
DIARY_MAX = {"diary_flow_volume": 7}   # every other diary item is on 0-5
SPLIT = (0.6, 0.8)


def split_by_time(days, gap):
    """train / validation / test for one person-round's hormone days (study days, ascending), in date order [4.6].
    With gap = g, a validation or test day must be at least g days after the last day of the part before it, so
    no 7-day input window (gap = 7) contains days of two parts; days in between are labelled "gap" and not used."""
    days = np.asarray(days)
    n = len(days)
    a, b = int(round(n * SPLIT[0])), int(round(n * SPLIT[1]))
    label = np.array(["train"] * a + ["validation"] * (b - a) + ["test"] * (n - b), dtype=object)
    if gap:
        last_train = days[a - 1]
        label[a:b][days[a:b] < last_train + gap] = "gap"
        val = days[a:b][label[a:b] == "validation"]
        label[b:][days[b:] < (val.max() if len(val) else last_train) + gap] = "gap"
    return list(label)


def diary_scaled(part, diary):
    return part[diary] / pd.Series({c: DIARY_MAX.get(c, 5) for c in diary})


def scaled(part, features):
    """z-score each column with the mean and SD of the previous days only."""
    hist = part[features].shift()
    exp = hist.expanding(min_periods=MIN_HISTORY)
    sd = exp.std(ddof=0).where(lambda s: s > 1e-8)
    return ((part[features] - exp.mean()) / sd).clip(-CLIP_SD, CLIP_SD)


MODALITIES = ["heart rate", "nightly temperature", "heart rate variability", "respiratory rate", "sleep",
              "activity", "glucose", "symptoms"]




# ---------------------------------------------------------------- graph inputs

def graph_inputs(t, days, gap_days, name):
    fm = pd.DataFrame([{"feature": f, "modality": m, "source": SOURCE[m]} for f, m in MODALITY.items()])
    by_mod = {m: list(fm.feature[fm.modality == m]) for m in MODALITIES}
    width = max(len(f) for f in by_mod.values())
    sensor = list(fm.feature[fm.source != "diary"]); diary = list(fm.feature[fm.source == "diary"])
    X, P, F, Y, idx = [], [], [], [], []
    for (pid, rnd), part in t.groupby(PR):
        part = part.sort_values("day_in_study").reset_index(drop=True)
        v = pd.concat([scaled(part, sensor), diary_scaled(part, diary)], axis=1)
        n = len(part)
        nodes = np.zeros((n, len(MODALITIES), width), np.float32)
        fmask = np.zeros((n, len(MODALITIES), width), bool)
        for i, m in enumerate(MODALITIES):
            vals = v[by_mod[m]].to_numpy(float)
            fmask[:, i, :vals.shape[1]] = np.isfinite(vals)
            nodes[:, i, :vals.shape[1]] = np.nan_to_num(vals)
        present = fmask.any(axis=2)
        pad = days - 1
        nodes = np.concatenate([np.zeros((pad,) + nodes.shape[1:], np.float32), nodes])
        fmask = np.concatenate([np.zeros((pad,) + fmask.shape[1:], bool), fmask])
        present = np.concatenate([np.zeros((pad, len(MODALITIES)), bool), present])
        y = np.log1p(part[["lh", "estrogen"]].to_numpy(float))
        rows = np.flatnonzero(np.isfinite(y).any(axis=1))
        for j, s in zip(rows, split_by_time(part.day_in_study.to_numpy()[rows], gap_days)):
            X.append(nodes[j:j + days]); F.append(fmask[j:j + days]); P.append(present[j:j + days]); Y.append(y[j])
            idx.append({"id": pid, "study_interval": rnd, "day_in_study": int(part.day_in_study[j]), "split_by_time": s})
    out = OUT / f"cs224w_{name}"
    out.mkdir(parents=True, exist_ok=True)
    feature_names = np.array([by_mod[m] + [""] * (width - len(by_mod[m])) for m in MODALITIES])
    np.savez_compressed(out / "graph_inputs.npz", node_features=np.stack(X), feature_present=np.stack(F),
                        node_present=np.stack(P), y=np.array(Y, np.float32), modalities=np.array(MODALITIES),
                        feature_names=feature_names, targets=np.array(["log1p_lh", "log1p_e3g"]))
    pd.DataFrame(idx).to_csv(out / "examples_index.csv", index=False)
    Pn = np.stack(P)
    print(f"cs224w_{name}: node_features {np.stack(X).shape} (examples x days x modalities x features); "
          f"{Pn.mean():.0%} of nodes present; per modality: " + ", ".join(f"{m} {Pn[:, :, i].mean():.0%}" for i, m in enumerate(MODALITIES)))




def main():
    global ROUNDS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="the mcPHASES v1.0.0 folder")
    ap.add_argument("--rounds", choices=["2022", "both"], default="2022", help="2022 only (default) or both rounds")
    ap.add_argument("--days", type=int, default=7, help="k: days per example, including the target day")
    ap.add_argument("--gap-days", type=int, default=7, help="buffer between parts in the split_by_time labels")
    a = ap.parse_args()
    ROUNDS = a.rounds
    OUT.mkdir(parents=True, exist_ok=True)
    name = "2022_only" if ROUNDS == "2022" else "both_rounds"
    print("1. cleaning and daily features")
    t = build(a.data_dir)
    t.to_csv(OUT / f"daily_table_{name}.csv", index=False)
    pd.DataFrame(LOG).to_csv(OUT / f"cleaning_summary_{name}.csv", index=False)
    print(f"   {t.id.nunique()} participants, {t.lh.notna().sum():,} LH and {t.estrogen.notna().sum():,} E3G days")
    print("2-3. scaling and graph inputs")
    graph_inputs(t, a.days, a.gap_days, name)


if __name__ == "__main__":
    main()
