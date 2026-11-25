"""Functions that run inside the worker processes.

CS4207 Coding Challenge, group 23: predict the payment method of a taxi trip
(card or cash). Like workers.py in the Week 9 lab, these live in their own
module so that ProcessPoolExecutor can import them in every child process.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one maths thread per process,
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # so workers do not fight over cores
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pickle
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CARD, CASH = 1, 2          # TLC payment_type codes; the label is 1 for cash, 0 for card
TRAIN, VAL, TEST = 0, 1, 2  # values of the "split" column

# Columns we read. tip_amount and total_amount are left out on purpose:
# card tips are recorded automatically but cash tips almost never are,
# so tip > 0 gives the answer away (data leakage).
WANTED = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
          "passenger_count", "trip_distance", "RatecodeID", "store_and_fwd_flag",
          "PULocationID", "DOLocationID", "payment_type", "fare_amount", "extra",
          "tolls_amount", "congestion_surcharge", "Airport_fee", "cbd_congestion_fee"]


def read_month(path):
    """Read the columns we need. Column names differ slightly between years
    (Airport_fee vs airport_fee, cbd_congestion_fee only exists from 2025),
    so match them case-insensitively and fill a missing fee column with 0."""
    available = {name.lower(): name for name in pq.ParquetFile(path).schema_arrow.names}
    present = [available[c.lower()] for c in WANTED if c.lower() in available]
    t = pd.read_parquet(path, columns=present)
    t.columns = [next(c for c in WANTED if c.lower() == name.lower()) for name in t.columns]
    for c in WANTED:
        if c not in t.columns:
            t[c] = 0.0
    return t


def prepare_month(path, out_path, role, sample_frac=1.0, val_frac=0.15, seed=4207):
    """Data parallelism: clean one monthly file and turn it into features.

    role is "train" (rows are split into train/validation) or "test"
    (every row is kept for the final evaluation). Writes a Parquet file of
    features and returns small summary counts, including per-zone cash counts
    from the TRAINING rows only (used later for the zone cash-rate features).
    """
    start = time.perf_counter()
    t = read_month(path)
    rows_in = len(t)

    t = t[t.payment_type.isin([CARD, CASH])]          # card or cash only
    dur = (t.tpep_dropoff_datetime - t.tpep_pickup_datetime).dt.total_seconds() / 60
    keep = ((t.fare_amount > 0) & (t.fare_amount < 500)
            & (t.trip_distance > 0) & (t.trip_distance < 100)
            & (dur >= 1) & (dur <= 180))
    t = t[keep].assign(duration_min=dur[keep])
    t = t.sort_values(["tpep_pickup_datetime", "PULocationID", "DOLocationID"],
                      kind="stable").reset_index(drop=True)   # fixed row order -> reproducible

    rng = np.random.default_rng(seed)
    if sample_frac < 1.0:                              # fixed random sample of rows
        t = t.iloc[np.sort(rng.permutation(len(t))[:int(sample_frac * len(t))])]
    if role == "train":
        split = np.where(rng.random(len(t)) < val_frac, VAL, TRAIN)
    else:
        split = np.full(len(t), TEST)

    pick = t.tpep_pickup_datetime
    hour, dow = pick.dt.hour, pick.dt.dayofweek
    rate = t.RatecodeID.fillna(1)
    f = pd.DataFrame({
        "trip_distance": t.trip_distance,
        "duration_min": t.duration_min,
        "speed_mph": (t.trip_distance / (t.duration_min / 60)).clip(upper=80),
        "fare_amount": t.fare_amount,
        "fare_per_mile": (t.fare_amount / t.trip_distance).clip(upper=50),
        "extra": t.extra.clip(lower=0),
        "tolls_amount": t.tolls_amount.clip(lower=0),
        "has_airport_fee": t.Airport_fee.fillna(0) > 0,
        "has_congestion_fee": t.congestion_surcharge.fillna(0) > 0,
        "has_cbd_fee": t.cbd_congestion_fee.fillna(0) > 0,
        "passenger_count": t.passenger_count.fillna(1).clip(0, 6),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_weekend": dow >= 5,
        "is_night": (hour >= 22) | (hour < 5),
        "is_rush_hour": (dow < 5) & hour.isin([7, 8, 9, 16, 17, 18, 19]),
        "same_zone": t.PULocationID == t.DOLocationID,
        "store_and_forward": t.store_and_fwd_flag == "Y",
        "rate_jfk": rate == 2,
        "rate_newark": rate == 3,
        "rate_nassau_westchester": rate == 4,
        "rate_negotiated": rate == 5,
        "rate_unknown": rate == 99,
        "vendor_1": t.VendorID == 1,
        "vendor_other": ~t.VendorID.isin([1, 2]),
        "PULocationID": t.PULocationID.astype("int16"),
        "DOLocationID": t.DOLocationID.astype("int16"),
        "split": split.astype("int8"),
        "cash": (t.payment_type == CASH).astype("int8"),
    }).reset_index(drop=True)
    bools = f.select_dtypes(bool).columns
    f[bools] = f[bools].astype("int8")
    f.to_parquet(out_path, index=False)

    tr = f[f.split == TRAIN]
    return {
        "path": path, "role": role, "pid": os.getpid(),
        "rows_in": rows_in, "rows_kept": len(f), "cash_share": float(f.cash.mean()),
        # per-zone counts from training rows only (no validation/test leakage)
        "pu_n": tr.groupby("PULocationID").size(), "pu_cash": tr.groupby("PULocationID").cash.sum(),
        "do_n": tr.groupby("DOLocationID").size(), "do_cash": tr.groupby("DOLocationID").cash.sum(),
        "seconds": time.perf_counter() - start,
    }


def make_model(kind, params, seed=4207):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=500, random_state=seed, **params))
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **params)
    raise ValueError(kind)


def best_threshold(y, p):
    """The probability cut-off that gives the best F1 for the cash class."""
    best = (0.0, 0.5)
    for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = p >= thr
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, float(thr))
    return best


def train_one_setting(name, kind, params, prep_dir, models_dir):
    """Task parallelism: one complete training run for one model setting.

    Fits on the training rows, scores on the validation rows and saves the
    fitted model. The test set is not touched here.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    start = time.perf_counter()
    X = np.load(f"{prep_dir}/train_X.npy", mmap_mode="r")
    y = np.load(f"{prep_dir}/train_y.npy")
    Xv = np.load(f"{prep_dir}/val_X.npy", mmap_mode="r")
    yv = np.load(f"{prep_dir}/val_y.npy")

    model = make_model(kind, params)
    model.fit(np.asarray(X), y)
    p = model.predict_proba(np.asarray(Xv))[:, 1]
    f1_tuned, thr = best_threshold(yv, p)
    with open(f"{models_dir}/{name}.pkl", "wb") as fh:
        pickle.dump(model, fh)
    return {"name": name, "kind": kind, "params": params,
            "val_auc": roc_auc_score(yv, p), "val_f1_at_0.5": f1_score(yv, p >= 0.5),
            "val_f1_tuned": f1_tuned, "threshold": thr,
            "seconds": time.perf_counter() - start, "pid": os.getpid()}
