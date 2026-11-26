"""Functions that run inside the worker processes.

CS4207 Coding Challenge, group 23: predict the payment method of a taxi trip
(card or cash). Like workers.py in the Week 9 lab, these live in their own
module so that ProcessPoolExecutor can import them in every child process.
The notebook payment_method.ipynb calls them.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one maths thread per process,
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # so workers do not fight over cores
os.environ.setdefault("MKL_NUM_THREADS", "1")       # and results do not depend on thread count

import hashlib
import pickle
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CARD, CASH = 1, 2          # TLC payment_type codes; the label is 1 for cash, 0 for card
FLEX_FARE = 0              # payment_type 0 = Flex Fare trip (no card/cash label, many nulls)
TRAIN, VAL, TEST = 0, 1, 2  # values of the "split" column

# Cleaning thresholds (justified in the notebook).
MAX_FARE = 500.0       # dollars; above this is a meter or keying error
MAX_DISTANCE = 100.0   # miles; NYC trips are rarely > 100 mi, errors reach 200,000+ mi
MIN_MINUTES, MAX_MINUTES = 1.0, 180.0

# Columns we read. tip_amount and total_amount are left out on purpose:
# card tips are recorded automatically but cash tips almost never are,
# so tip > 0 gives the answer away (data leakage).
WANTED = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
          "passenger_count", "trip_distance", "RatecodeID", "store_and_fwd_flag",
          "PULocationID", "DOLocationID", "payment_type", "fare_amount", "extra",
          "tolls_amount", "congestion_surcharge", "Airport_fee", "cbd_congestion_fee"]

# Task-parallel feature extraction: each group is built by its own task and the
# groups are always joined in THIS order (a fixed column order).
GROUP_ORDER = ["trip", "time", "flags", "zones"]
GROUP_COLUMNS = {
    "trip": ["trip_distance", "duration_min", "speed_mph", "fare_amount", "fare_per_mile",
             "extra", "tolls_amount", "passenger_count"],
    "time": ["hour_sin", "hour_cos", "is_weekend", "is_night", "is_rush_hour"],
    "flags": ["has_airport_fee", "has_congestion_fee", "has_cbd_fee", "same_zone",
              "store_and_forward", "rate_jfk", "rate_newark", "rate_nassau_westchester",
              "rate_negotiated", "rate_unknown", "vendor_1", "vendor_other"],
    "zones": ["pu_zone_cash_rate", "do_zone_cash_rate"],
}
GROUP_INPUTS = {   # cleaned columns each feature task needs to read
    "trip": ["trip_distance", "duration_min", "fare_amount", "extra", "tolls_amount",
             "passenger_count"],
    "time": ["tpep_pickup_datetime"],
    "flags": ["Airport_fee", "congestion_surcharge", "cbd_congestion_fee", "PULocationID",
              "DOLocationID", "store_and_fwd_flag", "RatecodeID", "VendorID"],
    "zones": ["PULocationID", "DOLocationID"],
}


def digest(obj):
    """Short fingerprint of a DataFrame or array, used to check two results are identical."""
    if isinstance(obj, pd.DataFrame):
        data = pd.util.hash_pandas_object(obj, index=False).values.tobytes()
        data += ",".join(obj.columns).encode()
    else:
        data = np.ascontiguousarray(obj).tobytes()
    return hashlib.md5(data).hexdigest()[:12]


def read_month(path):
    """Read only the columns we need (column pruning). Names differ slightly between
    years (Airport_fee vs airport_fee, cbd_congestion_fee only exists from 2025),
    so match them case-insensitively using the file footer, and fill a missing fee with 0."""
    available = {name.lower(): name for name in pq.ParquetFile(path).schema_arrow.names}
    present = [available[c.lower()] for c in WANTED if c.lower() in available]
    t = pd.read_parquet(path, columns=present)
    t.columns = [next(c for c in WANTED if c.lower() == name.lower()) for name in t.columns]
    for c in WANTED:
        if c not in t.columns:
            t[c] = 0.0
    return t


def clean_month(path, month, out_path, role, sample_frac=1.0, seed=4207):
    """DATA PARALLELISM: the same cleaning function is applied to every monthly file (shard).

    role is "train", "val" (the validation window) or "test". Training and validation
    months are sampled (sample_frac); test months are normally kept in full. seed=None means unseeded (used only
    in the non-determinism demonstration). Writes the cleaned rows to out_path and
    returns small partial results that the main process combines:
    row counts per cleaning step, and per-zone trip/cash counts (training months only).
    """
    start = time.perf_counter()
    t = read_month(path)
    steps = [("rows in file", len(t))]

    def drop(label, bad):
        nonlocal t
        t = t[~bad]
        steps.append((label, int(bad.sum())))

    drop("Flex Fare trips (payment_type 0)", t.payment_type == FLEX_FARE)
    drop("other payment types (no charge, dispute, ...)", ~t.payment_type.isin([CARD, CASH]))
    first = pd.Timestamp(f"{month}-01")
    nxt = first + pd.offsets.MonthBegin(1)
    drop("pickup outside the file's month", (t.tpep_pickup_datetime < first)
         | (t.tpep_pickup_datetime >= nxt))
    drop(f"fare <= 0 or >= ${MAX_FARE:.0f}", (t.fare_amount <= 0) | (t.fare_amount >= MAX_FARE))
    drop(f"distance <= 0 or >= {MAX_DISTANCE:.0f} miles",
         (t.trip_distance <= 0) | (t.trip_distance >= MAX_DISTANCE))
    dur = (t.tpep_dropoff_datetime - t.tpep_pickup_datetime).dt.total_seconds() / 60
    drop(f"duration < {MIN_MINUTES:.0f} or > {MAX_MINUTES:.0f} min",
         (dur < MIN_MINUTES) | (dur > MAX_MINUTES))
    t = t.assign(duration_min=dur.loc[t.index])
    rows_clean = len(t)

    # Remaining nulls are filled with the normal value, and how many is recorded.
    nulls = {"passenger_count": int(t.passenger_count.isna().sum()),
             "RatecodeID": int(t.RatecodeID.isna().sum())}
    t = t.assign(passenger_count=t.passenger_count.fillna(1).clip(0, 6),
                 RatecodeID=t.RatecodeID.fillna(1),
                 Airport_fee=t.Airport_fee.fillna(0),
                 congestion_surcharge=t.congestion_surcharge.fillna(0),
                 cbd_congestion_fee=t.cbd_congestion_fee.fillna(0),
                 store_and_fwd_flag=(t.store_and_fwd_flag == "Y"))

    t = t.sort_values(["tpep_pickup_datetime", "PULocationID", "DOLocationID"],
                      kind="stable").reset_index(drop=True)   # fixed row order inside the shard

    rng = np.random.default_rng(seed)
    if sample_frac < 1.0:
        t = t.iloc[np.sort(rng.permutation(len(t))[:int(sample_frac * len(t))])]
    split = np.full(len(t), {"train": TRAIN, "val": VAL, "test": TEST}[role])

    out = t.drop(columns=["payment_type", "tpep_dropoff_datetime"]).assign(
        split=split.astype("int8"), cash=(t.payment_type == CASH).astype("int8").values)
    out = out.reset_index(drop=True)
    out.to_parquet(out_path, index=False)

    tr = out[out.split == TRAIN]
    return {
        "month": month, "role": role, "pid": os.getpid(),
        "steps": steps, "rows_clean": rows_clean, "rows_kept": len(out),
        "nulls_filled": nulls, "cash_share": float(out.cash.mean()),
        # partial results (counts) for the zone cash rates; training rows only
        "pu_n": tr.groupby("PULocationID").size(), "pu_cash": tr.groupby("PULocationID").cash.sum(),
        "do_n": tr.groupby("DOLocationID").size(), "do_cash": tr.groupby("DOLocationID").cash.sum(),
        "md5": digest(out), "seconds": time.perf_counter() - start,
    }


def build_feature_group(group, clean_paths, out_path, zone_tables=None):
    """TASK PARALLELISM (feature extraction): build ONE group of feature columns.

    The groups need different input columns and do not depend on each other, so they
    can run at the same time. Shards are read in the order given (a fixed order).
    """
    start = time.perf_counter()
    t = pd.concat([pd.read_parquet(p, columns=GROUP_INPUTS[group]) for p in clean_paths],
                  ignore_index=True)
    if group == "trip":
        f = pd.DataFrame({
            "trip_distance": t.trip_distance,
            "duration_min": t.duration_min,
            "speed_mph": (t.trip_distance / (t.duration_min / 60)).clip(upper=80),
            "fare_amount": t.fare_amount,
            "fare_per_mile": (t.fare_amount / t.trip_distance).clip(upper=50),
            "extra": t.extra.clip(lower=0),
            "tolls_amount": t.tolls_amount.clip(lower=0),
            "passenger_count": t.passenger_count,
        })
    elif group == "time":
        pick = t.tpep_pickup_datetime
        hour, dow = pick.dt.hour, pick.dt.dayofweek
        f = pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_weekend": dow >= 5,
            "is_night": (hour >= 22) | (hour < 5),
            "is_rush_hour": (dow < 5) & hour.isin([7, 8, 9, 16, 17, 18, 19]),
        })
    elif group == "flags":
        rate = t.RatecodeID   # rate code is a category: one 0/1 column per code
        f = pd.DataFrame({
            "has_airport_fee": t.Airport_fee > 0,
            "has_congestion_fee": t.congestion_surcharge > 0,
            "has_cbd_fee": t.cbd_congestion_fee > 0,
            "same_zone": t.PULocationID == t.DOLocationID,
            "store_and_forward": t.store_and_fwd_flag,
            "rate_jfk": rate == 2,
            "rate_newark": rate == 3,
            "rate_nassau_westchester": rate == 4,
            "rate_negotiated": rate == 5,
            "rate_unknown": rate == 99,
            "vendor_1": t.VendorID == 1,
            "vendor_other": ~t.VendorID.isin([1, 2]),
        })
    elif group == "zones":
        # ~260 zones is too many categories for one-hot, so each zone is replaced by its
        # smoothed cash rate, learned from TRAINING rows only (target encoding without leakage)
        pu_rate, do_rate, prior = zone_tables
        f = pd.DataFrame({
            "pu_zone_cash_rate": t.PULocationID.map(pu_rate).fillna(prior),
            "do_zone_cash_rate": t.DOLocationID.map(do_rate).fillna(prior),
        })
    else:
        raise ValueError(group)
    X = f[GROUP_COLUMNS[group]].to_numpy(dtype=np.float32)
    np.save(out_path, X)
    return {"group": group, "columns": GROUP_COLUMNS[group], "path": out_path,
            "rows": len(X), "md5": digest(X), "seconds": time.perf_counter() - start,
            "pid": os.getpid()}


def assemble_features(group_files, naive=False, rows=None):
    """Join the feature groups side by side into one matrix.

    naive=True reproduces a bug we are demonstrating: looping over a *set* of group
    names, whose order depends on Python's per-run string hashing.
    """
    order = list(set(group_files)) if naive else GROUP_ORDER
    X = np.hstack([np.load(group_files[g], mmap_mode="r")[:rows] for g in order])
    columns = [c for g in order for c in GROUP_COLUMNS[g]]
    return columns, X


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
    """The probability cut-off that gives the best F1 for the cash class (on validation data)."""
    best = (0.0, 0.5)
    for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = p >= thr
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, float(thr))
    return best


def train_one_setting(name, kind, params, prep_dir, models_dir=None, max_rows=None):
    """TASK PARALLELISM (hyperparameter search): one complete training run for one setting.

    Fits on the training rows, scores on the VALIDATION rows and (optionally) saves the
    model. The test month is never touched here. max_rows limits the training rows
    (used for the speed-up benchmark, so it does not take too long on the shared server).
    """
    from sklearn.metrics import f1_score, roc_auc_score
    start = time.perf_counter()
    X = np.load(f"{prep_dir}/train_X.npy", mmap_mode="r")
    y = np.load(f"{prep_dir}/train_y.npy")
    if max_rows:
        X, y = X[:max_rows], y[:max_rows]
    Xv = np.load(f"{prep_dir}/val_X.npy", mmap_mode="r")
    yv = np.load(f"{prep_dir}/val_y.npy")

    model = make_model(kind, params)
    model.fit(np.asarray(X), y)
    p = model.predict_proba(np.asarray(Xv))[:, 1]
    f1_tuned, thr = best_threshold(yv, p)
    if models_dir:
        with open(f"{models_dir}/{name}.pkl", "wb") as fh:
            pickle.dump(model, fh)
    return {"name": name, "kind": kind, "params": params,
            "val_auc": roc_auc_score(yv, p), "val_f1_at_0.5": f1_score(yv, p >= 0.5),
            "val_f1_tuned": f1_tuned, "threshold": thr,
            "seconds": time.perf_counter() - start, "pid": os.getpid()}


def noop(x):
    """Empty task, used to measure the cost of starting a process pool."""
    return x
