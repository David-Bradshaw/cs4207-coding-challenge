"""CS4207 Coding Challenge, group 23: predict whether a taxi trip is paid by card or cash.

Pipeline (each part prints what it did and how long it took):
  1. Prepare  - data parallelism: one worker per monthly file cleans it and builds features
  2. Encode   - zone cash rates from TRAINING rows only, then build X / y matrices
  3. Search   - task parallelism: each model setting is an independent training job,
                chosen by F1 for the cash class on the validation set
  4. Test     - the chosen model is scored once on a later month it has never seen

Example on Rockhill:
  python payment_method.py --train 2026-01 2026-02 2026-03 --test 2026-04
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

import payment_workers as pw

SEED = 4207
SMOOTHING = 200   # zones with few training trips are pulled towards the overall cash rate

# Every setting is one independent task for Part 3.
GRID = [
    ("logreg_C1", "logreg", {"C": 1.0}),
    ("logreg_C1_bal", "logreg", {"C": 1.0, "class_weight": "balanced"}),
    ("logreg_C0.01_bal", "logreg", {"C": 0.01, "class_weight": "balanced"}),
    ("hgb_100x31", "hgb", {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1}),
    ("hgb_200x63", "hgb", {"max_iter": 200, "max_leaf_nodes": 63, "learning_rate": 0.1}),
    ("hgb_200x63_bal", "hgb", {"max_iter": 200, "max_leaf_nodes": 63, "learning_rate": 0.1,
                               "class_weight": "balanced"}),
]


def parse_args():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--data-dir", default="/home/cs4207/data")
    a.add_argument("--pattern", default="yellow_tripdata_{month}.parquet",
                   help="file name pattern inside --data-dir")
    a.add_argument("--train", nargs="+", default=["2026-01", "2026-02", "2026-03"],
                   help="months used for training and validation")
    a.add_argument("--test", nargs="+", default=["2026-04"],
                   help="later month(s) kept back for the final test")
    a.add_argument("--sample", type=float, default=0.25,
                   help="fraction of training-month rows to use (keeps the shared server responsive)")
    a.add_argument("--workers", type=int, default=6,
                   help="worker processes; Rockhill is shared, so please keep this modest")
    a.add_argument("--out", default="results")
    a.add_argument("--compare-serial", action="store_true",
                   help="also run Part 1 with one process, to measure the parallel speed-up")
    a.add_argument("--quick", action="store_true", help="only the two fastest models (for testing)")
    return a.parse_args()


def part1_prepare(args, prep_dir):
    jobs = [(m, "train", args.sample) for m in sorted(args.train)] + \
           [(m, "test", 1.0) for m in sorted(args.test)]
    paths = [os.path.join(args.data_dir, args.pattern.format(month=m)) for m, _, _ in jobs]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("Taxi files not found:\n  " + "\n  ".join(missing))
    outs = [f"{prep_dir}/{m}_{role}.parquet" for m, role, _ in jobs]
    roles = [role for _, role, _ in jobs]
    fracs = [frac for _, _, frac in jobs]
    seeds = [SEED + i for i in range(len(jobs))]       # seed depends on the shard -> same split every run

    serial_s = None
    if args.compare_serial:
        t0 = time.perf_counter()
        list(map(pw.prepare_month, paths, outs, roles, fracs, [0.15] * len(jobs), seeds))
        serial_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        stats = list(pool.map(pw.prepare_month, paths, outs, roles, fracs,
                              [0.15] * len(jobs), seeds))   # map keeps input order -> reproducible
    parallel_s = time.perf_counter() - t0

    print(f"\n=== Part 1: prepare {len(jobs)} monthly files (data parallelism) ===")
    print(f"{'file':42} {'role':5} {'rows in':>10} {'rows kept':>10} {'cash %':>7} {'secs':>6} {'pid':>7}")
    for s in stats:
        print(f"{os.path.basename(s['path']):42} {s['role']:5} {s['rows_in']:>10,} "
              f"{s['rows_kept']:>10,} {100 * s['cash_share']:>6.1f}% {s['seconds']:>6.1f} {s['pid']:>7}")
    print(f"parallel wall time: {parallel_s:.1f}s with {min(args.workers, len(jobs))} workers")
    if serial_s:
        print(f"serial wall time:   {serial_s:.1f}s  ->  speed-up {serial_s / parallel_s:.2f}x")
    return stats, outs, {"prepare_parallel_s": parallel_s, "prepare_serial_s": serial_s}


def zone_rates(stats, prefix):
    """Smoothed cash rate per zone, added up over the training months only."""
    train = [s for s in stats if s["role"] == "train"]
    n = pd.concat([s[f"{prefix}_n"] for s in train], axis=1).sum(axis=1)
    cash = pd.concat([s[f"{prefix}_cash"] for s in train], axis=1).sum(axis=1)
    prior = cash.sum() / n.sum()
    return (cash + SMOOTHING * prior) / (n + SMOOTHING), prior


def part2_encode(stats, outs, prep_dir):
    t0 = time.perf_counter()
    pu_rate, prior = zone_rates(stats, "pu")
    do_rate, _ = zone_rates(stats, "do")
    df = pd.concat([pd.read_parquet(p) for p in outs], ignore_index=True)
    df["pu_zone_cash_rate"] = df.PULocationID.map(pu_rate).fillna(prior)
    df["do_zone_cash_rate"] = df.DOLocationID.map(do_rate).fillna(prior)
    features = [c for c in df.columns if c not in ("PULocationID", "DOLocationID", "split", "cash")]

    sizes = {}
    for name, code in (("train", pw.TRAIN), ("val", pw.VAL), ("test", pw.TEST)):
        part = df[df.split == code]
        np.save(f"{prep_dir}/{name}_X.npy", part[features].to_numpy(dtype=np.float32))
        np.save(f"{prep_dir}/{name}_y.npy", part.cash.to_numpy(dtype=np.int8))
        sizes[name] = (len(part), float(part.cash.mean()))

    print("\n=== Part 2: encode zones and build X / y ===")
    print(f"{len(features)} features: {', '.join(features)}")
    for name, (n, share) in sizes.items():
        print(f"  {name:5} {n:>10,} trips   cash share {100 * share:.1f}%")
    print(f"overall training cash rate {100 * prior:.1f}%  ({time.perf_counter() - t0:.1f}s)")
    return features, sizes


def part3_search(args, prep_dir, models_dir):
    grid = GRID[:1] + GRID[3:4] if args.quick else GRID
    names, kinds, params = zip(*grid)
    k = len(grid)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, k)) as pool:
        runs = list(pool.map(pw.train_one_setting, names, kinds, params,
                             [prep_dir] * k, [models_dir] * k))
    wall = time.perf_counter() - t0
    table = pd.DataFrame(runs)
    print("\n=== Part 3: model search (task parallelism) ===")
    print(table[["name", "val_auc", "val_f1_at_0.5", "val_f1_tuned", "threshold", "seconds", "pid"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sequential = table.seconds.sum()
    print(f"parallel wall time {wall:.1f}s; one-after-another estimate {sequential:.1f}s "
          f"-> speed-up about {sequential / wall:.2f}x")
    best = table.sort_values(["val_f1_tuned", "val_auc"], ascending=False).iloc[0]
    print(f"chosen on validation F1 (cash): {best['name']} at threshold {best['threshold']:.2f}")
    return table, best, {"search_parallel_s": wall, "search_sequential_estimate_s": sequential}


def part4_test(best, prep_dir, models_dir, features, out_dir):
    X = np.load(f"{prep_dir}/test_X.npy")
    y = np.load(f"{prep_dir}/test_y.npy")
    with open(f"{models_dir}/{best['name']}.pkl", "rb") as fh:
        model = pickle.load(fh)
    p = model.predict_proba(X)[:, 1]
    pred = (p >= best["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"model": best["name"], "threshold": float(best["threshold"]), "test_rows": int(len(y)),
         "accuracy": accuracy_score(y, pred), "precision_cash": precision_score(y, pred),
         "recall_cash": recall_score(y, pred), "f1_cash": f1_score(y, pred),
         "f1_macro": f1_score(y, pred, average="macro"), "roc_auc": roc_auc_score(y, p),
         "baseline_always_card_accuracy": float(1 - y.mean()),
         "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    print("\n=== Part 4: final test on the held-back month(s) ===")
    print(f"model {m['model']}  threshold {m['threshold']:.2f}  on {m['test_rows']:,} trips")
    print(f"baseline (always guess card) accuracy {m['baseline_always_card_accuracy']:.4f}, F1 cash 0")
    for key in ("accuracy", "precision_cash", "recall_cash", "f1_cash", "f1_macro", "roc_auc"):
        print(f"  {key:15} {m[key]:.4f}")
    print(f"                 predicted card   predicted cash")
    print(f"  actual card    {tn:>14,}   {fp:>14,}")
    print(f"  actual cash    {fn:>14,}   {tp:>14,}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(np.array([[tn, fp], [fn, tp]]), display_labels=["card", "cash"])
        disp.plot(values_format=",", cmap="Blues")
        plt.title(f"Payment method, test set ({m['model']})")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
        print(f"saved {out_dir}/confusion_matrix.png")
    except ImportError:
        print("(matplotlib not installed, skipping the confusion matrix picture)")
    return m


def explain_logreg(models_dir, features, table):
    """Show which features push towards cash, using the best logistic regression."""
    lr = table[table.kind == "logreg"].sort_values("val_f1_tuned", ascending=False)
    if lr.empty:
        return None
    with open(f"{models_dir}/{lr.iloc[0]['name']}.pkl", "rb") as fh:
        coefs = pd.Series(pickle.load(fh)[-1].coef_[0], index=features).sort_values()
    print(f"\nLogistic regression weights ({lr.iloc[0]['name']}, standardised features):")
    print("  towards card:", ", ".join(f"{k} {v:+.2f}" for k, v in coefs.head(5).items()))
    print("  towards cash:", ", ".join(f"{k} {v:+.2f}" for k, v in coefs.tail(5)[::-1].items()))
    return coefs


def main():
    args = parse_args()
    prep_dir, models_dir = f"{args.out}/prepared", f"{args.out}/models"
    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    start = time.perf_counter()

    stats, outs, t1 = part1_prepare(args, prep_dir)
    features, sizes = part2_encode(stats, outs, prep_dir)
    table, best, t3 = part3_search(args, prep_dir, models_dir)
    metrics = part4_test(best, prep_dir, models_dir, features, args.out)
    coefs = explain_logreg(models_dir, features, table)

    table.to_csv(f"{args.out}/model_search.csv", index=False)
    if coefs is not None:
        coefs.to_csv(f"{args.out}/logreg_weights.csv", header=["weight"])
    summary = {"args": vars(args), "features": features, "sizes": sizes,
               "timings": {**t1, **t3, "total_s": time.perf_counter() - start}, "test": metrics}
    with open(f"{args.out}/metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nsaved {args.out}/metrics.json, model_search.csv  (total {summary['timings']['total_s']:.0f}s)")


if __name__ == "__main__":
    main()
