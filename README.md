# CS4207 Coding Challenge: Group 23, predict the payment method (card or cash)

Given a yellow-taxi trip, predict whether it was paid by **card** (`payment_type` 1) or **cash** (`payment_type` 2).
The positive class is **cash**, the minority (about 12% of card-or-cash trips), so we report precision, recall and F1 for cash, not only accuracy.

## Files

| File | What it does |
|---|---|
| `payment_method.py` | Main script: runs Parts 1 to 4 and saves the results |
| `payment_workers.py` | Functions that run in the worker processes (same idea as `workers.py` in Week 9) |

## What the program does

1. **Prepare (data parallelism).** Each worker process takes one monthly Parquet file. It keeps card and cash trips only and removes refunds, zero or impossible distances, and trips under 1 minute or over 3 hours. It then builds the features. The training months are split 85/15 into train and validation, using a seed that depends on the shard. `--compare-serial` also times a one-process run, so you can report the speed-up.
2. **Encode.** Builds a smoothed cash rate for each pickup zone and each drop-off zone, computed from the **training rows only** to avoid data leakage. It then saves the X and y matrices.
3. **Model search (task parallelism).** Six independent training jobs: three logistic regressions and three histogram gradient-boosting models, with and without class balancing. Each one is scored on the validation set, and its probability threshold is tuned for the best cash F1. The best model is chosen on validation F1.
4. **Test.** The chosen model is scored **once** on a later month it has never seen. The output includes a confusion matrix, accuracy, precision, recall, F1 and ROC AUC, plus the "always guess card" baseline.

Training on earlier months and testing on a later month follows the Week 8 advice: train on the past, predict the future.

### Data leakage we avoid on purpose
- **`tip_amount` and `total_amount` are not used.** Card tips are recorded automatically, but cash tips almost never are. In January 2026, only 24 of 314,043 cash trips had a tip above 0. A model given the tip would score close to 100% while learning nothing useful.
- Zone cash rates use training rows only. Validation and test rows never feed into them.
- `payment_type` 0 ("flex fare / unknown"), 3 (no charge) and 4 (dispute) are removed. Type 0 rows also have no passenger count, rate code or surcharges.

## Running it on Rockhill (from a lab PC)

Open **PowerShell** on the lab PC. This is the *Laptop window*.

**1. Copy the project to Rockhill.** Run this from the folder that *contains* `cs4207-coding-challenge`:
```
scp -r cs4207-coding-challenge group23@rockhill.csis.ul.ie:~/
```

**2. Log in.** Open a second PowerShell window. This is the *Rockhill window*:
```
ssh group23@rockhill.csis.ul.ie
```
Type the group password. Nothing appears while you type, which is normal. The starting password is printed in the class team list, which everyone can read, so change it with `passwd` and tell your team the new one.

**3. Set up Python (first time only):**
```
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install pandas pyarrow scikit-learn matplotlib
```
On later logins you only need `source ~/venv/bin/activate`.

**4. Quick test.** Uses 5% of the rows and only two models, so it takes about a minute:
```
cd ~/cs4207-coding-challenge
python payment_method.py --quick --sample 0.05 --out results_quick
```

**5. Full run.** Trains on Jan to Mar 2026 and tests on Apr 2026. `tee` also saves the printout to a file for your report:
```
python payment_method.py --compare-serial 2>&1 | tee run_log.txt
```
Useful options: `--train 2026-01 2026-02 2026-03 2026-04 2026-05 --test 2026-06`, `--sample 0.5` and `--workers 8`.
Rockhill is shared and often busy (`uptime` shows the load), so keep `--workers` modest.

**6. Copy the results back** (in the Laptop window):
```
scp group23@rockhill.csis.ul.ie:~/cs4207-coding-challenge/results/metrics.json .
scp group23@rockhill.csis.ul.ie:~/cs4207-coding-challenge/results/model_search.csv .
scp group23@rockhill.csis.ul.ie:~/cs4207-coding-challenge/results/confusion_matrix.png .
scp group23@rockhill.csis.ul.ie:~/cs4207-coding-challenge/run_log.txt .
```

**7. Tidy up and log out.** `results/prepared` holds the large working matrices:
```
rm -rf ~/cs4207-coding-challenge/results/prepared ~/cs4207-coding-challenge/results/models
deactivate
exit
```

## Outputs (in `results/`)
- `metrics.json`: settings, data sizes, timings (serial vs parallel) and test metrics
- `model_search.csv`: validation scores and time for every model setting
- `logreg_weights.csv`: which features push a prediction towards cash or card
- `confusion_matrix.png`: test-set confusion matrix

## Reproducibility
- Monthly files are processed in sorted order.
- Every shard has a fixed seed (`4207 + shard number`), and rows are sorted before sampling.
- `pool.map` returns results in input order, whichever worker finishes first.
- Models use `random_state=4207`.

Running the same command twice gives the same numbers.
