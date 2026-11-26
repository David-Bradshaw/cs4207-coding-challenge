# CS4207 Coding Challenge: Group 23, predict the payment method (card or cash)

Given a yellow-taxi trip, predict whether it was paid by **card** (`payment_type` 1) or **cash** (`payment_type` 2).
Cash is the minority (about 12% of card-or-cash trips), so we report precision, recall and F1 for cash, not only accuracy.

## Files

| File | What it does |
|---|---|
| `payment_method.ipynb` | **The solution notebook.** Runs everything, from cleaning to the final test, and keeps its outputs |
| `payment_workers.py` | Functions that run in the worker processes (same idea as `workers.py` in Week 9) |
| `requirements.txt` | Python packages needed |
| `outputs/` | Small result files written by the notebook (charts, CSV tables, `summary.json`) and used for the poster |
| `poster/` | A0 poster (PowerPoint and PDF) |

## What the notebook covers (rubric criteria)

| Part | Criterion |
|---|---|
| 0 | Configuration in one place; Python, library versions, cores and RAM recorded (criterion 6) |
| 1 | Footer inspection, column pruning, cleaning with stated thresholds, before/after row counts, Flex Fare nulls, spill-over pickups (criterion 1) |
| 1–2 | **Data parallelism:** one shard per month, partial zone counts merged in a fixed order, checked against the sequential result (criterion 3) |
| 3 | **Task parallelism #1, feature extraction:** 4 feature groups built as separate tasks and joined in a fixed column order (criterion 4) |
| 4 | **Task parallelism #2, hyperparameter search:** 6 independent model settings, best chosen on validation (criterion 4) |
| 5 | Sequential vs parallel: warm-up, 3 repeats, 1/2/4(/3/6) workers, speed-up and efficiency, chart, Amdahl's law, measured overheads, outputs checked equal (criterion 2) |
| 6 | Non-determinism: unseeded sampling, and set/hash ordering, each shown differing and then fixed (criterion 5) |
| 7–8 | Final test on April 2026 and a summary of every number used on the poster |

### Data leakage we avoid on purpose
- **`tip_amount` and `total_amount` are never read.** Card tips are recorded automatically, but cash tips almost never are, so a tip gives the answer away.
- Zone cash rates use training rows only.
- Time-based split: train and validate on January–March 2026, test on April 2026.

## Running it on Rockhill

Log in from a lab PC (`ssh group23@rockhill.csis.ul.ie`), then:

```
source ~/venv/bin/activate
cd ~/cs4207-coding-challenge
git fetch
git reset --hard origin/main
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute payment_method.ipynb --output payment_method.ipynb --ExecutePreprocessor.timeout=-1
```

This takes about 10–15 minutes. The notebook is saved **with its outputs**, and the files in `outputs/` are updated.
The first time only, create the virtual environment with `python3 -m venv ~/venv`.

To send the results back to GitHub (the password prompt needs a GitHub token, not your GitHub password):

```
git add payment_method.ipynb outputs
git commit -m "Notebook run on Rockhill"
git push
```

Then free the disk space: `rm -rf results`.

To test on other machines, set `TAXI_DATA_DIR`, `TAXI_PATTERN` and `TAXI_SAMPLE` before starting Jupyter. The defaults are the Rockhill paths.

## Reproducibility
- All settings and seeds are in the first code cell.
- Each shard has a fixed seed (`4207 + shard number`), and rows are stable-sorted before sampling.
- `pool.map` returns results in input order, whichever worker finishes first.
- Feature groups are joined in a fixed order (`GROUP_ORDER`).
- Each process uses one maths thread, and models use `random_state=4207`.
- `outputs/environment.json` records the library versions, hardware and run time.
