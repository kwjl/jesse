# autoresearch — Jesse ML

Autonomous ML experiment loop for Jesse trading strategies. This is a port of Karpathy's autoresearch concept into Jesse's research module. Instead of training a language model, you are training and tuning ML-enhanced trading strategies.

**The goal: maximise the combined score** (a weighted blend of ML prediction quality and backtest trading performance).

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar16`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files**:
   - This file (`autoresearch_program.md`) — the experiment protocol.
   - The target strategy file (e.g. `strategies/MyStrategy/__init__.py`) — the file you modify. Features, labels, estimator config, hyperparameters.
   - The experiment runner script (e.g. `run_experiment.py`) — the script that calls `autoresearch.run_experiment()` with the strategy configuration. You can also modify this.
   - `jesse/research/autoresearch.py` — the evaluation infrastructure. **Read-only.** Do not modify.
   - `jesse/research/ml.py` — the ML training pipeline. **Read-only.** Do not modify.
4. **Verify data exists**: Confirm the user has candle data loaded for the strategy's exchange/symbol/timeframe. If not, tell them to import candles first.
5. **Initialize results.tsv**: Create `results.tsv` with the header row. Run the experiment script once to establish YOUR baseline on the current strategy + data.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## What you can modify

You have two main files to experiment with:

### 1. The strategy file

The Jesse strategy is where ML features and labels are defined. You can modify:

- **Feature engineering** — what features are recorded via `self.record_features({...})`. Add new technical indicators, change lookback periods, combine features, remove noisy ones.
- **Label definition** — what the strategy records via `self.record_label(name, value)`. Change the label (e.g. fixed-threshold vs triple-barrier, different forward horizons).
- **ML inference logic** — how the trained model's predictions are used in trading decisions (entry/exit filters, position sizing, confidence thresholds).
- **Strategy hyperparameters** — anything in the strategy that affects trading behavior.

### 2. The experiment runner script

This is a Python script (typically a Jupyter notebook or `.py` file) that:
- Loads candle data
- Configures the `ExperimentConfig`
- Calls `run_experiment(cfg)`
- Logs results

You can modify:
- **Estimator choice** — swap between SVC, RandomForest, GradientBoosting, XGBoost, etc.
- **Estimator hyperparameters** — C, gamma, n_estimators, max_depth, etc.
- **Task type** — binary, multiclass, or regression.
- **Test ratio** — chronological train/test split.
- **Score weights** — `ml_weight` and `backtest_weight` to shift emphasis.

### What you CANNOT modify

- `jesse/research/autoresearch.py` — the evaluation infrastructure is fixed.
- `jesse/research/ml.py` — the ML training and metrics pipeline is fixed.
- `jesse/research/backtest.py` — the backtest engine is fixed.
- The candle data.

## The experiment runner script

Here is an example `run_experiment.py` to use as a starting point:

```python
"""
Autoresearch experiment runner for Jesse ML.

Usage: python run_experiment.py
"""
from jesse.research import get_candles, autoresearch
from jesse.research.autoresearch import ExperimentConfig, run_experiment
from sklearn.svm import SVC

# ── Backtest config ──────────────────────────────────────────────────────
config = {
    'starting_balance': 10_000,
    'fee': 0.001,
    'type': 'futures',
    'futures_leverage': 3,
    'futures_leverage_mode': 'cross',
    'exchange': 'Bybit USDT Perpetual',
    'warm_up_candles': 210,
}

routes = [
    {'exchange': 'Bybit USDT Perpetual', 'strategy': 'MyMLStrategy', 'symbol': 'BTC-USDT', 'timeframe': '4h'},
]

data_routes = []

# ── Load candles ─────────────────────────────────────────────────────────
candles = {}
warmup_candles = {}
for route in routes:
    key = f"{route['exchange']}-{route['symbol']}"
    result = get_candles(route['exchange'], route['symbol'], '1m',
                         '2024-01-01', '2024-12-31')
    candles[key] = {
        'exchange': route['exchange'],
        'symbol': route['symbol'],
        'candles': result['candles'],
    }
    warmup_candles[key] = {
        'exchange': route['exchange'],
        'symbol': route['symbol'],
        'candles': result['warmup_candles'],
    }

# ── Estimator (EDIT THIS) ───────────────────────────────────────────────
estimator = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

# ── Run experiment ───────────────────────────────────────────────────────
cfg = ExperimentConfig(
    config=config,
    routes=routes,
    data_routes=data_routes,
    candles=candles,
    warmup_candles=warmup_candles,
    estimator=estimator,
    task='binary',
    test_ratio=0.2,
    ml_weight=0.4,
    backtest_weight=0.6,
    results_tsv='results.tsv',
    verbose=True,
)

result = run_experiment(cfg)
```

## Output format

When the experiment finishes, it prints a summary like this:

```
────────────────────────────────────────────────────────────────
  EXPERIMENT RESULT
────────────────────────────────────────────────────────────────
  ML:  ROC AUC = 0.6234  |  Accuracy = 58.3%  |  MCC = +0.167
  BT:  Sharpe = 1.450  |  PNL = +23.45%  |  Trades = 87  |  DD = -12.30%
  ---
  combined_score: 0.543210
  data_points:    412
  wall_seconds:   34.2
────────────────────────────────────────────────────────────────
```

## Logging results

When an experiment is done, log it to `results.tsv` using `autoresearch.log_result()` or manually. The TSV has 8 columns:

```
commit	score	ml_metric	sharpe	pnl_pct	trades	status	description
```

1. git commit hash (short, 7 chars)
2. combined score (e.g. 0.543210)
3. primary ML metric (ROC AUC for binary, accuracy for multiclass, R² for regression)
4. Sharpe ratio from backtest
5. net profit percentage from backtest
6. total trades
7. status: `keep`, `discard`, or `crash`
8. short text description of what this experiment tried

Example:

```
commit	score	ml_metric	sharpe	pnl_pct	trades	status	description
a1b2c3d	0.412000	0.5200	0.340	+5.20	42	keep	baseline SVC rbf
e4f5g6h	0.543210	0.6234	1.450	+23.45	87	keep	add RSI divergence feature + increase C to 10
i7j8k9l	0.000000	0.0000	0.000	+0.00	0	crash	OOM on GradientBoosting with 500 estimators
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar16`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Choose an experiment: modify the strategy's feature engineering, label logic, estimator, or hyperparameters by editing the strategy file and/or runner script.
3. `git add <strategy_file> <runner_script> && git commit -m "experiment: <description>"`
4. Run the experiment: `python run_experiment.py > run.log 2>&1`
5. Read the results: `grep "combined_score:\|wall_seconds:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the TSV.
8. If the combined score improved (higher), `git add results.tsv && git commit --amend --no-edit` to include the log, advancing the branch.
9. If the combined score is equal or worse, record the discard commit hash, then `git reset --hard <previous kept commit>` to discard it cleanly.

## Experiment ideas

These are ordered roughly from easiest/fastest to most ambitious:

### Feature engineering (strategy file)
- Add momentum indicators: RSI, MACD histogram, Stochastic %K/%D
- Add volatility features: ATR, Bollinger Band width, Keltner channel position
- Add volume features: OBV slope, VWAP distance, volume profile
- Add multi-timeframe features: higher TF trend alignment, TF momentum divergence
- Remove noisy features: check the feature importance report, drop consistently low-ranked features
- Engineer composite features: RSI divergence, trend strength * volume confirmation

### Label engineering (strategy file)
- Change forward horizon: 5 candles → 10 → 20
- Try triple-barrier labels: combine TP, SL, and time exit into -1/0/+1
- Use continuous labels (regression): forward return, risk-adjusted return

### Estimator tuning (runner script)
- SVC with different kernels: 'linear', 'rbf', 'poly'
- SVC with different C: 0.1, 1.0, 10.0, 100.0
- RandomForestClassifier: vary n_estimators, max_depth, min_samples_leaf
- GradientBoostingClassifier: learning_rate, n_estimators, subsample
- XGBoost (if installed): very strong on tabular data
- LightGBM (if installed): fast, handles feature interactions well

### Inference logic (strategy file)
- Adjust confidence threshold: only take trades when predict_proba > 0.6 / 0.65 / 0.7
- Filter by trend alignment: only take ML signals in the direction of the higher-TF trend
- Position sizing by confidence: scale position size with model confidence

### Score tuning (runner script)
- Shift ml_weight / backtest_weight balance
- If ML accuracy is stuck, lean more on backtest weight to let trading logic drive improvement
- If backtest is noisy, lean more on ML weight to stabilise

## Timeouts and crashes

- Each experiment should complete in a few minutes. If it takes more than 15 minutes, kill it and treat as failure.
- If a run crashes due to a fixable bug (typo, missing import, wrong shape), fix and re-run.
- If the idea itself is fundamentally broken, log as `crash`, revert, and move on.

## NEVER STOP

Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the feature importance reports, try combining near-misses, try more radical changes. The loop runs until the human interrupts you, period.
