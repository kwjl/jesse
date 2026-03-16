"""
jesse/research/autoresearch.py

Autonomous ML experiment loop for Jesse strategies.

Ports the autoresearch concept (Karpathy/autoresearch → autoresearch-mlx) into
Jesse's research module.  An AI agent edits a strategy's feature engineering,
estimator choice, and hyperparameters, then calls ``run_experiment()`` to get a
combined ML + backtest score.  Results are logged to a TSV and the agent
keeps or discards changes via git — exactly like autoresearch does for language
model training.

Public API
----------
run_experiment      – single experiment: gather ML data → train model → score
compute_score       – combine ML metrics and backtest metrics into one number
log_result          – append a row to the results TSV
load_results        – read back the TSV as a list of dicts
print_leaderboard   – pretty-print the top experiments
"""

from __future__ import annotations

import csv
import datetime
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .ml import gather_ml_data, train_model


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentConfig:
    """Knobs for a single autoresearch experiment.

    The caller (strategy notebook / agent script) fills this in and passes it
    to ``run_experiment()``.
    """

    # Jesse backtest plumbing — same shapes as research.backtest()
    config: dict = field(default_factory=dict)
    routes: List[Dict] = field(default_factory=list)
    data_routes: List[Dict] = field(default_factory=list)
    candles: dict = field(default_factory=dict)
    warmup_candles: Optional[dict] = None

    # ML settings
    estimator: Any = None          # sklearn-compatible estimator
    task: str = "binary"           # "binary" | "multiclass" | "regression"
    test_ratio: float = 0.2

    # Scoring weights (ML vs backtest)
    ml_weight: float = 0.4
    backtest_weight: float = 0.6

    # Where to save model artefacts (None = don't save)
    save_model_to: Optional[str] = None

    # Results log path
    results_tsv: str = "results.tsv"

    # Verbosity
    verbose: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Run one autonomous experiment iteration.

    Steps:
        1. Run a Jesse backtest in ML gather mode, collecting labelled features.
        2. Train the estimator on the gathered data.
        3. Compute a combined score from ML metrics and backtest metrics.
        4. Return everything the caller needs to decide keep vs discard.

    Parameters
    ----------
    cfg : ExperimentConfig
        Fully populated experiment configuration.

    Returns
    -------
    dict with keys:
        ``score``            – combined float (higher is better)
        ``ml_metrics``       – dict from ``train_model``
        ``backtest_metrics`` – dict from the Jesse backtest
        ``train_result``     – full ``train_model`` return dict
        ``gather_result``    – full ``gather_ml_data`` return dict
        ``data_points``      – number of ML samples collected
        ``wall_seconds``     – wall-clock time for this experiment
    """
    if cfg.estimator is None:
        raise ValueError("ExperimentConfig.estimator must be set.")

    t0 = time.time()

    # ── Step 1: gather labelled data via backtest ────────────────────────
    gather_result = gather_ml_data(
        config=cfg.config,
        routes=cfg.routes,
        data_routes=cfg.data_routes,
        candles=cfg.candles,
        warmup_candles=cfg.warmup_candles,
        csv_path=None,           # skip CSV write; we train in-memory
        verbose=cfg.verbose,
    )

    data_points = gather_result.get("data_points", [])
    backtest_metrics = gather_result.get("backtest_metrics", {})

    if not data_points:
        wall = time.time() - t0
        return {
            "score": 0.0,
            "ml_metrics": {},
            "backtest_metrics": backtest_metrics,
            "train_result": None,
            "gather_result": gather_result,
            "data_points": 0,
            "wall_seconds": wall,
        }

    # ── Step 2: train the model ──────────────────────────────────────────
    train_result = train_model(
        data=data_points,
        estimator=cfg.estimator,
        task=cfg.task,
        test_ratio=cfg.test_ratio,
        save_to=cfg.save_model_to,
        verbose=cfg.verbose,
    )

    ml_metrics = train_result.get("metrics", {})

    # ── Step 3: combined score ───────────────────────────────────────────
    score = compute_score(
        ml_metrics=ml_metrics,
        backtest_metrics=backtest_metrics,
        task=cfg.task,
        ml_weight=cfg.ml_weight,
        backtest_weight=cfg.backtest_weight,
    )

    wall = time.time() - t0

    if cfg.verbose:
        _print_experiment_summary(score, ml_metrics, backtest_metrics, cfg.task, wall, len(data_points))

    return {
        "score": score,
        "ml_metrics": ml_metrics,
        "backtest_metrics": backtest_metrics,
        "train_result": train_result,
        "gather_result": gather_result,
        "data_points": len(data_points),
        "wall_seconds": wall,
    }


def compute_score(
    ml_metrics: dict,
    backtest_metrics: dict,
    task: str = "binary",
    ml_weight: float = 0.4,
    backtest_weight: float = 0.6,
) -> float:
    """Combine ML quality and backtest performance into a single score.

    Both sub-scores are normalised to [0, 1] before weighting so neither
    metric can dominate purely by scale.

    ML sub-score (0–1):
        binary      → ROC AUC (already 0–1, 0.5 = random).
        multiclass  → accuracy (0–1).
        regression  → clamp(R², 0, 1).

    Backtest sub-score (0–1):
        A blend of Sharpe ratio (clamped to [0, 3] / 3) and
        net profit percentage (sigmoid-scaled around 0).

    The final score is ``ml_weight * ml_sub + backtest_weight * bt_sub``.
    Higher is always better.
    """
    # ── ML sub-score ─────────────────────────────────────────────────────
    if task == "binary":
        ml_sub = ml_metrics.get("roc_auc", 0.5)
    elif task == "multiclass":
        ml_sub = ml_metrics.get("accuracy", 0.0)
    else:  # regression
        ml_sub = max(0.0, min(1.0, ml_metrics.get("r2", 0.0)))

    # ── Backtest sub-score ───────────────────────────────────────────────
    sharpe = backtest_metrics.get("sharpe_ratio", 0.0)
    pnl_pct = backtest_metrics.get("net_profit_percentage", 0.0)
    total_trades = backtest_metrics.get("total", 0)

    # Normalise Sharpe: clamp to [0, 3], divide by 3
    sharpe_norm = max(0.0, min(3.0, sharpe)) / 3.0

    # Normalise PNL: sigmoid centred at 0, scaling factor 50
    pnl_norm = 1.0 / (1.0 + math.exp(-pnl_pct / 50.0))

    # Penalise if too few trades (not statistically meaningful)
    trade_penalty = min(1.0, total_trades / 30.0) if total_trades > 0 else 0.0

    bt_sub = (0.5 * sharpe_norm + 0.5 * pnl_norm) * trade_penalty

    # ── Combine ──────────────────────────────────────────────────────────
    total_weight = ml_weight + backtest_weight
    if total_weight == 0:
        return 0.0
    return (ml_weight * ml_sub + backtest_weight * bt_sub) / total_weight


def log_result(
    results_tsv: str,
    commit: str,
    score: float,
    ml_metric: float,
    backtest_sharpe: float,
    backtest_pnl: float,
    trades: int,
    status: str,
    description: str,
) -> None:
    """Append one experiment row to the results TSV.

    Creates the file with a header row if it doesn't exist.

    Columns:
        commit  score  ml_metric  sharpe  pnl_pct  trades  status  description
    """
    header = "commit\tscore\tml_metric\tsharpe\tpnl_pct\ttrades\tstatus\tdescription"
    write_header = not os.path.exists(results_tsv) or os.path.getsize(results_tsv) == 0

    with open(results_tsv, "a", newline="") as f:
        if write_header:
            f.write(header + "\n")
        row = (
            f"{commit}\t{score:.6f}\t{ml_metric:.6f}\t{backtest_sharpe:.4f}\t"
            f"{backtest_pnl:.2f}\t{trades}\t{status}\t{description}\n"
        )
        f.write(row)


def load_results(results_tsv: str) -> List[dict]:
    """Read back a results TSV into a list of dicts."""
    if not os.path.exists(results_tsv):
        return []
    rows = []
    with open(results_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({
                "commit": row.get("commit", ""),
                "score": float(row.get("score", 0)),
                "ml_metric": float(row.get("ml_metric", 0)),
                "sharpe": float(row.get("sharpe", 0)),
                "pnl_pct": float(row.get("pnl_pct", 0)),
                "trades": int(row.get("trades", 0)),
                "status": row.get("status", ""),
                "description": row.get("description", ""),
            })
    return rows


def print_leaderboard(results_tsv: str, top_n: int = 10) -> None:
    """Pretty-print the top experiments from a results TSV."""
    rows = load_results(results_tsv)
    kept = [r for r in rows if r["status"] == "keep"]
    kept.sort(key=lambda r: r["score"], reverse=True)
    display = kept[:top_n]

    if not display:
        print("  No kept experiments yet.")
        return

    W = 80
    print("\n" + "=" * W)
    print("  AUTORESEARCH LEADERBOARD".center(W))
    print("=" * W)
    print(
        f"  {'#':<4} {'Commit':<10} {'Score':>8} {'ML':>8} "
        f"{'Sharpe':>8} {'PNL%':>8} {'Trades':>7}  Description"
    )
    print(f"  {'─'*4} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}  {'─'*20}")
    for i, r in enumerate(display, 1):
        print(
            f"  {i:<4} {r['commit']:<10} {r['score']:>8.4f} {r['ml_metric']:>8.4f} "
            f"{r['sharpe']:>8.4f} {r['pnl_pct']:>7.2f}% {r['trades']:>7}  {r['description']}"
        )
    print("=" * W)


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _print_experiment_summary(
    score: float,
    ml_metrics: dict,
    backtest_metrics: dict,
    task: str,
    wall_seconds: float,
    n_samples: int,
) -> None:
    """Print a compact experiment summary, autoresearch-style."""
    W = 64
    print()
    print("─" * W)
    print("  EXPERIMENT RESULT")
    print("─" * W)

    # ML line
    if task == "binary":
        ml_label = "ROC AUC"
        ml_val = ml_metrics.get("roc_auc", 0.0)
        acc = ml_metrics.get("accuracy", 0.0)
        mcc = ml_metrics.get("mcc", 0.0)
        print(f"  ML:  {ml_label} = {ml_val:.4f}  |  Accuracy = {acc*100:.1f}%  |  MCC = {mcc:+.3f}")
    elif task == "multiclass":
        ml_label = "Accuracy"
        ml_val = ml_metrics.get("accuracy", 0.0)
        mcc = ml_metrics.get("mcc", 0.0)
        print(f"  ML:  {ml_label} = {ml_val*100:.1f}%  |  MCC = {mcc:+.3f}")
    else:
        mae = ml_metrics.get("mae", 0.0)
        r2 = ml_metrics.get("r2", 0.0)
        rho = ml_metrics.get("spearman", 0.0)
        print(f"  ML:  MAE = {mae:.6f}  |  R² = {r2:.4f}  |  Spearman = {rho:.4f}")

    # Backtest line
    sharpe = backtest_metrics.get("sharpe_ratio", 0.0)
    pnl = backtest_metrics.get("net_profit_percentage", 0.0)
    trades = int(backtest_metrics.get("total", 0))
    dd = backtest_metrics.get("max_drawdown", 0.0)
    print(f"  BT:  Sharpe = {sharpe:.3f}  |  PNL = {pnl:+.2f}%  |  Trades = {trades}  |  DD = {dd:+.2f}%")

    # Combined
    print(f"  ---")
    print(f"  combined_score: {score:.6f}")
    print(f"  data_points:    {n_samples}")
    print(f"  wall_seconds:   {wall_seconds:.1f}")
    print("─" * W)
