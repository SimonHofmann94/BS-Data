"""Hyperparameter grid search script for learning rate and weight decay.

Runs all combinations of LR × WD × seeds in a single execution.

Usage:
    # Full grid search (default values)
    python code/experiments/run_hparam_sweep.py

    # Custom seeds
    python code/experiments/run_hparam_sweep.py --seeds 42 43 44

    # Resume after interruption
    python code/experiments/run_hparam_sweep.py --skip-completed

    # Preview without running
    python code/experiments/run_hparam_sweep.py --dry-run

Results saved under: code/experiments/results/hparam_summaries/<timestamped_sweep>/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import csv
import re

# ---------------------------------------------------------------------------
# Configuration of sweep levels (adjust as needed)
# ---------------------------------------------------------------------------
LR_VALUES = [1e-1, 1e-3, 5e-4]
WD_VALUES = [1e-4]
DEFAULT_SEEDS = [42, 43]

RESULTS_ROOT = Path("code/experiments/results")
SUMMARY_ROOT = RESULTS_ROOT / "hparam_summaries"
SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

# Potential metric keys to extract for convenience
METRIC_CANDIDATES = [
    "best_val_f1_macro",
    "val_f1_macro_best",
    "val_f1_macro",
    "f1_macro",
    "best_f1_macro"
]


def _slugify(text: str) -> str:
    """Sanitize a string so it can be used in a folder name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return slug.strip("_") or "default"


def _format_param_block(name: str, values: List[Any]) -> str:
    if not values:
        return name
    str_values = "-".join(f"{v}" for v in values)
    return _slugify(f"{name}_{str_values}")


def create_summary_dir(lr_values: List[float], wd_values: List[float], seeds: List[int], *, tag: str | None = None) -> Path:
    """Create a unique summary directory for the current sweep configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        _format_param_block("lr", lr_values),
        _format_param_block("wd", wd_values),
        _format_param_block("seeds", seeds)
    ]
    if tag:
        parts.append(_slugify(tag))
    descriptor = "__".join(parts)
    summary_dir = SUMMARY_ROOT / f"sweep_{timestamp}__{descriptor}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_run_dir(base_dir: Path) -> Path | None:
    """Find the most recently created experiment directory."""
    exp_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("experiment_")]
    if not exp_dirs:
        return None
    # Sort by creation time
    latest_dir = max(exp_dirs, key=lambda d: d.stat().st_ctime)
    return latest_dir


def _sanitize_for_dir_name(name: str) -> str:
    """Match train.py's experiment name sanitization for directory names."""
    return "".join(
        ch if (ch.isalnum() or ch in ("-", "_", ".", "=")) else "_"
        for ch in str(name)
    )


def find_latest_run_dir_for_experiment(base_dir: Path, experiment_name: str) -> Path | None:
    """Find the most recent run directory for a specific experiment name.
    Train script formats as: experiment_<sanitized_name>_<timestamp>
    """
    if not base_dir.exists():
        return None
    sanitized = _sanitize_for_dir_name(experiment_name)
    prefix = f"experiment_{sanitized}_"
    candidates = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_ctime)


def _find_existing_run(base_dir: Path, experiment_name: str) -> Path | None:
    """Return an existing completed run dir for this experiment if present."""
    run_dir = find_latest_run_dir_for_experiment(base_dir, experiment_name)
    if run_dir and (run_dir / "results.json").exists():
        return run_dir
    return None


def run_training(experiment_name: str, lr: float, wd: float, seed: int, *, skip_completed: bool = False, force_rerun: bool = False, dry_run: bool = False, quick_test: bool = False) -> Path:
    """Run a single training job via subprocess and return its output directory.
    Hydra's run directory pattern includes experiment.name so we scan for it.
    """
    existing = _find_existing_run(RESULTS_ROOT, experiment_name)
    if existing and (existing / "results.json").exists() and skip_completed and not force_rerun:
        print(f"[SKIP] {experiment_name} already completed -> {existing}")
        return existing

    command = [
        sys.executable, "code/train.py",
        f"optimizer.lr={lr}",
        f"optimizer.weight_decay={wd}",
        # Also set training.* in case the trainer reads from training section
        f"training.learning_rate={lr}",
        f"training.weight_decay={wd}",
        f"experiment.seed={seed}",
        f"experiment.name={experiment_name}",  # Pass name for logging and metadata
    ]
    
    # Quick test mode: minimal epochs and batch size
    if quick_test:
        command.extend([
            "training.num_epochs=2",
            "data.batch_size=32",
            "training.early_stopping_patience=1"
        ])

    if dry_run:
        print(f"[DRY-RUN] {' '.join(command)}")
        # Simulate directory
        return existing if existing else Path("/tmp/dry_run")

    print(f"\n[RUN] {' '.join(command)}")
    start_time = time.time()
    try:
        # Execute training; the script creates a directory: experiment_<name>_<ts>
        subprocess.run(command, check=True)

        # Find the directory that was just created for this experiment
        latest_run_dir = find_latest_run_dir_for_experiment(RESULTS_ROOT, experiment_name)

        if latest_run_dir:
            print(f"✅ SUCCESS: Training for {experiment_name} completed.")
            print(f"   - Output directory: {latest_run_dir}")
            
            # Add a README to the folder for clarity
            readme_content = f"""
# Experiment: {experiment_name}

This directory contains the results for a single training run with the following hyperparameters:

- **Learning Rate**: `{lr}`
- **Weight Decay**: `{wd}`
- **Seed**: `{seed}`
- **Quick Test**: `{quick_test}`

- **Execution Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            with open(latest_run_dir / "README.md", "w") as f:
                f.write(readme_content)
                
            return latest_run_dir
        else:
            print(f"❌ FAILED: Could not find output directory for {experiment_name} in {RESULTS_ROOT}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error during training for {experiment_name}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during training for {experiment_name}: {e}")
        return None


def extract_metric(results_json: Dict[str, Any]) -> tuple[float | None, str | None]:
    """Try extracting a macro F1 (or similar) metric from results.json.
    Returns (value, key_path) where key_path may be nested like 'val.best_f1_macro'.
    """
    # Direct level
    for key in METRIC_CANDIDATES:
        if key in results_json and isinstance(results_json[key], (int, float)):
            return float(results_json[key]), key

    # Nested search (one level deep)
    for parent, v in results_json.items():
        if isinstance(v, dict):
            for key in METRIC_CANDIDATES:
                if key in v and isinstance(v[key], (int, float)):
                    return float(v[key]), f"{parent}.{key}"
    return None, None


def summarize_run(run_dir: Path, lr: float, wd: float, seed: int, phase: str) -> Dict[str, Any]:
    results_file = run_dir / "results.json"
    data: Dict[str, Any] = {}
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed reading {results_file}: {e}")
    metric_val, metric_key = extract_metric(data)
    return {
        "phase": phase,
        "experiment_dir": str(run_dir),
        "lr": lr,
        "weight_decay": wd,
        "seed": seed,
        "metric_key_used": metric_key,
        "metric_value": metric_val,
        "raw_results": data
    }


def write_aggregate(summaries: List[Dict[str, Any]], phase: str, lr_values: List[float], wd_values: List[float], seeds: List[int], summary_dir: Path) -> None:
    json_path = summary_dir / f"{phase}_sweep_summary.json"
    csv_path = summary_dir / f"{phase}_sweep_summary.csv"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[SAVE] JSON summary written: {json_path} Total runs: {len(summaries)}")

    # CSV concise view
    fieldnames = ["phase", "lr", "weight_decay", "seed", "metric_value", "experiment_dir", "metric_key_used"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"[SAVE] CSV summary written: {csv_path}")
    
    # Create README for summary directory
    readme_path = summary_dir / "README.md"
    readme_content = f"""# Hyperparameter Sweep Summary

## Overview
This directory contains aggregated results from the hyperparameter grid search experiment, which you can find singularly in the respective run directories.

## Configuration
- **Learning Rates**: {lr_values}
- **Weight Decays**: {wd_values}
- **Seeds per Configuration**: {seeds}
- **Total Runs**: {len(summaries)}

## Created
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    try:
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"[SAVE] Summary README written: {readme_path}")
    except Exception as e:
        print(f"[WARN] Failed to create summary README: {e}")


def aggregate_stats(summaries: List[Dict[str, Any]], param_name: str) -> Dict[str, Any]:
    """Compute mean/std per hyperparameter level."""
    from math import sqrt
    stats: Dict[str, Any] = {}
    by_level: Dict[str, List[float]] = {}
    for s in summaries:
        mv = s.get("metric_value")
        if mv is None:
            continue
        level_key = f"{param_name}={s[param_name]}"
        by_level.setdefault(level_key, []).append(mv)
    for level, vals in by_level.items():
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, (len(vals) - 1))
        stats[level] = {
            "n": len(vals),
            "mean": mean,
            "std": sqrt(var),
            "cv": (sqrt(var) / mean) if mean != 0 else None,
            "min": min(vals),
            "max": max(vals)
        }
    return stats


# ---------------------------------------------------------------------------
# Main grid search logic
# ---------------------------------------------------------------------------

def run_grid_search(seeds: List[int], skip_completed: bool, force_rerun: bool, dry_run: bool, quick_test: bool, lr_values: List[float], wd_values: List[float], summary_dir: Path) -> None:
    """Full grid search: all LR × WD combinations."""
    summaries: List[Dict[str, Any]] = []
    total_runs = len(lr_values) * len(wd_values) * len(seeds)
    
    print("\n" + "="*70)
    print("GRID SEARCH CONFIGURATION")
    print("="*70)
    print(f"Learning rates: {lr_values}")
    print(f"Weight decays: {wd_values}")
    print(f"Seeds: {seeds}")
    print(f"\nTotal combinations: {len(lr_values)} LR × {len(wd_values)} WD × {len(seeds)} seeds = {total_runs} runs")
    if quick_test:
        print(f"⚡ QUICK TEST MODE: 2 epochs per run")
        print(f"Estimated time: ~{total_runs * 0.5:.1f} minutes")
    else:
        print(f"Estimated time: ~{total_runs * 1.5:.1f} hours (~{total_runs * 1.5 / 24:.1f} days)")
    print("="*70 + "\n")
    
    # Print execution order
    print("EXECUTION ORDER:")
    print("-" * 70)
    run_count = 0
    for lr in lr_values:
        for wd in wd_values:
            for seed in seeds:
                run_count += 1
                exp_name = f"grid_lr{lr}_wd{wd}_s{seed}"
                print(f"{run_count:3d}. {exp_name:<40} (lr={lr}, wd={wd}, seed={seed})")
    print("="*70 + "\n")
    
    if dry_run:
        print("[DRY-RUN MODE] Preview of runs:\n")
        return
    
    run_count = 0
    for lr in lr_values:
        for wd in wd_values:
            for seed in seeds:
                run_count += 1
                exp_name = f"grid_lr{lr}_wd{wd}_s{seed}"
                print(f"\n[Progress: {run_count}/{total_runs}] Starting {exp_name}")
                run_dir = run_training(exp_name, lr=lr, wd=wd, seed=seed, skip_completed=skip_completed, force_rerun=force_rerun, dry_run=dry_run, quick_test=quick_test)
                
                if run_dir is None:
                    print(f"[WARN] Skipping summary for failed run: {exp_name}")
                    continue
                
                summaries.append(summarize_run(run_dir, lr=lr, wd=wd, seed=seed, phase="grid"))
    
    write_aggregate(summaries, phase="grid", lr_values=lr_values, wd_values=wd_values, seeds=seeds, summary_dir=summary_dir)
    
    # Stats by LR
    lr_stats = aggregate_stats(summaries, param_name="lr")
    lr_stats_path = summary_dir / "grid_lr_stats.json"
    with open(lr_stats_path, "w") as f:
        json.dump(lr_stats, f, indent=2)
    print(f"[SAVE] Per-LR stats: {lr_stats_path}")
    
    # Stats by WD
    wd_stats = aggregate_stats(summaries, param_name="weight_decay")
    wd_stats_path = summary_dir / "grid_wd_stats.json"
    with open(wd_stats_path, "w") as f:
        json.dump(wd_stats, f, indent=2)
    print(f"[SAVE] Per-WD stats: {wd_stats_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search for learning rate and weight decay.")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="List of seeds (default: 5 preset seeds)")
    p.add_argument("--skip-completed", action="store_true", help="Skip runs with existing results.json")
    p.add_argument("--force-rerun", action="store_true", help="Re-run even if results exist")
    p.add_argument("--dry-run", action="store_true", help="Preview without executing")
    p.add_argument("--quick-test", action="store_true", help="Fast test mode: 2 epochs, 1 LR, 1 WD, 1 seed (~30 seconds)")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Quick test mode: minimal configuration
    if args.quick_test:
        seeds = [42]  # Just 1 seed
        lr_values = [0.001]  # Just 1 learning rate
        wd_values = [1e-4]   # Just 1 weight decay
        print("\n⚡ QUICK TEST MODE ENABLED")
        print("   - 1 learning rate, 1 weight decay, 1 seed = 1 run")
        print("   - 2 epochs per run")
        print("   - Expected duration: ~30 seconds\n")
    else:
        seeds = args.seeds if args.seeds else DEFAULT_SEEDS
        lr_values = LR_VALUES
        wd_values = WD_VALUES
    
    summary_dir = create_summary_dir(lr_values, wd_values, seeds, tag="quick-test" if args.quick_test else None)
    print(f"Summaries will be stored in: {summary_dir}\n")
    
    run_grid_search(
        seeds=seeds,
        skip_completed=args.skip_completed,
        force_rerun=args.force_rerun,
        dry_run=args.dry_run,
        quick_test=args.quick_test,
        lr_values=lr_values,
        wd_values=wd_values,
        summary_dir=summary_dir
    )

    print("\n" + "="*70)
    print("SWEEP COMPLETED")
    print("="*70)
    print(f"Results saved to: {summary_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
