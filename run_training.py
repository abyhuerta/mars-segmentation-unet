"""
Master orchestrator: runs all four training methods sequentially, then generates plots.
Usage:  python run_training.py
Each method runs in its own subprocess so a crash in one does not abort the others.
Logs are written to logs/<method>/epoch_N.json by each training script.
A final timing summary is written to logs/run_summary.json.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

METHODS = [
    ("baseline",    "train_baseline.py"),
    ("amp",         "train_amp.py"),
    ("compile",     "train_compile.py"),
    ("streams",     "train_streams.py"),
    ("amp_compile", "train_amp_compile.py"),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_method(label: str, script: str) -> dict:
    script_path = os.path.join(SCRIPT_DIR, script)
    print(f"\n{'='*70}")
    print(f"  Starting [{label.upper()}]  —  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    wall_start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=SCRIPT_DIR,
    )
    wall_elapsed = time.perf_counter() - wall_start

    status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{label.upper()}] {status}  |  Total wall time: {wall_elapsed/60:.1f} min")

    return {
        "method":          label,
        "script":          script,
        "returncode":      result.returncode,
        "status":          status,
        "wall_time_sec":   wall_elapsed,
        "wall_time_min":   wall_elapsed / 60,
        "finished_at":     datetime.now().isoformat(),
    }


def main():
    overall_start = time.perf_counter()
    print(f"\nStarting full training run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Methods: {', '.join(m for m, _ in METHODS)}")
    print(f"Working directory: {SCRIPT_DIR}\n")

    os.makedirs(os.path.join(SCRIPT_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "checkpoints"), exist_ok=True)

    results = []
    for label, script in METHODS:
        entry = run_method(label, script)
        results.append(entry)

    # --- Generate plots ---
    plot_script = os.path.join(SCRIPT_DIR, "plot_results.py")
    if os.path.exists(plot_script):
        print(f"\n{'='*70}")
        print(f"  Generating plots  —  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")
        plot_result = subprocess.run(
            [sys.executable, plot_script],
            cwd=SCRIPT_DIR,
        )
        plot_status = "ok" if plot_result.returncode == 0 else f"FAILED (exit {plot_result.returncode})"
        print(f"\n[PLOTS] {plot_status}")
    else:
        plot_status = "skipped (plot_results.py not found)"
        print(f"\n[PLOTS] {plot_status}")

    overall_elapsed = time.perf_counter() - overall_start

    # --- Summary ---
    summary = {
        "run_date":            datetime.now().isoformat(),
        "total_wall_time_sec": overall_elapsed,
        "total_wall_time_hr":  overall_elapsed / 3600,
        "plot_status":         plot_status,
        "methods":             results,
    }
    summary_path = os.path.join(SCRIPT_DIR, "logs", "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'='*70}")
    print(f"  ALL DONE  —  Total: {overall_elapsed/3600:.2f} h  ({overall_elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\n{'Method':<12} {'Status':<22} {'Time (min)':>10}")
    print("-" * 46)
    for r in results:
        print(f"{r['method']:<12} {r['status']:<22} {r['wall_time_min']:>10.1f}")
    print(f"\nSummary saved to {summary_path}")
    print(f"Plots in {os.path.join(SCRIPT_DIR, 'results', 'plots')}\n")


if __name__ == "__main__":
    main()
