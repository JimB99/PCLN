#!/usr/bin/env python3
"""
Real-time monitoring of benchmark progress

Shows:
- Currently running experiment
- Elapsed time
- Estimated completion time
- Loss progress (if available)
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import argparse


def get_latest_log():
    """Find the most recent benchmark log"""
    log_dir = Path("results")
    log_files = sorted(log_dir.glob("benchmark_log_*.txt"), reverse=True)
    
    if log_files:
        return log_files[0]
    return None


def parse_log_progress(log_file):
    """Parse current progress from log file"""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Get last few lines to find current state
    recent_lines = lines[-50:]
    content = "".join(recent_lines)
    
    # Look for currently running experiment
    running_exp = None
    for exp_name in ["exp1_baseline", "exp2_char_level", "exp3_dynamic_neurons", "exp4_all_features"]:
        if exp_name in content:
            running_exp = exp_name
    
    return {
        "log_file": str(log_file),
        "current_exp": running_exp,
        "content": content,
        "last_update": datetime.fromtimestamp(log_file.stat().st_mtime)
    }


def check_training_progress(exp_dir):
    """Check progress of a training in progress"""
    results = {
        "exists": False,
        "has_training_log": False,
        "has_checkpoints": False,
        "latest_checkpoint": None
    }
    
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        return results
    
    results["exists"] = True
    
    # Check for training log
    log_file = exp_path / "training.log"
    if log_file.exists():
        results["has_training_log"] = True
        results["log_size_mb"] = log_file.stat().st_size / (1024 * 1024)
        results["last_modified"] = datetime.fromtimestamp(log_file.stat().st_mtime)
    
    # Check for checkpoints
    checkpoint_dir = exp_path / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            results["has_checkpoints"] = True
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            results["latest_checkpoint"] = latest.name
            results["num_checkpoints"] = len(checkpoints)
    
    return results


def monitor_benchmarks():
    """Main monitoring function"""
    log_file = get_latest_log()
    
    if not log_file:
        print("No benchmark logs found. Start benchmarking with: python scripts/run_benchmarks.py")
        return
    
    print(f"\n{'='*80}")
    print("BENCHMARK MONITORING")
    print(f"{'='*80}")
    print(f"Log file: {log_file}")
    print(f"Last update: {datetime.fromtimestamp(log_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse progress
    progress = parse_log_progress(log_file)
    current_exp = progress["current_exp"]
    
    if current_exp:
        print(f"\nCurrently running: {current_exp.replace('_', ' ').title()}")
    else:
        print("\nNo experiment currently running (or completed)")
    
    # Check each experiment's progress
    print(f"\n{'='*80}")
    print("EXPERIMENT PROGRESS")
    print(f"{'='*80}")
    
    experiments = ["exp1_baseline", "exp2_char_level", "exp3_dynamic_neurons", "exp4_all_features"]
    results_dir = Path("results")
    
    for exp in experiments:
        exp_dir = results_dir / exp
        status = check_training_progress(exp_dir)
        
        status_symbol = "[DONE]" if status["exists"] else "[TODO]"
        exp_name = exp.replace("_", " ").title()
        
        print(f"\n{status_symbol} {exp_name}")
        
        if status["exists"]:
            if status["has_training_log"]:
                print(f"  Training log: {status['log_size_mb']:.1f} MB")
                print(f"  Last modified: {status['last_modified'].strftime('%H:%M:%S')}")
            
            if status["has_checkpoints"]:
                print(f"  Checkpoints: {status['num_checkpoints']} saved")
                print(f"  Latest: {status['latest_checkpoint']}")
        else:
            print(f"  Not started")
    
    print(f"\n{'='*80}")
    print("To continue monitoring:")
    print(f"  python scripts/monitor_benchmarks.py")
    print(f"\nTo analyze completed results:")
    print(f"  python scripts/analyze_benchmarks.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor benchmark progress")
    parser.add_argument(
        "--watch",
        type=int,
        help="Refresh every N seconds (continuous monitoring mode)"
    )
    
    args = parser.parse_args()
    
    if args.watch:
        try:
            while True:
                monitor_benchmarks()
                print(f"Refreshing in {args.watch} seconds... (Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    else:
        monitor_benchmarks()
