#!/usr/bin/env python3
"""
Comprehensive Phase 4 Benchmarking Suite

Runs 4 experimental configurations sequentially:
1. Exp1: Baseline (word-level, dense neurons, no features)
2. Exp2: Character-level tokenization
3. Exp3: Dynamic neurons (word-level)
4. Exp4: All features combined (char-level + dynamic neurons + sparse MoE)

Collects metrics: loss curves, timing, generation samples, memory usage
Generates comparative analysis report
"""

import os
import sys
import subprocess
import json
import time
import copy
from datetime import datetime
from pathlib import Path
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration for each experiment
EXPERIMENTS = {
    "exp1_baseline": {
        "name": "Baseline (Word-Level, Dense)",
        "description": "Standard word-level tokenization + dense neurons (no dynamic neurons, no MoE)",
        "config": {
            "dataset": "wikitext2",
            "use_char_level": False,
            "use_dynamic_neurons": False,
            "use_sparse_moe": False,
            "epochs": 8,
            "batch_size": 8,
            "d_model": 256,
            "seq_len": 64,
            "learning_rate": 0.001,
            "num_train_samples": 1000,
            "num_val_samples": 100,
            "save_dir": "./results/exp1_baseline"
        }
    },
    "exp2_char_level": {
        "name": "Character-Level Tokenization",
        "description": "WikiText-2 with character-level encoding via get_wikitext2_char_dataloader",
        "config": {
            "dataset": "wikitext2",
            "use_char_level": True,
            "use_dynamic_neurons": False,
            "use_sparse_moe": False,
            "epochs": 8,
            "batch_size": 8,
            "d_model": 256,
            "seq_len": 64,
            "learning_rate": 0.001,
            "num_train_samples": 1000,
            "num_val_samples": 100,
            "save_dir": "./results/exp2_char_level"
        }
    },
    "exp3_dynamic_neurons": {
        "name": "Dynamic Neurons (Word-Level)",
        "description": "Standard word-level + dynamic neurons enabled (512 neurons, top-32)",
        "config": {
            "dataset": "wikitext2",
            "use_char_level": False,
            "use_dynamic_neurons": True,
            "num_neurons": 512,
            "top_k_neurons": 64,
            "use_sparse_moe": False,
            "epochs": 8,
            "batch_size": 8,
            "d_model": 256,
            "seq_len": 64,
            "learning_rate": 0.001,
            "num_train_samples": 1000,
            "num_val_samples": 100,
            "save_dir": "./results/exp3_dynamic_neurons"
        }
    },
    "exp4_all_features": {
        "name": "All Features Combined",
        "description": "Char-level WikiText-2 + dynamic neurons + sparse MoE (sequential PCN blocks)",
        "config": {
            "dataset": "wikitext2",
            "use_char_level": True,
            "use_dynamic_neurons": True,
            "num_neurons": 512,
            "top_k_neurons": 64,
            "use_sparse_moe": True,
            "num_experts": 4,
            "top_k_experts": 2,
            "epochs": 8,
            "batch_size": 8,
            "d_model": 256,
            "seq_len": 64,
            "learning_rate": 0.001,
            "num_train_samples": 1000,
            "num_val_samples": 100,
            "save_dir": "./results/exp4_all_features"
        }
    }
}


LOW_VRAM_NEURON_CONFIG = {"num_neurons": 128, "top_k_neurons": 32}
LOW_VRAM_VRAM_BYTES = 6 * 1024 ** 3


def detect_low_vram() -> bool:
    """True on GPUs with <6 GB VRAM (e.g. GTX 1650)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory < LOW_VRAM_VRAM_BYTES
    except Exception:
        pass
    return False


def apply_low_vram_settings(experiments: dict) -> None:
    """Shrink dynamic-neuron experiments so they fit 4 GB GPUs."""
    for key in ("exp3_dynamic_neurons", "exp4_all_features"):
        if key in experiments:
            experiments[key]["config"].update(LOW_VRAM_NEURON_CONFIG)


def build_cli_args(config):
    """Convert config dict to CLI arguments for train_full.py"""
    args = []
    
    for key, value in config.items():
        if key == "save_dir":
            # Use checkpoint-dir instead of save-dir
            args.extend(["--checkpoint-dir", str(value)])
        elif key == "use_char_level":
            if value:
                args.append("--use-char-level")
        elif key == "use_dynamic_neurons":
            if value:
                args.append("--use-dynamic-neurons")
        elif key == "use_sparse_moe":
            if value:
                args.append("--use-sparse-moe")
        elif key == "num_train_samples":
            args.extend(["--num-train-samples", str(value)])
        elif key == "num_val_samples":
            args.extend(["--num-val-samples", str(value)])
        elif key == "dataset":
            args.extend(["--dataset", str(value)])
        elif key == "epochs":
            args.extend(["--epochs", str(value)])
        elif key == "batch_size":
            args.extend(["--batch-size", str(value)])
        elif key == "d_model":
            args.extend(["--d-model", str(value)])
        elif key == "seq_len":
            args.extend(["--seq-len", str(value)])
        elif key == "learning_rate":
            args.extend(["--learning-rate", str(value)])
        else:
            # Handle other parameters
            if value is not None:
                arg_name = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        args.append(arg_name)
                else:
                    args.extend([arg_name, str(value)])
    
    return args


def run_experiment(exp_key, exp_info, log_file):
    """Run a single experiment and collect metrics"""
    print(f"\n{'='*80}")
    print(f"Starting: {exp_info['name']}")
    print(f"{'='*80}")
    print(f"Description: {exp_info['description']}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Build CLI arguments
    cli_args = build_cli_args(exp_info["config"])
    cmd = [sys.executable, "scripts/train_full.py"] + cli_args
    
    print(f"\nCommand:\n{' '.join(cmd)}\n")
    
    # Log command
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Experiment: {exp_info['name']}\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"{'='*80}\n\n")
    
    # Record timing
    start_time = time.time()
    
    try:
        # Run training script
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=14400  # 4 hour timeout per experiment
        )
        
        elapsed_time = time.time() - start_time
        
        # Log output
        with open(log_file, "a") as f:
            f.write(f"Elapsed time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)\n")
            f.write(f"Exit code: {result.returncode}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
            f.write(f"\n{'='*80}\n\n")
        
        # Print summary
        print(f"\n[OK] Completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        print(f"Exit code: {result.returncode}")
        
        if result.returncode != 0:
            print(f"[WARNING] Non-zero exit code (expected if training interrupted normally)")
        
        return {
            "exp_key": exp_key,
            "exp_name": exp_info["name"],
            "elapsed_time": elapsed_time,
            "exit_code": result.returncode,
            "timestamp": datetime.now().isoformat()
        }
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Experiment timed out after 4 hours")
        with open(log_file, "a") as f:
            f.write("Experiment timed out after 4 hours\n")
        return {
            "exp_key": exp_key,
            "exp_name": exp_info["name"],
            "elapsed_time": 14400,
            "exit_code": -1,
            "timeout": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[ERROR] Error running experiment: {e}")
        with open(log_file, "a") as f:
            f.write(f"Error: {e}\n")
        return {
            "exp_key": exp_key,
            "exp_name": exp_info["name"],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def pre_download_wikitext2(log_file):
    """Pre-download WikiText-2 to avoid downloading in first experiment"""
    print("\n" + "="*80)
    print("Phase A: Data Preparation - Pre-downloading WikiText-2")
    print("="*80)
    
    print("Attempting to download WikiText-2...")
    
    try:
        from src.data import download_wikitext2_direct
        
        start_time = time.time()
        
        # Try direct download first
        try:
            path = download_wikitext2_direct(split='train')
            elapsed = time.time() - start_time
            print(f"[OK] WikiText-2 pre-downloaded successfully ({elapsed:.1f}s)")
            print(f"  Location: {path}")
            
            with open(log_file, "a") as f:
                f.write(f"Data Preparation: WikiText-2 pre-downloaded in {elapsed:.1f}s\n")
            
            return True
        except Exception as e:
            print(f"[WARNING] Direct download didn't work: {e}")
            print("  Will fall back to standard loading during training")
            
            with open(log_file, "a") as f:
                f.write(f"Data Preparation: Direct download skipped ({e})\n")
            
            return False
            
    except Exception as e:
        print(f"[ERROR] Could not pre-download: {e}")
        print("  Data loading will happen during first training run")
        
        with open(log_file, "a") as f:
            f.write(f"Data Preparation: Error during download: {e}\n")
        
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 Benchmarking Suite")
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Experiments to run: 'all', 'exp1', 'exp2', 'exp3', 'exp4', or comma-separated"
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation phase"
    )
    parser.add_argument(
        "--continue-from",
        type=str,
        help="Continue from a specific experiment (skip previous ones)"
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Use 128 neurons for exp3/exp4 (required on GTX 1650 4GB)"
    )
    parser.add_argument(
        "--no-low-vram",
        action="store_true",
        help="Disable auto low-VRAM detection"
    )

    args = parser.parse_args()

    low_vram = args.low_vram or (not args.no_low_vram and detect_low_vram())
    active_experiments = copy.deepcopy(EXPERIMENTS)
    if low_vram:
        apply_low_vram_settings(active_experiments)
        print("Low VRAM mode: exp3/exp4 use num_neurons=128, top_k_neurons=32")
    
    # Setup logging
    log_dir = Path("results")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"benchmark_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"\n{'='*80}")
    print(f"PCLN Phase 4 - Comprehensive Benchmarking Suite")
    print(f"{'='*80}")
    print(f"Log file: {log_file}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Write header
    with open(log_file, "w") as f:
        f.write(f"PCLN Phase 4 Benchmarking Suite\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Experiments: {args.experiments}\n")
        f.write(f"  Skip data prep: {args.skip_data_prep}\n")
        f.write(f"  Continue from: {args.continue_from}\n")
        f.write(f"  Low VRAM mode: {low_vram}\n\n")
    
    # Determine which experiments to run
    if args.experiments == "all":
        exp_keys = list(EXPERIMENTS.keys())
    else:
        exp_keys = args.experiments.split(",")
    
    # Handle continue_from
    if args.continue_from:
        try:
            start_idx = exp_keys.index(args.continue_from)
            exp_keys = exp_keys[start_idx:]
            print(f"Continuing from {args.continue_from}")
        except ValueError:
            print(f"Warning: Unknown experiment '{args.continue_from}', running all")
    
    # Phase A: Data preparation
    if not args.skip_data_prep:
        pre_download_wikitext2(log_file)
    
    # Phases B-E: Run experiments
    results = []
    total_start_time = time.time()
    
    try:
        for exp_key in exp_keys:
            if exp_key not in active_experiments:
                print(f"Warning: Unknown experiment '{exp_key}', skipping")
                continue

            result = run_experiment(exp_key, active_experiments[exp_key], log_file)
            results.append(result)
            
            # Print interim summary
            print(f"\n{len(results)} / {len(exp_keys)} experiments completed")
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print("Benchmarking interrupted by user")
        print(f"{'='*80}")
        with open(log_file, "a") as f:
            f.write(f"\n\nBenchmarking interrupted by user at {datetime.now().isoformat()}\n")
    
    # Summary
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/3600:.1f} hours)")
    print(f"Experiments completed: {len(results)} / {len(exp_keys)}")
    
    for result in results:
        exp_name = result.get("exp_name", "Unknown")
        elapsed = result.get("elapsed_time", 0)
        print(f"  {exp_name}: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    print(f"\nLog file: {log_file}")
    print(f"Results: {Path('results').absolute()}")
    
    # Save JSON summary
    summary_file = log_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "started": datetime.now().isoformat(),
            "total_elapsed": total_elapsed,
            "experiments": results,
            "configuration": {
                "experiments": args.experiments,
                "skip_data_prep": args.skip_data_prep
            }
        }, f, indent=2)
    
    print(f"\n[OK] Summary JSON: {summary_file}")
    
    print(f"\n{'='*80}")
    print("Next steps:")
    print("  1. Review log file for any errors or warnings")
    print("  2. Check results/ folders for loss curves and checkpoints")
    print("  3. Run analysis_benchmarks.py to generate comparative report")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
