#!/usr/bin/env python3
"""
Phase 4 Benchmarking Analysis

Parses results from 4 experiments and generates comparative report:
- Loss curve comparison
- Convergence analysis
- Generation quality samples
- Performance metrics (speed, memory, etc.)
- Comparative table
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import re
import argparse


def parse_checkpoint_logs(checkpoint_dir):
    """Extract metrics from checkpoint folder"""
    metrics = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "generation_samples": [],
        "best_loss": None,
        "best_epoch": None
    }
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for loss logs
    loss_files = list(checkpoint_dir.glob("loss_*.json"))
    
    if not loss_files:
        # Try finding any JSON files
        loss_files = list(checkpoint_dir.glob("*.json"))
    
    for loss_file in loss_files:
        try:
            with open(loss_file, 'r') as f:
                data = json.load(f)
                # Try various JSON formats
                if "history" in data:
                    history = data["history"]
                    metrics["train_loss"] = history.get("train_loss", [])
                    metrics["val_loss"] = history.get("val_loss", [])
                    metrics["epochs"] = list(range(len(metrics["train_loss"])))
                elif "val_loss" in data:
                    metrics["val_loss"] = [data["val_loss"]]
                    metrics["train_loss"] = [data.get("train_loss", data["val_loss"])]
                    metrics["epochs"] = [0]
        except:
            pass
    
    # Calculate best loss
    if metrics["val_loss"]:
        best_idx = min(range(len(metrics["val_loss"])), key=lambda i: metrics["val_loss"][i])
        metrics["best_loss"] = metrics["val_loss"][best_idx]
        metrics["best_epoch"] = metrics["epochs"][best_idx]
    
    return metrics


def parse_training_log(log_file):
    """Extract key metrics from training.log written by train_full.py."""
    metrics = {
        "final_train_loss": None,
        "final_val_loss": None,
        "best_val_loss": None,
        "epochs_completed": 0,
        "train_loss": [],
        "val_loss": [],
        "generation_samples": [],
        "errors": []
    }

    if not Path(log_file).exists():
        return metrics

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    epoch_pattern = (
        r"Epoch\s+(\d+)\s+finished\s+\|\s+train_loss=([\d.]+)\s+\|\s+val_loss=([\d.]+)"
    )
    matches = re.findall(epoch_pattern, content)

    if matches:
        metrics["epochs_completed"] = int(matches[-1][0])
        metrics["train_loss"] = [float(m[1]) for m in matches]
        metrics["val_loss"] = [float(m[2]) for m in matches]
        metrics["final_train_loss"] = metrics["train_loss"][-1]
        metrics["final_val_loss"] = metrics["val_loss"][-1]
        metrics["best_val_loss"] = min(metrics["val_loss"])

    gen_pattern = r"Generated:\s*(.+?)(?:\n|$)"
    gen_matches = re.findall(gen_pattern, content)
    if gen_matches:
        metrics["generation_samples"] = gen_matches[:3]

    if re.search(r"Error|Exception|Traceback", content):
        metrics["errors"].append("Found error patterns in log")

    return metrics


def analyze_experiment(exp_dir, exp_key):
    """Analyze a single experiment directory"""
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        return None
    
    analysis = {
        "exp_key": exp_key,
        "exists": True,
        "checkpoint_dir": exp_path,
        "training_log": exp_path / "training.log",
    }

    if analysis["training_log"].exists():
        log_metrics = parse_training_log(str(analysis["training_log"]))
        analysis.update(log_metrics)

    json_metrics = parse_checkpoint_logs(str(exp_path))
    if json_metrics.get("val_loss"):
        analysis["checkpoint_metrics"] = json_metrics
    
    return analysis


def generate_comparison_report(analyses):
    """Generate comparative analysis report"""
    report = []
    report.append("="*100)
    report.append("PCLN PHASE 4 - BENCHMARKING COMPARISON REPORT")
    report.append("="*100)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary table
    report.append("SUMMARY METRICS")
    report.append("-"*100)
    report.append(f"{'Experiment':<25} {'Final Val Loss':<20} {'Best Val Loss':<20} {'Generation Sample':<35}")
    report.append("-"*100)
    
    for analysis in analyses:
        if not analysis:
            continue
        
        final_loss = analysis.get("final_val_loss", "N/A")
        best_loss = analysis.get("best_val_loss", "N/A")
        sample = ""
        if analysis.get("generation_samples"):
            sample = analysis["generation_samples"][0][:30] + "..."
        
        final_loss_str = f"{final_loss:.4f}" if isinstance(final_loss, float) else str(final_loss)
        best_loss_str = f"{best_loss:.4f}" if isinstance(best_loss, float) else str(best_loss)
        
        exp_name = analysis.get("exp_key", "Unknown").replace("_", " ").title()
        
        report.append(f"{exp_name:<25} {final_loss_str:<20} {best_loss_str:<20} {sample:<35}")
    
    report.append("-"*100)
    report.append("")
    
    # Detailed analysis
    report.append("DETAILED ANALYSIS")
    report.append("-"*100)
    
    for analysis in analyses:
        if not analysis:
            continue
        
        exp_name = analysis.get("exp_key", "Unknown").replace("_", " ").title()
        report.append(f"\n{exp_name}:")
        report.append(f"  Final training loss: {analysis.get('final_train_loss', 'N/A')}")
        report.append(f"  Final validation loss: {analysis.get('final_val_loss', 'N/A')}")
        report.append(f"  Best validation loss: {analysis.get('best_val_loss', 'N/A')}")
        report.append(f"  Epochs completed: {analysis.get('epochs_completed', 'N/A')}")
        
        if analysis.get("generation_samples"):
            report.append(f"  Generation samples:")
            for i, sample in enumerate(analysis["generation_samples"], 1):
                report.append(f"    {i}. {sample[:80]}")
        
        if analysis.get("errors"):
            report.append(f"  Errors/Warnings:")
            for error in analysis["errors"]:
                report.append(f"    • {error}")
    
    report.append("")
    report.append("-"*100)
    
    # Key findings
    report.append("\nKEY FINDINGS")
    report.append("-"*100)
    
    valid_analyses = [a for a in analyses if a and a.get("best_val_loss")]
    if valid_analyses:
        best_analysis = min(valid_analyses, key=lambda a: a.get("best_val_loss", float('inf')))
        best_name = best_analysis.get("exp_key", "").replace("_", " ").title()
        best_loss = best_analysis.get("best_val_loss")
        
        report.append(f"• Best validation loss achieved by: {best_name}")
        report.append(f"  Loss value: {best_loss:.4f}")
        
        # Compare to baseline
        baseline = next((a for a in analyses if a and "baseline" in a.get("exp_key", "")), None)
        if baseline and baseline.get("best_val_loss"):
            baseline_loss = baseline.get("best_val_loss")
            
            improvement = ((baseline_loss - best_loss) / baseline_loss) * 100
            report.append(f"• Improvement over baseline: {improvement:.2f}%")
    
    report.append("")
    report.append("="*100)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 4 Benchmarking Results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Results directory containing experiment folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for report (default: print to stdout and save as BENCHMARK_REPORT.md)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print(f"Analyzing results from: {results_dir}")
    
    # Analyze each experiment
    experiment_dirs = {
        "exp1_baseline": results_dir / "exp1_baseline",
        "exp2_char_level": results_dir / "exp2_char_level",
        "exp3_dynamic_neurons": results_dir / "exp3_dynamic_neurons",
        "exp4_all_features": results_dir / "exp4_all_features"
    }
    
    analyses = []
    found_count = 0
    
    for exp_key, exp_dir in experiment_dirs.items():
        if exp_dir.exists():
            found_count += 1
            analysis = analyze_experiment(exp_dir, exp_key)
            analyses.append(analysis)
            print(f"[OK] Found: {exp_key}")
        else:
            print(f"[TODO] Not found: {exp_key}")
            analyses.append(None)
    
    if found_count == 0:
        print("\nWarning: No experiment directories found!")
        return 1
    
    # Generate report
    report = generate_comparison_report(analyses)
    
    # Output
    output_file = Path(args.output) if args.output else results_dir / "BENCHMARK_REPORT.md"
    
    # Print to console
    print("\n" + report)
    
    # Save to file
    with open(output_file, "w") as f:
        f.write(report)
    
    print(f"\n[OK] Report saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
