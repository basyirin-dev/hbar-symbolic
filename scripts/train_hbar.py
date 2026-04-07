#!/usr/bin/env python3
"""H-Bar Training Script for Conditions B (Additive) and C (Multiplicative).

This script executes the H-Bar interventions designed to bridge the ~30%
generalization gap found in the baseline and reach near-perfect (95-99%)
OOD accuracy predicted by the H-Bar paper.

Conditions:
    - Condition B (Additive): L_total = L_task + λ_σ(1-σ_A)L_comp
    - Condition C (Multiplicative): L_total = L_task · (1 + λ_σ(1-σ_A)L_comp)

The multiplicative coupling creates more aggressive training dynamics and
may lead to faster "crystallization" but could be prone to gradient instability.

Usage on Kaggle:
    # Single run (Condition C - default)
    !python scripts/train_hbar.py --domain scan --condition multiplicative

    # Pilot study (N=15 runs)
    !python scripts/train_hbar.py --domain scan --condition multiplicative --n_runs 15

    # Condition B (Additive)
    !python scripts/train_hbar.py --domain scan --condition additive --n_runs 5

Outputs:
    - hbar_{condition}_metrics.csv: Training metrics for each run
    - pilot_results_summary.csv: Aggregated results across all runs
    - model_params_{condition}_run_{n}.msgpack: Saved model parameters
"""

import argparse
import csv
import os
import sys
import time

import jax
import jax.numpy as jnp

# ============================================================================
# HARDWARE PRE-ALLOCATION FIX FOR JAX ON KAGGLE
# ============================================================================
# Kaggle GPUs (P100/T4) have limited memory. By default, JAX pre-allocates
# ~90% of GPU memory, which can cause OOM errors. This environment variable
# must be set BEFORE importing jax to enable dynamic memory allocation.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

from hbar.benchmarks.grammar_engine import GrammarEngine
from hbar.engine.evaluator import Evaluator
from hbar.engine.trainer import (
    run_hbar_training,
    run_hbar_training_multiplicative,
    save_params,
    HBarTrainingResults,
)
from hbar.models.config import TransformerConfig, FusionConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="H-Bar Training Script for Conditions B (Additive) and C (Multiplicative)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="scan",
        choices=["scan", "cogs"],
        help="Domain to train on (default: scan)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="multiplicative",
        choices=["additive", "multiplicative"],
        help="Loss coupling condition: additive (B) or multiplicative (C) (default: multiplicative)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of independent training runs with different seeds (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=5000,
        help="Total training steps (default: 5000)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Evaluation interval in steps (default: 500)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam (default: 1e-3)",
    )
    parser.add_argument(
        "--lambda-sigma",
        type=float,
        default=0.5,
        help="Maximum compositional penalty weight λ_σ (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory)",
    )
    return parser.parse_args()


def run_single_training(
    args: argparse.Namespace,
    seed: int,
    run_idx: int,
) -> HBarTrainingResults:
    """Execute a single H-Bar training run.

    Args:
        args: Parsed command-line arguments.
        seed: Random seed for this run.
        run_idx: Index of this run (for file naming).

    Returns:
        HBarTrainingResults containing final parameters and metrics.
    """
    print(f"\n{'='*60}")
    print(f"Run {run_idx + 1}/{args.n_runs} (seed={seed})")
    print(f"Condition: {args.condition.upper()}")
    print(f"{'='*60}")

    # Set random seed
    rng = jax.random.PRNGKey(seed)

    # Initialize components
    grammar_engine = GrammarEngine(seed=seed)
    evaluator = Evaluator(domain=args.domain, data_dir="data")

    tokenizer = grammar_engine.get_tokenizer(args.domain)
    vocab_size = len(tokenizer.word2id)
    max_seq_len = 50 if args.domain == "scan" else 80

    # Create model configuration with FusionConfig
    fusion_cfg = FusionConfig(
        w_gca=0.4,
        w_rga=0.35,
        w_ac=0.25,
        target_sigma_critical=0.5,
        kappa_alpha=2.0,
    )
    config = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        dropout_rate=0.1,
        fusion_config=fusion_cfg,
    )

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model: {config.n_layers} layers, {config.n_heads} heads, d_model={config.d_model}")

    # Determine output filenames
    condition_prefix = "hbar_additive" if args.condition == "additive" else "hbar_multiplicative"
    metrics_filename = f"{condition_prefix}_run_{run_idx}_metrics.csv"
    params_filename = f"model_params_{condition_prefix}_run_{run_idx}.msgpack"

    # Run training
    start_time = time.time()

    rng, train_rng = jax.random.split(rng)
    if args.condition == "additive":
        results = run_hbar_training(
            config=config,
            grammar_engine=grammar_engine,
            evaluator=evaluator,
            rng=train_rng,
            batch_size=args.batch_size,
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            learning_rate=args.learning_rate,
            lambda_sigma=args.lambda_sigma,
            log_dir=args.output_dir,
            log_filename=metrics_filename,
        )
    else:
        results = run_hbar_training_multiplicative(
            config=config,
            grammar_engine=grammar_engine,
            evaluator=evaluator,
            rng=train_rng,
            batch_size=args.batch_size,
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            learning_rate=args.learning_rate,
            lambda_sigma=args.lambda_sigma,
            log_dir=args.output_dir,
            log_filename=metrics_filename,
        )

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

    # Save model parameters
    params_path = os.path.join(args.output_dir, params_filename)
    save_params(results.final_params, params_path)

    # Print final summary
    if results.metrics_history:
        final_metrics = results.metrics_history[-1]
        print(f"\nFinal Metrics (Step {final_metrics.step}):")
        print(f"  ID Accuracy:    {final_metrics.id_accuracy:.4f}")
        print(f"  OOD Accuracy:   {final_metrics.ood_accuracy:.4f}")
        print(f"  σ̂_A (sigma):    {final_metrics.sigma_ode:.4f}")
        print(f"  α_A:            {final_metrics.alpha_A:.4f}")

    return results


def init_summary_csv(output_dir: str, condition: str) -> None:
    """Initialize the pilot results summary CSV file.

    Args:
        output_dir: Directory for output files.
        condition: 'additive' or 'multiplicative'.
    """
    summary_path = os.path.join(output_dir, "pilot_results_summary.csv")

    # Only write header if file doesn't exist
    file_exists = os.path.exists(summary_path)
    fieldnames = [
        "run_idx",
        "seed",
        "condition",
        "domain",
        "final_id_accuracy",
        "final_ood_accuracy",
        "final_sigma_hat",
        "final_alpha_A",
        "phase2_entry_step",
        "elapsed_time",
    ]

    with open(summary_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # We'll append rows after each run


def append_to_summary(
    output_dir: str,
    run_idx: int,
    seed: int,
    condition: str,
    domain: str,
    results: HBarTrainingResults,
    elapsed_time: float,
) -> None:
    """Append a single run's results to the summary CSV.

    Args:
        output_dir: Directory for output files.
        run_idx: Index of this run.
        seed: Random seed used.
        condition: 'additive' or 'multiplicative'.
        domain: 'scan' or 'cogs'.
        results: Training results.
        elapsed_time: Time taken for this run in seconds.
    """
    summary_path = os.path.join(output_dir, "pilot_results_summary.csv")
    fieldnames = [
        "run_idx",
        "seed",
        "condition",
        "domain",
        "final_id_accuracy",
        "final_ood_accuracy",
        "final_sigma_hat",
        "final_alpha_A",
        "phase2_entry_step",
        "elapsed_time",
    ]

    # Get final metrics
    final_metrics = results.metrics_history[-1] if results.metrics_history else None

    # Detect Phase 2 entry step (first step where sigma_ode > 0.5)
    phase2_entry_step = None
    for m in results.metrics_history:
        if m.sigma_ode > 0.5:
            phase2_entry_step = m.step
            break

    with open(summary_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            "run_idx": run_idx,
            "seed": seed,
            "condition": condition,
            "domain": domain,
            "final_id_accuracy": final_metrics.id_accuracy if final_metrics else None,
            "final_ood_accuracy": final_metrics.ood_accuracy if final_metrics else None,
            "final_sigma_hat": final_metrics.sigma_ode if final_metrics else None,
            "final_alpha_A": final_metrics.alpha_A if final_metrics else None,
            "phase2_entry_step": phase2_entry_step,
            "elapsed_time": elapsed_time,
        })


def print_pilot_summary(output_dir: str) -> None:
    """Print a summary of all pilot runs.

    Args:
        output_dir: Directory containing pilot_results_summary.csv.
    """
    summary_path = os.path.join(output_dir, "pilot_results_summary.csv")

    if not os.path.exists(summary_path):
        print("No pilot summary found.")
        return

    # Read and compute statistics
    ood_accuracies = []
    sigma_hats = []
    phase2_steps = []

    with open(summary_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["final_ood_accuracy"]:
                ood_accuracies.append(float(row["final_ood_accuracy"]))
            if row["final_sigma_hat"]:
                sigma_hats.append(float(row["final_sigma_hat"]))
            if row["phase2_entry_step"]:
                phase2_steps.append(int(row["phase2_entry_step"]))

    print("\n" + "="*60)
    print("PILOT STUDY SUMMARY")
    print("="*60)
    print(f"Total runs: {len(ood_accuracies)}")

    if ood_accuracies:
        import statistics
        mean_ood = statistics.mean(ood_accuracies)
        std_ood = statistics.stdev(ood_accuracies) if len(ood_accuracies) > 1 else 0.0
        print(f"\nOOD Accuracy:")
        print(f"  Mean:   {mean_ood:.4f}")
        print(f"  Std:    {std_ood:.4f}")
        print(f"  Min:    {min(ood_accuracies):.4f}")
        print(f"  Max:    {max(ood_accuracies):.4f}")

    if sigma_hats:
        mean_sigma = statistics.mean(sigma_hats)
        std_sigma = statistics.stdev(sigma_hats) if len(sigma_hats) > 1 else 0.0
        print(f"\nσ̂_A (Schema Coherence):")
        print(f"  Mean:   {mean_sigma:.4f}")
        print(f"  Std:    {std_sigma:.4f}")

    if phase2_steps:
        mean_phase2 = statistics.mean(phase2_steps)
        print(f"\nPhase 2 Entry:")
        print(f"  Runs entered Phase 2: {len(phase2_steps)}/{len(ood_accuracies)}")
        print(f"  Mean entry step:      {mean_phase2:.1f}")

    # Check if H-Bar effect is confirmed
    print("\n" + "-"*60)
    print("H-BAR EFFECT VERIFICATION:")
    if ood_accuracies and mean_ood > 0.90:
        print(f"  ✓ OOD Accuracy ({mean_ood:.1%}) > 90% - H-Bar effect confirmed!")
    else:
        print(f"  ✗ OOD Accuracy below 90% threshold")

    if sigma_hats and mean_sigma > 0.9:
        print(f"  ✓ σ̂_A ({mean_sigma:.3f}) > 0.9 - High schema coherence achieved!")
    else:
        print(f"  ✗ σ̂_A below 0.9 threshold")


def main():
    """Main training function."""
    args = parse_args()

    # Print hardware information
    print("="*60)
    print("H-Bar Symbolic - H-Bar Training")
    print("="*60)
    print(f"\nHardware Information:")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  JAX backend: {jax.default_backend()}")

    print(f"\nConfiguration:")
    print(f"  Domain: {args.domain}")
    print(f"  Condition: {args.condition}")
    print(f"  Number of runs: {args.n_runs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Evaluation interval: {args.eval_interval}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Lambda sigma: {args.lambda_sigma}")
    print(f"  Base seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")

    # Initialize summary CSV
    init_summary_csv(args.output_dir, args.condition)

    # Run training runs
    start_time = time.time()
    for run_idx in range(args.n_runs):
        seed = args.seed + run_idx * 1000

        # Run training
        run_start = time.time()
        results = run_single_training(args, seed, run_idx)
        run_elapsed = time.time() - run_start

        # Append to summary
        append_to_summary(
            args.output_dir,
            run_idx,
            seed,
            args.condition,
            args.domain,
            results,
            run_elapsed,
        )

    total_elapsed = time.time() - start_time

    # Print pilot summary if multiple runs
    if args.n_runs > 1:
        print_pilot_summary(args.output_dir)

    print(f"\n{'='*60}")
    print(f"ALL RUNS COMPLETED")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"{'='*60}")

    # Output file locations
    condition_prefix = "hbar_additive" if args.condition == "additive" else "hbar_multiplicative"
    print(f"\nOutput files:")
    print(f"  Summary: {os.path.join(args.output_dir, 'pilot_results_summary.csv')}")
    for i in range(args.n_runs):
        print(f"  Run {i}: {os.path.join(args.output_dir, f'{condition_prefix}_run_{i}_metrics.csv')}")
        print(f"         {os.path.join(args.output_dir, f'model_params_{condition_prefix}_run_{i}.msgpack')}")


if __name__ == "__main__":
    main()
