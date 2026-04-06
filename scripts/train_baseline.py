#!/usr/bin/env python3
"""Baseline Training Script for H-Bar Symbolic - Kaggle Entry Point.

This script implements the baseline condition for the H-Bar framework,
designed to demonstrate the "Illusion of Mastery" failure mode on the
SCAN Add-Jump compositional generalization benchmark.

The baseline uses standard Adam optimizer (learning_rate=1e-3) without
any H-Bar signal modulation. It trains a 2-layer Transformer for 5,000
steps on in-distribution samples only, then evaluates on both ID and OOD
splits to measure the generalization gap.

Expected Outcome:
    - ID Accuracy: >95% (model masters in-distribution patterns)
    - OOD Accuracy: <50% (model fails on novel compositions with 'jump')
    - σ̂_A: <0.5 (large generalization gap confirming the σ-trap)

Usage on Kaggle:
    1. Enable GPU (T4 or P100) and Internet
    2. Run: !python scripts/train_baseline.py --domain scan
    3. Download baseline_metrics.csv and model_params.msgpack

Outputs:
    - baseline_metrics.csv: Training metrics at each evaluation interval
    - model_params.msgpack: Saved model parameters for Phase 2 analysis
"""

import argparse
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

# Now import JAX (must be after setting env vars)
import jax

from hbar.benchmarks.grammar_engine import GrammarEngine
from hbar.engine.evaluator import Evaluator
from hbar.engine.tokenizer import create_scan_tokenizer
from hbar.engine.trainer import run_baseline_training, save_params
from hbar.models.config import TransformerConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="H-Bar Baseline Training on SCAN Add-Jump split"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="scan",
        choices=["scan", "cogs"],
        help="Domain to train on (default: scan)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory)",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Print hardware information
    print("=" * 60)
    print("H-Bar Symbolic - Baseline Training")
    print("=" * 60)
    print(f"\nHardware Information:")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  JAX version: {jax.__version__}")

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    print(f"\nConfiguration:")
    print(f"  Domain: {args.domain}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Evaluation interval: {args.eval_interval}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")

    # ========================================================================
    # Initialize components
    # ========================================================================
    print("\nInitializing components...")

    # Create grammar engine for training data generation
    rng, engine_rng = jax.random.split(rng)
    grammar_engine = GrammarEngine(seed=args.seed)

    # Create evaluator for periodic evaluation
    evaluator = Evaluator(domain=args.domain, data_dir="data")

    # Create tokenizer to get vocabulary size
    tokenizer = grammar_engine.get_tokenizer(args.domain)
    vocab_size = len(tokenizer.word2id)
    print(f"  Vocabulary size: {vocab_size}")

    # Set max sequence length based on domain
    max_seq_len = 50 if args.domain == "scan" else 80

    # Create model configuration
    config = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        dropout_rate=0.1,
    )
    print(f"  Model: {config.n_layers} layers, {config.n_heads} heads, d_model={config.d_model}")

    # ========================================================================
    # Run training
    # ========================================================================
    print("\nStarting training...")
    start_time = time.time()

    rng, train_rng = jax.random.split(rng)
    results = run_baseline_training(
        config=config,
        grammar_engine=grammar_engine,
        evaluator=evaluator,
        rng=train_rng,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
        log_dir=args.output_dir,
        log_filename="baseline_metrics.csv",
    )

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

    # ========================================================================
    # Save results
    # ========================================================================
    print("\nSaving results...")

    # Save model parameters
    params_path = os.path.join(args.output_dir, "model_params.msgpack")
    save_params(results.final_params, params_path)

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    if results.metrics_history:
        final_metrics = results.metrics_history[-1]
        print(f"\nFinal Metrics (Step {final_metrics.step}):")
        print(f"  Train Loss:     {final_metrics.train_loss:.4f}")
        print(f"  ID Accuracy:    {final_metrics.id_accuracy:.4f}")
        print(f"  OOD Accuracy:   {final_metrics.ood_accuracy:.4f}")
        print(f"  ID Loss:        {final_metrics.id_loss:.4f}")
        print(f"  OOD Loss:       {final_metrics.ood_loss:.4f}")
        print(f"  σ̂_A (sigma):    {final_metrics.ground_truth_sigma:.4f}")

        # Check for "Illusion of Mastery" pattern
        print("\n" + "-" * 60)
        print("ILLUSION OF MASTERY CHECK:")
        if final_metrics.id_accuracy > 0.90:
            print(f"  ✓ ID Accuracy ({final_metrics.id_accuracy:.1%}) > 90% - Model masters ID patterns")
        else:
            print(f"  ✗ ID Accuracy ({final_metrics.id_accuracy:.1%}) < 90% - Model may need more training")

        if final_metrics.ood_accuracy < 0.50:
            print(f"  ✓ OOD Accuracy ({final_metrics.ood_accuracy:.1%}) < 50% - Model fails on OOD compositions")
        else:
            print(f"  ✗ OOD Accuracy ({final_metrics.ood_accuracy:.1%}) > 50% - Unexpected generalization")

        if final_metrics.ground_truth_sigma < 0.5:
            print(f"  ✓ σ̂_A ({final_metrics.ground_truth_sigma:.3f}) < 0.5 - Large generalization gap confirmed")
        else:
            print(f"  ✗ σ̂_A ({final_metrics.ground_truth_sigma:.3f}) > 0.5 - Gap smaller than expected")

        print("-" * 60)

    print(f"\nOutput files:")
    print(f"  Metrics: {os.path.join(args.output_dir, 'baseline_metrics.csv')}")
    print(f"  Params:  {params_path}")
    print("\nReady for Phase 2 analysis!")


if __name__ == "__main__":
    main()
