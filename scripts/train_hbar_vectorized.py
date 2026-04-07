#!/usr/bin/env python3
"""Optimized H-Bar Training Script - All 5 Tiers.

This script runs H-Bar training using the fully optimized engine that
implements all 5 tiers of JAX optimization:

- Tier 1: Full-trajectory compilation (jax.lax.scan) + vmap over N
- Tier 2: Zero-transfer data pipeline (pre-tokenized, on-device sampling)
- Tier 3: Concatenated forward passes + O(1) RDM computation
- Tier 4: H-Bar specific optimizations (frozen RDMs, fixed-step ODE)
- Tier 5: XLA memory management (metric downsampling, static shapes)

Usage:
    python scripts/train_hbar_vectorized.py \
        --domain scan \
        --condition multiplicative \
        --n_runs 15 \
        --output_dir ./results

Features:
    - Pre-tokenized data pipeline (no I/O during training)
    - Concatenated forward passes for ID/OOD/Aug streams
    - JIT-compiled training steps
    - Early stopping via crystallization detection (σ̃_A > 0.90)
    - Mixed precision (bfloat16) for 2x speedup
    - CSV logging for all runs
"""

import argparse
import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp

from hbar.engine.evaluator import Evaluator
from hbar.engine.vectorized_trainer import run_optimized_training, TrainingResults
from hbar.models.config import TransformerConfig, FusionConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized H-Bar Training")
    parser.add_argument(
        "--domain",
        type=str,
        default="scan",
        choices=["scan", "cogs"],
        help="Benchmark domain",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="multiplicative",
        choices=["additive", "multiplicative"],
        help="H-Bar loss condition",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of parallel training runs (note: current implementation runs sequentially)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per model",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=5000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam",
    )
    parser.add_argument(
        "--lambda_sigma",
        type=float,
        default=0.5,
        help="Compositional penalty weight",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Print configuration
    print("=" * 60)
    print("H-Bar Symbolic - Optimized Training (5 Tiers)")
    print("=" * 60)
    print(f"\nHardware Information:")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"\nConfiguration:")
    print(f"  Domain: {args.domain}")
    print(f"  Condition: {args.condition}")
    print(f"  Number of runs: {args.n_runs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Lambda sigma: {args.lambda_sigma}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize random key
    rng = jax.random.PRNGKey(args.base_seed)

    # Create model configuration with mixed precision
    fusion_config = FusionConfig(
        w_gca=0.4,
        w_rga=0.35,
        w_ac=0.25,
        target_sigma_critical=0.5,
        kappa_alpha=2.0,
    )

    config = TransformerConfig(
        vocab_size=128,
        max_seq_len=50,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        dropout_rate=0.1,
        dtype=jnp.bfloat16,  # Mixed precision
        param_dtype=jnp.float32,
        fusion_config=fusion_config,
    )

    # Initialize evaluator
    print(f"Initializing evaluator for {args.domain}...")
    evaluator = Evaluator(
        domain=args.domain,
        data_dir="data",
    )

    # Run optimized training
    print(f"\nStarting optimized training...")
    start_time = time.time()

    rng, train_rng = jax.random.split(rng)
    results = run_optimized_training(
        config=config,
        evaluator=evaluator,
        rng=train_rng,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        lambda_sigma=args.lambda_sigma,
        log_dir=args.output_dir,
        log_filename=f"hbar_{args.condition}_optimized_metrics.csv",
    )

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

    # Save summary results
    summary_path = os.path.join(args.output_dir, f"pilot_{args.condition}_summary.csv")
    save_pilot_summary(results, args, summary_path)
    print(f"Summary saved to {summary_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("PILOT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Condition: {args.condition}")
    print(f"  N runs: {args.n_runs}")
    print(f"  Crystallized: {results.n_crystallized}/{args.n_runs}")
    print(f"  Final σ_A: {results.final_sigma_A:.4f}")
    print(f"  Final α_A: {results.final_alpha_A:.4f}")
    print(f"  Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

    if results.crystallization_step is not None:
        print(f"  Crystallization step: {results.crystallization_step}")

    print(f"\nNext steps:")
    print(f"  1. Review {summary_path} for per-run metrics")
    print(f"  2. Compare with baseline results")
    print(f"  3. Run the other condition (additive/multiplicative)")


def save_pilot_summary(
    results: TrainingResults,
    args: argparse.Namespace,
    output_path: str,
):
    """Save pilot study summary to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "crystallized",
            "crystallization_step",
            "final_sigma_A",
            "final_alpha_A",
            "condition",
            "domain",
            "n_runs",
            "batch_size",
            "total_steps",
            "learning_rate",
            "lambda_sigma",
            "base_seed",
            "timestamp",
        ])

        writer.writerow([
            0,
            results.n_crystallized > 0,
            results.crystallization_step if results.crystallization_step else -1,
            results.final_sigma_A,
            results.final_alpha_A,
            args.condition,
            args.domain,
            args.n_runs,
            args.batch_size,
            args.total_steps,
            args.learning_rate,
            args.lambda_sigma,
            args.base_seed,
            datetime.now().isoformat(),
        ])


if __name__ == "__main__":
    main()
