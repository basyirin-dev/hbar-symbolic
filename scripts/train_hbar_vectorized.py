#!/usr/bin/env python3
"""Vectorized H-Bar Training Script.

This script runs N=15 parallel training runs using JAX vmap and lax.scan
for maximum efficiency. It compresses the 8-hour sequential experiment into
~20-30 minutes on a single Kaggle GPU.

Usage:
    python scripts/train_hbar_vectorized.py \
        --domain scan \
        --condition multiplicative \
        --n_runs 15 \
        --output_dir ./results

Features:
    - Vectorized training with jax.vmap across N runs
    - Compiled loops with jax.lax.scan
    - Early stopping via crystallization detection (σ̃_A > 0.90)
    - Mixed precision (bfloat16) for 2x speedup
    - CSV logging for all runs
    - Automatic result aggregation
"""

import argparse
import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp

from hbar.benchmarks.grammar_engine import GrammarEngine
from hbar.engine.evaluator import Evaluator
from hbar.engine.vectorized_trainer import run_vectorized_training, VectorizedTrainingResults
from hbar.models.config import TransformerConfig, FusionConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized H-Bar Training")
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
        default=15,
        help="Number of parallel training runs",
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
    print("H-Bar Symbolic - Vectorized Training")
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

    # Initialize grammar engine
    print(f"Initializing grammar engine for {args.domain}...")
    if args.domain not in ["scan", "cogs"]:
        raise ValueError(f"Unsupported domain: {args.domain}")

    grammar_engine = GrammarEngine(seed=args.base_seed)

    # Initialize evaluator
    print(f"Initializing evaluator...")
    rng, eval_rng = jax.random.split(rng)
    evaluator = Evaluator(
        config=config,
        domain=args.domain,
        grammar_engine=grammar_engine,
        rng=eval_rng,
    )

    # Run vectorized training
    print(f"\nStarting vectorized training...")
    start_time = time.time()

    rng, train_rng = jax.random.split(rng)
    results = run_vectorized_training(
        config=config,
        grammar_engine=grammar_engine,
        evaluator=evaluator,
        rng=train_rng,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        eval_interval=500,
        learning_rate=args.learning_rate,
        lambda_sigma=args.lambda_sigma,
        log_dir=args.output_dir,
        log_filename=f"hbar_{args.condition}_vectorized_metrics.csv",
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
    print(f"  Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

    if results.crystallization_steps:
        crystallized = [s for s in results.crystallization_steps if s >= 0]
        if crystallized:
            print(f"  Mean crystallization step: {sum(crystallized)/len(crystallized):.0f}")
            print(f"  Min crystallization step: {min(crystallized)}")
            print(f"  Max crystallization step: {max(crystallized)}")

    print(f"\nNext steps:")
    print(f"  1. Review {summary_path} for per-run metrics")
    print(f"  2. Compare with baseline results")
    print(f"  3. Run the other condition (additive/multiplicative)")


def save_pilot_summary(
    results: VectorizedTrainingResults,
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

        for i, (params, hbar_state) in enumerate(
            zip(results.final_params, results.final_hbar_states)
        ):
            crystallized = results.crystallization_steps[i] >= 0
            cryst_step = results.crystallization_steps[i]

            writer.writerow([
                i,
                crystallized,
                cryst_step,
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
