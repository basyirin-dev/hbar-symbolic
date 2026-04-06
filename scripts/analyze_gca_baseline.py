"""Analyze GCA (Gradient-Composition Alignment) signal on the trained baseline model.

This script loads the saved baseline model parameters and computes the GCA
signal over 100 batches to characterize the "baseline noise" level of
gradient alignment between in-distribution (ID) and out-of-distribution (OOD)
compositional samples.

The GCA signal g_A measures the Pearson correlation between:
- ∇_θ L_train: Gradients from in-distribution loss
- ∇_θ L_comp: Gradients from OOD compositional loss

Expected result for baseline (σ-trap): Low GCA (0.1-0.3), indicating that
learning ID patterns does not align with learning compositional rules.

Usage:
    python scripts/analyze_gca_baseline.py --params model_params.msgpack
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import optax
from flax import serialization

from hbar.benchmarks.grammar_engine import GrammarEngine
from hbar.engine.data_utils import get_hbar_batch, HBarBatch
from hbar.engine.signals import compute_gca
from hbar.engine.trainer import TrainState, compute_dual_gradients
from hbar.engine.tokenizer import create_scan_tokenizer
from hbar.models.config import TransformerConfig
from hbar.models.transformer import Seq2SeqTransformer


def load_baseline_state(
    params_path: str,
    config: TransformerConfig,
    rng: jax.Array,
) -> TrainState:
    """Load the baseline model parameters into a TrainState.

    Args:
        params_path: Path to the saved model parameters file.
        config: TransformerConfig matching the trained model.
        rng: JAX PRNGKey for initialization.

    Returns:
        TrainState with loaded parameters and initialized optimizer.
    """
    # Load saved parameters
    with open(params_path, "rb") as f:
        params = serialization.from_bytes(None, f.read())
    print(f"Loaded parameters from {params_path}")

    # Initialize model to get apply_fn
    model = Seq2SeqTransformer(config)
    dummy_src = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    dummy_tgt = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_src, dummy_tgt, training=False)

    # Create optimizer (same as training)
    optimizer = optax.adam(learning_rate=1e-3)

    # Create TrainState with loaded params
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return state


def analyze_gca(
    state: TrainState,
    grammar_engine: GrammarEngine,
    num_batches: int = 100,
    batch_size: int = 32,
    domain: str = "scan",
    rng: jax.Array = jax.random.PRNGKey(42),
) -> dict[str, float]:
    """Compute GCA signal over multiple batches.

    For each batch, computes the Pearson correlation between ID and OOD
    gradient vectors. Aggregates results to produce mean ± SEM statistics.

    Args:
        state: TrainState with loaded model parameters.
        grammar_engine: GrammarEngine for generating HBarBatch triples.
        num_batches: Number of batches to evaluate. Default 100.
        batch_size: Samples per batch. Default 32 (memory-efficient for T4).
        domain: Benchmark domain. Default "scan".
        rng: JAX PRNGKey for batch generation.

    Returns:
        Dictionary with GCA statistics:
            - mean: Mean GCA across all batches
            - std: Standard deviation
            - sem: Standard error of the mean
            - min: Minimum GCA observed
            - max: Maximum GCA observed
    """
    print(f"Computing GCA over {num_batches} batches (batch_size={batch_size})...")

    gca_values = []

    for i in range(num_batches):
        rng, batch_key = jax.random.split(rng)

        # Generate HBarBatch with ID, OOD, and aug streams
        hbar_batch = get_hbar_batch(
            key=batch_key,
            batch_size=batch_size,
            domain=domain,
            grammar_engine=grammar_engine,
        )

        # Compute dual gradients and GCA
        grad_id, grad_ood = compute_dual_gradients(state, hbar_batch)
        gca = compute_gca(grad_id, grad_ood)
        gca_values.append(float(gca))

        if (i + 1) % 10 == 0:
            gca_array = jnp.array(gca_values)
            mean_gca = jnp.mean(gca_array)
            print(
                f"  Batch {i + 1}/{num_batches} - "
                f"Current GCA: {gca:.4f}, "
                f"Running Mean: {mean_gca:.4f}"
            )

    # Compute summary statistics
    gca_array = jnp.array(gca_values)
    mean_gca = jnp.mean(gca_array)
    std_gca = jnp.std(gca_array)
    sem_gca = std_gca / jnp.sqrt(num_batches)

    return {
        "mean": float(mean_gca),
        "std": float(std_gca),
        "sem": float(sem_gca),
        "min": float(jnp.min(gca_array)),
        "max": float(jnp.max(gca_array)),
    }


def print_results(stats: dict[str, float]) -> None:
    """Print GCA analysis results with interpretation.

    Args:
        stats: Dictionary of GCA statistics from analyze_gca().
    """
    print(f"\n{'=' * 60}")
    print(f"GCA ANALYSIS RESULTS - Baseline Model")
    print(f"{'=' * 60}")
    print(f"  Mean GCA (g_A):  {stats['mean']:.4f} ± {stats['sem']:.4f} (SEM)")
    print(f"  Std Deviation:   {stats['std']:.4f}")
    print(f"  Min GCA:         {stats['min']:.4f}")
    print(f"  Max GCA:         {stats['max']:.4f}")
    print(f"{'=' * 60}")

    # Interpretation
    mean = stats["mean"]
    print(f"\nINTERPRETATION:")
    if mean > 0.7:
        print(f"  ✓ HIGH GCA ({mean:.4f} > 0.7)")
        print(f"    Model is crystallizing compositional rules.")
        print(f"    ID and OOD gradients are well-aligned.")
    elif mean > 0.3:
        print(f"  ⚠ MODERATE GCA ({mean:.4f} in [0.3, 0.7])")
        print(f"    Partial compositional alignment detected.")
        print(f"    Model shows some generalization capability.")
    elif mean > 0.0:
        print(f"  ✗ LOW GCA ({mean:.4f} in [0.0, 0.3])")
        print(f"    Model is in the σ-trap.")
        print(f"    Learning ID statistics conflicts with OOD structure.")
    else:
        print(f"  ✗ NEGATIVE GCA ({mean:.4f} < 0.0)")
        print(f"    Learning ID patterns actively harms OOD performance!")
        print(f"    Severe overfitting to surface statistics.")

    print(f"\n{'=' * 60}")


def main():
    """Parse arguments and run GCA analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze GCA signal on trained baseline model"
    )
    parser.add_argument(
        "--params",
        type=str,
        default="./model_params.msgpack",
        help="Path to saved model parameters (default: ./model_params.msgpack)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of batches to evaluate (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="scan",
        help="Benchmark domain (default: scan)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Print hardware info
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.devices()[0].device_kind}")

    # Model config (must match trained model)
    tokenizer = create_scan_tokenizer()
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        max_seq_len=50,
        dropout_rate=0.1,
    )
    print(f"Model: {config.n_layers} layers, {config.n_heads} heads, d_model={config.d_model}")

    # Initialize RNG
    rng = jax.random.PRNGKey(args.seed)

    # Initialize grammar engine
    grammar_engine = GrammarEngine(seed=args.seed)

    # Load model state
    rng, init_rng = jax.random.split(rng)
    state = load_baseline_state(args.params, config, init_rng)

    # Run GCA analysis
    stats = analyze_gca(
        state=state,
        grammar_engine=grammar_engine,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        domain=args.domain,
        rng=rng,
    )

    # Print results
    print_results(stats)


if __name__ == "__main__":
    main()
