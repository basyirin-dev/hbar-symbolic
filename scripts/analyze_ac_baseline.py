#!/usr/bin/env python3
"""Combined GCA and AC baseline analysis script.

This script loads a trained baseline model and computes both GCA (Gradient-
Composition Alignment) and AC (Augmentation Consistency) signals over 100
batches, reporting their correlation and individual statistics.

Expected behavior:
    - GCA should be low (0.0–0.3) or negative (σ-trap confirmed).
    - AC should be moderate (0.5–0.8) — higher than GCA because self-attention
      provides some shallow invariance even without compositional rules.

Usage:
    python scripts/analyze_ac_baseline.py \
        --params ./model_params.msgpack \
        --num-batches 100 \
        --batch-size 32 \
        --domain scan \
        --seed 42
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import optax
from flax import serialization

from hbar.benchmarks.grammar_engine import GrammarEngine
from hbar.engine.data_utils import get_hbar_batch
from hbar.engine.signals import compute_gca
from hbar.engine.trainer import TrainState, compute_dual_gradients, get_ac_signal
from hbar.engine.tokenizer import create_scan_tokenizer, create_cogs_tokenizer
from hbar.models.config import TransformerConfig
from hbar.models.transformer import Seq2SeqTransformer


def load_baseline_state(
    params_path: str,
    config: TransformerConfig,
    init_rng: jax.Array,
):
    """Load model parameters and create a TrainState."""
    with open(params_path, "rb") as f:
        params = serialization.from_bytes(None, f.read())

    model = Seq2SeqTransformer(config)
    # Re-initialize to get proper state structure (params shape must match)
    dummy_src = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    dummy_tgt = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(init_rng, dummy_src, dummy_tgt, training=False)

    # Replace params with loaded ones
    opt = optax.adam(1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    return state, model


def analyze_signals(args):
    """Run combined GCA + AC analysis."""
    # Setup
    print(f"JAX devices: {jax.devices()}")

    # Tokenizer
    if args.domain == "scan":
        tokenizer = create_scan_tokenizer()
    else:
        tokenizer = create_cogs_tokenizer()

    # Model config
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

    # Load parameters
    print(f"Loading parameters from {args.params}...", end=" ", flush=True)
    rng = jax.random.PRNGKey(args.seed)
    state, model = load_baseline_state(args.params, config, rng)
    print("done.")

    # Grammar engine for batch generation
    grammar = GrammarEngine(seed=args.seed)

    # Collect signals
    print(f"\nComputing GCA and AC over {args.num_batches} batches (batch_size={args.batch_size})...")
    gca_values = []
    ac_values = []

    for i in range(args.num_batches):
        rng, bk = jax.random.split(rng)

        # Generate triple-stream HBarBatch
        hbar_batch = get_hbar_batch(
            key=bk,
            batch_size=args.batch_size,
            domain=args.domain,
            grammar_engine=grammar,
        )

        # Compute GCA (gradient-based)
        grad_id, grad_ood = compute_dual_gradients(state, hbar_batch)
        gca = compute_gca(grad_id, grad_ood)
        gca_values.append(float(gca))

        # Compute AC (representation-based)
        ac = get_ac_signal(state, hbar_batch, model)
        ac_values.append(float(ac))

        if (i + 1) % 10 == 0:
            gca_mean = jnp.mean(jnp.array(gca_values))
            ac_mean = jnp.mean(jnp.array(ac_values))
            print(
                f"  Batch {i + 1}/{args.num_batches} - "
                f"GCA: {gca:.4f} (μ={gca_mean:.4f}) | "
                f"AC:  {ac:.4f} (μ={ac_mean:.4f})"
            )

    # Statistical summary
    gca_arr = jnp.array(gca_values)
    ac_arr = jnp.array(ac_values)

    gca_mean = float(jnp.mean(gca_arr))
    gca_sem = float(jnp.std(gca_arr) / jnp.sqrt(args.num_batches))
    gca_std = float(jnp.std(gca_arr))

    ac_mean = float(jnp.mean(ac_arr))
    ac_sem = float(jnp.std(ac_arr) / jnp.sqrt(args.num_batches))
    ac_std = float(jnp.std(ac_arr))

    # Correlation between GCA and AC
    correlation = float(jnp.corrcoef(gca_arr, ac_arr)[0, 1])

    # Print results
    print(f"\n{'=' * 60}")
    print(f"COMBINED SIGNAL ANALYSIS — Baseline Model")
    print(f"{'=' * 60}")
    print(f"\nGCA (Gradient-Composition Alignment):")
    print(f"  Mean (g_A):     {gca_mean:.4f} ± {gca_sem:.4f} (SEM)")
    print(f"  Std Deviation:  {gca_std:.4f}")
    print(f"  Range:          [{float(jnp.min(gca_arr)):.4f}, {float(jnp.max(gca_arr)):.4f}]")

    print(f"\nAC (Augmentation Consistency):")
    print(f"  Mean (c_A):     {ac_mean:.4f} ± {ac_sem:.4f} (SEM)")
    print(f"  Std Deviation:  {ac_std:.4f}")
    print(f"  Range:          [{float(jnp.min(ac_arr)):.4f}, {float(jnp.max(ac_arr)):.4f}]")

    print(f"\nSignal Correlation (Pearson r):")
    print(f"  r(g_A, c_A):    {correlation:.4f}")

    print(f"\n{'=' * 60}")
    print(f"INTERPRETATION")
    print(f"{'=' * 60}")

    # GCA interpretation
    if gca_mean < 0.0:
        print(f"  ✗ GCA NEGATIVE ({gca_mean:.4f} < 0.0)")
        print(f"    Learning ID patterns actively harms OOD performance!")
    elif gca_mean < 0.3:
        print(f"  ✗ GCA LOW ({gca_mean:.4f} < 0.3)")
        print(f"    Model is in σ-trap: gradients are misaligned.")
    elif gca_mean < 0.7:
        print(f"  ⚠ GCA MODERATE ({gca_mean:.4f})")
        print(f"    Partial compositional alignment detected.")
    else:
        print(f"  ✓ GCA HIGH ({gca_mean:.4f} > 0.7)")
        print(f"    Model is crystallizing compositional rules.")

    # AC interpretation
    if ac_mean < 0.5:
        print(f"\n  ✗ AC LOW ({ac_mean:.4f} < 0.5)")
        print(f"    Representations drift significantly under augmentation.")
    elif ac_mean < 0.8:
        print(f"\n  ⚠ AC MODERATE ({ac_mean:.4f})")
        print(f"    Partial structural invariance — some schema capture.")
    else:
        print(f"\n  ✓ AC HIGH ({ac_mean:.4f} > 0.8)")
        print(f"    Strong invariance — compositional schema well-encoded.")

    # Correlation interpretation
    print(f"\n  Signal Correlation: r = {correlation:.4f}")
    if correlation > 0.7:
        print(f"    GCA and AC are strongly coupled — both reflect the same underlying")
        print(f"    compositional structure (or lack thereof).")
    elif correlation > 0.3:
        print(f"    Moderate coupling between gradient alignment and representational")
        print(f"    invariance.")
    else:
        print(f"    GCA and AC are weakly coupled — they capture different aspects")
        print(f"    of the model's failure mode.")

    # H-Bar prediction check
    print(f"\n{'=' * 60}")
    print(f"H-BAR PHASE 2 PREDICTION CHECK")
    print(f"{'=' * 60}")

    if gca_mean < 0.0 and ac_mean > gca_mean:
        print(f"  ✓ AC ({ac_mean:.4f}) > GCA ({gca_mean:.4f})")
        print(f"    Model has shallow invariance without compositional rules.")
        print(f"    This is the characteristic σ-trap signature: the Transformer")
        print(f"    self-attention provides some token-level consistency (AC),")
        print(f"    but the gradient geometry is broken (negative GCA).")
    elif gca_mean < 0.3 and ac_mean > gca_mean:
        print(f"  ✓ AC ({ac_mean:.4f}) > GCA ({gca_mean:.4f})")
        print(f"    Consistent with σ-trap: AC reflects partial invariance,")
        print(f"    while GCA reveals gradient misalignment.")
    else:
        print(f"  ✗ Unexpected pattern: AC ({ac_mean:.4f}) ≤ GCA ({gca_mean:.4f})")
        print(f"    This contradicts the σ-trap prediction.")

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Domain:         {args.domain}")
    print(f"  Batches:        {args.num_batches}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Seed:           {args.seed}")
    print(f"  Model params:   {args.params}")
    print(f"\n  g_A (GCA):      {gca_mean:.4f} ± {gca_sem:.4f}")
    print(f"  c_A (AC):       {ac_mean:.4f} ± {ac_sem:.4f}")
    print(f"  Correlation:    {correlation:.4f}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined GCA + AC baseline analysis for H-Bar signals."
    )
    parser.add_argument(
        "--params",
        type=str,
        default="./model_params.msgpack",
        help="Path to saved model parameters.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of HBarBatch samples to analyze.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for signal computation.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="scan",
        choices=["scan", "cogs"],
        help="Domain for analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    analyze_signals(args)


if __name__ == "__main__":
    main()
