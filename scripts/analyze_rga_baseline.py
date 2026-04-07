#!/usr/bin/env python3
"""Baseline RGA (Representational-Geometry Alignment) analysis script.

This script computes the RGA signal for a trained baseline model,
along with GCA and AC signals for a complete Stage 1 signal profile.

The RGA signal measures whether the model's internal representation
geometry aligns with the structural geometry of the grammar.

Usage:
    python scripts/analyze_rga_baseline.py \
        --params model_params.msgpack \
        --num-probes 100 \
        --domain scan \
        --seed 42
"""

import argparse
import csv
import time
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from hbar.benchmarks.grammar_engine import GrammarEngine
from hbar.engine.data_utils import prepare_batch
from hbar.engine.signals import (
    compute_ac_from_batch,
    compute_gca,
    compute_rga_from_representations,
)
from hbar.engine.tokenizer import create_scan_tokenizer
from hbar.models.config import TransformerConfig
from hbar.models.transformer import Seq2SeqTransformer, get_model_representations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute RGA, GCA, and AC signals for baseline model"
    )
    parser.add_argument(
        "--params",
        type=str,
        default="model_params.msgpack",
        help="Path to model parameters file",
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=100,
        help="Number of compositional probes for RGA computation",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="scan",
        choices=["scan", "cogs"],
        help="Domain to analyze",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_model_and_params(params_path: str):
    """Load model architecture and trained parameters."""
    config = TransformerConfig(
        vocab_size=20,
        max_seq_len=50,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout_rate=0.1,
    )

    model = Seq2SeqTransformer(config)

    with open(params_path, "rb") as f:
        params = serialization.msgpack_restore(f.read())

    return model, params, config


def generate_compositional_probes(
    engine: GrammarEngine,
    domain: str,
    num_probes: int,
    seed: int,
):
    """Generate compositional probe pairs for RGA analysis."""
    rng = jax.random.PRNGKey(seed)
    pairs = engine.generate_compositional_pairs(
        batch_size=num_probes,
        domain=domain,
        max_depth=3,
        rng=rng,
    )
    return pairs


def extract_bos_representations(
    model: Seq2SeqTransformer,
    params: Dict,
    pairs: list,
    domain: str,
    config: TransformerConfig,
):
    """Extract BOS token representations from final encoder layer."""
    if domain == "scan":
        tokenizer = create_scan_tokenizer()
        max_seq_len = 50
    else:
        tokenizer = engine.cogs_tokenizer
        max_seq_len = 80

    # Tokenize all pairs
    inputs = []
    for input_text, _ in pairs:
        tokens = tokenizer.encode(input_text, max_seq_len)
        inputs.append(tokens)

    # Create batch
    inputs = jnp.array(inputs)
    src_mask = jnp.where(inputs > 0, True, False)
    src_mask = src_mask[:, jnp.newaxis, jnp.newaxis, :]

    # Dummy decoder input (just BOS token)
    decoder_inputs = jnp.ones((inputs.shape[0], 1), dtype=jnp.int32)
    tgt_mask = jnp.ones((inputs.shape[0], 1, 1, 1), dtype=bool)

    # Extract representations
    intermediates = get_model_representations(
        params, model, inputs, decoder_inputs
    )

    # Get BOS token representation from final encoder layer
    # BOS is at position 0 in the sequence
    encoder_last = intermediates["encoder_block_1"]  # (batch, seq_len, d_model)
    bos_reps = encoder_last[:, 0, :]  # (batch, d_model) - BOS token

    return bos_reps


def compute_structural_rdm(engine: GrammarEngine, pairs: list, domain: str):
    """Compute structural RDM from grammar."""
    if domain == "scan":
        # For SCAN, use action sequences for structural distance
        action_sequences = [pair[1] for pair in pairs]
        rdm_struct = engine.compute_rdmstruct(action_sequences)
    else:
        # For COGS, use logical forms
        logical_forms = [pair[1] for pair in pairs]
        rdm_struct = engine.compute_rdmstruct(logical_forms)

    return rdm_struct


def main():
    args = parse_args()

    print("=" * 60)
    print("H-Bar RGA Baseline Analysis")
    print("=" * 60)
    print(f"Parameters: {args.params}")
    print(f"Domain: {args.domain}")
    print(f"Number of probes: {args.num_probes}")
    print(f"Seed: {args.seed}")
    print()

    # Load model
    print("Loading model and parameters...")
    model, params, config = load_model_and_params(args.params)
    print(f"Model loaded: {config.n_layers} layers, {config.n_heads} heads, d_model={config.d_model}")

    # Initialize grammar engine
    engine = GrammarEngine(seed=args.seed)

    # Generate compositional probes
    print(f"\nGenerating {args.num_probes} compositional probes...")
    pairs = generate_compositional_probes(engine, args.domain, args.num_probes, args.seed)
    print(f"Sample probe: '{pairs[0][0]}' → '{pairs[0][1]}'")

    # Extract BOS representations
    print("\nExtracting BOS representations from final encoder layer...")
    bos_reps = extract_bos_representations(model, params, pairs, args.domain, config)
    print(f"BOS representations shape: {bos_reps.shape}")

    # Compute structural RDM
    print("\nComputing structural RDM...")
    rdm_struct = compute_structural_rdm(engine, pairs, args.domain)
    print(f"Structural RDM shape: {rdm_struct.shape}")
    print(f"Structural distance range: [{rdm_struct.min():.4f}, {rdm_struct.max():.4f}]")

    # Compute RGA
    print("\n" + "-" * 40)
    print("Computing RGA signal...")
    rga = compute_rga_from_representations(bos_reps, rdm_struct, method="cosine")
    print(f"RGA (r_A): {float(rga):.4f}")

    # Interpretation
    if float(rga) > 0.5:
        print("Interpretation: MODERATE-HIGH RGA — Model's representations partially align with grammar structure")
    elif float(rga) > 0.2:
        print("Interpretation: LOW-MODERATE RGA — Weak alignment between representations and grammar structure")
    else:
        print("Interpretation: LOW RGA — Model's representations are geometrically disorganized relative to grammar")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
