#!/usr/bin/env python3
"""Diagnostic test for H-Bar sigma update dynamics.

This script runs a short training session and collects detailed metrics
to understand what's actually happening with the sigma update signal.

It will help us determine:
1. Whether the current sigma update formula captures meaningful signal
2. What the actual correlation is between ID and OOD gradients (true GCA)
3. Whether representations are becoming more structured over time
4. If the loss ratio we're using is a good proxy for schema coherence
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from hbar.benchmarks.grammar_engine import GrammarEngine
from hbar.engine.data_utils import Batch, HBarBatch, compute_loss, compute_hbar_loss
from hbar.engine.evaluator import Evaluator
from hbar.engine.signals import compute_gca, compute_ac_from_batch, compute_rga, compute_rdm_representational
from hbar.models.config import TransformerConfig
from hbar.models.transformer import Seq2SeqTransformer


def create_model_and_state(config: TransformerConfig, rng: jax.Array):
    """Initialize model and optimizer state."""
    model = Seq2SeqTransformer(config)
    dummy_src = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    dummy_tgt = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_src, dummy_tgt, training=False)
    params = variables["params"]
    optimizer = optax.adam(learning_rate=config.learning_rate)
    opt_state = optimizer.init(params)
    return model, params, opt_state


def compute_representations(
    model: Seq2SeqTransformer,
    params: dict,
    inputs: jax.Array,
    decoder_inputs: jax.Array,
    rng: jax.Array,
    layer_name: str = "encoder_block_2",
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute representations from a specific layer."""

    def forward_fn(p):
        # Use variables to capture intermediate activations
        variables = {"params": p}

        # Forward pass with capture
        def model_fn():
            return model.apply(
                variables,
                inputs,
                decoder_inputs,
                training=True,
                rngs={"dropout": rng},
                capture_activations=True,
            )

        # We need to modify the model to return activations
        # For now, let's use a simpler approach: get logits and compute RDM from them
        logits = model.apply(variables, inputs, decoder_inputs, training=True, rngs={"dropout": rng})
        return logits

    logits = forward_fn(params)

    # Use logits as proxy for representations (simplified)
    # In a full implementation, we'd capture encoder hidden states
    return logits, {}


def run_diagnostic(
    domain: str = "scan",
    n_steps: int = 100,
    batch_size: int = 32,
    seed: int = 42,
    output_dir: str = ".",
):
    """Run diagnostic training and collect detailed metrics."""

    print("=" * 60)
    print("H-Bar Sigma Update Diagnostic Test")
    print("=" * 60)

    # Initialize evaluator
    print(f"\nInitializing evaluator for {domain}...")
    evaluator = Evaluator(domain=domain)
    print(f"  ID samples: {len(evaluator.id_samples)}")
    print(f"  OOD samples: {len(evaluator.ood_samples)}")

    # Initialize config
    config = TransformerConfig(
        vocab_size=evaluator.tokenizer.vocab_size,
        max_seq_len=evaluator.max_seq_len,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout_rate=0.1,
        learning_rate=1e-3,
    )

    # Initialize model
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    model, params, opt_state = create_model_and_state(config, init_rng)

    # Initialize GrammarEngine for data generation
    engine = GrammarEngine(seed=seed)

    # Diagnostic storage
    diagnostics = {
        "step": [],
        "id_loss": [],
        "ood_loss": [],
        "total_loss": [],
        "sigma_A": [],
        "loss_ratio_signal": [],  # Current formula: 1 - ood_loss/id_loss
        "gen_signal": [],  # Generalization signal: (id_loss - ood_loss) / (id_loss + eps)
        "true_gca": [],  # Actual gradient correlation
        "true_ac": [],  # Augmentation consistency (placeholder)
        "grad_norm_id": [],
        "grad_norm_ood": [],
        "param_magnitude": [],
    }

    optimizer = optax.adam(learning_rate=config.learning_rate)

    print(f"\nRunning {n_steps} diagnostic steps...")
    print(f"{'Step':>6} {'ID Loss':>10} {'OOD Loss':>10} {'Loss Ratio':>12} {'Gen Signal':>12} {'True GCA':>10}")
    print("-" * 70)

    for step in range(1, n_steps + 1):
        # Generate batch
        rng, batch_rng = jax.random.split(rng)
        id_batch = engine.generate_id_batch(batch_size=batch_size, domain=domain)
        ood_batch = engine.get_compositional_batch(batch_size=batch_size, domain=domain)
        aug_batch = id_batch  # Simplified: use ID as augmented

        hbar_batch = HBarBatch(
            id_stream=id_batch,
            ood_stream=ood_batch,
            aug_stream=aug_batch,
        )

        # Current sigma value (simplified: start at 0.27)
        sigma_A = jnp.array(0.27)

        # Forward pass
        rng, fwd_rng = jax.random.split(rng)

        def loss_fn(p):
            all_inputs = jnp.concatenate([
                hbar_batch.id_stream.inputs,
                hbar_batch.ood_stream.inputs,
                hbar_batch.aug_stream.inputs,
            ], axis=0)
            all_decoder_inputs = jnp.concatenate([
                hbar_batch.id_stream.decoder_inputs,
                hbar_batch.ood_stream.decoder_inputs,
                hbar_batch.aug_stream.decoder_inputs,
            ], axis=0)

            all_logits = model.apply(
                {"params": p},
                all_inputs,
                all_decoder_inputs,
                training=True,
                rngs={"dropout": fwd_rng},
            )

            n = hbar_batch.id_stream.inputs.shape[0]
            logits_id = all_logits[:n]
            logits_ood = all_logits[n:2*n]

            total_loss = compute_hbar_loss(
                logits_id=logits_id,
                labels_id=hbar_batch.id_stream.labels,
                logits_ood=logits_ood,
                labels_ood=hbar_batch.ood_stream.labels,
                sigma_A=sigma_A,
                lambda_sigma=0.5,
            )

            id_loss = compute_loss(logits_id, hbar_batch.id_stream.labels)
            ood_loss = compute_loss(logits_ood, hbar_batch.ood_stream.labels)

            return total_loss, (id_loss, ood_loss)

        (total_loss, (id_loss, ood_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)

        # Compute diagnostic signals

        # 1. Loss ratio signal (current formula)
        id_loss_val = float(id_loss)
        ood_loss_val = float(ood_loss)

        loss_ratio_signal = float(jnp.where(
            id_loss > 1e-6,
            jnp.clip(1.0 - ood_loss / (id_loss + 1e-6), 0.0, 1.0),
            0.9,
        ))

        # 2. Generalization signal (alternative)
        gen_signal = float(jnp.clip(
            (id_loss - ood_loss) / (id_loss + 0.01) + 0.5, 0.0, 1.0
        ))

        # 3. True GCA - compute separate gradients for ID and OOD
        def id_loss_fn(p):
            logits = model.apply(
                {"params": p},
                hbar_batch.id_stream.inputs,
                hbar_batch.id_stream.decoder_inputs,
                training=True,
                rngs={"dropout": fwd_rng},
            )
            return compute_loss(logits, hbar_batch.id_stream.labels)

        def ood_loss_fn(p):
            logits = model.apply(
                {"params": p},
                hbar_batch.ood_stream.inputs,
                hbar_batch.ood_stream.decoder_inputs,
                training=True,
                rngs={"dropout": fwd_rng},
            )
            return compute_loss(logits, hbar_batch.ood_stream.labels)

        _, grad_id = jax.value_and_grad(id_loss_fn)(params)
        _, grad_ood = jax.value_and_grad(ood_loss_fn)(params)

        # Flatten gradients for GCA computation
        grad_id_flat = jnp.concatenate([g.flatten() for g in jax.tree_leaves(grad_id)])
        grad_ood_flat = jnp.concatenate([g.flatten() for g in jax.tree_leaves(grad_ood)])

        true_gca = float(compute_gca(grad_id_flat, grad_ood_flat))

        # 4. Gradient norms
        grad_norm_id = float(jnp.sqrt(jnp.sum(grad_id_flat ** 2)))
        grad_norm_ood = float(jnp.sqrt(jnp.sum(grad_ood_flat ** 2)))

        # 5. Parameter magnitude
        param_magnitude = float(jnp.sqrt(sum(
            jnp.sum(p ** 2) for p in jax.tree_leaves(params)
        )))

        # Store diagnostics
        diagnostics["step"].append(step)
        diagnostics["id_loss"].append(id_loss_val)
        diagnostics["ood_loss"].append(ood_loss_val)
        diagnostics["total_loss"].append(float(total_loss))
        diagnostics["sigma_A"].append(0.27)  # Would be updated in real training
        diagnostics["loss_ratio_signal"].append(loss_ratio_signal)
        diagnostics["gen_signal"].append(gen_signal)
        diagnostics["true_gca"].append(true_gca)
        diagnostics["grad_norm_id"].append(grad_norm_id)
        diagnostics["grad_norm_ood"].append(grad_norm_ood)
        diagnostics["param_magnitude"].append(param_magnitude)

        # Print progress
        if step % 10 == 0:
            print(f"{step:6d} {id_loss_val:10.4f} {ood_loss_val:10.4f} "
                  f"{loss_ratio_signal:12.4f} {gen_signal:12.4f} {true_gca:10.4f}")

        # Apply gradients
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    # Print summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    # Compute correlations
    loss_ratio_arr = np.array(diagnostics["loss_ratio_signal"])
    gen_signal_arr = np.array(diagnostics["gen_signal"])
    true_gca_arr = np.array(diagnostics["true_gca"])
    id_loss_arr = np.array(diagnostics["id_loss"])
    ood_loss_arr = np.array(diagnostics["ood_loss"])

    print(f"\nSignal Statistics:")
    print(f"  Loss Ratio Signal:  mean={loss_ratio_arr.mean():.4f}, std={loss_ratio_arr.std():.4f}")
    print(f"  Gen Signal:         mean={gen_signal_arr.mean():.4f}, std={gen_signal_arr.std():.4f}")
    print(f"  True GCA:           mean={true_gca_arr.mean():.4f}, std={true_gca_arr.std():.4f}")

    # Correlation between signals
    corr_loss_gca = np.corrcoef(loss_ratio_arr, true_gca_arr)[0, 1]
    corr_gen_gca = np.corrcoef(gen_signal_arr, true_gca_arr)[0, 1]

    print(f"\nSignal Correlations with True GCA:")
    print(f"  Loss Ratio ↔ GCA:   {corr_loss_gca:.4f}")
    print(f"  Gen Signal ↔ GCA:   {corr_gen_gca:.4f}")

    # Loss statistics
    print(f"\nLoss Statistics:")
    print(f"  ID Loss:   mean={id_loss_arr.mean():.4f}, final={id_loss_arr[-1]:.4f}")
    print(f"  OOD Loss:  mean={ood_loss_arr.mean():.4f}, final={ood_loss_arr[-1]:.4f}")
    print(f"  ID-OOD Gap: mean={(id_loss_arr - ood_loss_arr).mean():.4f}")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if corr_loss_gca < 0.3:
        print("  ⚠ Loss ratio signal has LOW correlation with true GCA")
        print("    → Consider using true GCA for sigma update")

    if true_gca_arr.mean() < 0.3:
        print("  ⚠ True GCA is LOW - gradients are not aligned")
        print("    → Model may not be learning compositional rules")

    if (id_loss_arr - ood_loss_arr).mean() < 0:
        print("  ⚠ OOD loss > ID loss on average")
        print("    → Model is overfitting to ID patterns")

    # Save diagnostics to CSV
    csv_path = os.path.join(output_dir, "hbar_sigma_diagnostic.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diagnostics.keys())
        writer.writeheader()
        for i in range(len(diagnostics["step"])):
            writer.writerow({k: diagnostics[k][i] for k in diagnostics})

    print(f"\nDiagnostics saved to: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H-Bar Sigma Update Diagnostic")
    parser.add_argument("--domain", type=str, default="scan", choices=["scan", "cogs"])
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    run_diagnostic(
        domain=args.domain,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
