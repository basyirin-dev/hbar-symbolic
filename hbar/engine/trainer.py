"""H-Bar Baseline Training Engine for compositional generalization benchmarks.

This module implements the standard SGD training loop for the baseline condition
in the H-Bar framework. The baseline uses Adam optimizer without any H-Bar signal
modulation, designed to demonstrate the "Illusion of Mastery" failure mode:
high in-distribution (ID) accuracy but low out-of-distribution (OOD) accuracy.

The training loop is JIT-compiled for performance and designed to be
Kaggle-compatible with CSV logging for result persistence.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import flax
import flax.struct
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from hbar.engine.data_utils import Batch, compute_loss
from hbar.engine.evaluator import Evaluator, EvaluationResult
from hbar.models.config import TransformerConfig
from hbar.models.transformer import Seq2SeqTransformer


@flax.struct.dataclass
class TrainingMetrics:
    """Metrics collected during training.

    This dataclass tracks the evolution of training and evaluation metrics
    over the course of training, enabling analysis of the "Illusion of Mastery"
    and H-Bar signal dynamics.

    Attributes:
        step: Training step number.
        train_loss: Current training loss.
        id_accuracy: Accuracy on in-distribution evaluation set.
        ood_accuracy: Accuracy on out-of-distribution evaluation set.
        id_loss: Loss on in-distribution evaluation set.
        ood_loss: Loss on out-of-distribution evaluation set.
        ground_truth_sigma: σ̂_A = Acc_OOD / Acc_ID (Equation 7).
    """

    step: int
    train_loss: float
    id_accuracy: float
    ood_accuracy: float
    id_loss: float
    ood_loss: float
    ground_truth_sigma: float


@dataclass
class TrainingResults:
    """Complete results from a training run.

    Attributes:
        final_params: Final model parameters after training.
        metrics_history: List of TrainingMetrics at each evaluation point.
        config: The TransformerConfig used for training.
    """

    final_params: flax.core.FrozenDict
    metrics_history: List[TrainingMetrics]
    config: TransformerConfig


class TrainState(train_state.TrainState):
    """Extended TrainState with batch statistics for batch norm.

    This extends the standard Flax TrainState to support batch statistics
    if needed for future extensions. For the baseline, we only track params.
    """

    batch_stats: flax.core.FrozenDict = flax.struct.field(default_factory=dict)


def init_train_state(
    config: TransformerConfig,
    rng: jax.Array,
    learning_rate: float = 1e-3,
) -> Tuple[TrainState, Seq2SeqTransformer]:
    """Initialize the training state for the baseline model.

    This function creates a Flax TrainState that bundles the model parameters,
    optimizer state, and apply function. The optimizer is Adam with the
    specified learning rate, which is the standard choice for Transformer
    training and represents "standard SGD dynamics" in modern deep learning.

    Args:
        config: TransformerConfig with model hyperparameters.
        rng: JAX PRNGKey for parameter initialization.
        learning_rate: Learning rate for Adam optimizer. Default 1e-3.

    Returns:
        Tuple of (TrainState, model) where:
            - TrainState contains initialized parameters and optimizer
            - model is the initialized Seq2SeqTransformer instance
    """
    # Create model instance
    model = Seq2SeqTransformer(config)

    # Create dummy inputs for initialization
    # Shape: (1, max_seq_len) - batch size 1 for initialization
    dummy_src = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    dummy_tgt = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)

    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, dummy_src, dummy_tgt, training=False)
    params = variables["params"]

    # Create Adam optimizer
    optimizer = optax.adam(learning_rate=learning_rate)

    # Create TrainState
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return state, model


def create_train_step() -> Callable:
    """Create a JIT-compiled training step function.

    Returns:
        A callable that takes (state, batch, rng) and returns
        (new_state, loss, accuracy).
    """

    @jax.jit
    def train_step(
        state: TrainState,
        batch: Batch,
        rng: jax.Array,
    ) -> Tuple[TrainState, float, float]:
        """Execute a single training step.

        This function performs the core mathematical operation of training:
        1. Forward pass to get logits
        2. Compute cross-entropy loss with padding mask
        3. Compute gradients via automatic differentiation
        4. Update parameters using the optimizer

        The loss uses the labels from the batch, which are already shifted
        by 1 position for decoder teacher forcing (prepared by prepare_batch).

        Args:
            state: Current training state with parameters and optimizer.
            batch: Batch of training data (inputs, decoder_inputs, labels, masks).
            rng: PRNGKey for dropout (if any).

        Returns:
            Tuple of (new_state, loss, accuracy) where:
                - new_state: Updated training state with new parameters
                - loss: Scalar training loss for this batch
                - accuracy: Token-level accuracy on this batch
        """

        def loss_fn(params: flax.core.FrozenDict) -> Tuple[jax.Array, jax.Array]:
            """Compute cross-entropy loss for the batch.

            Args:
                params: Model parameters.

            Returns:
                Tuple of (loss, accuracy).
            """
            # Forward pass - only use id_stream from HBarBatch
            # The baseline model only sees in-distribution samples
            logits = state.apply_fn(
                {"params": params},
                batch.inputs,
                batch.decoder_inputs,
                training=True,
                rngs={"dropout": rng},
            )

            # Compute cross-entropy loss (ignores padding tokens)
            loss = compute_loss(logits, batch.labels)

            # Compute accuracy for monitoring
            accuracy = compute_accuracy(logits, batch.labels)

            return loss, accuracy

        # Compute loss and gradients
        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )

        # Update parameters
        new_state = state.apply_gradients(grads=grads)

        # Return JAX arrays directly (cannot use float() inside JIT)
        return new_state, loss, accuracy

    return train_step


def compute_accuracy(
    logits: jax.Array,
    labels: jax.Array,
    pad_token_id: int = 0,
) -> jax.Array:
    """Compute token-level accuracy for sequence prediction.

    Computes the accuracy over all non-padding tokens.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        labels: Target token IDs of shape (batch, seq_len).
        pad_token_id: Token ID to ignore in accuracy computation.

    Returns:
        Scalar accuracy value between 0 and 1.
    """
    # Get predicted token IDs
    predictions = jnp.argmax(logits, axis=-1)

    # Compare with labels
    correct = predictions == labels

    # Mask out padding tokens
    mask = (labels != pad_token_id).astype(jnp.float32)
    correct_masked = correct * mask

    # Compute accuracy over non-padding tokens
    num_valid_tokens = jnp.sum(mask)
    accuracy = jnp.sum(correct_masked) / num_valid_tokens

    return accuracy


def run_baseline_training(
    config: TransformerConfig,
    grammar_engine: Any,
    evaluator: Evaluator,
    rng: jax.Array,
    batch_size: int = 64,
    total_steps: int = 5000,
    eval_interval: int = 500,
    learning_rate: float = 1e-3,
    log_dir: str = ".",
    log_filename: str = "baseline_metrics.csv",
    eval_batch_size: int = 256,
) -> TrainingResults:
    """Run the baseline training loop.

    This function implements the main training loop for the baseline condition.
    It trains the model for a specified number of steps, periodically evaluating
    on both ID and OOD splits to track the "Illusion of Mastery" phenomenon.

    The training data is generated on-the-fly using the grammar engine's
    generate_id_batch() method, which ensures only in-distribution samples
    (no novel compositions) are seen during training.

    Args:
        config: TransformerConfig with model hyperparameters.
        grammar_engine: GrammarEngine instance for generating training batches.
        evaluator: Evaluator instance for periodic evaluation.
        rng: JAX PRNGKey for random operations.
        batch_size: Number of samples per training batch. Default 64.
        total_steps: Total number of training steps. Default 5000.
        eval_interval: Evaluate every N steps. Default 500.
        learning_rate: Learning rate for Adam optimizer. Default 1e-3.
        log_dir: Directory for saving CSV logs. Default current directory.
        log_filename: Name of the CSV log file. Default "baseline_metrics.csv".

    Returns:
        TrainingResults containing final parameters and metrics history.
    """
    # Initialize training state
    rng, init_rng = jax.random.split(rng)
    state, model = init_train_state(config, init_rng, learning_rate)

    # Create JIT-compiled training step
    train_step = create_train_step()

    # Create CSV logger
    log_path = os.path.join(log_dir, log_filename)
    csv_file = open(log_path, "w", newline="")
    fieldnames = [
        "step",
        "train_loss",
        "id_accuracy",
        "ood_accuracy",
        "id_loss",
        "ood_loss",
        "ground_truth_sigma",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Metrics history
    metrics_history: List[TrainingMetrics] = []

    print(f"Starting baseline training for {total_steps} steps...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Evaluation interval: {eval_interval}")
    print(f"  Log file: {log_path}")

    # Training loop
    for step in range(1, total_steps + 1):
        # Split RNG for this step
        rng, train_rng = jax.random.split(rng)

        # Generate training batch (ID only - baseline only sees ID samples)
        # Note: generate_id_batch returns a Batch object
        train_batch = grammar_engine.generate_id_batch(
            batch_size=batch_size,
            domain="scan",
            rng=train_rng,
        )

        # Execute training step
        state, train_loss, train_acc = train_step(state, train_batch, train_rng)

        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"  Step {step}/{total_steps} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # Periodic evaluation
        if step % eval_interval == 0 or step == total_steps:
            print(f"\nStep {step}/{total_steps} - Train Loss: {train_loss:.4f}")

            # Evaluate on ID and OOD splits
            eval_result = evaluator.evaluate(state.params, model, batch_size=eval_batch_size)

            print(f"  ID Accuracy:  {eval_result.acc_id:.4f}")
            print(f"  OOD Accuracy: {eval_result.acc_ood:.4f}")
            print(f"  σ̂_A:          {eval_result.ground_truth_sigma:.4f}")

            # Record metrics
            metrics = TrainingMetrics(
                step=step,
                train_loss=train_loss,
                id_accuracy=eval_result.acc_id,
                ood_accuracy=eval_result.acc_ood,
                id_loss=eval_result.loss_id,
                ood_loss=eval_result.loss_ood,
                ground_truth_sigma=eval_result.ground_truth_sigma,
            )
            metrics_history.append(metrics)

            # Log to CSV
            writer.writerow(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "id_accuracy": eval_result.acc_id,
                    "ood_accuracy": eval_result.acc_ood,
                    "id_loss": eval_result.loss_id,
                    "ood_loss": eval_result.loss_ood,
                    "ground_truth_sigma": eval_result.ground_truth_sigma,
                }
            )
            csv_file.flush()

    csv_file.close()
    print(f"\nTraining complete! Results saved to {log_path}")

    return TrainingResults(
        final_params=state.params,
        metrics_history=metrics_history,
        config=config,
    )


def save_params(params: flax.core.FrozenDict, filepath: str) -> None:
    """Save model parameters to a file.

    Uses Flax's serialization utilities to save parameters in msgpack format,
    which is compact and can be loaded back into Flax.

    Args:
        params: Model parameters to save.
        filepath: Path to save the parameters file.
    """
    from flax import serialization

    with open(filepath, "wb") as f:
        f.write(serialization.to_bytes(params))
    print(f"Parameters saved to {filepath}")


def load_params(
    filepath: str,
    model: Seq2SeqTransformer,
    rng: jax.Array,
) -> flax.core.FrozenDict:
    """Load model parameters from a file.

    Args:
        filepath: Path to the parameters file.
        model: Model instance for shape inference.
        rng: PRNGKey for initialization (used if file not found).

    Returns:
        Loaded model parameters.
    """
    from flax import serialization

    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            params = serialization.from_bytes(None, f.read())
        print(f"Parameters loaded from {filepath}")
        return params
    else:
        print(f"File not found: {filepath}. Initializing new parameters.")
        state, _ = init_train_state(
            TransformerConfig(vocab_size=128),
            rng,
        )
        return state.params
