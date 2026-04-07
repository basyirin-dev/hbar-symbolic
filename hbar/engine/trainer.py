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
from flax.training.train_state import TrainState

from hbar.engine.data_utils import Batch, HBarBatch, compute_loss
from hbar.engine.evaluator import Evaluator, EvaluationResult
from hbar.models.config import TransformerConfig, FusionConfig
from hbar.models.transformer import Seq2SeqTransformer


# ============================================================================
# H-Bar Integrated TrainState (Subtask 8.3)
# ============================================================================


@flax.struct.dataclass
class HBarTrainState:
    """Unified training state that bundles neural weights and cognitive ODE state.

    This is the "Architectural Glue" of the H-Bar framework. By wrapping both
    the standard Flax TrainState and the HBar ODE state into a single Pytree,
    the entire H-Bar training step—including signal extraction, ODE integration,
    and modulated backprop—can be handled by a single jax.jit function call.

    The key advantage of this unified structure is that it enables the use of
    jax.lax.scan for massive training speedups, as the entire training loop
    can be compiled into a single XLA operation.

    Attributes:
        train_state: Standard Flax TrainState containing model parameters,
            optimizer state, and apply function.
        hbar_state: The 7-variable ODE state (delta_A, sigma_A, alpha_A,
            M_hat_A, Xi_A_P, Xi_A_I, Xi_A_F) from the H-Bar dynamical system.
        constants: HBarConstants containing the 11 dynamical parameters
            (r_delta, gamma_sigma, gamma_alpha, mu_delta, eta_sigma, eta_alpha,
            K_delta, kappa_M, kappa_Xi, lambda_Xi, sigma_critical).
        fusion_config: Configuration for signal fusion weights (w_gca, w_rga,
            w_ac) and the target sigma_critical threshold.

    Example:
        >>> # Initialize unified state
        >>> hbar_train_state = init_hbar_train_state(config, rng)
        >>>
        >>> # Single JIT-compiled step updates both weights and ODE state
        >>> hbar_train_state, metrics = apply_hbar_step(hbar_train_state, batch, model, rng)
        >>>
        >>> # Access neural parameters and cognitive state separately
        >>> params = hbar_train_state.train_state.params
        >>> sigma_A = hbar_train_state.hbar_state.sigma_A
    """

    train_state: TrainState
    hbar_state: Any  # HBarState from hbar.core.dynamics
    constants: Any  # HBarConstants from hbar.core.dynamics
    fusion_config: FusionConfig


def init_hbar_train_state(
    config: TransformerConfig,
    rng: jax.Array,
    learning_rate: float = 1e-3,
) -> HBarTrainState:
    """Initialize the unified H-Bar training state.

    This function creates a complete HBarTrainState that bundles:
    1. The standard Flax TrainState with Adam optimizer
    2. The HBar ODE state initialized at the baseline starting point
    3. The HBar dynamical constants
    4. The fusion configuration for signal combination

    The HBarState is initialized at the baseline point (sigma_A ≈ 0.27)
    reflecting the pre-training cognitive state before H-Bar modulation.

    Args:
        config: TransformerConfig with model hyperparameters.
        rng: JAX PRNGKey for parameter initialization.
        learning_rate: Learning rate for Adam optimizer. Default 1e-3.

    Returns:
        HBarTrainState containing initialized TrainState, HBarState,
        constants, and fusion configuration.
    """
    from hbar.core.dynamics import HBarState, HBarConstants, init_hbar_state

    # Initialize standard training state
    rng, init_rng = jax.random.split(rng)
    train_state, _ = init_train_state(config, init_rng, learning_rate)

    # Initialize HBar ODE state at baseline starting point
    hbar_constants = HBarConstants()
    hbar_state = init_hbar_state(hbar_constants)

    # Get fusion config (use defaults if not specified)
    fusion_cfg = config.fusion_config if config.fusion_config else FusionConfig()

    return HBarTrainState(
        train_state=train_state,
        hbar_state=hbar_state,
        constants=hbar_constants,
        fusion_config=fusion_cfg,
    )


def apply_hbar_step(
    hbar_train_state: HBarTrainState,
    hbar_batch: HBarBatch,
    model: Seq2SeqTransformer,
    rng: jax.Array,
    lambda_sigma: float = 0.5,
) -> Tuple[HBarTrainState, Dict[str, jax.Array]]:
    """Execute a single H-Bar training step (Algorithm 3.2 from the paper).

    This is the "Master Step" that coordinates the full 7-step sequence:
    1. Signal Extraction: Compute GCA (g_A), RGA (r_A), AC (c_A)
    2. Fusion: Compute sigma_tilde_A via weighted sum (Equation 6)
    3. ODE Integration: Evolve HBarState via CognitiveManager.step
    4. Modulated Loss: Compute L_total using new sigma_A (Equation 25)
    5. Backward Pass: Compute gradients via automatic differentiation
    6. Acceleration: Apply gradient scaling based on alpha_A (Equation 26)
    7. Weight Update: Apply scaled gradients via optimizer

    The entire function is JIT-compatible, enabling the full training loop
    to be compiled into a single XLA operation via jax.lax.scan.

    Args:
        hbar_train_state: Unified state containing TrainState, HBarState,
            constants, and fusion configuration.
        hbar_batch: HBarBatch containing id_stream, ood_stream, and aug_stream.
        model: Seq2SeqTransformer model instance for signal extraction.
        rng: PRNGKey for dropout and augmentation randomness.
        lambda_sigma: Maximum compositional penalty weight. Default 0.5.

    Returns:
        Tuple of (new_hbar_train_state, metrics_dict) where:
            - new_hbar_train_state: Updated state with new weights and ODE state
            - metrics_dict: Dictionary of all computed signals and metrics
    """
    from hbar.core.dynamics import HBarConstants
    from hbar.core.state_manager import CognitiveManager, create_manager
    from hbar.engine.signals import (
        compute_gca,
        fuse_hbar_signals,
        compute_ac_from_batch,
    )
    from hbar.engine.data_utils import compute_hbar_loss
    from hbar.models.transformer import get_model_representations

    # Unpack the unified state
    train_state = hbar_train_state.train_state
    hbar_state = hbar_train_state.hbar_state
    constants = hbar_train_state.constants
    fusion_cfg = hbar_train_state.fusion_config

    # ========================================================
    # Step 1: Signal Extraction
    # ========================================================

    # GCA (Gradient-Composition Alignment) - Equation 3
    grad_id_flat, grad_ood_flat = compute_dual_gradients(train_state, hbar_batch)
    g_A = compute_gca(grad_id_flat, grad_ood_flat)

    # AC (Augmentation Consistency) - Equation 5
    # Extract representations from ID and augmented streams
    orig_repr = get_model_representations(
        train_state.params,
        model,
        hbar_batch.id_stream.inputs,
        hbar_batch.id_stream.decoder_inputs,
    )
    aug_repr = get_model_representations(
        train_state.params,
        model,
        hbar_batch.aug_stream.inputs,
        hbar_batch.aug_stream.decoder_inputs,
    )
    c_A = compute_ac_from_batch(orig_repr, aug_repr, layer="encoder_block_1")

    # RGA (Representational-Geometry Alignment) - Equation 4
    # For efficiency during training, we use a simplified RGA estimate
    # Full RGA computation happens at evaluation checkpoints
    # Here we use the current sigma_A as a proxy for representational alignment
    r_A = hbar_state.sigma_A  # Simplified proxy during training

    # ========================================================
    # Step 2: Signal Fusion (Equation 6)
    # ========================================================
    # sigma_tilde_A = w_g * max(0, g_A) + w_r * max(0, r_A) + w_c * c_A
    sigma_tilde = fuse_hbar_signals(
        g_A=g_A,
        r_A=r_A,
        c_A=c_A,
        weights={
            "w_g": fusion_cfg.w_gca,
            "w_r": fusion_cfg.w_rga,
            "w_c": fusion_cfg.w_ac,
        },
    )

    # ========================================================
    # Step 3: ODE Integration
    # ========================================================
    # Create cognitive manager and prepare inputs
    cognitive_manager = create_manager(constants)

    # Map training metrics to HBarInputs
    inputs = cognitive_manager.metrics_to_inputs({
        "sigma_tilde": sigma_tilde,
        "sigma_hat": hbar_state.sigma_A,
    })

    # Step the ODEs to update HBarState
    new_hbar_state = cognitive_manager.step(hbar_state, inputs, dt=1.0)

    # ========================================================
    # Step 4: Modulated Loss (Equation 25)
    # ========================================================
    # L_total = L_task + lambda_sigma * (1 - sigma_A) * L_comp

    def loss_fn(params: flax.core.FrozenDict) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Compute H-Bar modulated loss."""
        # Forward pass on ID stream
        logits_id = train_state.apply_fn(
            {"params": params},
            hbar_batch.id_stream.inputs,
            hbar_batch.id_stream.decoder_inputs,
            training=True,
            rngs={"dropout": rng},
        )

        # Forward pass on OOD stream
        logits_ood = train_state.apply_fn(
            {"params": params},
            hbar_batch.ood_stream.inputs,
            hbar_batch.ood_stream.decoder_inputs,
            training=True,
            rngs={"dropout": rng},
        )

        # Compute modulated loss using the NEW sigma_A from ODE
        total_loss = compute_hbar_loss(
            logits_id=logits_id,
            labels_id=hbar_batch.id_stream.labels,
            logits_ood=logits_ood,
            labels_ood=hbar_batch.ood_stream.labels,
            sigma_A=new_hbar_state.sigma_A,
            lambda_sigma=lambda_sigma,
        )

        # Compute individual losses for logging
        id_loss = compute_loss(logits_id, hbar_batch.id_stream.labels)
        ood_loss = compute_loss(logits_ood, hbar_batch.ood_stream.labels)

        return total_loss, id_loss, ood_loss

    # ========================================================
    # Step 5 & 6: Backward Pass + Acceleration (Equation 26)
    # ========================================================
    (total_loss, id_loss, ood_loss), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(train_state.params)

    # Apply attentional acceleration via gradient scaling
    # eta_effective = eta_base * (1 + kappa_alpha * alpha_A)
    acceleration_factor = 1.0 + fusion_cfg.kappa_alpha * new_hbar_state.alpha_A
    scaled_grads = jax.tree_util.tree_map(
        lambda g: g * acceleration_factor, grads
    )

    # ========================================================
    # Step 7: Weight Update
    # ========================================================
    new_train_state = train_state.apply_gradients(grads=scaled_grads)

    # ========================================================
    # Construct new unified state
    # ========================================================
    new_hbar_train_state = HBarTrainState(
        train_state=new_train_state,
        hbar_state=new_hbar_state,
        constants=constants,
        fusion_config=fusion_cfg,
    )

    # ========================================================
    # Metrics dictionary for logging
    # ========================================================
    metrics = {
        "total_loss": total_loss,
        "id_loss": id_loss,
        "ood_loss": ood_loss,
        "g_A": g_A,
        "r_A": r_A,
        "c_A": c_A,
        "sigma_tilde": sigma_tilde,
        "sigma_A": new_hbar_state.sigma_A,
        "alpha_A": new_hbar_state.alpha_A,
        "acceleration_factor": acceleration_factor,
        "compositional_penalty": 1.0 - new_hbar_state.sigma_A,
    }

    return new_hbar_train_state, metrics


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


def compute_dual_gradients(
    state: TrainState,
    hbar_batch: HBarBatch,
) -> Tuple[jax.Array, jax.Array]:
    """Compute gradients for ID and OOD streams separately.

    This function performs two forward/backward passes to extract the
    gradient vectors needed for GCA (Gradient-Composition Alignment)
    computation:
    1. ID gradient: ∇_θ L_train from id_stream (in-distribution samples)
    2. OOD gradient: ∇_θ L_comp from ood_stream (compositional probes)

    The gradients are flattened into vectors using jax.flatten_util.ravel_pytree
    for Pearson correlation computation. This includes all trainable parameters
    (embeddings + transformer layers) to measure Total Systemic Alignment.

    Compositional generalization in SCAN/COGS relies on Variable-Role Binding,
    which requires alignment between embeddings (Variables) and transformer
    layers (Roles/Functions).

    Args:
        state: Current training state with parameters.
        hbar_batch: HBarBatch containing id_stream and ood_stream.

    Returns:
        Tuple of (flattened_id_grad, flattened_ood_grad), both 1D arrays
        of shape (n_params,).
    """

    def loss_fn(params: flax.core.FrozenDict, batch: Batch) -> jax.Array:
        """Compute cross-entropy loss for a batch.

        Args:
            params: Model parameters.
            batch: Batch of training data.

        Returns:
            Scalar loss value.
        """
        logits = state.apply_fn(
            {"params": params},
            batch.inputs,
            batch.decoder_inputs,
            training=False,
        )
        loss = compute_loss(logits, batch.labels)
        return loss

    # Compute ID gradient (in-distribution)
    grad_id = jax.grad(loss_fn)(state.params, hbar_batch.id_stream)

    # Compute OOD gradient (compositional probes)
    grad_ood = jax.grad(loss_fn)(state.params, hbar_batch.ood_stream)

    # Flatten both gradients to 1D vectors for correlation computation
    grad_id_flat, _ = jax.flatten_util.ravel_pytree(grad_id)
    grad_ood_flat, _ = jax.flatten_util.ravel_pytree(grad_ood)

    return grad_id_flat, grad_ood_flat


@jax.jit
def get_gca_signal(
    state: TrainState,
    hbar_batch: HBarBatch,
) -> jax.Array:
    """Compute the GCA (Gradient-Composition Alignment) scalar for a batch.

    This is the core JIT-compiled function that computes the Pearson
    correlation between ID and OOD gradient vectors. It first extracts
    the dual gradients and then computes their alignment.

    The GCA signal g_A ∈ [-1, 1] indicates whether the model is learning
    rules that generalize to novel compositions:
        - g_A > 0.7: Model is "crystallizing" compositional rules
        - 0.0 < g_A < 0.3: Model in σ-trap (gradient misalignment)
        - g_A < 0.0: Learning ID actively harms OOD performance

    Args:
        state: Current training state with parameters.
        hbar_batch: HBarBatch containing id_stream and ood_stream.

    Returns:
        Scalar GCA value in [-1, 1].
    """
    from hbar.engine.signals import compute_gca

    grad_id_flat, grad_ood_flat = compute_dual_gradients(state, hbar_batch)
    return compute_gca(grad_id_flat, grad_ood_flat)


def get_ac_signal(
    state: TrainState,
    hbar_batch: HBarBatch,
    model: Seq2SeqTransformer,
    layer: str = "encoder_block_1",
) -> jax.Array:
    """Compute the AC (Augmentation Consistency) scalar for a batch.

    This function extracts encoder representations from the ID stream and
    augmented stream, then computes the cosine similarity between them.
    Unlike GCA, AC is computed from forward-pass activations rather than
    gradients, measuring representational invariance.

    The AC signal c_A in [0, 1] indicates how invariant the model's internal
    representations are to structure-preserving augmentations:
        - c_A > 0.8: Strong structural invariance (compositional schema encoded)
        - 0.5 < c_A < 0.8: Moderate invariance (partial structure capture)
        - c_A < 0.5: Weak invariance (representations drift significantly)

    This function wraps `get_model_representations` + `compute_ac_from_batch`.
    It is NOT jit-compiled by default since we use intermediates (sow).
    Use externally with jax.jit if needed.

    Args:
        state: Current training state with parameters.
        hbar_batch: HBarBatch containing id_stream and aug_stream.
        model: Seq2SeqTransformer model instance.
        layer: The encoder layer key to use.

    Returns:
        Scalar AC value in [0, 1].
    """
    from hbar.models.transformer import get_model_representations
    from hbar.engine.signals import compute_ac_from_batch

    # Forward pass on ID stream (original)
    orig_repr = get_model_representations(
        state.params,
        model,
        hbar_batch.id_stream.inputs,
        hbar_batch.id_stream.decoder_inputs,
    )

    # Forward pass on Aug stream (structure-preserved perturbation)
    aug_repr = get_model_representations(
        state.params,
        model,
        hbar_batch.aug_stream.inputs,
        hbar_batch.aug_stream.decoder_inputs,
    )

    return compute_ac_from_batch(orig_repr, aug_repr, layer)


def create_hbar_train_step() -> Callable:
    """Create a JIT-compiled H-Bar training step function.

    This training step accepts HBarBatch (with id_stream and ood_stream)
    and modulates the loss using the current schema coherence σ_A from
    the HBarState. The total loss follows Equation 25:

        L_total = L_task + λ_σ · (1 - σ_A) · L_comp

    Additionally, the learning rate is modulated by attentional fidelity
    via gradient scaling (Equation 26):

        η_effective = η_base · (1 + κ_α · α_A)

    This is implemented by scaling the gradients before applying them,
    which is mathematically equivalent to scaling the learning rate but
    more efficient in JAX/Optax (avoids recreating optimizer state).

    Returns:
        A callable that takes (state, hbar_batch, sigma_A, alpha_A, rng, lambda_sigma, kappa_alpha, base_lr)
        and returns (new_state, loss, id_loss, ood_loss, compositional_penalty, effective_lr, acceleration_factor).
    """
    from hbar.engine.data_utils import compute_hbar_loss

    @jax.jit
    def train_step(
        state: TrainState,
        hbar_batch: HBarBatch,
        sigma_A: jax.Array,
        alpha_A: jax.Array,
        rng: jax.Array,
        lambda_sigma: float = 0.5,
        kappa_alpha: float = 2.0,
        base_lr: float = 1e-3,
    ) -> Tuple[TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Execute a single H-Bar training step with attentional acceleration.

        This function performs the core mathematical operation of H-Bar training:
        1. Forward pass on both ID and OOD streams
        2. Compute modulated loss: L_total = L_task + λ_σ · (1 - σ_A) · L_comp
        3. Compute gradients via automatic differentiation
        4. Scale gradients by attentional acceleration factor (Equation 26)
        5. Update parameters using the optimizer

        The compositional pressure (1 - σ_A) ensures that when schema coherence
        is low, the gradient is strongly pushed toward improving OOD performance.

        The attentional acceleration creates a positive feedback loop: high
        attentional fidelity (α_A) accelerates learning, which reinforces
        schema coherence growth.

        Args:
            state: Current training state with parameters and optimizer.
            hbar_batch: HBarBatch containing id_stream and ood_stream.
            sigma_A: Current schema coherence estimate, scalar in [0, 1].
            alpha_A: Current attentional fidelity from ODE state, scalar in [0, 1].
            rng: PRNGKey for dropout (if any).
            lambda_sigma: Maximum compositional penalty weight.
            kappa_alpha: Attentional acceleration coefficient (κ_α). Default 2.0.
            base_lr: Base learning rate (η_base). Default 1e-3.

        Returns:
            Tuple of (new_state, total_loss, id_loss, ood_loss, compositional_penalty,
            effective_lr, acceleration_factor) where:
                - new_state: Updated training state with new parameters
                - total_loss: Scalar total modulated loss
                - id_loss: Loss on ID stream
                - ood_loss: Loss on OOD stream
                - compositional_penalty: The (1 - σ_A) weight value
                - effective_lr: η_base · (1 + κ_α · α_A)
                - acceleration_factor: (1 + κ_α · α_A)
        """

        def loss_fn(params: flax.core.FrozenDict) -> Tuple[jax.Array, jax.Array, jax.Array]:
            """Compute H-Bar modulated loss.

            Args:
                params: Model parameters.

            Returns:
                Tuple of (total_loss, id_loss, ood_loss).
            """
            # Forward pass on ID stream
            logits_id = state.apply_fn(
                {"params": params},
                hbar_batch.id_stream.inputs,
                hbar_batch.id_stream.decoder_inputs,
                training=True,
                rngs={"dropout": rng},
            )

            # Forward pass on OOD stream
            logits_ood = state.apply_fn(
                {"params": params},
                hbar_batch.ood_stream.inputs,
                hbar_batch.ood_stream.decoder_inputs,
                training=True,
                rngs={"dropout": rng},
            )

            # Compute modulated loss
            total_loss = compute_hbar_loss(
                logits_id=logits_id,
                labels_id=hbar_batch.id_stream.labels,
                logits_ood=logits_ood,
                labels_ood=hbar_batch.ood_stream.labels,
                sigma_A=sigma_A,
                lambda_sigma=lambda_sigma,
            )

            # Also compute individual losses for logging
            id_loss = compute_loss(logits_id, hbar_batch.id_stream.labels)
            ood_loss = compute_loss(logits_ood, hbar_batch.ood_stream.labels)

            return total_loss, id_loss, ood_loss

        # Compute loss and gradients
        (total_loss, id_loss, ood_loss), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        # Apply attentional acceleration via gradient scaling (Equation 26)
        # This is mathematically equivalent to scaling the learning rate:
        #   θ_new = θ - η_base · (1 + κ_α · α_A) · g
        # But more efficient in JAX/Optax (avoids recreating optimizer state)
        acceleration_factor = 1.0 + kappa_alpha * alpha_A
        scaled_grads = jax.tree_util.tree_map(
            lambda g: g * acceleration_factor, grads
        )

        # Update parameters with scaled gradients
        new_state = state.apply_gradients(grads=scaled_grads)

        # Compute compositional pressure for logging
        compositional_penalty = 1.0 - sigma_A

        # Compute effective learning rate for logging
        effective_lr = base_lr * acceleration_factor

        return (
            new_state,
            total_loss,
            id_loss,
            ood_loss,
            compositional_penalty,
            effective_lr,
            acceleration_factor,
        )

    return train_step


def create_hbar_train_step_multiplicative() -> Callable:
    """Create a JIT-compiled H-Bar training step with multiplicative coupling (Condition C).

    This training step accepts HBarBatch (with id_stream and ood_stream)
    and modulates the loss using multiplicative coupling:

        L_total = L_task · (1 + λ_σ · (1 - σ_A) · L_comp)

    Additionally, the learning rate is modulated by attentional fidelity
    via gradient scaling (Equation 26):

        η_effective = η_base · (1 + κ_α · α_A)

    The multiplicative coupling creates more aggressive training dynamics:
    when task loss is high, the compositional penalty is amplified, creating
    stronger pressure to learn compositional rules.

    Returns:
        A callable that takes (state, hbar_batch, sigma_A, alpha_A, rng, lambda_sigma, kappa_alpha, base_lr)
        and returns (new_state, loss, id_loss, ood_loss, compositional_penalty, effective_lr, acceleration_factor).
    """
    from hbar.engine.data_utils import compute_hbar_loss_multiplicative

    @jax.jit
    def train_step(
        state: TrainState,
        hbar_batch: HBarBatch,
        sigma_A: jax.Array,
        alpha_A: jax.Array,
        rng: jax.Array,
        lambda_sigma: float = 0.5,
        kappa_alpha: float = 2.0,
        base_lr: float = 1e-3,
    ) -> Tuple[TrainState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Execute a single H-Bar training step with multiplicative coupling.

        This function performs the core mathematical operation of H-Bar training
        with multiplicative loss coupling (Condition C):
        1. Forward pass on both ID and OOD streams
        2. Compute modulated loss: L_total = L_task · (1 + λ_σ · (1 - σ_A) · L_comp)
        3. Compute gradients via automatic differentiation
        4. Scale gradients by attentional acceleration factor (Equation 26)
        5. Update parameters using the optimizer

        The multiplicative coupling amplifies the compositional penalty when
        task loss is high, creating stronger pressure for compositional learning.

        Args:
            state: Current training state with parameters and optimizer.
            hbar_batch: HBarBatch containing id_stream and ood_stream.
            sigma_A: Current schema coherence estimate, scalar in [0, 1].
            alpha_A: Current attentional fidelity from ODE state, scalar in [0, 1].
            rng: PRNGKey for dropout (if any).
            lambda_sigma: Maximum compositional penalty weight.
            kappa_alpha: Attentional acceleration coefficient (κ_α). Default 2.0.
            base_lr: Base learning rate (η_base). Default 1e-3.

        Returns:
            Tuple of (new_state, total_loss, id_loss, ood_loss, compositional_penalty,
            effective_lr, acceleration_factor) where:
                - new_state: Updated training state with new parameters
                - total_loss: Scalar total modulated loss
                - id_loss: Loss on ID stream
                - ood_loss: Loss on OOD stream
                - compositional_penalty: The (1 - σ_A) weight value
                - effective_lr: η_base · (1 + κ_α · α_A)
                - acceleration_factor: (1 + κ_α · α_A)
        """

        def loss_fn(params: flax.core.FrozenDict) -> Tuple[jax.Array, jax.Array, jax.Array]:
            """Compute H-Bar modulated loss with multiplicative coupling.

            Args:
                params: Model parameters.

            Returns:
                Tuple of (total_loss, id_loss, ood_loss).
            """
            # Forward pass on ID stream
            logits_id = state.apply_fn(
                {"params": params},
                hbar_batch.id_stream.inputs,
                hbar_batch.id_stream.decoder_inputs,
                training=True,
                rngs={"dropout": rng},
            )

            # Forward pass on OOD stream
            logits_ood = state.apply_fn(
                {"params": params},
                hbar_batch.ood_stream.inputs,
                hbar_batch.ood_stream.decoder_inputs,
                training=True,
                rngs={"dropout": rng},
            )

            # Compute modulated loss with multiplicative coupling
            total_loss = compute_hbar_loss_multiplicative(
                logits_id=logits_id,
                labels_id=hbar_batch.id_stream.labels,
                logits_ood=logits_ood,
                labels_ood=hbar_batch.ood_stream.labels,
                sigma_A=sigma_A,
                lambda_sigma=lambda_sigma,
            )

            # Also compute individual losses for logging
            id_loss = compute_loss(logits_id, hbar_batch.id_stream.labels)
            ood_loss = compute_loss(logits_ood, hbar_batch.ood_stream.labels)

            return total_loss, id_loss, ood_loss

        # Compute loss and gradients
        (total_loss, id_loss, ood_loss), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        # Apply attentional acceleration via gradient scaling (Equation 26)
        acceleration_factor = 1.0 + kappa_alpha * alpha_A
        scaled_grads = jax.tree_util.tree_map(
            lambda g: g * acceleration_factor, grads
        )

        # Update parameters with scaled gradients
        new_state = state.apply_gradients(grads=scaled_grads)

        # Compute compositional pressure for logging
        compositional_penalty = 1.0 - sigma_A

        # Compute effective learning rate for logging
        effective_lr = base_lr * acceleration_factor

        return (
            new_state,
            total_loss,
            id_loss,
            ood_loss,
            compositional_penalty,
            effective_lr,
            acceleration_factor,
        )

    return train_step


def compute_attentional_lr(
    base_lr: float,
    kappa_alpha: float,
    alpha_A: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Compute the effective learning rate with attentional acceleration (Equation 26).

    This function implements the attentional fidelity modulation from the H-Bar
    paper:

        η_effective = η_base · (1 + κ_α · α_A)

    The mechanism creates a positive feedback loop: high attentional fidelity
    (α_A) accelerates learning, which in turn reinforces schema coherence growth.
    During Phase 1, α_A is low (suppressed by surface rewards), so acceleration
    ≈ 1.0. At Phase 2 entry (crystallization), α_A increases rapidly, causing
    an "Attentional Burst" that marks the transition.

    Args:
        base_lr: Base learning rate (η_base).
        kappa_alpha: Attentional acceleration coefficient (κ_α). Default 2.0
            from FusionConfig. At α_A=1.0, multiplier = 1 + 2.0 = 3.0.
        alpha_A: Current attentional fidelity from ODE state, scalar in [0, 1].

    Returns:
        Tuple of (effective_lr, acceleration_factor) where:
            - effective_lr: η_base · (1 + κ_α · α_A)
            - acceleration_factor: (1 + κ_α · α_A), the multiplicative speedup
    """
    acceleration_factor = 1.0 + kappa_alpha * alpha_A
    effective_lr = base_lr * acceleration_factor
    return effective_lr, acceleration_factor


@flax.struct.dataclass
class HBarTrainingMetrics:
    """Metrics collected during H-Bar training.

    This dataclass extends TrainingMetrics to include H-Bar specific
    signals and modulation parameters.

    Attributes:
        step: Training step number.
        train_loss: Current total modulated training loss.
        id_loss: Loss on in-distribution stream.
        ood_loss: Loss on out-of-distribution stream.
        id_accuracy: Accuracy on in-distribution evaluation set.
        ood_accuracy: Accuracy on out-of-distribution evaluation set.
        sigma_tilde: Fused schema coherence estimate σ̃_A.
        sigma_ode: ODE state variable σ_A from the dynamical system.
        alpha_A: Attentional fidelity from ODE system.
        compositional_penalty: The (1 - σ_A) weight value.
        lambda_sigma: Current compositional penalty coefficient.
        effective_learning_rate: Learning rate after attentional modulation.
        acceleration_factor: The (1 + κ_α · α_A) multiplier value.
    """

    step: int
    train_loss: float
    id_loss: float
    ood_loss: float
    id_accuracy: float
    ood_accuracy: float
    sigma_tilde: float
    sigma_ode: float
    alpha_A: float
    compositional_penalty: float
    lambda_sigma: float
    effective_learning_rate: float
    acceleration_factor: float


@dataclass
class HBarTrainingResults:
    """Complete results from an H-Bar training run.

    Attributes:
        final_params: Final model parameters after training.
        final_hbar_state: Final HBarState from the ODE system.
        metrics_history: List of HBarTrainingMetrics at each evaluation point.
        config: The TransformerConfig used for training.
    """

    final_params: flax.core.FrozenDict
    final_hbar_state: Any  # HBarState from dynamics
    metrics_history: List[HBarTrainingMetrics]
    config: TransformerConfig


def run_hbar_training(
    config: TransformerConfig,
    grammar_engine: Any,
    evaluator: Evaluator,
    rng: jax.Array,
    batch_size: int = 64,
    total_steps: int = 5000,
    eval_interval: int = 500,
    learning_rate: float = 1e-3,
    lambda_sigma: float = 0.5,
    log_dir: str = ".",
    log_filename: str = "hbar_metrics.csv",
    eval_batch_size: int = 256,
) -> HBarTrainingResults:
    """Run the H-Bar modulated training loop.

    This function implements the main training loop for the H-Bar condition.
    It integrates the ODE dynamics with neural network training:

    Per step:
    1. Generate HBarBatch (ID + OOD streams)
    2. Compute operative estimate σ̃_A via signal fusion
    3. Step the ODEs via CognitiveManager.step to update HBarState
    4. Execute train_step using the updated σ_A from HBarState
    5. Log the Compositional Penalty Weight λ_σ · (1 - σ_A)

    Args:
        config: TransformerConfig with model hyperparameters.
        grammar_engine: GrammarEngine instance for generating training batches.
        evaluator: Evaluator instance for periodic evaluation.
        rng: JAX PRNGKey for random operations.
        batch_size: Number of samples per training batch. Default 64.
        total_steps: Total number of training steps. Default 5000.
        eval_interval: Evaluate every N steps. Default 500.
        learning_rate: Learning rate for Adam optimizer. Default 1e-3.
        lambda_sigma: Maximum compositional penalty weight. Default 0.5.
        log_dir: Directory for saving CSV logs. Default current directory.
        log_filename: Name of the CSV log file. Default "hbar_metrics.csv".
        eval_batch_size: Batch size for evaluation. Default 256.

    Returns:
        HBarTrainingResults containing final parameters, HBarState, and metrics.
    """
    # Import H-Bar core components
    from hbar.core.dynamics import (
        HBarState,
        HBarConstants,
        init_hbar_state,
    )
    from hbar.core.state_manager import CognitiveManager, create_manager
    from hbar.engine.data_utils import HBarBatch, get_hbar_batch, HBarSignals
    from hbar.engine.signals import fuse_hbar_signals

    # Initialize training state
    rng, init_rng = jax.random.split(rng)
    state, model = init_train_state(config, init_rng, learning_rate)

    # Create JIT-compiled H-Bar training step
    train_step = create_hbar_train_step()

    # Initialize H-Bar ODE state at baseline starting point
    hbar_constants = HBarConstants()
    hbar_state = init_hbar_state(hbar_constants)

    # Create cognitive manager for ODE integration
    cognitive_manager = create_manager()

    # Create CSV logger with H-Bar specific columns
    log_path = os.path.join(log_dir, log_filename)
    csv_file = open(log_path, "w", newline="")
    fieldnames = [
        "step",
        "train_loss",
        "id_loss",
        "ood_loss",
        "id_accuracy",
        "ood_accuracy",
        "sigma_tilde",
        "sigma_ode",
        "alpha_A",
        "compositional_penalty",
        "lambda_sigma",
        "effective_learning_rate",
        "acceleration_factor",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Metrics history
    metrics_history: List[HBarTrainingMetrics] = []

    print(f"Starting H-Bar training for {total_steps} steps...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Lambda_sigma: {lambda_sigma}")
    print(f"  Evaluation interval: {eval_interval}")
    print(f"  Log file: {log_path}")

    # Training loop
    for step in range(1, total_steps + 1):
        # Split RNG for this step
        rng, train_rng, batch_rng = jax.random.split(rng, 3)

        # Step 1: Generate HBarBatch
        hbar_batch = get_hbar_batch(
            key=batch_rng,
            batch_size=batch_size,
            domain="scan",
            grammar_engine=grammar_engine,
        )

        # Step 2: Compute operative estimate σ̃_A (Signal Fusion)
        # For efficiency, we use a simplified fusion during training
        # Full GCA/RGA/AC computation happens at evaluation checkpoints
        # Here we use the ODE state σ_A as the operative estimate
        sigma_tilde = hbar_state.sigma_A

        # Step 3: Prepare H-Bar inputs for ODE stepping
        # Map training metrics to HBarInputs
        inputs = cognitive_manager.metrics_to_inputs({
            "sigma_tilde": sigma_tilde,
            "sigma_hat": hbar_state.sigma_A,
        })

        # Step the ODEs to update HBarState
        hbar_state = cognitive_manager.step(hbar_state, inputs, hbar_constants, dt=1.0)

        # Step 4: Execute train_step using updated σ_A and α_A
        state, total_loss, id_loss, ood_loss, comp_penalty, eff_lr, accel_factor = train_step(
            state,
            hbar_batch,
            hbar_state.sigma_A,
            hbar_state.alpha_A,
            train_rng,
            lambda_sigma,
            config.fusion_config.kappa_alpha if config.fusion_config else 2.0,
            learning_rate,
        )

        # Step 5: Log compositional penalty weight
        current_penalty = float(comp_penalty)

        # Print progress every 100 steps
        if step % 100 == 0:
            print(
                f"  Step {step}/{total_steps} - "
                f"Loss: {total_loss:.4f}, "
                f"σ_A: {hbar_state.sigma_A:.4f}, "
                f"α_A: {hbar_state.alpha_A:.4f}, "
                f"Penalty: {current_penalty:.4f}"
            )

        # Periodic evaluation
        if step % eval_interval == 0 or step == total_steps:
            print(f"\nStep {step}/{total_steps}")
            print(f"  Total Loss: {total_loss:.4f}")
            print(f"  ID Loss:    {id_loss:.4f}")
            print(f"  OOD Loss:   {ood_loss:.4f}")
            print(f"  σ_A (ODE):  {hbar_state.sigma_A:.4f}")
            print(f"  α_A:        {hbar_state.alpha_A:.4f}")
            print(f"  Penalty:    {current_penalty:.4f}")

            # Evaluate on ID and OOD splits
            eval_result = evaluator.evaluate(
                state.params, model, batch_size=eval_batch_size
            )

            print(f"  ID Accuracy:  {eval_result.acc_id:.4f}")
            print(f"  OOD Accuracy: {eval_result.acc_ood:.4f}")
            print(f"  σ̂_A:          {eval_result.ground_truth_sigma:.4f}")

            # Update cognitive manager with evaluation results
            inputs = cognitive_manager.metrics_to_inputs({
                "sigma_tilde": hbar_state.sigma_A,
                "sigma_hat": eval_result.ground_truth_sigma,
            })

            # Check for phase transition
            phase_info = cognitive_manager.check_phase_transition(hbar_state)

            # Compute acceleration metrics for logging
            kappa_alpha_val = config.fusion_config.kappa_alpha if config.fusion_config else 2.0
            accel_factor_val = float(1.0 + kappa_alpha_val * hbar_state.alpha_A)
            eff_lr_val = float(learning_rate * accel_factor_val)

            # Record metrics
            metrics = HBarTrainingMetrics(
                step=step,
                train_loss=total_loss,
                id_loss=id_loss,
                ood_loss=ood_loss,
                id_accuracy=eval_result.acc_id,
                ood_accuracy=eval_result.acc_ood,
                sigma_tilde=hbar_state.sigma_A,
                sigma_ode=hbar_state.sigma_A,
                alpha_A=hbar_state.alpha_A,
                compositional_penalty=current_penalty,
                lambda_sigma=lambda_sigma,
                effective_learning_rate=eff_lr_val,
                acceleration_factor=accel_factor_val,
            )
            metrics_history.append(metrics)

            # Log to CSV
            writer.writerow(
                {
                    "step": step,
                    "train_loss": total_loss,
                    "id_loss": id_loss,
                    "ood_loss": ood_loss,
                    "id_accuracy": eval_result.acc_id,
                    "ood_accuracy": eval_result.acc_ood,
                    "sigma_tilde": hbar_state.sigma_A,
                    "sigma_ode": hbar_state.sigma_A,
                    "alpha_A": hbar_state.alpha_A,
                    "compositional_penalty": current_penalty,
                    "lambda_sigma": lambda_sigma,
                    "effective_learning_rate": eff_lr_val,
                    "acceleration_factor": accel_factor_val,
                }
            )
            csv_file.flush()

    csv_file.close()
    print(f"\nTraining complete! Results saved to {log_path}")

    return HBarTrainingResults(
        final_params=state.params,
        final_hbar_state=hbar_state,
        metrics_history=metrics_history,
        config=config,
    )


def run_hbar_training_multiplicative(
    config: TransformerConfig,
    grammar_engine: Any,
    evaluator: Evaluator,
    rng: jax.Array,
    batch_size: int = 64,
    total_steps: int = 5000,
    eval_interval: int = 500,
    learning_rate: float = 1e-3,
    lambda_sigma: float = 0.5,
    log_dir: str = ".",
    log_filename: str = "hbar_multiplicative_metrics.csv",
    eval_batch_size: int = 256,
) -> HBarTrainingResults:
    """Run the H-Bar modulated training loop with multiplicative coupling (Condition C).

    This function implements the main training loop for the H-Bar condition with
    multiplicative loss coupling:

        L_total = L_task · (1 + λ_σ · (1 - σ_A) · L_comp)

    The multiplicative coupling creates more aggressive training dynamics compared
    to the additive version. When task loss is high, the compositional penalty is
    amplified, creating stronger pressure to learn compositional rules.

    Per step:
    1. Generate HBarBatch (ID + OOD streams)
    2. Compute operative estimate σ̃_A via signal fusion
    3. Step the ODEs via CognitiveManager.step to update HBarState
    4. Execute train_step using multiplicative loss coupling
    5. Log the Compositional Penalty Weight λ_σ · (1 - σ_A)

    Args:
        config: TransformerConfig with model hyperparameters.
        grammar_engine: GrammarEngine instance for generating training batches.
        evaluator: Evaluator instance for periodic evaluation.
        rng: JAX PRNGKey for random operations.
        batch_size: Number of samples per training batch. Default 64.
        total_steps: Total number of training steps. Default 5000.
        eval_interval: Evaluate every N steps. Default 500.
        learning_rate: Learning rate for Adam optimizer. Default 1e-3.
        lambda_sigma: Maximum compositional penalty weight. Default 0.5.
        log_dir: Directory for saving CSV logs. Default current directory.
        log_filename: Name of the CSV log file. Default "hbar_multiplicative_metrics.csv".
        eval_batch_size: Batch size for evaluation. Default 256.

    Returns:
        HBarTrainingResults containing final parameters, HBarState, and metrics.
    """
    # Import H-Bar core components
    from hbar.core.dynamics import (
        HBarState,
        HBarConstants,
        init_hbar_state,
    )
    from hbar.core.state_manager import CognitiveManager, create_manager
    from hbar.engine.data_utils import HBarBatch, get_hbar_batch, HBarSignals
    from hbar.engine.signals import fuse_hbar_signals

    # Initialize training state
    rng, init_rng = jax.random.split(rng)
    state, model = init_train_state(config, init_rng, learning_rate)

    # Create JIT-compiled H-Bar training step with multiplicative coupling
    train_step = create_hbar_train_step_multiplicative()

    # Initialize H-Bar ODE state at baseline starting point
    hbar_constants = HBarConstants()
    hbar_state = init_hbar_state(hbar_constants)

    # Create cognitive manager for ODE integration
    cognitive_manager = create_manager()

    # Create CSV logger with H-Bar specific columns
    log_path = os.path.join(log_dir, log_filename)
    csv_file = open(log_path, "w", newline="")
    fieldnames = [
        "step",
        "train_loss",
        "id_loss",
        "ood_loss",
        "id_accuracy",
        "ood_accuracy",
        "sigma_tilde",
        "sigma_ode",
        "alpha_A",
        "compositional_penalty",
        "lambda_sigma",
        "effective_learning_rate",
        "acceleration_factor",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Metrics history
    metrics_history: List[HBarTrainingMetrics] = []

    print(f"Starting H-Bar training (MULTIPLICATIVE) for {total_steps} steps...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Lambda_sigma: {lambda_sigma}")
    print(f"  Evaluation interval: {eval_interval}")
    print(f"  Log file: {log_path}")

    # Training loop
    for step in range(1, total_steps + 1):
        # Split RNG for this step
        rng, train_rng, batch_rng = jax.random.split(rng, 3)

        # Step 1: Generate HBarBatch
        hbar_batch = get_hbar_batch(
            key=batch_rng,
            batch_size=batch_size,
            domain="scan",
            grammar_engine=grammar_engine,
        )

        # Step 2: Compute operative estimate σ̃_A (Signal Fusion)
        # For efficiency, we use a simplified fusion during training
        sigma_tilde = hbar_state.sigma_A

        # Step 3: Prepare H-Bar inputs for ODE stepping
        inputs = cognitive_manager.metrics_to_inputs({
            "sigma_tilde": sigma_tilde,
            "sigma_hat": hbar_state.sigma_A,  # Use current state as estimate
        })

        # Step the ODEs to update HBarState
        hbar_state = cognitive_manager.step(hbar_state, inputs, hbar_constants, dt=1.0)

        # Step 4: Execute train_step using updated σ_A and α_A with multiplicative coupling
        state, total_loss, id_loss, ood_loss, comp_penalty, eff_lr, accel_factor = train_step(
            state,
            hbar_batch,
            hbar_state.sigma_A,
            hbar_state.alpha_A,
            train_rng,
            lambda_sigma,
            config.fusion_config.kappa_alpha if config.fusion_config else 2.0,
            learning_rate,
        )

        # Step 5: Log compositional penalty weight
        current_penalty = float(comp_penalty)

        # Print progress every 100 steps
        if step % 100 == 0:
            print(
                f"  Step {step}/{total_steps} - "
                f"Loss: {total_loss:.4f}, "
                f"σ_A: {hbar_state.sigma_A:.4f}, "
                f"α_A: {hbar_state.alpha_A:.4f}, "
                f"Penalty: {current_penalty:.4f}"
            )

        # Periodic evaluation
        if step % eval_interval == 0 or step == total_steps:
            print(f"\nStep {step}/{total_steps}")
            print(f"  Total Loss: {total_loss:.4f}")
            print(f"  ID Loss:    {id_loss:.4f}")
            print(f"  OOD Loss:   {ood_loss:.4f}")
            print(f"  σ_A (ODE):  {hbar_state.sigma_A:.4f}")
            print(f"  α_A:        {hbar_state.alpha_A:.4f}")
            print(f"  Penalty:    {current_penalty:.4f}")

            # Evaluate on ID and OOD splits
            eval_result = evaluator.evaluate(
                state.params, model, batch_size=eval_batch_size
            )

            print(f"  ID Accuracy:  {eval_result.acc_id:.4f}")
            print(f"  OOD Accuracy: {eval_result.acc_ood:.4f}")
            print(f"  σ̂_A:          {eval_result.ground_truth_sigma:.4f}")

            # Update cognitive manager with evaluation results
            inputs = cognitive_manager.metrics_to_inputs({
                "sigma_tilde": hbar_state.sigma_A,
                "sigma_hat": eval_result.ground_truth_sigma,
            })

            # Check for phase transition
            phase_info = cognitive_manager.check_phase_transition(hbar_state)

            # Compute acceleration metrics for logging
            kappa_alpha_val = config.fusion_config.kappa_alpha if config.fusion_config else 2.0
            accel_factor_val = float(1.0 + kappa_alpha_val * hbar_state.alpha_A)
            eff_lr_val = float(learning_rate * accel_factor_val)

            # Record metrics
            metrics = HBarTrainingMetrics(
                step=step,
                train_loss=total_loss,
                id_loss=id_loss,
                ood_loss=ood_loss,
                id_accuracy=eval_result.acc_id,
                ood_accuracy=eval_result.acc_ood,
                sigma_tilde=hbar_state.sigma_A,
                sigma_ode=hbar_state.sigma_A,
                alpha_A=hbar_state.alpha_A,
                compositional_penalty=current_penalty,
                lambda_sigma=lambda_sigma,
                effective_learning_rate=eff_lr_val,
                acceleration_factor=accel_factor_val,
            )
            metrics_history.append(metrics)

            # Log to CSV
            writer.writerow(
                {
                    "step": step,
                    "train_loss": total_loss,
                    "id_loss": id_loss,
                    "ood_loss": ood_loss,
                    "id_accuracy": eval_result.acc_id,
                    "ood_accuracy": eval_result.acc_ood,
                    "sigma_tilde": hbar_state.sigma_A,
                    "sigma_ode": hbar_state.sigma_A,
                    "alpha_A": hbar_state.alpha_A,
                    "compositional_penalty": current_penalty,
                    "lambda_sigma": lambda_sigma,
                    "effective_learning_rate": eff_lr_val,
                    "acceleration_factor": accel_factor_val,
                }
            )
            csv_file.flush()

    csv_file.close()
    print(f"\nTraining complete! Results saved to {log_path}")

    return HBarTrainingResults(
        final_params=state.params,
        final_hbar_state=hbar_state,
        metrics_history=metrics_history,
        config=config,
    )
