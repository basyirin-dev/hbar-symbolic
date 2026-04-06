"""H-Bar Evaluator for ground-truth σ_A calculation.

This module implements the evaluation engine that computes the ground-truth
schema coherence signal (σ̂_A) as defined in Equation 7 of the H-Bar paper:

    σ̂_A = Acc_OOD / Acc_ID

This ratio measures the model's ability to generalize compositionally.
A value close to 1.0 indicates good compositional generalization,
while a value close to 0.0 indicates the model is trapped in the
"illusion of mastery" (high ID accuracy but low OOD accuracy).

The evaluator loads frozen evaluation sets from the data/ directory
to ensure reproducibility across all training runs.
"""

from functools import partial
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from hbar.engine.data_utils import Batch, prepare_batch
from hbar.engine.tokenizer import Tokenizer, create_scan_tokenizer


@dataclass(frozen=True)
class EvaluationResult:
    """Results from a full evaluation run.

    Attributes:
        acc_id: Accuracy on in-distribution evaluation set.
        acc_ood: Accuracy on out-of-distribution evaluation set.
        loss_id: Loss on in-distribution evaluation set.
        loss_ood: Loss on out-of-distribution evaluation set.
        ground_truth_sigma: σ̂_A = Acc_OOD / Acc_ID (Eq. 7).
        n_id: Number of ID samples evaluated.
        n_ood: Number of OOD samples evaluated.
    """

    acc_id: float
    acc_ood: float
    loss_id: float
    loss_ood: float
    ground_truth_sigma: float
    n_id: int
    n_ood: int


class Evaluator:
    """H-Bar evaluation engine for ground-truth σ_A calculation.

    This class loads frozen evaluation sets and computes the ground-truth
    schema coherence signal used for Stage 2 calibration in the H-Bar framework.

    The evaluator is designed to be called periodically during training
    (every T_eval steps) to track the model's compositional generalization
    ability and update the multi-signal fusion weights.

    Attributes:
        domain: The domain ('scan' or 'cogs') being evaluated.
        id_samples: List of (input, output) pairs for ID evaluation.
        ood_samples: List of (input, output) pairs for OOD evaluation.
        tokenizer: Tokenizer for the domain.
        max_seq_len: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        domain: str = "scan",
        data_dir: str = "data",
        id_eval_file: Optional[str] = None,
        ood_eval_file: Optional[str] = None,
    ):
        """Initialize the evaluator.

        Args:
            domain: The domain ('scan' or 'cogs') to evaluate.
            data_dir: Directory containing the frozen evaluation sets.
            id_eval_file: Optional custom filename for ID eval set.
            ood_eval_file: Optional custom filename for OOD eval set.
        """
        self.domain = domain
        self.data_dir = data_dir

        # Set default filenames based on domain
        if id_eval_file is None:
            id_eval_file = f"{domain}_id_eval.json"
        if ood_eval_file is None:
            ood_eval_file = f"{domain}_ood_eval.json"

        # Load evaluation sets
        self.id_samples = self._load_eval_set(os.path.join(data_dir, id_eval_file))
        self.ood_samples = self._load_eval_set(os.path.join(data_dir, ood_eval_file))

        # Initialize tokenizer
        if domain == "scan":
            self.tokenizer = create_scan_tokenizer()
            self.max_seq_len = 50
        else:
            # For COGS, create tokenizer with appropriate vocabulary
            self.tokenizer = self._create_cogs_tokenizer()
            self.max_seq_len = 80

        print(f"Evaluator initialized for {domain}:")
        print(f"  ID samples: {len(self.id_samples)}")
        print(f"  OOD samples: {len(self.ood_samples)}")

    def _load_eval_set(self, filepath: str) -> List[Tuple[str, str]]:
        """Load an evaluation set from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            List of (input_text, output_text) tuples.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return [(item["input"], item["output"]) for item in data]

    def _create_cogs_tokenizer(self) -> Tokenizer:
        """Create tokenizer for COGS domain.

        Returns:
            Tokenizer initialized with COGS vocabulary.
        """
        from hbar.benchmarks.cogs_grammar import COGSGrammar

        grammar = COGSGrammar()
        vocab = grammar.get_vocabulary()
        return Tokenizer(vocab)

    def get_tokenizer(self) -> Tokenizer:
        """Get the tokenizer for this evaluator's domain.

        Returns:
            Tokenizer instance.
        """
        return self.tokenizer

    def prepare_evaluation_batches(
        self,
        batch_size: int = 32,
    ) -> List[Batch]:
        """Prepare evaluation data as a list of batches.

        Args:
            batch_size: Batch size for evaluation.

        Returns:
            List of Batch objects for evaluation.
        """
        id_batches = []
        ood_batches = []

        # Prepare ID batches
        for i in range(0, len(self.id_samples), batch_size):
            batch_samples = self.id_samples[i : i + batch_size]
            batch = prepare_batch(batch_samples, self.tokenizer, self.max_seq_len)
            id_batches.append(batch)

        # Prepare OOD batches
        for i in range(0, len(self.ood_samples), batch_size):
            batch_samples = self.ood_samples[i : i + batch_size]
            batch = prepare_batch(batch_samples, self.tokenizer, self.max_seq_len)
            ood_batches.append(batch)

        return id_batches, ood_batches

    def create_eval_step(self, model: Any) -> Callable:
        """Create a JIT-compiled evaluation step function.

        Args:
            model: The model to evaluate (captured via closure, not traced).

        Returns:
            A callable that takes (params, batch) and returns (loss, accuracy).
        """

        @jax.jit
        def eval_step(
            params: Any,
            batch: Batch,
        ) -> Tuple[jax.Array, jax.Array]:
            """Evaluate model on a single batch.

            Args:
                params: Model parameters.
                batch: Batch of evaluation data.

            Returns:
                Tuple of (loss, accuracy).
            """
            # Run forward pass
            logits = model.apply(
                {"params": params},
                batch.inputs,
                batch.decoder_inputs,
                training=False,
            )

            # Compute cross-entropy loss
            one_hot = jax.nn.one_hot(batch.labels, logits.shape[-1])
            log_probs = jax.nn.log_softmax(logits)
            # Create padding mask from labels with shape (batch, seq_len, 1) for broadcasting
            mask = (batch.labels != 0).astype(jnp.float32)[:, :, jnp.newaxis]
            loss = -jnp.sum(one_hot * log_probs * mask) / jnp.sum(mask)

            # Compute accuracy
            predictions = jnp.argmax(logits, axis=-1)
            correct = (predictions == batch.labels) * mask.astype(jnp.int32)
            accuracy = jnp.sum(correct) / jnp.sum(mask)

            return loss, accuracy

        return eval_step

    def evaluate(
        self,
        params: Any,
        model: Any,
        batch_size: int = 32,
    ) -> EvaluationResult:
        """Run full evaluation and compute ground-truth σ_A.

        Args:
            params: Model parameters.
            model: The model to evaluate.
            batch_size: Batch size for evaluation.

        Returns:
            EvaluationResult with all metrics.
        """
        eval_step = self.create_eval_step(model)
        id_batches, ood_batches = self.prepare_evaluation_batches(batch_size)

        # Evaluate on ID set
        id_losses = []
        id_accs = []
        for batch in id_batches:
            loss, acc = eval_step(params, batch)
            id_losses.append(float(loss))
            id_accs.append(float(acc))

        acc_id = float(jnp.mean(jnp.array(id_accs)))
        loss_id = float(jnp.mean(jnp.array(id_losses)))

        # Evaluate on OOD set
        ood_losses = []
        ood_accs = []
        for batch in ood_batches:
            loss, acc = eval_step(params, batch)
            ood_losses.append(float(loss))
            ood_accs.append(float(acc))

        acc_ood = float(jnp.mean(jnp.array(ood_accs)))
        loss_ood = float(jnp.mean(jnp.array(ood_losses)))

        # Compute ground-truth σ_A (Eq. 7)
        # Handle division by zero: if acc_id is 0, return 0.0
        if acc_id > 0:
            ground_truth_sigma = acc_ood / acc_id
        else:
            ground_truth_sigma = 0.0

        return EvaluationResult(
            acc_id=acc_id,
            acc_ood=acc_ood,
            loss_id=loss_id,
            loss_ood=loss_ood,
            ground_truth_sigma=ground_truth_sigma,
            n_id=len(self.id_samples),
            n_ood=len(self.ood_samples),
        )

    @staticmethod
    def calculate_calibration_error(
        sigma_tilde: float,
        sigma_hat: float,
    ) -> float:
        """Calculate calibration error between proxy and ground-truth σ_A.

        This is used to update the multi-signal fusion weights (w_g, w_r, w_c)
        in Phase 2 of the H-Bar framework.

        Args:
            sigma_tilde: Training-time proxy estimate (σ̃_A).
            sigma_hat: Ground-truth estimate (σ̂_A).

        Returns:
            Absolute calibration error |σ̃_A - σ̂_A|.
        """
        return abs(sigma_tilde - sigma_hat)


def create_evaluator(
    domain: str = "scan",
    data_dir: str = "data",
) -> Evaluator:
    """Factory function to create an evaluator for a specific domain.

    Args:
        domain: The domain ('scan' or 'cogs').
        data_dir: Directory containing frozen evaluation sets.

    Returns:
        Evaluator instance.
    """
    return Evaluator(domain=domain, data_dir=data_dir)
