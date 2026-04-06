"""JIT-compatible batch preprocessing for seq2seq training.

This module provides functions for converting raw text pairs into
training-ready batches for the Seq2SeqTransformer. All operations
are designed to produce JAX arrays suitable for JIT compilation.

The HBarBatch class provides a multi-stream batch structure for
H-Bar signal extraction, containing ID, OOD, and augmentation streams.
"""

from dataclasses import dataclass
from typing import Sequence, Optional

import flax.struct
import jax
import jax.numpy as jnp

from hbar.engine.tokenizer import Tokenizer, PAD_TOKEN_ID, BOS_TOKEN_ID


@flax.struct.dataclass
class Batch:
    """Container for a batch of training data.

    This dataclass holds all components needed for a Transformer forward pass:
    - inputs: Encoder input token IDs
    - decoder_inputs: Decoder input token IDs (shifted right)
    - labels: Target token IDs for loss computation
    - src_mask: Padding mask for encoder
    - tgt_mask: Combined padding + causal mask for decoder

    Attributes:
        inputs: Encoder input of shape (batch, src_seq_len).
        decoder_inputs: Decoder input of shape (batch, tgt_seq_len).
        labels: Target labels of shape (batch, tgt_seq_len).
        src_mask: Encoder padding mask of shape (batch, 1, 1, src_seq_len).
        tgt_mask: Decoder mask of shape (batch, 1, tgt_seq_len, tgt_seq_len).
    """
    inputs: jax.Array
    decoder_inputs: jax.Array
    labels: jax.Array
    src_mask: jax.Array
    tgt_mask: jax.Array


def prepare_decoder_io(
    output_ids: jax.Array,
    bos_token_id: int = BOS_TOKEN_ID,
    pad_token_id: int = PAD_TOKEN_ID,
) -> tuple[jax.Array, jax.Array]:
    """Prepare decoder input and labels from output token IDs.

    For teacher forcing during training, the decoder receives the output
    sequence shifted right by one position (with BOS prepended), and the
    labels are the output sequence (which the model should predict).

    Args:
        output_ids: Output token IDs of shape (seq_len,).
        bos_token_id: Beginning-of-sequence token ID.
        pad_token_id: Padding token ID.

    Returns:
        A tuple of:
            - decoder_inputs: Input IDs shifted right with BOS prepended.
            - labels: Original output IDs for loss computation.
    """
    # Decoder input: [BOS, output[:-1]] (shifted right)
    decoder_input = jnp.concatenate([
        jnp.array([bos_token_id]),
        output_ids[:-1],
    ])

    # Labels are the original output
    labels = output_ids

    return decoder_input, labels


def prepare_batch(
    pairs: Sequence[tuple[str, str]],
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> Batch:
    """Prepare a batch of training data from raw text pairs.

    This function converts (input_command, output_action) string pairs
    into a Batch object containing all components needed for training.

    Args:
        pairs: Sequence of (input_text, output_text) tuples.
        tokenizer: Tokenizer instance for encoding text.
        max_seq_len: Maximum sequence length for padding/truncation.

    Returns:
        Batch object containing inputs, decoder_inputs, labels, and masks.
    """
    batch_size = len(pairs)

    # Encode all inputs and outputs
    input_ids_list = []
    output_ids_list = []

    for input_text, output_text in pairs:
        input_ids = tokenizer.encode(input_text, max_seq_len)
        output_ids = tokenizer.encode(output_text, max_seq_len)
        input_ids_list.append(input_ids)
        output_ids_list.append(output_ids)

    # Stack into batches
    input_ids = jnp.stack(input_ids_list, axis=0)  # (batch, max_seq_len)
    output_ids = jnp.stack(output_ids_list, axis=0)  # (batch, max_seq_len)

    # Prepare decoder inputs and labels
    decoder_inputs_list = []
    labels_list = []

    for out_ids in output_ids_list:
        dec_input, labels = prepare_decoder_io(out_ids)
        decoder_inputs_list.append(dec_input)
        labels_list.append(labels)

    decoder_inputs = jnp.stack(decoder_inputs_list, axis=0)
    labels = jnp.stack(labels_list, axis=0)

    # Generate masks
    from hbar.engine.encoding import get_padding_mask, get_decoder_mask

    # Encoder padding mask
    src_mask = get_padding_mask(input_ids, tokenizer.get_pad_token_id())

    # Decoder combined mask (padding + causal)
    tgt_mask = get_decoder_mask(decoder_inputs, tokenizer.get_pad_token_id())

    return Batch(
        inputs=input_ids,
        decoder_inputs=decoder_inputs,
        labels=labels,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
    )


def compute_loss(
    logits: jax.Array,
    labels: jax.Array,
    pad_token_id: int = PAD_TOKEN_ID,
) -> jax.Array:
    """Compute cross-entropy loss for sequence prediction.

    Computes the mean cross-entropy loss over all non-padding tokens.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        labels: Target token IDs of shape (batch, seq_len).
        pad_token_id: Token ID to ignore in loss computation.

    Returns:
        Scalar loss value.
    """
    # One-hot encode labels
    vocab_size = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, vocab_size)

    # Compute log softmax
    log_probs = jax.nn.log_softmax(logits)

    # Cross-entropy loss
    loss_per_token = -jnp.sum(one_hot * log_probs, axis=-1)

    # Mask out padding tokens
    mask = (labels != pad_token_id).astype(jnp.float32)
    loss_per_token = loss_per_token * mask

    # Mean over non-padding tokens
    num_valid_tokens = jnp.sum(mask)
    loss = jnp.sum(loss_per_token) / num_valid_tokens

    return loss


def compute_accuracy(
    logits: jax.Array,
    labels: jax.Array,
    pad_token_id: int = PAD_TOKEN_ID,
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
    correct = (predictions == labels)

    # Mask out padding tokens
    mask = (labels != pad_token_id).astype(jnp.float32)
    correct_masked = correct * mask

    # Compute accuracy over non-padding tokens
    num_valid_tokens = jnp.sum(mask)
    accuracy = jnp.sum(correct_masked) / num_valid_tokens

    return accuracy


@flax.struct.dataclass
class HBarBatch:
    """Multi-stream batch for H-Bar signal extraction.

    This dataclass extends the standard Batch to include three parallel
    streams needed for computing H-Bar signals:

    - id_stream: In-distribution samples for δ_A (Parametric Depth) optimization.
      These are simple, standard samples from the training distribution.

    - ood_stream: Out-of-distribution compositional probes from G(d) for
      g_A (GCA - Gradient-Composition Alignment) calculation. These samples
      test novel combinations of known primitives.

    - aug_stream: Structure-preserving augmentations for c_A (AC - Augmentation
      Consistency) calculation. These are created by swapping primitives
      (e.g., 'jump' → 'run') while keeping syntactic structure identical.

    This structure is a valid JAX pytree, allowing it to be passed into
    jit-compiled functions and used with jax.tree_util operations.

    Attributes:
        id_stream: Standard Batch of in-distribution samples.
        ood_stream: Batch of compositional probe samples.
        aug_stream: Batch of augmented samples.
    """
    id_stream: Batch
    ood_stream: Batch
    aug_stream: Batch


def get_hbar_batch(
    key: jax.Array,
    batch_size: int,
    domain: str = "scan",
    grammar_engine: Optional[object] = None,
    max_seq_len: Optional[int] = None,
) -> HBarBatch:
    """Generate a triple-stream H-Bar batch.

    This function generates all three streams needed for H-Bar signal
    extraction in a single call, splitting the PRNGKey for independent
    generation in each stream.

    The three streams are:
    1. ID stream: Simple in-distribution samples via GrammarEngine.generate_id_batch()
    2. OOD stream: Compositional probes via GrammarEngine.get_compositional_batch()
    3. Aug stream: Structure-preserving augmentations of ID samples

    Args:
        key: JAX PRNGKey for deterministic generation.
        batch_size: Number of samples per stream.
        domain: The domain ('scan' or 'cogs').
        grammar_engine: GrammarEngine instance for batch generation.
            If None, creates a new GrammarEngine with seed from key.
        max_seq_len: Maximum sequence length. If None, uses domain default
            (50 for SCAN, 80 for COGS).

    Returns:
        HBarBatch containing all three streams.
    """
    # Import here to avoid circular dependencies
    from hbar.benchmarks.grammar_engine import GrammarEngine
    from hbar.engine.augmentation import (
        vmap_augment_batch,
        generate_augmentation_keys,
    )

    # Set default max_seq_len
    if max_seq_len is None:
        max_seq_len = 50 if domain == "scan" else 80

    # Create grammar engine if not provided
    if grammar_engine is None:
        seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
        grammar_engine = GrammarEngine(seed=seed)

    # Split key for three parallel operations
    id_key, ood_key, aug_key = jax.random.split(key, 3)

    # Generate ID stream (in-distribution samples)
    # Note: This uses Python random internally, so we seed from JAX key
    # Convert key to int safely (avoid overflow)
    id_seed = int(jax.random.randint(id_key, (), 0, 2**31 - 1))
    id_engine = GrammarEngine(seed=id_seed)
    id_stream = id_engine.generate_id_batch(batch_size, domain=domain)

    # Generate OOD stream (compositional probes)
    ood_seed = int(jax.random.randint(ood_key, (), 0, 2**31 - 1))
    ood_engine = GrammarEngine(seed=ood_seed)
    ood_stream = ood_engine.get_compositional_batch(
        batch_size, domain=domain, rng=ood_key
    )

    # Generate Aug stream (structure-preserving augmentations)
    # Take the ID inputs and apply primitive substitutions
    tokenizer = grammar_engine.get_tokenizer(domain)

    # Generate augmentation keys for each sample
    aug_keys = generate_augmentation_keys(aug_key, batch_size)

    # Apply vectorized augmentation to ID stream inputs
    aug_inputs = vmap_augment_batch(
        id_stream.inputs, aug_keys, tokenizer, domain
    )

    # For the aug_stream, we need to create a full Batch
    # The augmentation only changes the input tokens, not the structure
    # We keep the same decoder_inputs, labels, and masks since the structure is preserved
    # But we need to apply the same token substitutions to decoder_inputs and labels
    aug_decoder_inputs = vmap_augment_batch(
        id_stream.decoder_inputs, aug_keys, tokenizer, domain
    )
    aug_labels = vmap_augment_batch(
        id_stream.labels, aug_keys, tokenizer, domain
    )

    aug_stream = Batch(
        inputs=aug_inputs,
        decoder_inputs=aug_decoder_inputs,
        labels=aug_labels,
        src_mask=id_stream.src_mask,
        tgt_mask=id_stream.tgt_mask,
    )

    return HBarBatch(
        id_stream=id_stream,
        ood_stream=ood_stream,
        aug_stream=aug_stream,
    )


def prepare_hbar_batch_from_pairs(
    id_pairs: Sequence[tuple[str, str]],
    ood_pairs: Sequence[tuple[str, str]],
    aug_pairs: Sequence[tuple[str, str]],
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> HBarBatch:
    """Prepare an HBarBatch from pre-generated pairs.

    This is useful when you want more control over the pair generation
    process, such as when using specific compositional probes.

    Args:
        id_pairs: In-distribution (input, output) pairs.
        ood_pairs: Out-of-distribution compositional probe pairs.
        aug_pairs: Augmented pairs (structure-preserving variants).
        tokenizer: Tokenizer instance.
        max_seq_len: Maximum sequence length.

    Returns:
        HBarBatch containing all three streams.
    """
    id_stream = prepare_batch(id_pairs, tokenizer, max_seq_len)
    ood_stream = prepare_batch(ood_pairs, tokenizer, max_seq_len)
    aug_stream = prepare_batch(aug_pairs, tokenizer, max_seq_len)

    return HBarBatch(
        id_stream=id_stream,
        ood_stream=ood_stream,
        aug_stream=aug_stream,
    )
