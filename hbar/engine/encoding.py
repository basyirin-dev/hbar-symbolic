"""Functional mask generation for JAX Transformer attention mechanisms.

This module provides JIT-compatible functions for generating attention masks
used in the Transformer's multi-head attention layers. All functions are
designed to work with padded sequences and produce boolean masks compatible
with JAX's functional paradigm.

Note: Functions that take sequence length as a parameter (get_causal_mask)
cannot be JIT-compiled because shape parameters must be static. Instead,
they should be called outside of JIT-compiled code or with static_argnums.
"""

import jax
import jax.numpy as jnp


def get_padding_mask(
    token_ids: jax.Array,
    pad_token_id: int = 0,
) -> jax.Array:
    """Generate padding mask for attention mechanisms.

    Creates a boolean mask indicating which tokens are valid (non-padding).
    The mask has shape (batch, 1, 1, seq_len) for broadcasting with
    attention scores in multi-head attention.

    This function is JIT-compatible as it doesn't depend on dynamic shapes.

    Args:
        token_ids: Token ID array of shape (batch, seq_len) or (seq_len,).
        pad_token_id: The ID of the padding token (default: 0).

    Returns:
        Boolean mask of shape (batch, 1, 1, seq_len) where True indicates
        a valid token and False indicates a padding token.
    """
    # Handle unbatched input
    if token_ids.ndim == 1:
        token_ids = token_ids[jnp.newaxis, :]

    # Create mask: True for valid tokens, False for padding
    mask = token_ids != pad_token_id

    # Reshape for broadcasting with attention scores
    # (batch, seq_len) -> (batch, 1, 1, seq_len)
    mask = mask[:, jnp.newaxis, jnp.newaxis, :]

    return mask


def get_causal_mask(seq_len: int) -> jax.Array:
    """Generate causal (triangular) mask for decoder self-attention.

    Creates a lower triangular mask that prevents the decoder from
    attending to future tokens during training.

    Note: This function cannot be JIT-compiled because seq_len is used
    as a shape parameter, which must be static in JAX. Call this function
    outside of jax.jit or use static_argnums when JIT-compiling.

    Args:
        seq_len: Length of the sequence (must be a concrete Python int).

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True indicates
        the token can be attended to and False indicates it should be masked.
    """
    # Create lower triangular mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask


def get_decoder_mask(
    token_ids: jax.Array,
    pad_token_id: int = 0,
) -> jax.Array:
    """Generate combined padding and causal mask for decoder self-attention.

    This function creates a mask that combines padding information with
    causal masking, suitable for decoder self-attention during training.

    Note: This function is JIT-compatible because it derives seq_len from
    the input tensor shape rather than taking it as a parameter.

    Args:
        token_ids: Token ID array of shape (batch, seq_len).
        pad_token_id: The ID of the padding token (default: 0).

    Returns:
        Boolean mask of shape (batch, 1, seq_len, seq_len) that can be
        directly applied to attention scores.
    """
    batch_size, seq_len = token_ids.shape

    # Get padding mask: (batch, 1, 1, seq_len)
    padding_mask = get_padding_mask(token_ids, pad_token_id)

    # Get causal mask: (seq_len, seq_len)
    causal_mask = get_causal_mask(seq_len)

    # Expand padding mask to (batch, 1, seq_len, seq_len)
    padding_mask_expanded = jnp.broadcast_to(
        padding_mask,
        (batch_size, 1, seq_len, seq_len),
    )

    # Combine masks: both must be True for attention to be allowed
    combined = jnp.logical_and(padding_mask_expanded, causal_mask[jnp.newaxis, jnp.newaxis, :, :])
    return combined


def apply_mask(
    attention_scores: jax.Array,
    mask: jax.Array,
    mask_value: float = -1e9,
) -> jax.Array:
    """Apply boolean mask to attention scores.

    Sets attention scores to a large negative value where the mask is False,
    effectively zeroing out those attention weights after softmax.

    This function is JIT-compatible.

    Args:
        attention_scores: Attention scores of shape (batch, heads, q_len, k_len).
        mask: Boolean mask of shape broadcastable to attention_scores.
        mask_value: Value to set for masked positions (default: -1e9).

    Returns:
        Masked attention scores.
    """
    return jnp.where(mask, attention_scores, mask_value)
