"""H-Bar signal computation engine.

This module implements the mathematical core of H-Bar signals:
- AC (Augmentation Consistency): c_A signal from Equation 5
- Future: GCA (Gradient-Composition Alignment), RGA (Representational-Geometry Alignment)

All functions are purely functional and JIT-compatible.
"""

from typing import Dict

import jax
import jax.numpy as jnp


def compute_augmentation_consistency(
    original_representations: Dict[str, jax.Array],
    augmented_representations: Dict[str, jax.Array],
    layer: str = "encoder_block_1",
) -> jax.Array:
    """Compute Augmentation Consistency (AC) signal c_A.

    Measures the cosine similarity between original and augmented representations
    at a specific layer (default: final encoder layer as the semantic bottleneck).

    The AC signal quantifies how invariant the model's representations are to
    structure-preserving augmentations. Higher values indicate better compositional
    consistency.

    Args:
        original_representations: Dictionary of activation tensors from the original
            batch, as returned by `get_model_representations()`.
        augmented_representations: Dictionary of activation tensors from the
            augmented batch.
        layer: The layer key to use for computing similarity. Default is the
            final encoder layer ("encoder_block_1" for 2-layer models).

    Returns:
        Scalar value in [0, 1] representing the mean cosine similarity across
        the batch. 1.0 indicates identical representations.

    Note:
        The final encoder layer is used as the primary "semantic bottleneck"
        where compositional schema should be most evident (per H-Bar paper
        Section 3.8.1).
    """
    # Get representations for the specified layer
    orig_repr = original_representations[layer]  # (batch, seq_len, d_model)
    aug_repr = augmented_representations[layer]

    # Compute cosine similarity along the feature dimension (d_model)
    # First, compute the dot product
    dot_product = jnp.sum(orig_repr * aug_repr, axis=-1)  # (batch, seq_len)

    # Compute norms
    orig_norm = jnp.sqrt(jnp.sum(orig_repr**2, axis=-1) + 1e-8)  # (batch, seq_len)
    aug_norm = jnp.sqrt(jnp.sum(aug_repr**2, axis=-1) + 1e-8)  # (batch, seq_len)

    # Compute cosine similarity
    cosine_sim = dot_product / (orig_norm * aug_norm)  # (batch, seq_len)

    # Mask out padding positions (cosine sim should be 0 for padding)
    # We use the original representation to detect padding (all zeros or near-zero)
    is_padding = orig_norm < 1e-6  # (batch, seq_len)
    cosine_sim = jnp.where(is_padding, 0.0, cosine_sim)

    # Average over sequence length and batch
    # Use mean over non-padding positions
    num_valid = jnp.sum(~is_padding)
    c_A = jnp.sum(cosine_sim) / num_valid

    # Clamp to [0, 1] range (cosine similarity can be negative, but we want
    # to measure consistency, so we map to [0, 1])
    c_A = (c_A + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]

    return jnp.clip(c_A, 0.0, 1.0)


def compute_layer_weighted_ac(
    original_representations: Dict[str, jax.Array],
    augmented_representations: Dict[str, jax.Array],
    layer_weights: Dict[str, float] | None = None,
) -> jax.Array:
    """Compute weighted AC signal across multiple layers.

    This function computes cosine similarity for each specified layer and
    combines them with learned or predefined weights. This can provide a
    more robust signal by integrating information from multiple levels
    of representation.

    Args:
        original_representations: Dictionary of activation tensors from original batch.
        augmented_representations: Dictionary of activation tensors from augmented batch.
        layer_weights: Dictionary mapping layer names to their weights. If None,
            uses uniform weights for all encoder block layers.

    Returns:
        Scalar value in [0, 1] representing the weighted mean cosine similarity.
    """
    # Default: use all encoder block layers with uniform weights
    if layer_weights is None:
        layer_weights = {}
        for key in original_representations:
            if key.startswith("encoder_block_"):
                layer_weights[key] = 1.0

    if not layer_weights:
        return jnp.array(1.0)

    # Compute weighted sum of cosine similarities
    total_weight = 0.0
    weighted_sum = 0.0

    for layer, weight in layer_weights.items():
        if layer in original_representations and layer in augmented_representations:
            layer_ac = compute_augmentation_consistency(
                original_representations, augmented_representations, layer
            )
            weighted_sum += weight * layer_ac
            total_weight += weight

    if total_weight == 0:
        return jnp.array(1.0)

    return weighted_sum / total_weight


def compute_representation_norm(
    representations: Dict[str, jax.Array],
    layer: str = "encoder_block_1",
) -> jax.Array:
    """Compute the average L2 norm of representations at a layer.

    This can be useful for monitoring representation magnitude during training.

    Args:
        representations: Dictionary of activation tensors.
        layer: The layer key to use.

    Returns:
        Scalar representing the mean L2 norm across batch and sequence.
    """
    repr_layer = representations[layer]  # (batch, seq_len, d_model)

    # Compute L2 norm along feature dimension
    norms = jnp.sqrt(jnp.sum(repr_layer**2, axis=-1))  # (batch, seq_len)

    # Average over batch and sequence
    return jnp.mean(norms)


def compute_ac_from_batch(
    original_representations: Dict[str, jax.Array],
    augmented_representations: Dict[str, jax.Array],
    layer: str = "encoder_block_1",
) -> jax.Array:
    """Compute AC signal c_A directly from representation dictionaries.

    This is a convenience wrapper around `compute_augmentation_consistency`
    that explicitly extracts representations from the encoder layer for
    the H-BarBatch workflow.

    Args:
        original_representations: Dict of activation tensors from original batch.
        augmented_representations: Dict of activation tensors from augmented batch.
        layer: The layer key to use for computing similarity.

    Returns:
        Scalar value in [0, 1] representing mean cosine similarity.
    """
    return compute_augmentation_consistency(
        original_representations, augmented_representations, layer
    )


def compute_gca(grad_id: jax.Array, grad_ood: jax.Array) -> jax.Array:
    """Compute Gradient-Composition Alignment (GCA) signal g_A.

    Pearson correlation coefficient between the flattened gradient vectors
    of the ID loss (∇_θ L_train) and OOD compositional loss (∇_θ L_comp).

    The GCA signal measures the alignment between "learning" and "generalizing"
    gradients. Per Equation 3 of the H-Bar paper, this quantifies whether the
    model is learning rules that apply to both in-distribution and
    out-of-distribution samples.

    Interpretation:
        - High GCA (>0.7): Model is "crystallizing" compositional rules that
          generalize well to novel compositions.
        - Low/Near-Zero GCA (0.0-0.3): ID patterns are unrelated to OOD
          structure, indicating the model is learning surface statistics.
        - Negative GCA (<0.0): Learning ID patterns actively harms OOD
          performance (severe overfitting to surface statistics).

    This computation includes all trainable parameters (embeddings + transformer
    layers) to measure Total Systemic Alignment. Compositional generalization
    in SCAN/COGS relies on Variable-Role Binding, which requires alignment
    between embeddings (Variables) and transformer layers (Roles/Functions).

    Args:
        grad_id: Flattened gradient vector from ID loss, shape (n_params,).
        grad_ood: Flattened gradient vector from OOD compositional loss,
            shape (n_params,).

    Returns:
        Scalar in [-1, 1] representing the Pearson correlation coefficient
        between the two gradient vectors.
    """
    # Center the gradients (subtract mean)
    mean_id = jnp.mean(grad_id)
    mean_ood = jnp.mean(grad_ood)

    id_centered = grad_id - mean_id
    ood_centered = grad_ood - mean_ood

    # Compute Pearson correlation numerator (covariance-like term)
    numerator = jnp.sum(id_centered * ood_centered)

    # Compute denominator (product of standard deviations)
    # Add epsilon for numerical stability with sparse gradients
    denominator = jnp.sqrt(
        jnp.sum(id_centered**2) * jnp.sum(ood_centered**2) + 1e-8
    )

    return numerator / denominator
