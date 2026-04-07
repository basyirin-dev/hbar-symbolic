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


def compute_rdm_representational(
    representations: jax.Array,
    method: str = "cosine",
) -> jax.Array:
    """Compute Representational Dissimilarity Matrix (RDM) from activation vectors.

    Given N activation vectors (e.g., BOS token representations from the final
    encoder layer), computes the pairwise distances to form an N×N RDM.

    The RDM captures the geometric structure of the model's internal representation
    space. Per the H-Bar paper, this should align with the structural RDM of the
    grammar if the model has learned compositional rules.

    Args:
        representations: Array of shape (N, d_model) containing activation vectors.
            Typically the BOS token representation from the final encoder layer,
            which acts as a sentence-level summary.
        method: Distance metric to use. Options:
            - "cosine": 1 - cosine_similarity (default)
            - "euclidean": Euclidean distance
            - "correlation": 1 - Pearson correlation

    Returns:
        N×N symmetric distance matrix with zeros on the diagonal.
    """
    N = representations.shape[0]

    if method == "cosine":
        # Normalize vectors
        norms = jnp.linalg.norm(representations, axis=-1, keepdims=True)
        norms = jnp.maximum(norms, 1e-8)  # Avoid division by zero
        normed = representations / norms

        # Cosine similarity matrix
        sim_matrix = normed @ normed.T  # (N, N)

        # Clip to [-1, 1] for numerical stability
        sim_matrix = jnp.clip(sim_matrix, -1.0, 1.0)

        # Convert to distance: 1 - similarity
        rdm = 1.0 - sim_matrix

    elif method == "euclidean":
        # Pairwise Euclidean distances
        diff = representations[:, jnp.newaxis, :] - representations[jnp.newaxis, :, :]
        rdm = jnp.sqrt(jnp.sum(diff**2, axis=-1))

    elif method == "correlation":
        # 1 - Pearson correlation
        mean = jnp.mean(representations, axis=-1, keepdims=True)
        centered = representations - mean

        norm = jnp.linalg.norm(centered, axis=-1, keepdims=True)
        norm = jnp.maximum(norm, 1e-8)
        normed = centered / norm

        corr_matrix = normed @ normed.T
        corr_matrix = jnp.clip(corr_matrix, -1.0, 1.0)

        rdm = 1.0 - corr_matrix

    else:
        raise ValueError(f"Unknown distance method: {method}")

    return rdm


def compute_rga(
    rdm_rep: jax.Array,
    rdm_struct: jax.Array,
) -> jax.Array:
    """Compute Representational-Geometry Alignment (RGA) signal r_A.

    Measures the alignment between the model's representational RDM and the
    structural RDM of the grammar using Spearman rank correlation.

    Per Equation 4 of the H-Bar paper, RGA quantifies whether the model's
    internal geometry reflects the compositional structure of the grammar.
    High RGA indicates that items with similar grammatical structure are
    represented similarly in the model's latent space.

    Args:
        rdm_rep: N×N representational dissimilarity matrix from the model.
        rdm_struct: N×N structural dissimilarity matrix from the grammar.
            Should be normalized to the same scale as rdm_rep.

    Returns:
        Scalar in [-1, 1] representing the Spearman rank correlation between
        the upper triangles of the two RDMs.
    """
    # Extract upper triangle (excluding diagonal) to avoid redundancy
    # and self-similarity
    N = rdm_rep.shape[0]
    mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)

    rep_flat = rdm_rep[mask]
    struct_flat = rdm_struct[mask]

    # Compute Spearman rank correlation
    # Rank the values
    rep_rank = _rank_data(rep_flat)
    struct_rank = _rank_data(struct_flat)

    # Compute Pearson correlation on ranks (Spearman)
    mean_rep = jnp.mean(rep_rank)
    mean_struct = jnp.mean(struct_rank)

    rep_centered = rep_rank - mean_rep
    struct_centered = struct_rank - mean_struct

    numerator = jnp.sum(rep_centered * struct_centered)
    denominator = jnp.sqrt(
        jnp.sum(rep_centered**2) * jnp.sum(struct_centered**2) + 1e-8
    )

    return numerator / denominator


def _rank_data(data: jax.Array) -> jax.Array:
    """Compute ranks for an array of values (handles ties with average rank).

    This is a JAX-compatible implementation of scipy.stats.rankdata with
    method='average' for tie handling.

    Args:
        data: 1D array of values to rank.

    Returns:
        1D array of ranks (float) with average ranks for ties.
    """
    # Get sorted indices
    sort_idx = jnp.argsort(data)

    # Create ranks (1-indexed)
    n = len(data)
    ranks = jnp.arange(1, n + 1, dtype=jnp.float32)

    # Assign ranks in original order
    ranks_sorted = ranks[sort_idx]

    # Handle ties by averaging ranks for equal values
    # Sort data values in the same order
    sorted_data = data[sort_idx]

    # Find tie groups
    # A tie occurs when consecutive sorted values are equal
    is_tie = jnp.concatenate([
        jnp.array([False]),
        sorted_data[:-1] == sorted_data[1:]
    ])

    # For each tie group, compute average rank
    if jnp.any(is_tie):
        # Find start and end of each tie group
        tie_start = jnp.where(~is_tie, size=n, fill_value=n)[0]
        tie_end = jnp.where(~is_tie, size=n, fill_value=n)[0]

        # Simple approach: use scipy-like ranking
        # For each unique value, assign average rank
        unique_vals, counts = jnp.unique(sorted_data, return_counts=True)

        # Compute cumulative sum to get positions
        cumsum = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(counts)
        ])

        # Average rank for each unique value
        avg_ranks = (cumsum[:-1] + cumsum[1:]) / 2.0 + 1.0

        # Map back to original positions
        val_to_rank = {float(v): float(r) for v, r in zip(unique_vals, avg_ranks)}
        avg_ranks_sorted = jnp.array([val_to_rank[float(v)] for v in sorted_data])

        # Unsort to original order
        unsort_idx = jnp.argsort(sort_idx)
        return avg_ranks_sorted[unsort_idx]
    else:
        # No ties, just unsort
        unsort_idx = jnp.argsort(sort_idx)
        return ranks_sorted[unsort_idx]


def compute_rga_from_representations(
    representations: jax.Array,
    structural_distances: jax.Array,
    method: str = "cosine",
) -> jax.Array:
    """Compute RGA signal directly from representations and structural distances.

    This is a convenience wrapper that combines RDM computation and RGA calculation.

    Args:
        representations: Array of shape (N, d_model) containing activation vectors.
        structural_distances: N×N structural dissimilarity matrix from the grammar.
        method: Distance metric for RDM_rep ("cosine", "euclidean", "correlation").

    Returns:
        Scalar in [-1, 1] representing the Spearman rank correlation.
    """
    rdm_rep = compute_rdm_representational(representations, method)
    return compute_rga(rdm_rep, structural_distances)


def fuse_hbar_signals(
    g_A: jax.Array,
    r_A: jax.Array,
    c_A: jax.Array,
    weights: Dict[str, float] | None = None,
) -> jax.Array:
    """Compute fused H-Bar signal σ̃_A via Equation 6.

    σ̃_A = w_g · max(0, g_A) + w_r · max(0, r_A) + w_c · c_A

    The max(0, x) rectifiers ensure negative alignment signals do not
    contribute to schema coherence — they are treated as zero coherence.
    This is critical because negative GCA/RGA indicates active harm to
    generalization, which should not reduce the fused signal below zero.

    The additive form (vs multiplicative) allows individual signals to
    "tug" the model out of the σ-trap even if others are near zero. For
    example, if GCA is high but RGA is low, the model can still achieve
    moderate σ̃_A. A multiplicative form would collapse if any signal is low.

    Args:
        g_A: Gradient-Composition Alignment signal (range: [-1, 1]).
        r_A: Representational-Geometry Alignment signal (range: [-1, 1]).
        c_A: Augmentation Consistency signal (range: [0, 1]).
        weights: Dictionary with keys 'w_g', 'w_r', 'w_c'. If None,
            uses default weights from H-Bar paper: w_g=0.4, w_r=0.35, w_c=0.25.

    Returns:
        Scalar value in [0, 1] representing the fused schema coherence estimate.

    Example:
        >>> # Baseline σ-trap signals
        >>> g_A = jnp.array(-0.0249)  # Negative GCA
        >>> r_A = jnp.array(0.0604)   # Low RGA
        >>> c_A = jnp.array(0.9901)   # High AC
        >>> sigma_tilde = fuse_hbar_signals(g_A, r_A, c_A)
        >>> # Result: ≈ 0.2686 (far below σ_critical ≈ 0.5)
    """
    # Default weights from H-Bar paper
    if weights is None:
        weights = {"w_g": 0.4, "w_r": 0.35, "w_c": 0.25}

    w_g = weights.get("w_g", 0.4)
    w_r = weights.get("w_r", 0.35)
    w_c = weights.get("w_c", 0.25)

    # Apply rectifiers: max(0, x) for g_A and r_A
    # Negative alignment should not contribute to schema coherence
    g_A_rectified = jnp.maximum(0.0, g_A)
    r_A_rectified = jnp.maximum(0.0, r_A)

    # Compute weighted sum (Equation 6)
    sigma_tilde = w_g * g_A_rectified + w_r * r_A_rectified + w_c * c_A

    # Clip to [0, 1] range for interpretability
    return jnp.clip(sigma_tilde, 0.0, 1.0)
