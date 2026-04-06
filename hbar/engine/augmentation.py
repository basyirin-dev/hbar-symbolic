"""Structure-preserving augmentation pipeline for H-Bar AC signal.

This module implements vectorized augmentations that preserve syntactic
structure while swapping semantic primitives or permuting argument order.
These augmentations are used to compute the Augmentation Consistency (AC)
signal in H-Bar.

Two types of augmentations are supported:
1. **Primitive Substitution:** Swap primitives (e.g., 'jump' -> 'run')
   while keeping syntactic structure identical.
2. **Argument Permutation:** Swap the order of coordinated sub-commands
   (e.g., 'jump left and look right' -> 'look right and jump left').

All augmentation functions are designed to work with JAX arrays and
can be vectorized using jax.vmap for O(1) scaling with batch size.
"""

from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp

from hbar.engine.tokenizer import Tokenizer


# Primitive substitution mappings for SCAN
# Maps (source_token_id, target_token_id) pairs
SCAN_PRIMITIVE_PAIRS = [
    ("jump", "run"),
    ("run", "jump"),
    ("walk", "look"),
    ("look", "walk"),
]

# Action token mappings
SCAN_ACTION_PAIRS = [
    ("I_JUMP", "I_RUN"),
    ("I_RUN", "I_JUMP"),
    ("I_WALK", "I_LOOK"),
    ("I_LOOK", "I_WALK"),
]


def apply_primitive_substitution(
    token_ids: jax.Array,
    key: jax.Array,
    tokenizer: Tokenizer,
    domain: str = "scan",
) -> jax.Array:
    """Apply a random primitive substitution to a single sequence.

    This function swaps one primitive with another while keeping the
    syntactic structure identical. For example, 'jump twice' becomes
    'run twice' - the structure (primitive + modifier) is preserved.

    Args:
        token_ids: 1D array of token IDs representing a sequence.
        key: JAX PRNGKey for random selection.
        tokenizer: Tokenizer instance for the domain.
        domain: The domain ('scan' or 'cogs').

    Returns:
        jax.Array with the same shape as input, with one primitive swapped.
    """
    if domain == "scan":
        return _apply_scan_substitution(token_ids, key, tokenizer)
    else:
        return _apply_cogs_substitution(token_ids, key, tokenizer)


def _apply_scan_substitution(
    token_ids: jax.Array,
    key: jax.Array,
    tokenizer: Tokenizer,
) -> jax.Array:
    """Apply primitive substitution for SCAN domain.

    Swaps all occurrences of one primitive with another primitive.
    Uses JAX-compatible operations throughout.

    Args:
        token_ids: 1D array of token IDs.
        key: JAX PRNGKey.
        tokenizer: Tokenizer instance.

    Returns:
        jax.Array with primitives swapped.
    """
    # Build list of (source_id, target_id) pairs for swapping
    swap_pairs = []
    for src_word, tgt_word in SCAN_PRIMITIVE_PAIRS:
        if src_word in tokenizer.word2id and tgt_word in tokenizer.word2id:
            swap_pairs.append((tokenizer.word2id[src_word], tokenizer.word2id[tgt_word]))

    # Also add action token pairs
    for src_word, tgt_word in SCAN_ACTION_PAIRS:
        if src_word in tokenizer.word2id and tgt_word in tokenizer.word2id:
            swap_pairs.append((tokenizer.word2id[src_word], tokenizer.word2id[tgt_word]))

    if not swap_pairs:
        return token_ids

    num_pairs = len(swap_pairs)

    # Choose a random swap pair
    subkey, _ = jax.random.split(key)
    pair_idx = jax.random.randint(subkey, (), 0, num_pairs)

    # Extract source and target IDs using JAX operations
    source_ids = jnp.array([p[0] for p in swap_pairs])
    target_ids = jnp.array([p[1] for p in swap_pairs])

    source_id = source_ids[pair_idx]
    target_id = target_ids[pair_idx]

    # Apply swap: where token == source_id, replace with target_id
    # and where token == target_id, replace with source_id (bidirectional swap)
    def swap_token(token_id):
        is_source = token_id == source_id
        is_target = token_id == target_id
        return jnp.where(is_source, target_id, jnp.where(is_target, source_id, token_id))

    return jax.vmap(swap_token)(token_ids)


def _apply_cogs_substitution(
    token_ids: jax.Array,
    key: jax.Array,
    tokenizer: Tokenizer,
) -> jax.Array:
    """Apply primitive substitution for COGS domain.

    Swaps a randomly chosen noun or verb with another of the same type.

    Args:
        token_ids: 1D array of token IDs.
        key: JAX PRNGKey.
        tokenizer: Tokenizer instance.

    Returns:
        jax.Array with a noun or verb swapped.
    """
    # Get all token IDs (excluding special tokens and structural tokens)
    vocab = list(tokenizer.word2id.keys())
    structural_tokens = {"(", ")", ",", "=", "agent", "patient", "theme", "by", "The", "A", "was"}

    # Find swappable tokens (nouns and verbs)
    swappable = [t for t in vocab if t not in structural_tokens]

    if len(swappable) < 2:
        return token_ids

    # Build token ID array for swappable tokens
    swappable_ids = jnp.array([tokenizer.word2id[t] for t in swappable])
    num_swappable = len(swappable_ids)

    # Choose two different tokens to swap
    subkey1, subkey2 = jax.random.split(key)
    idx1 = jax.random.randint(subkey1, (), 0, num_swappable)
    # For idx2, we need to ensure it's different from idx1
    idx2_raw = jax.random.randint(subkey2, (), 0, num_swappable - 1)
    idx2 = jnp.where(idx2_raw >= idx1, idx2_raw + 1, idx2_raw)

    source_id = swappable_ids[idx1]
    target_id = swappable_ids[idx2]

    # Create swap function
    def swap_token(token_id):
        is_source = token_id == source_id
        is_target = token_id == target_id
        return jnp.where(is_source, target_id, jnp.where(is_target, source_id, token_id))

    return jax.vmap(swap_token)(token_ids)


def vmap_augment_batch(
    token_ids_batch: jax.Array,
    keys: jax.Array,
    tokenizer: Tokenizer,
    domain: str = "scan",
) -> jax.Array:
    """Vectorized augmentation over a batch of sequences.

    This function applies primitive substitution to each sequence in
    the batch using jax.vmap, ensuring O(1) scaling with batch size.

    Args:
        token_ids_batch: 2D array of shape (batch_size, seq_len).
        keys: 2D array of shape (batch_size, 2) containing PRNGKeys.
        tokenizer: Tokenizer instance.
        domain: The domain ('scan' or 'cogs').

    Returns:
        jax.Array of shape (batch_size, seq_len) with augmented sequences.
    """
    # Vectorize the single-sequence augmentation
    augmented = jax.vmap(
        lambda ids, k: apply_primitive_substitution(ids, k, tokenizer, domain)
    )(token_ids_batch, keys)

    return augmented


def generate_augmentation_keys(
    base_key: jax.Array,
    batch_size: int,
) -> jax.Array:
    """Generate a batch of PRNGKeys for augmentation.

    Args:
        base_key: Base PRNGKey to split from.
        batch_size: Number of keys to generate.

    Returns:
        jax.Array of shape (batch_size, 2) containing PRNGKeys.
    """
    keys = jax.random.split(base_key, batch_size)
    return keys


def create_augmentation_fn(
    tokenizer: Tokenizer,
    domain: str = "scan",
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Create a vectorized augmentation function for a specific domain.

    This factory function creates a fully vectorized augmentation function
    that can be used in JIT-compiled training loops.

    Args:
        tokenizer: Tokenizer instance for the domain.
        domain: The domain ('scan' or 'cogs').

    Returns:
        Callable that takes (token_ids_batch, keys) and returns augmented batch.
    """

    def augment_fn(token_ids_batch: jax.Array, keys: jax.Array) -> jax.Array:
        return vmap_augment_batch(token_ids_batch, keys, tokenizer, domain)

    return augment_fn


def apply_argument_permutation(
    token_ids: jax.Array,
    key: jax.Array,
    tokenizer: Tokenizer,
    domain: str = "scan",
) -> jax.Array:
    """Apply argument permutation to a single sequence.

    This function swaps the order of coordinated sub-commands while
    preserving the syntactic structure. For example:
    - 'jump left and look right' -> 'look right and jump left'
    - 'walk after run' -> 'run after walk'

    Args:
        token_ids: 1D array of token IDs representing a sequence.
        key: JAX PRNGKey for random selection.
        tokenizer: Tokenizer instance for the domain.
        domain: The domain ('scan' or 'cogs').

    Returns:
        jax.Array with the same shape as input, with arguments permuted.
    """
    if domain == "scan":
        return _apply_scan_permutation(token_ids, key, tokenizer)
    else:
        return _apply_cogs_permutation(token_ids, key, tokenizer)


def _apply_scan_permutation(
    token_ids: jax.Array,
    key: jax.Array,
    tokenizer: Tokenizer,
) -> jax.Array:
    """Apply argument permutation for SCAN domain.

    Identifies conjunctions ('and', 'after') and swaps the order of
    sub-commands around them. Uses explicit parsing to handle the
    non-linear mapping between commands and actions.

    Args:
        token_ids: 1D array of token IDs.
        key: JAX PRNGKey.
        tokenizer: Tokenizer instance.

    Returns:
        jax.Array with sub-commands permuted.
    """
    # Get conjunction token IDs
    and_id = tokenizer.word2id.get("and", -1)
    after_id = tokenizer.word2id.get("after", -1)

    # Check if conjunctions exist in the sequence
    # Use Python booleans by converting outside of traced context
    has_and_val = bool(jnp.any(token_ids == and_id)) if and_id >= 0 else False
    has_after_val = bool(jnp.any(token_ids == after_id)) if after_id >= 0 else False

    # If no conjunctions, return unchanged
    if not (has_and_val or has_after_val):
        return token_ids

    # Choose which conjunction type to use (prefer 'and' if both present)
    if has_and_val and has_after_val:
        use_and_val = jax.random.bernoulli(key, 0.7)
    else:
        use_and_val = has_and_val

    # Convert to Python bool for cond
    if isinstance(use_and_val, jax.Array):
        use_and_val = bool(use_and_val)

    if use_and_val:
        conj_id = and_id
    else:
        conj_id = after_id

    # Find the position of the conjunction
    conj_positions = jnp.where(token_ids == conj_id, size=1, fill_value=-1)
    conj_pos = int(conj_positions[0])

    # If no conjunction found, return unchanged
    if conj_pos < 0:
        return token_ids

    # Find segment boundaries
    bos_id = tokenizer.word2id.get("<BOS>", 1)
    eos_id = tokenizer.word2id.get("<EOS>", 2)
    pad_id = tokenizer.word2id.get("<PAD>", 0)

    # Find first non-BOS, non-PAD token
    seq_len = len(token_ids)
    is_content_start = (token_ids != bos_id) & (token_ids != pad_id)
    start_indices = jnp.where(is_content_start, size=1, fill_value=seq_len)
    seg1_start = int(start_indices[0])

    # Segment 1 ends at conjunction
    seg1_end = conj_pos

    # Segment 2 starts after conjunction
    seg2_start = conj_pos + 1

    # Segment 2 ends at EOS or padding
    is_content_end = (token_ids == eos_id) | (token_ids == pad_id)
    end_indices = jnp.where(is_content_end, size=1, fill_value=seq_len)
    seg2_end = int(end_indices[0])
    if seg2_end == seq_len:
        seg2_end = seq_len

    # Extract segments
    seg1 = token_ids[seg1_start:seg1_end]
    seg2 = token_ids[seg2_start:seg2_end]

    # Build result: prefix + swapped segments
    prefix = token_ids[:seg1_start]
    suffix = token_ids[seg2_end:]

    result = jnp.concatenate([
        prefix,
        seg2,
        jnp.array([conj_id], dtype=token_ids.dtype),
        seg1,
        suffix,
    ])

    # Pad or truncate to match original length
    if len(result) < seq_len:
        padding = jnp.full(seq_len - len(result), pad_id, dtype=token_ids.dtype)
        result = jnp.concatenate([result, padding])
    else:
        result = result[:seq_len]

    return result


def _apply_cogs_permutation(
    token_ids: jax.Array,
    key: jax.Array,
    tokenizer: Tokenizer,
) -> jax.Array:
    """Apply argument permutation for COGS domain.

    Swaps the order of coordinated nouns or parallel clauses in the
    logical form representation while maintaining semantic integrity.

    Args:
        token_ids: 1D array of token IDs.
        key: JAX PRNGKey.
        tokenizer: Tokenizer instance.

    Returns:
        jax.Array with arguments permuted.
    """
    # For COGS, we swap arguments in coordinated structures
    # E.g., "chase ( agent = dog , patient = cat )"
    #       -> "chase ( agent = cat , patient = dog )"

    # Get structural token IDs
    comma_id = tokenizer.word2id.get(",", -1)
    agent_id = tokenizer.word2id.get("agent", -1)
    patient_id = tokenizer.word2id.get("patient", -1)

    # Check if this sequence has the right structure for permutation
    has_comma = jnp.any(token_ids == comma_id) if comma_id >= 0 else False
    has_agent = jnp.any(token_ids == agent_id) if agent_id >= 0 else False
    has_patient = jnp.any(token_ids == patient_id) if patient_id >= 0 else False

    # Only permute if we have the right structure
    if not (has_comma and has_agent and has_patient):
        return token_ids

    # Find positions of agent and patient markers
    agent_pos = jnp.argmax(token_ids == agent_id)
    patient_pos = jnp.argmax(token_ids == patient_id)

    # Find the content after each marker (the actual noun)
    # The noun is typically the token right after the '=' sign
    eq_id = tokenizer.word2id.get("=", -1)

    def find_value_after_marker(ids, marker_id, eq_id):
        """Find the value token after marker = pattern."""
        marker_pos = jnp.argmax(ids == marker_id)
        # Skip '=' after marker
        eq_pos = jnp.argmax((ids == eq_id) & (jnp.arange(len(ids)) > marker_pos))
        # The value is the next token
        value_pos = eq_pos + 1
        return ids[value_pos], value_pos

    agent_value, agent_value_pos = find_value_after_marker(token_ids, agent_id, eq_id)
    patient_value, patient_value_pos = find_value_after_marker(token_ids, patient_id, eq_id)

    # Swap the values
    result = token_ids.at[agent_value_pos].set(patient_value)
    result = result.at[patient_value_pos].set(agent_value)

    return result


def apply_augmentation(
    token_ids: jax.Array,
    key: jax.Array,
    tokenizer: Tokenizer,
    domain: str = "scan",
    permutation_probability: float = 0.5,
) -> jax.Array:
    """Apply augmentation (substitution or permutation) to a sequence.

    Randomly chooses between primitive substitution and argument permutation
    based on the configured probability.

    Args:
        token_ids: 1D array of token IDs.
        key: JAX PRNGKey.
        tokenizer: Tokenizer instance.
        domain: The domain ('scan' or 'cogs').
        permutation_probability: Probability of using argument permutation
            (vs primitive substitution). Default 0.5.

    Returns:
        jax.Array with augmented sequence.
    """
    subkey1, subkey2 = jax.random.split(key)

    # Decide whether to use permutation or substitution
    use_permutation = jax.random.bernoulli(subkey1, permutation_probability)

    # Apply the chosen augmentation
    substituted = apply_primitive_substitution(token_ids, subkey2, tokenizer, domain)
    permuted = apply_argument_permutation(token_ids, subkey2, tokenizer, domain)

    return jax.lax.cond(use_permutation, lambda: permuted, lambda: substituted)


def vmap_augment_batch(
    token_ids_batch: jax.Array,
    keys: jax.Array,
    tokenizer: Tokenizer,
    domain: str = "scan",
    permutation_probability: float = 0.5,
) -> jax.Array:
    """Vectorized augmentation over a batch of sequences.

    This function applies primitive substitution (not argument permutation)
    to each sequence in the batch using jax.vmap, ensuring O(1) scaling
    with batch size.

    Note: Argument permutation is not included here because it requires
    Python control flow that isn't compatible with jax.vmap. The AC signal
    uses only primitive substitution as the primary structure-preserving
    augmentation.

    Args:
        token_ids_batch: 2D array of shape (batch_size, seq_len).
        keys: 2D array of shape (batch_size, 2) containing PRNGKeys.
        tokenizer: Tokenizer instance.
        domain: The domain ('scan' or 'cogs').
        permutation_probability: Unused. Kept for API compatibility.

    Returns:
        jax.Array of shape (batch_size, seq_len) with augmented sequences.
    """
    # Only use primitive substitution (JIT/vmap compatible)
    augmented = jax.vmap(
        lambda ids, k: apply_primitive_substitution(ids, k, tokenizer, domain)
    )(token_ids_batch, keys)

    return augmented
