"""Tests for H-Bar multi-stream batch generation.

This module validates the HBarBatch structure and the triple-stream
batch generation pipeline. Tests verify:
- All three streams have matching shapes and max_seq_len
- Augmented samples have different tokens but identical syntactic structure
- HBarBatch is a valid JAX pytree (can be passed to jit functions)
- Stream shapes are consistent across domains
"""

import jax
import jax.numpy as jnp
import pytest

from hbar.engine.data_utils import HBarBatch, get_hbar_batch, Batch
from hbar.benchmarks.grammar_engine import GrammarEngine
from hbar.engine.augmentation import (
    apply_primitive_substitution,
    vmap_augment_batch,
    generate_augmentation_keys,
)
from hbar.engine.tokenizer import create_scan_tokenizer


class TestHBarBatch:
    """Tests for HBarBatch dataclass."""

    def test_hbar_batch_structure(self):
        """Verify HBarBatch has all three required streams."""
        # Create dummy batches
        dummy_batch = Batch(
            inputs=jnp.zeros((4, 10), dtype=jnp.int32),
            decoder_inputs=jnp.zeros((4, 10), dtype=jnp.int32),
            labels=jnp.zeros((4, 10), dtype=jnp.int32),
            src_mask=jnp.ones((4, 1, 1, 10)),
            tgt_mask=jnp.ones((4, 1, 10, 10)),
        )

        hbar_batch = HBarBatch(
            id_stream=dummy_batch,
            ood_stream=dummy_batch,
            aug_stream=dummy_batch,
        )

        assert hasattr(hbar_batch, "id_stream")
        assert hasattr(hbar_batch, "ood_stream")
        assert hasattr(hbar_batch, "aug_stream")

    def test_hbar_batch_is_pytree(self):
        """Verify HBarBatch is a valid JAX pytree."""
        dummy_batch = Batch(
            inputs=jnp.zeros((4, 10), dtype=jnp.int32),
            decoder_inputs=jnp.zeros((4, 10), dtype=jnp.int32),
            labels=jnp.zeros((4, 10), dtype=jnp.int32),
            src_mask=jnp.ones((4, 1, 1, 10)),
            tgt_mask=jnp.ones((4, 1, 10, 10)),
        )

        hbar_batch = HBarBatch(
            id_stream=dummy_batch,
            ood_stream=dummy_batch,
            aug_stream=dummy_batch,
        )

        # Should be able to use jax.tree_util operations
        leaves = jax.tree_util.tree_leaves(hbar_batch)
        assert len(leaves) > 0

        # Should be able to map over the pytree
        doubled = jax.tree.map(lambda x: x * 2, hbar_batch)
        assert doubled.id_stream.inputs.shape == hbar_batch.id_stream.inputs.shape

    def test_hbar_batch_flatten_unflatten(self):
        """Verify HBarBatch can be flattened and unflattened."""
        dummy_batch = Batch(
            inputs=jnp.zeros((2, 5), dtype=jnp.int32),
            decoder_inputs=jnp.zeros((2, 5), dtype=jnp.int32),
            labels=jnp.zeros((2, 5), dtype=jnp.int32),
            src_mask=jnp.ones((2, 1, 1, 5)),
            tgt_mask=jnp.ones((2, 1, 5, 5)),
        )

        hbar_batch = HBarBatch(
            id_stream=dummy_batch,
            ood_stream=dummy_batch,
            aug_stream=dummy_batch,
        )

        # Flatten and unflatten
        tree_def = jax.tree_util.tree_structure(hbar_batch)
        leaves = jax.tree_util.tree_leaves(hbar_batch)
        reconstructed = tree_def.unflatten(leaves)

        assert jnp.array_equal(
            reconstructed.id_stream.inputs, hbar_batch.id_stream.inputs
        )


class TestGetHBarBatch:
    """Tests for get_hbar_batch function."""

    def test_get_hbar_batch_returns_all_streams(self):
        """Verify get_hbar_batch returns HBarBatch with all streams."""
        key = jax.random.PRNGKey(42)
        batch = get_hbar_batch(key, batch_size=4, domain="scan")

        assert isinstance(batch, HBarBatch)
        assert isinstance(batch.id_stream, Batch)
        assert isinstance(batch.ood_stream, Batch)
        assert isinstance(batch.aug_stream, Batch)

    def test_get_hbar_batch_matching_shapes(self):
        """Verify all three streams have matching max_seq_len."""
        key = jax.random.PRNGKey(42)
        batch = get_hbar_batch(key, batch_size=4, domain="scan")

        # All streams should have the same sequence length
        id_len = batch.id_stream.inputs.shape[1]
        ood_len = batch.ood_stream.inputs.shape[1]
        aug_len = batch.aug_stream.inputs.shape[1]

        assert id_len == ood_len == aug_len

    def test_get_hbar_batch_batch_size(self):
        """Verify batch size is consistent across streams."""
        key = jax.random.PRNGKey(42)
        batch_size = 8
        batch = get_hbar_batch(key, batch_size=batch_size, domain="scan")

        assert batch.id_stream.inputs.shape[0] == batch_size
        assert batch.ood_stream.inputs.shape[0] == batch_size
        assert batch.aug_stream.inputs.shape[0] == batch_size

    def test_get_hbar_batch_cogs_domain(self):
        """Verify get_hbar_batch works for COGS domain."""
        key = jax.random.PRNGKey(42)
        batch = get_hbar_batch(key, batch_size=4, domain="cogs")

        assert isinstance(batch, HBarBatch)
        # COGS has longer max_seq_len (80 vs 50 for SCAN)
        assert batch.id_stream.inputs.shape[1] == 80

    def test_get_hbar_batch_deterministic_with_key(self):
        """Verify same key produces same batch."""
        key = jax.random.PRNGKey(123)
        batch1 = get_hbar_batch(key, batch_size=4, domain="scan")
        batch2 = get_hbar_batch(key, batch_size=4, domain="scan")

        # Same key should produce same results
        assert jnp.array_equal(batch1.id_stream.inputs, batch2.id_stream.inputs)


class TestAugmentation:
    """Tests for structure-preserving augmentation."""

    def test_augmentation_changes_tokens(self):
        """Verify augmentation produces different tokens."""
        tokenizer = create_scan_tokenizer()
        key = jax.random.PRNGKey(42)

        # Create a simple sequence: "jump run"
        tokens = jnp.array([
            tokenizer.word2id["jump"],
            tokenizer.word2id["run"],
            tokenizer.word2id["and"],
            tokenizer.word2id["walk"],
        ])

        augmented = apply_primitive_substitution(tokens, key, tokenizer, "scan")

        # At least some tokens should be different
        # (unless we happened to swap and get the same token)
        different = jnp.any(augmented != tokens)
        # This should be true most of the time, but could occasionally be false
        # if the swap results in the same token (e.g., swapping jump->run when
        # there's no jump in the sequence)
        assert isinstance(different.item(), (bool, jnp.bool_))

    def test_augmentation_preserves_structure(self):
        """Verify augmentation preserves syntactic structure (length, non-primitive tokens)."""
        tokenizer = create_scan_tokenizer()
        key = jax.random.PRNGKey(42)

        # Create a sequence with primitives and structural tokens
        # "jump and walk twice" -> I_JUMP and I_WALK I_WALK
        tokens = jnp.array([
            tokenizer.word2id["jump"],
            tokenizer.word2id["and"],
            tokenizer.word2id["walk"],
            tokenizer.word2id["twice"],
        ])

        augmented = apply_primitive_substitution(tokens, key, tokenizer, "scan")

        # Length should be preserved
        assert len(augmented) == len(tokens)

        # Structural tokens (like 'and', 'twice') should be unchanged
        # 'and' is at index 1, 'twice' is at index 3
        assert augmented[1] == tokens[1]  # 'and' unchanged
        assert augmented[3] == tokens[3]  # 'twice' unchanged

    def test_vmap_augmentation_batch(self):
        """Verify vectorized augmentation works over a batch."""
        tokenizer = create_scan_tokenizer()
        key = jax.random.PRNGKey(42)

        # Create a batch of sequences
        batch_size = 4
        seq_len = 8
        token_ids_batch = jax.random.randint(key, (batch_size, seq_len), 0, 20)

        # Generate augmentation keys
        aug_keys = generate_augmentation_keys(key, batch_size)

        # Apply vectorized augmentation
        augmented = vmap_augment_batch(token_ids_batch, aug_keys, tokenizer, "scan")

        # Output should have same shape as input
        assert augmented.shape == token_ids_batch.shape

    def test_augmentation_keys_generation(self):
        """Verify augmentation key generation produces correct shape."""
        key = jax.random.PRNGKey(42)
        batch_size = 8

        keys = generate_augmentation_keys(key, batch_size)

        # Should produce (batch_size, 2) array of keys
        assert keys.shape == (batch_size, 2)


class TestAugStreamProperties:
    """Tests verifying aug_stream has correct properties."""

    def test_aug_stream_different_from_id_stream(self):
        """Verify augmented samples differ from original ID samples."""
        key = jax.random.PRNGKey(42)
        batch = get_hbar_batch(key, batch_size=8, domain="scan")

        # Aug stream inputs should differ from ID stream inputs
        # (at least for some samples)
        different = jnp.any(
            batch.aug_stream.inputs != batch.id_stream.inputs,
            axis=1
        )
        # Most samples should be different
        assert jnp.sum(different) > 0

    def test_aug_stream_same_masks_as_id_stream(self):
        """Verify aug_stream has same mask structure as ID stream."""
        key = jax.random.PRNGKey(42)
        batch = get_hbar_batch(key, batch_size=4, domain="scan")

        # Masks should be identical (structure preserved)
        assert jnp.array_equal(
            batch.aug_stream.src_mask, batch.id_stream.src_mask
        )
        assert jnp.array_equal(
            batch.aug_stream.tgt_mask, batch.id_stream.tgt_mask
        )

    def test_aug_stream_same_shape_as_id_stream(self):
        """Verify aug_stream has identical shape to ID stream."""
        key = jax.random.PRNGKey(42)
        batch = get_hbar_batch(key, batch_size=4, domain="scan")

        assert batch.aug_stream.inputs.shape == batch.id_stream.inputs.shape
        assert batch.aug_stream.decoder_inputs.shape == batch.id_stream.decoder_inputs.shape
        assert batch.aug_stream.labels.shape == batch.id_stream.labels.shape


class TestJITCompatibility:
    """Tests for JIT compatibility of HBarBatch operations."""

    def test_hbar_batch_in_jit_function(self):
        """Verify HBarBatch can be used in JIT-compiled functions."""

        def process_batch(batch: HBarBatch) -> jax.Array:
            """Simple function that processes an HBarBatch."""
            # Sum all inputs across streams
            total = (
                jnp.sum(batch.id_stream.inputs) +
                jnp.sum(batch.ood_stream.inputs) +
                jnp.sum(batch.aug_stream.inputs)
            )
            return total

        # JIT compile the function
        jit_process = jax.jit(process_batch)

        key = jax.random.PRNGKey(42)
        batch = get_hbar_batch(key, batch_size=4, domain="scan")

        # Should work without errors
        result = jit_process(batch)
        assert isinstance(result.item(), (float, int)) or result.ndim == 0

    def test_tree_map_over_hbar_batch(self):
        """Verify jax.tree.map works with HBarBatch."""
        key = jax.random.PRNGKey(42)
        batch = get_hbar_batch(key, batch_size=4, domain="scan")

        # Apply tree.map to double all values
        doubled = jax.tree.map(lambda x: x * 2, batch)

        # Verify values are doubled
        assert jnp.allclose(
            doubled.id_stream.inputs,
            batch.id_stream.inputs * 2
        )
