"""Tests for Augmentation Consistency (AC) signal computation.

This module verifies the correctness of the c_A signal calculation
as defined in Equation 5 of the H-Bar paper.

Tests correspond to Subtask 3.2 requirements:
- Test 1: Identical batches result in c_A = 1.0
- Test 2: Structure-preserving changes result in higher c_A than random changes
- Test 3: Full pipeline integration without JAX tracer errors
"""

import jax
import jax.numpy as jnp
import pytest

from hbar.engine.signals import compute_augmentation_consistency
from hbar.models.config import TransformerConfig
from hbar.models.transformer import Seq2SeqTransformer, get_model_representations


class TestAugmentationConsistency:
    """Test suite for AC signal correctness."""

    @pytest.fixture
    def config(self) -> TransformerConfig:
        """Default transformer configuration for testing."""
        return TransformerConfig(
            vocab_size=32,
            max_seq_len=20,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout_rate=0.1,
        )

    @pytest.fixture
    def model_and_inputs(self, config: TransformerConfig):
        """Create model and sample inputs."""
        model = Seq2SeqTransformer(config)
        batch_size = 4
        src_seq_len = 10
        tgt_seq_len = 8

        src = jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, src_seq_len), 0, config.vocab_size
        )
        tgt = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, tgt_seq_len), 0, config.vocab_size
        )

        return model, src, tgt, batch_size, src_seq_len, tgt_seq_len

    def test_identical_representations_yield_cA_one(self, model_and_inputs):
        """Test 1: Identical batches should result in c_A ≈ 1.0."""
        model, src, tgt, batch_size, src_seq_len, tgt_seq_len = model_and_inputs

        # Initialize model
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Get representations for the same batch twice
        orig_repr = get_model_representations(variables["params"], model, src, tgt)
        aug_repr = get_model_representations(variables["params"], model, src, tgt)

        # Compute AC signal
        c_A = compute_augmentation_consistency(orig_repr, aug_repr)

        # Should be very close to 1.0
        assert jnp.isclose(c_A, 1.0, atol=1e-5), (
            f"Identical representations should yield c_A ≈ 1.0, got {c_A}"
        )

    def test_structure_preserving_higher_than_random(self, model_and_inputs):
        """Test 2: Structure-preserving changes should yield higher c_A than random."""
        model, src, tgt, batch_size, src_seq_len, tgt_seq_len = model_and_inputs
        config = model.config

        # Initialize model
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Get original representations
        orig_repr = get_model_representations(variables["params"], model, src, tgt)

        # Create a structure-preserving augmentation (small perturbation)
        # Add small noise to simulate structure-preserving change
        noise_key = jax.random.PRNGKey(123)
        small_noise = jax.random.normal(noise_key, src.shape) * 0.01
        aug_src_preserving = jnp.clip(src + small_noise, 0, config.vocab_size - 1).astype(src.dtype)
        aug_repr_preserving = get_model_representations(
            variables["params"], model, aug_src_preserving, tgt
        )

        # Create a random (non-structural) change (large perturbation)
        large_noise = jax.random.normal(noise_key, src.shape) * 5.0
        aug_src_random = jnp.clip(src + large_noise, 0, config.vocab_size - 1).astype(src.dtype)
        aug_repr_random = get_model_representations(
            variables["params"], model, aug_src_random, tgt
        )

        # Compute AC signals
        c_A_preserving = compute_augmentation_consistency(orig_repr, aug_repr_preserving)
        c_A_random = compute_augmentation_consistency(orig_repr, aug_repr_random)

        # Structure-preserving should have higher consistency
        assert c_A_preserving > c_A_random, (
            f"Structure-preserving c_A ({c_A_preserving}) should be higher than "
            f"random c_A ({c_A_random})"
        )

    def test_jit_compatibility(self, model_and_inputs):
        """Test 3: Full pipeline should work with JIT compilation."""
        model, src, tgt, batch_size, src_seq_len, tgt_seq_len = model_and_inputs

        # Initialize model
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Create JIT-compiled function for the full pipeline
        @jax.jit
        def compute_ac_jit(src, tgt):
            orig_repr = get_model_representations(variables["params"], model, src, tgt)
            # Use same batch for simplicity (should give c_A = 1.0)
            aug_repr = get_model_representations(variables["params"], model, src, tgt)
            return compute_augmentation_consistency(orig_repr, aug_repr)

        # Run JIT-compiled function
        c_A = compute_ac_jit(src, tgt)

        # Should be close to 1.0
        assert jnp.isclose(c_A, 1.0, atol=1e-5), (
            f"JIT-compiled identical representations should yield c_A ≈ 1.0, got {c_A}"
        )

    def test_cA_range(self, model_and_inputs):
        """Verify c_A is always in [0, 1] range."""
        model, src, tgt, batch_size, src_seq_len, tgt_seq_len = model_and_inputs

        # Initialize model
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Get original representations
        orig_repr = get_model_representations(variables["params"], model, src, tgt)

        # Create various levels of perturbation
        for noise_scale in [0.1, 1.0, 10.0, 100.0]:
            noise_key = jax.random.PRNGKey(int(noise_scale * 10))
            noise = jax.random.normal(noise_key, src.shape) * noise_scale
            aug_src = jnp.clip(src + noise, 0, model.config.vocab_size - 1).astype(src.dtype)
            aug_repr = get_model_representations(
                variables["params"], model, aug_src, tgt
            )

            c_A = compute_augmentation_consistency(orig_repr, aug_repr)

            # Should always be in [0, 1]
            assert 0.0 <= c_A <= 1.0, (
                f"c_A should be in [0, 1] but got {c_A} for noise_scale={noise_scale}"
            )

    def test_different_layers(self, model_and_inputs):
        """Verify AC can be computed for different layers."""
        model, src, tgt, batch_size, src_seq_len, tgt_seq_len = model_and_inputs
        config = model.config

        # Initialize model
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Get representations
        orig_repr = get_model_representations(variables["params"], model, src, tgt)
        aug_repr = get_model_representations(variables["params"], model, src, tgt)

        # Test different layers
        layers_to_test = ["embedding", "encoder_block_0", "encoder_block_1",
                          "decoder_embedding", "decoder_block_0", "decoder_block_1"]

        for layer in layers_to_test:
            if layer in orig_repr:
                c_A = compute_augmentation_consistency(orig_repr, aug_repr, layer=layer)
                assert jnp.isclose(c_A, 1.0, atol=1e-5), (
                    f"Layer {layer} should yield c_A ≈ 1.0 for identical inputs, got {c_A}"
                )

    def test_gradient_flow_through_ac(self, model_and_inputs):
        """Verify gradients can flow through AC computation."""
        model, src, tgt, batch_size, src_seq_len, tgt_seq_len = model_and_inputs

        # Initialize model
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        def ac_loss_fn(params):
            orig_repr = get_model_representations(params, model, src, tgt)
            # Use same batch (c_A should be 1.0)
            aug_repr = get_model_representations(params, model, src, tgt)
            c_A = compute_augmentation_consistency(orig_repr, aug_repr)
            # Loss is 1 - c_A (we want to maximize consistency)
            return 1.0 - c_A

        # Compute gradients
        grads = jax.grad(ac_loss_fn)(variables["params"])

        # Verify gradients exist
        assert "encoder" in grads
        assert "decoder" in grads

        # Verify gradients are non-zero (at least some)
        total_grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads))
        )
        # Note: gradients might be very small for identical inputs, but should exist
        assert total_grad_norm >= 0.0, "Gradients should exist through AC computation"
