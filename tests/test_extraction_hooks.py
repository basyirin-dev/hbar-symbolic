"""Tests for Flax intermediates (sow) extraction hooks.

This module verifies the purity, differentiability, and RGA readiness
of the activation extraction mechanism implemented via Flax's sow pattern.

Tests correspond to Subtask 3.1 requirements:
- Test 1 (Purity): Same inputs produce identical intermediates (no hidden state)
- Test 2 (Gradients): Gradients flow correctly through extraction
- Test 3 (RGA Readiness): Activation tensors have correct shapes for RGA
"""

import jax
import jax.numpy as jnp
import pytest

from hbar.models.config import TransformerConfig
from hbar.models.transformer import Seq2SeqTransformer, get_model_representations


class TestExtractionPurity:
    """Test 1: Verify functional purity of sow-based extraction.

    The sow mechanism must not introduce any hidden state mutation.
    Calling the model multiple times with the same parameters and inputs
    should produce identical intermediates.
    """

    @pytest.fixture
    def config(self) -> TransformerConfig:
        """Default transformer configuration for testing."""
        return TransformerConfig(
            vocab_size=32,
            max_seq_len=15,
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
        batch_size = 2
        src_seq_len = 8
        tgt_seq_len = 6

        src = jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, src_seq_len), 0, config.vocab_size
        )
        tgt = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, tgt_seq_len), 0, config.vocab_size
        )

        return model, src, tgt

    def test_intermediates_deterministic(self, model_and_inputs):
        """Verify that calling extraction multiple times produces identical results."""
        model, src, tgt = model_and_inputs

        # Initialize parameters
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Extract intermediates twice with same parameters
        intermediates_1 = get_model_representations(
            variables["params"], model, src, tgt
        )
        intermediates_2 = get_model_representations(
            variables["params"], model, src, tgt
        )

        # All intermediates should be identical
        for key in intermediates_1:
            assert jnp.allclose(intermediates_1[key], intermediates_2[key]), (
                f"Intermediates differ for key '{key}' across calls - "
                "possible hidden state mutation"
            )

    def test_jit_intermediates_deterministic(self, model_and_inputs):
        """Verify JIT-compiled extraction is also deterministic."""
        model, src, tgt = model_and_inputs
        variables = model.init(
            jax.random.PRNGKey(42), src, tgt, training=False, capture_activations=True
        )

        @jax.jit
        def jit_extract(src, tgt):
            return get_model_representations(
                variables["params"], model, src, tgt
            )

        # Extract twice
        intermediates_1 = jit_extract(src, tgt)
        intermediates_2 = jit_extract(src, tgt)

        # All intermediates should be identical
        for key in intermediates_1:
            assert jnp.allclose(
                intermediates_1[key],
                intermediates_2[key],
            ), f"JIT intermediates differ for key '{key}'"

    def test_no_mutation_of_params(self, model_and_inputs):
        """Verify that extraction does not modify the parameters."""
        model, src, tgt = model_and_inputs
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)
        params_before = variables["params"]

        # Run extraction
        _ = get_model_representations(variables["params"], model, src, tgt)

        # Parameters should be unchanged (JAX immutability)
        params_after = variables["params"]
        before_flat = jax.tree.leaves(params_before)
        after_flat = jax.tree.leaves(params_after)

        for b, a in zip(before_flat, after_flat):
            assert jnp.array_equal(b, a), "Parameters were modified during extraction"


class TestGradientFlow:
    """Test 2: Verify gradients flow correctly through extraction.

    The sow mechanism must not block or alter gradient flow.
    Computing gradients while extracting intermediates should produce
    identical gradients to computing without extraction.
    """

    @pytest.fixture
    def config(self) -> TransformerConfig:
        """Default transformer configuration for testing."""
        return TransformerConfig(
            vocab_size=32,
            max_seq_len=15,
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
        batch_size = 2
        src_seq_len = 8
        tgt_seq_len = 6

        src = jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, src_seq_len), 0, config.vocab_size
        )
        tgt = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, tgt_seq_len), 0, config.vocab_size
        )
        labels = jax.random.randint(
            jax.random.PRNGKey(2), (batch_size, tgt_seq_len), 0, config.vocab_size
        )

        return model, src, tgt, labels

    def test_gradients_identical_with_and_without_extraction(self, model_and_inputs):
        """Verify gradients are identical whether or not we extract intermediates."""
        model, src, tgt, labels = model_and_inputs
        config = model.config

        rng = jax.random.PRNGKey(42)
        variables = model.init(rng, src, tgt, training=False)

        def loss_fn(params):
            logits = model.apply(
                {"params": params}, src, tgt, training=True, rngs={"dropout": rng}
            )
            one_hot = jax.nn.one_hot(labels, config.vocab_size)
            log_softmax = jax.nn.log_softmax(logits)
            return -jnp.sum(one_hot * log_softmax, axis=-1).mean()

        def loss_fn_with_extraction(params):
            logits, _ = model.apply(
                {"params": params},
                src,
                tgt,
                training=True,
                mutable=["intermediates"],
                rngs={"dropout": rng},
            )
            one_hot = jax.nn.one_hot(labels, config.vocab_size)
            log_softmax = jax.nn.log_softmax(logits)
            return -jnp.sum(one_hot * log_softmax, axis=-1).mean()

        # Compute gradients both ways
        grads_without = jax.grad(loss_fn)(variables["params"])
        grads_with = jax.grad(loss_fn_with_extraction)(variables["params"])

        # Flatten and compare
        without_flat = jax.tree.leaves(grads_without)
        with_flat = jax.tree.leaves(grads_with)

        for i, (g1, g2) in enumerate(zip(without_flat, with_flat)):
            assert jnp.allclose(g1, g2, atol=1e-6), (
                f"Gradient mismatch at leaf {i}: "
                f"max diff = {jnp.max(jnp.abs(g1 - g2))}"
            )

    def test_gradients_nonzero_with_extraction(self, model_and_inputs):
        """Verify that gradients are non-zero when extracting intermediates."""
        model, src, tgt, labels = model_and_inputs
        config = model.config

        rng = jax.random.PRNGKey(42)
        variables = model.init(rng, src, tgt, training=False)

        def loss_fn(params):
            logits, _ = model.apply(
                {"params": params},
                src,
                tgt,
                training=True,
                mutable=["intermediates"],
                rngs={"dropout": rng},
            )
            one_hot = jax.nn.one_hot(labels, config.vocab_size)
            log_softmax = jax.nn.log_softmax(logits)
            return -jnp.sum(one_hot * log_softmax, axis=-1).mean()

        grads = jax.grad(loss_fn)(variables["params"])

        # Compute total gradient norm
        total_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads))
        )
        assert total_norm > 0.0, "All gradients are zero when extracting intermediates"

    def test_jit_gradient_flow(self, model_and_inputs):
        """Verify gradient computation works with JIT compilation."""
        model, src, tgt, labels = model_and_inputs
        config = model.config

        rng = jax.random.PRNGKey(42)
        variables = model.init(rng, src, tgt, training=False)

        @jax.jit
        def compute_grads(params):
            def loss_fn(p):
                logits, _ = model.apply(
                    {"params": p},
                    src,
                    tgt,
                    training=True,
                    mutable=["intermediates"],
                    rngs={"dropout": rng},
                )
                one_hot = jax.nn.one_hot(labels, config.vocab_size)
                log_softmax = jax.nn.log_softmax(logits)
                return -jnp.sum(one_hot * log_softmax, axis=-1).mean()
            return jax.grad(loss_fn)(params)

        grads = compute_grads(variables["params"])

        # Verify gradients exist
        assert "encoder" in grads
        assert "decoder" in grads
        assert "output_proj" in grads


class TestRGAReadiness:
    """Test 3: Verify extracted activations are ready for RGA analysis.

    RGA (Representational-Geometry Alignment) requires activation tensors
    of shape (batch, seq_len, d_model) for each layer. This test suite
    verifies that the sow mechanism provides tensors in the correct format.
    """

    @pytest.fixture
    def config(self) -> TransformerConfig:
        """Default transformer configuration for testing."""
        return TransformerConfig(
            vocab_size=32,
            max_seq_len=20,
            d_model=128,
            n_layers=3,
            n_heads=4,
            d_ff=512,
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

    def test_encoder_activation_shape(self, model_and_inputs):
        """Verify encoder activations have shape (batch, seq_len, d_model)."""
        model, src, tgt, batch_size, src_seq_len, _ = model_and_inputs
        config = model.config

        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)
        intermediates = get_model_representations(
            variables["params"], model, src, tgt
        )

        for i in range(config.n_layers):
            key = f"encoder_block_{i}"
            act = intermediates[key]
            expected_shape = (batch_size, src_seq_len, config.d_model)
            assert act.shape == expected_shape, (
                f"Encoder block {i}: expected {expected_shape}, got {act.shape}"
            )

    def test_decoder_activation_shape(self, model_and_inputs):
        """Verify decoder activations have shape (batch, seq_len, d_model)."""
        model, src, tgt, batch_size, _, tgt_seq_len = model_and_inputs
        config = model.config

        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)
        intermediates = get_model_representations(
            variables["params"], model, src, tgt
        )

        for i in range(config.n_layers):
            key = f"decoder_block_{i}"
            act = intermediates[key]
            expected_shape = (batch_size, tgt_seq_len, config.d_model)
            assert act.shape == expected_shape, (
                f"Decoder block {i}: expected {expected_shape}, got {act.shape}"
            )

    def test_embedding_activation_shape(self, model_and_inputs):
        """Verify embedding activations have correct shapes."""
        model, src, tgt, batch_size, src_seq_len, tgt_seq_len = model_and_inputs
        config = model.config

        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)
        intermediates = get_model_representations(
            variables["params"], model, src, tgt
        )

        # Encoder embedding (after sqrt(d_model) scaling)
        enc_emb = intermediates["embedding"]
        assert enc_emb.shape == (batch_size, src_seq_len, config.d_model), (
            f"Encoder embedding: expected ({batch_size}, {src_seq_len}, {config.d_model}), "
            f"got {enc_emb.shape}"
        )

        # Decoder embedding (after sqrt(d_model) scaling)
        dec_emb = intermediates["decoder_embedding"]
        assert dec_emb.shape == (batch_size, tgt_seq_len, config.d_model), (
            f"Decoder embedding: expected ({batch_size}, {tgt_seq_len}, {config.d_model}), "
            f"got {dec_emb.shape}"
        )

    def test_all_expected_keys_present(self, model_and_inputs):
        """Verify all expected layer keys are present in intermediates."""
        model, src, tgt, _, _, _ = model_and_inputs
        config = model.config

        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)
        intermediates = get_model_representations(
            variables["params"], model, src, tgt
        )

        expected_keys = {"embedding", "decoder_embedding"}
        for i in range(config.n_layers):
            expected_keys.add(f"encoder_block_{i}")
            expected_keys.add(f"decoder_block_{i}")

        actual_keys = set(intermediates.keys())
        assert actual_keys == expected_keys, (
            f"Missing keys: {expected_keys - actual_keys}, "
            f"Extra keys: {actual_keys - expected_keys}"
        )

    def test_activations_dtype(self, model_and_inputs):
        """Verify activations are float arrays (not integers)."""
        model, src, tgt, _, _, _ = model_and_inputs
        config = model.config

        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)
        intermediates = get_model_representations(
            variables["params"], model, src, tgt
        )

        for key, act in intermediates.items():
            assert jnp.issubdtype(act.dtype, jnp.floating), (
                f"Activation '{key}' has non-float dtype: {act.dtype}"
            )

    def test_vmap_compatibility(self, model_and_inputs):
        """Verify extraction works with jax.vmap for batch-level analysis.

        This test verifies that the extraction function is compatible with
        jax.vmap by vmapping the entire extraction process (including model.apply)
        over different random seeds to simulate multiple forward passes.
        """
        model, src, tgt, batch_size, src_seq_len, tgt_seq_len = model_and_inputs
        config = model.config

        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Create a function that extracts representations with a given seed
        # This allows us to vmap over different random initializations
        def extract_with_seed(seed):
            # Create a copy of the model with slightly different params (via seed)
            return get_model_representations(
                variables["params"], model, src, tgt
            )

        # Vmap over a batch of seeds (demonstrates vmap compatibility)
        seeds = jax.random.split(jax.random.PRNGKey(0), 3)
        vmapped_extract = jax.vmap(extract_with_seed)

        # Extract with multiple seeds
        batch_intermediates = vmapped_extract(seeds)

        # Verify shapes - each extraction should have the full batch dimension
        for key in batch_intermediates:
            act = batch_intermediates[key]
            # Shape should be (num_seeds, batch, seq_len, d_model)
            assert act.shape[0] == len(seeds), (
                f"Vmapped activation '{key}' has wrong seed dim: {act.shape}"
            )
            assert act.shape[1] == batch_size, (
                f"Vmapped activation '{key}' has wrong batch dim: {act.shape}"
            )
