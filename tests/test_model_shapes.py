"""Shape verification and JIT compatibility tests for the H-Bar Transformer."""

import jax
import jax.numpy as jnp
import pytest

from hbar.models.config import TransformerConfig
from hbar.models.transformer import Seq2SeqTransformer, get_model_representations


class TestTransformerShapes:
    """Test suite for verifying output shapes and activation tracking."""

    @pytest.fixture
    def config(self) -> TransformerConfig:
        """Default transformer configuration for testing."""
        return TransformerConfig(
            vocab_size=64,
            max_seq_len=20,
            d_model=128,
            n_layers=2,
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

    def test_output_logits_shape(self, model_and_inputs):
        """Verify logits shape is (batch, tgt_seq_len, vocab_size)."""
        model, src, tgt, batch_size, _, tgt_seq_len = model_and_inputs
        config = model.config

        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)
        logits = model.apply(variables, src, tgt, training=False)

        expected_shape = (batch_size, tgt_seq_len, config.vocab_size)
        assert logits.shape == expected_shape, (
            f"Expected logits shape {expected_shape}, got {logits.shape}"
        )

    def test_intermediates_keys(self, model_and_inputs):
        """Verify intermediates dict contains keys for all encoder and decoder layers."""
        model, src, tgt, _, _, _ = model_and_inputs
        config = model.config

        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)
        intermediates = get_model_representations(
            variables["params"], model, src, tgt
        )

        # Check encoder block keys
        for i in range(config.n_layers):
            key = f"encoder_block_{i}"
            assert key in intermediates, f"Missing encoder activation key: {key}"

        # Check decoder block keys
        for i in range(config.n_layers):
            key = f"decoder_block_{i}"
            assert key in intermediates, f"Missing decoder activation key: {key}"

        # Check embedding keys
        assert "embedding" in intermediates, "Missing encoder embedding key"
        assert "decoder_embedding" in intermediates, "Missing decoder embedding key"

    def test_intermediates_shape(self, model_and_inputs):
        """Verify each activation tensor has shape (batch, seq_len, d_model=128)."""
        model, src, tgt, batch_size, src_seq_len, tgt_seq_len = model_and_inputs
        config = model.config

        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)
        intermediates = get_model_representations(
            variables["params"], model, src, tgt
        )

        # Check encoder activations shapes
        for i in range(config.n_layers):
            enc_act = intermediates[f"encoder_block_{i}"]
            expected_shape = (batch_size, src_seq_len, config.d_model)
            assert enc_act.shape == expected_shape, (
                f"Encoder block {i}: expected {expected_shape}, got {enc_act.shape}"
            )

        # Check decoder activations shapes
        for i in range(config.n_layers):
            dec_act = intermediates[f"decoder_block_{i}"]
            expected_shape = (batch_size, tgt_seq_len, config.d_model)
            assert dec_act.shape == expected_shape, (
                f"Decoder block {i}: expected {expected_shape}, got {dec_act.shape}"
            )

        # Check embedding shapes
        assert intermediates["embedding"].shape == (batch_size, src_seq_len, config.d_model)
        assert intermediates["decoder_embedding"].shape == (batch_size, tgt_seq_len, config.d_model)


class TestJITCompatibility:
    """Test suite for JIT compatibility and gradient flow."""

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

    def test_jit_compatibility(self, config: TransformerConfig):
        """Verify forward pass works with jax.jit."""
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

        # Initialize parameters
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Create JIT-compiled forward pass
        @jax.jit
        def forward_fn(src, tgt):
            return model.apply(variables, src, tgt, training=False)

        # Run JIT-compiled function
        logits = forward_fn(src, tgt)

        # Verify output shape
        assert logits.shape == (batch_size, tgt_seq_len, config.vocab_size)

    def test_jit_with_intermediates(self, config: TransformerConfig):
        """Verify extraction with mutable=['intermediates'] works with jax.jit."""
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

        # Initialize parameters
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Create JIT-compiled extraction function
        @jax.jit
        def extract_fn(src, tgt):
            return model.apply(
                variables,
                src,
                tgt,
                training=False,
                capture_activations=True,
                mutable=["intermediates"],
            )

        # Run JIT-compiled function
        logits, intermediates = extract_fn(src, tgt)

        # Verify outputs
        assert logits.shape == (batch_size, tgt_seq_len, config.vocab_size)
        # Check that intermediates collection has encoder and decoder entries
        assert "encoder" in intermediates["intermediates"]
        assert "decoder" in intermediates["intermediates"]

    def test_gradient_flow(self, config: TransformerConfig):
        """Verify gradients flow through the model using jax.grad."""
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
        # Create dummy labels for loss computation
        labels = jax.random.randint(
            jax.random.PRNGKey(2), (batch_size, tgt_seq_len), 0, config.vocab_size
        )

        # Initialize parameters
        rng = jax.random.PRNGKey(42)
        variables = model.init(rng, src, tgt, training=False)

        def loss_fn(params):
            logits = model.apply(
                {"params": params}, src, tgt, training=True, rngs={"dropout": rng}
            )
            # Cross-entropy loss
            one_hot = jax.nn.one_hot(labels, config.vocab_size)
            log_softmax = jax.nn.log_softmax(logits)
            loss = -jnp.sum(one_hot * log_softmax, axis=-1).mean()
            return loss

        # Compute gradients
        grads = jax.grad(loss_fn)(variables["params"])

        # Verify that gradients exist for all parameters
        assert "encoder" in grads
        assert "decoder" in grads
        assert "output_proj" in grads

        # Verify gradients are non-zero (at least some)
        total_grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads))
        )
        assert total_grad_norm > 0.0, "All gradients are zero!"

    def test_gradient_flow_with_intermediates(self, config: TransformerConfig):
        """Verify gradients flow correctly when extracting intermediates."""
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

        rng = jax.random.PRNGKey(42)
        variables = model.init(rng, src, tgt, training=False)

        def loss_fn_with_extraction(params):
            logits, intermediates = model.apply(
                {"params": params},
                src,
                tgt,
                training=True,
                mutable=["intermediates"],
                rngs={"dropout": rng},
            )
            # Cross-entropy loss
            one_hot = jax.nn.one_hot(labels, config.vocab_size)
            log_softmax = jax.nn.log_softmax(logits)
            loss = -jnp.sum(one_hot * log_softmax, axis=-1).mean()
            return loss

        # Compute gradients while extracting intermediates
        grads_with_extraction = jax.grad(loss_fn_with_extraction)(variables["params"])

        # Verify gradients exist and are non-zero
        total_grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads_with_extraction))
        )
        assert total_grad_norm > 0.0, "All gradients are zero with intermediates!"

    def test_deterministic_dropout(self, config: TransformerConfig):
        """Verify dropout is disabled when training=False."""
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

        # Initialize parameters
        variables = model.init(jax.random.PRNGKey(42), src, tgt, training=False)

        # Run forward pass twice with training=False
        logits1 = model.apply(variables, src, tgt, training=False)
        logits2 = model.apply(variables, src, tgt, training=False)

        # Outputs should be identical when dropout is disabled
        assert jnp.allclose(logits1, logits2), (
            "Outputs differ with training=False, dropout may not be disabled"
        )

    def test_training_mode_affects_output(self, config: TransformerConfig):
        """Verify that training=True produces different outputs due to dropout."""
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

        # Initialize parameters
        rng = jax.random.PRNGKey(42)
        variables = model.init(rng, src, tgt, training=False)

        # Run forward pass with training=False and training=True
        logits_eval = model.apply(variables, src, tgt, training=False)
        logits_train = model.apply(
            variables, src, tgt, training=True, rngs={"dropout": rng}
        )

        # Outputs should differ when dropout is enabled
        are_different = not jnp.allclose(logits_eval, logits_train)
        assert are_different, (
            "Outputs are identical with training=True/False, dropout may not be working"
        )
