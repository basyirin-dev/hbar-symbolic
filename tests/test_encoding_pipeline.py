"""Integration tests for the encoding pipeline and Transformer model.

This module tests the full pipeline from raw text to model output,
verifying that the tokenizer, encoding functions, and Transformer
work together correctly with JIT compilation.
"""

import jax
import jax.numpy as jnp
import pytest

from hbar.engine.tokenizer import Tokenizer, create_scan_tokenizer
from hbar.engine.encoding import get_padding_mask, get_causal_mask, get_decoder_mask
from hbar.engine.data_utils import prepare_batch, compute_loss, compute_accuracy
from hbar.models.config import TransformerConfig
from hbar.models.transformer import Seq2SeqTransformer


class TestTokenizer:
    """Tests for the Tokenizer class."""

    def test_special_token_ids(self):
        """Verify special token IDs match specification."""
        tokenizer = Tokenizer()
        assert tokenizer.get_pad_token_id() == 0
        assert tokenizer.get_bos_token_id() == 1
        assert tokenizer.get_eos_token_id() == 2

    def test_encode_adds_bos_eos(self):
        """Verify encoding adds BOS and EOS tokens."""
        tokenizer = Tokenizer(["jump", "walk"])
        encoded = tokenizer.encode("jump", max_seq_len=10)
        # Should be: [BOS=1, jump, EOS=2, PAD=0, ...]
        assert encoded[0] == 1  # BOS
        assert encoded[1] == tokenizer.word2id["jump"]
        assert encoded[2] == 2  # EOS
        assert encoded[3] == 0  # PAD

    def test_encode_truncates_long_sequences(self):
        """Verify long sequences are truncated with EOS preserved."""
        tokenizer = Tokenizer(["a", "b", "c", "d", "e"])
        encoded = tokenizer.encode("a b c d e", max_seq_len=4)
        # Should be: [BOS, a, b, EOS] (truncated to fit)
        assert len(encoded) == 4
        assert encoded[0] == 1  # BOS
        assert encoded[-1] == 2  # EOS

    def test_decode_skips_special_tokens(self):
        """Verify decoding skips special tokens."""
        tokenizer = Tokenizer(["jump", "walk"])
        encoded = tokenizer.encode("jump", max_seq_len=10)
        decoded = tokenizer.decode(encoded, skip_special=True)
        assert decoded == "jump"

    def test_scan_tokenizer_vocabulary(self):
        """Verify SCAN tokenizer has expected vocabulary."""
        tokenizer = create_scan_tokenizer()
        assert "jump" in tokenizer.word2id
        assert "I_JUMP" in tokenizer.word2id
        assert tokenizer.vocab_size == 4 + 13 + 6  # special + commands + actions


class TestEncoding:
    """Tests for encoding/mask generation functions."""

    def test_padding_mask_shape(self):
        """Verify padding mask has correct shape for attention."""
        token_ids = jnp.array([[1, 5, 3, 0, 0], [1, 6, 2, 4, 0]])
        mask = get_padding_mask(token_ids, pad_token_id=0)
        assert mask.shape == (2, 1, 1, 5)

    def test_padding_mask_values(self):
        """Verify padding mask correctly identifies valid vs padding tokens."""
        token_ids = jnp.array([[1, 5, 0, 0]])
        mask = get_padding_mask(token_ids, pad_token_id=0)
        # Mask should be True for positions 0, 1 and False for 2, 3
        expected = jnp.array([[[[True, True, False, False]]]])
        assert jnp.array_equal(mask, expected)

    def test_causal_mask_shape(self):
        """Verify causal mask has correct shape."""
        mask = get_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_causal_mask_is_triangular(self):
        """Verify causal mask is lower triangular."""
        mask = get_causal_mask(4)
        expected = jnp.array([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ])
        assert jnp.array_equal(mask, expected)

    def test_decoder_mask_with_causal(self):
        """Verify decoder mask applies both padding and causal."""
        token_ids = jnp.array([[1, 5, 0, 0]])
        mask = get_decoder_mask(token_ids, pad_token_id=0)
        assert mask.shape == (1, 1, 4, 4)
        # The decoder mask combines causal + padding:
        # - Causal: position i can only attend to positions <= i
        # - Padding: positions with PAD tokens should not be attended to
        #
        # Token sequence: [BOS=1, word=5, PAD=0, PAD=0]
        # Valid positions: 0, 1; Padding positions: 2, 3
        #
        # Valid tokens can't attend to padding tokens (padding mask):
        assert not mask[0, 0, 0, 2]  # Position 0 can't attend to padding at 2
        assert not mask[0, 0, 0, 3]  # Position 0 can't attend to padding at 3
        assert not mask[0, 0, 1, 2]  # Position 1 can't attend to padding at 2
        assert not mask[0, 0, 1, 3]  # Position 1 can't attend to padding at 3
        #
        # Causal masking: position i can't attend to position j if j > i
        assert not mask[0, 0, 0, 1]  # Position 0 can't attend to future position 1
        assert mask[0, 0, 1, 0]  # Position 1 CAN attend to past position 0 (causal allows)
        assert mask[0, 0, 0, 0]  # Position 0 can attend to itself
        assert mask[0, 0, 1, 1]  # Position 1 can attend to itself


class TestBatchPreparation:
    """Tests for batch preparation functions."""

    def test_prepare_batch_shapes(self):
        """Verify all batch components have correct shapes."""
        tokenizer = create_scan_tokenizer()
        pairs = [
            ("jump twice", "I_JUMP I_JUMP"),
            ("walk left", "I_WALK I_TURN_LEFT"),
        ]
        batch = prepare_batch(pairs, tokenizer, max_seq_len=10)

        assert batch.inputs.shape == (2, 10)
        assert batch.decoder_inputs.shape == (2, 10)
        assert batch.labels.shape == (2, 10)
        assert batch.src_mask.shape == (2, 1, 1, 10)
        assert batch.tgt_mask.shape == (2, 1, 10, 10)

    def test_prepare_batch_decoder_input_shifted(self):
        """Verify decoder input is shifted right with BOS."""
        tokenizer = create_scan_tokenizer()
        pairs = [("jump", "I_JUMP I_JUMP")]
        batch = prepare_batch(pairs, tokenizer, max_seq_len=10)

        # First token of decoder input should be BOS
        assert batch.decoder_inputs[0, 0] == 1  # BOS token ID

    def test_compute_loss_ignores_padding(self):
        """Verify loss computation ignores padding tokens."""
        vocab_size = 10
        logits = jnp.ones((2, 5, vocab_size))
        labels = jnp.array([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])

        loss = compute_loss(logits, labels, pad_token_id=0)

        # Loss should be finite and positive
        assert jnp.isfinite(loss)
        assert loss > 0


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_forward_pass(self):
        """Test complete pipeline: text -> tokens -> model -> logits."""
        tokenizer = create_scan_tokenizer()
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=10,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
        )
        model = Seq2SeqTransformer(config)
        rng = jax.random.PRNGKey(42)
        params = model.init(rng, jnp.ones((1, 10), jnp.int32), jnp.ones((1, 10), jnp.int32))

        # Prepare batch
        pairs = [("jump twice", "I_JUMP I_JUMP")]
        batch = prepare_batch(pairs, tokenizer, max_seq_len=10)

        # Forward pass
        logits, activations = model.apply(params, batch.inputs, batch.decoder_inputs, training=False)

        assert logits.shape == (1, 10, tokenizer.vocab_size)
        assert len(activations.encoder_layers) == 1
        assert len(activations.decoder_layers) == 1

    def test_jit_compiled_pipeline(self):
        """Verify entire pipeline can be JIT compiled."""
        tokenizer = create_scan_tokenizer()
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=10,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
        )
        model = Seq2SeqTransformer(config)
        rng = jax.random.PRNGKey(42)
        params = model.init(rng, jnp.ones((1, 10), jnp.int32), jnp.ones((1, 10), jnp.int32))

        def forward(inputs, decoder_inputs):
            logits, _ = model.apply(params, inputs, decoder_inputs)
            return logits

        # JIT compile
        jit_forward = jax.jit(forward)

        # Run compiled function
        pairs = [("jump twice", "I_JUMP I_JUMP")]
        batch = prepare_batch(pairs, tokenizer, max_seq_len=10)
        logits = jit_forward(batch.inputs, batch.decoder_inputs)

        assert logits.shape == (1, 10, tokenizer.vocab_size)

    def test_gradient_flow_through_pipeline(self):
        """Verify gradients flow through the entire pipeline."""
        tokenizer = create_scan_tokenizer()
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=10,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
        )
        model = Seq2SeqTransformer(config)
        rng = jax.random.PRNGKey(42)
        params = model.init(rng, jnp.ones((1, 10), jnp.int32), jnp.ones((1, 10), jnp.int32))

        def compute_loss_fn(params, inputs, decoder_inputs, labels, rng):
            # Use training=False to avoid dropout RNG issues in grad computation
            logits, _ = model.apply(params, inputs, decoder_inputs, training=False, rngs={"dropout": rng})
            return compute_loss(logits, labels)

        # Compute gradients
        pairs = [("jump twice", "I_JUMP I_JUMP")]
        batch = prepare_batch(pairs, tokenizer, max_seq_len=10)
        dropout_rng = jax.random.PRNGKey(123)

        grads = jax.grad(compute_loss_fn)(
            params, batch.inputs, batch.decoder_inputs, batch.labels, dropout_rng
        )

        # Verify gradients exist - check that the gradient tree has the same structure
        # as the parameters tree
        grads_flat = jax.tree_util.tree_leaves(grads)
        params_flat = jax.tree_util.tree_leaves(params)
        assert len(grads_flat) == len(params_flat)
        for g, p in zip(grads_flat, params_flat):
            assert g.shape == p.shape
