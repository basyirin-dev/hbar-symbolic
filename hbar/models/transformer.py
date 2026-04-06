"""Flax Linen Encoder-Decoder Transformer for H-Bar Model V3.0+.

This module implements a complete seq2seq Transformer architecture with activation
hooks for Representational-Geometry Alignment (RGA) signal extraction. The
architecture follows Section 11.1 of the H-Bar paper and is designed for
compositional generalization benchmarks (SCAN/COGS).

The Encoder-Decoder design is required because SCAN and COGS are cross-modal
mapping tasks (natural language → action sequences / logical forms).
"""

from typing import Any, Dict, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from hbar.models.config import ActivationsDict, TransformerConfig


class Embed(nn.Module):
    """Token embedding with learned positional encodings.

    This module combines token embeddings with learned positional encodings
    to provide position-aware representations for the Transformer.

    Attributes:
        num_embeddings: Size of the token vocabulary.
        features: Dimensionality of the embedding space (d_model).
        max_len: Maximum sequence length for positional encodings.
    """
    num_embeddings: int
    features: int
    max_len: int = 50

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        capture_activations: bool = False,
    ) -> jax.Array:
        """Embed tokens and add positional encodings.

        Args:
            inputs: Token indices with shape (batch, seq_len).
            capture_activations: Whether to capture this layer's output
                via self.sow for later extraction.

        Returns:
            Embedded representations with shape (batch, seq_len, features).
        """
        # Token embedding
        token_embed = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            embedding_init=nn.initializers.xavier_uniform(),
        )
        token_emb = token_embed(inputs)

        # Learned positional encoding
        pos_embed = nn.Embed(
            num_embeddings=self.max_len,
            features=self.features,
            embedding_init=nn.initializers.xavier_uniform(),
        )
        # Get sequence length - handle both 1D (vmapped) and 2D (batch) inputs
        if inputs.ndim == 1:
            # 1D input: (seq_len,) - from vmap or single sample
            seq_len = inputs.shape[0]
        else:
            # 2D input: (batch, seq_len)
            seq_len = inputs.shape[1]
        positions = jnp.arange(seq_len)
        pos_emb = pos_embed(positions)

        x = token_emb + pos_emb

        # Capture activation if requested
        if capture_activations:
            self.sow('intermediates', 'embedding', x)

        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple attention heads,
    following the standard Transformer architecture.

    Attributes:
        d_model: Dimensionality of the model (for projections).
        n_heads: Number of attention heads.
        dropout_rate: Dropout rate for attention weights.
    """
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute multi-head attention.

        Args:
            query: Query tensor with shape (batch, seq_len, d_model).
            key: Key tensor with shape (batch, seq_len, d_model).
            value: Value tensor with shape (batch, seq_len, d_model).
            mask: Optional attention mask with shape (batch, 1, seq_len, seq_len)
                or broadcastable to it.
            deterministic: Whether to apply dropout (False during training).

        Returns:
            A tuple of:
                - Output tensor with shape (batch, seq_len, d_model).
                - Attention weights with shape (batch, n_heads, seq_len, seq_len).
        """
        head_dim = self.d_model // self.n_heads

        # Linear projections for Q, K, V
        query_proj = nn.Dense(
            features=self.d_model,
            use_bias=True,
            kernel_init=nn.initializers.xavier_uniform(),
        )(query)
        key_proj = nn.Dense(
            features=self.d_model,
            use_bias=True,
            kernel_init=nn.initializers.xavier_uniform(),
        )(key)
        value_proj = nn.Dense(
            features=self.d_model,
            use_bias=True,
            kernel_init=nn.initializers.xavier_uniform(),
        )(value)

        # Reshape for multi-head attention
        query_reshaped = query_proj.reshape(
            query_proj.shape[0], query_proj.shape[1], self.n_heads, head_dim
        )
        key_reshaped = key_proj.reshape(
            key_proj.shape[0], key_proj.shape[1], self.n_heads, head_dim
        )
        value_reshaped = value_proj.reshape(
            value_proj.shape[0], value_proj.shape[1], self.n_heads, head_dim
        )

        # Transpose for attention computation
        query_transposed = query_reshaped.transpose(0, 2, 1, 3)
        key_transposed = key_reshaped.transpose(0, 2, 1, 3)
        value_transposed = value_reshaped.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", query_transposed, key_transposed)
        attn_weights = attn_weights / jnp.sqrt(head_dim)

        # Apply mask if provided
        if mask is not None:
            attn_weights = jnp.where(mask == 0, -1e9, attn_weights)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout_rate)(
            attn_weights, deterministic=deterministic
        )

        # Apply attention to values
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value_transposed)

        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            attn_output.shape[0], attn_output.shape[2], self.d_model
        )

        # Output projection
        output_proj = nn.Dense(
            features=self.d_model,
            use_bias=True,
            kernel_init=nn.initializers.xavier_uniform(),
        )(attn_output)

        return output_proj, attn_weights


class TransformerBlock(nn.Module):
    """Single Transformer layer with self-attention and MLP.

    Implements the standard Transformer block: self-attention with residual
    connection and layer normalization, followed by position-wise feed-forward
    network with residual connection and layer normalization.

    Attributes:
        d_model: Dimensionality of the model.
        n_heads: Number of attention heads.
        d_ff: Hidden dimension of the feed-forward network.
        dropout_rate: Dropout rate.
    """
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        encoder_output: jax.Array | None = None,
        mask: jax.Array | None = None,
        deterministic: bool = True,
        capture_activations: bool = False,
    ) -> Tuple[jax.Array, jax.Array | None]:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor with shape (batch, seq_len, d_model).
            encoder_output: Optional encoder output for cross-attention with
                shape (batch, encoder_seq_len, d_model).
            mask: Optional attention mask.
            deterministic: Whether to apply dropout.
            capture_activations: Whether to capture this layer's output
                via self.sow for later extraction.

        Returns:
            A tuple of:
                - Output tensor with shape (batch, seq_len, d_model).
                - Attention weights from cross-attention (if applicable).
        """
        # Self-attention
        mha = MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
        )
        attn_output, attn_weights = mha(x, x, x, mask=mask, deterministic=deterministic)

        # Residual connection and layer norm
        x = nn.LayerNorm()(x + attn_output)

        # Cross-attention (if encoder_output provided)
        cross_attn_weights = None
        if encoder_output is not None:
            cross_mha = MultiHeadAttention(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
            )
            cross_attn_output, cross_attn_weights = cross_mha(
                x, encoder_output, encoder_output, mask=None, deterministic=deterministic
            )
            x = nn.LayerNorm()(x + cross_attn_output)

        # Feed-forward network
        x_ff = nn.Dense(
            features=self.d_ff,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)
        x_ff = nn.relu(x_ff)
        x_ff = nn.Dropout(rate=self.dropout_rate)(x_ff, deterministic=deterministic)
        mlp_output = nn.Dense(
            features=self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x_ff)

        # Residual connection and layer norm
        x = nn.LayerNorm()(x + mlp_output)

        # Capture activation if requested
        if capture_activations:
            self.sow('intermediates', 'block_output', x)

        return x, cross_attn_weights


class Encoder(nn.Module):
    """Transformer encoder stack.

    Stacks multiple TransformerBlocks to form the encoder, which processes
    the source sequence and produces contextualized representations.

    Attributes:
        vocab_size: Size of the token vocabulary.
        max_seq_len: Maximum sequence length.
        d_model: Dimensionality of the model.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        d_ff: Hidden dimension of the feed-forward networks.
        dropout_rate: Dropout rate.
    """
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        src: jax.Array,
        deterministic: bool = True,
        capture_activations: bool = False,
    ) -> jax.Array:
        """Encode the source sequence.

        Args:
            src: Source token indices with shape (batch, src_seq_len).
            deterministic: Whether to apply dropout.
            capture_activations: Whether to capture intermediate activations
                via self.sow for later extraction. When True, embeddings and
                each encoder block output are sown to the 'intermediates'
                collection with keys 'embedding', 'encoder_block_0', etc.

        Returns:
            Encoder output with shape (batch, src_seq_len, d_model).
        """
        # Embedding
        embed = Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            max_len=self.max_seq_len,
        )
        x = embed(src, capture_activations=capture_activations)

        # Scale by sqrt(d_model) as in standard Transformer
        x = x * jnp.sqrt(self.d_model)

        # Capture scaled embedding if requested (after scaling)
        if capture_activations:
            self.sow('intermediates', 'embedding', x)

        # Stack of transformer blocks
        for i in range(self.n_layers):
            block = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
            )
            x, _ = block(x, deterministic=deterministic, capture_activations=capture_activations)

            # Capture block output with proper naming
            if capture_activations:
                self.sow('intermediates', f'encoder_block_{i}', x)

        return x


class Decoder(nn.Module):
    """Transformer decoder stack with cross-attention.

    Stacks multiple TransformerBlocks with self-attention and cross-attention
    to form the decoder, which generates the target sequence conditioned on
    the encoder output.

    Attributes:
        vocab_size: Size of the token vocabulary.
        max_seq_len: Maximum sequence length.
        d_model: Dimensionality of the model.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        d_ff: Hidden dimension of the feed-forward networks.
        dropout_rate: Dropout rate.
    """
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        tgt: jax.Array,
        encoder_output: jax.Array,
        deterministic: bool = True,
        capture_activations: bool = False,
    ) -> jax.Array:
        """Decode the target sequence.

        Args:
            tgt: Target token indices with shape (batch, tgt_seq_len).
            encoder_output: Encoder output with shape (batch, src_seq_len, d_model).
            deterministic: Whether to apply dropout.
            capture_activations: Whether to capture intermediate activations
                via self.sow for later extraction. When True, embeddings and
                each decoder block output are sown to the 'intermediates'
                collection with keys 'decoder_embedding', 'decoder_block_0', etc.

        Returns:
            Decoder output with shape (batch, tgt_seq_len, d_model).
        """
        # Embedding
        embed = Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            max_len=self.max_seq_len,
        )
        x = embed(tgt, capture_activations=capture_activations)

        # Scale by sqrt(d_model) as in standard Transformer
        x = x * jnp.sqrt(self.d_model)

        # Capture scaled embedding if requested (after scaling)
        if capture_activations:
            self.sow('intermediates', 'decoder_embedding', x)

        # Create causal mask for self-attention
        # Handle both 1D (vmapped) and 2D (batch) inputs
        if tgt.ndim == 1:
            tgt_len = tgt.shape[0]
        else:
            tgt_len = tgt.shape[1]
        causal_mask = jnp.tril(jnp.ones((tgt_len, tgt_len)))
        causal_mask = causal_mask.reshape(1, 1, tgt_len, tgt_len)

        # Stack of transformer blocks
        for i in range(self.n_layers):
            block = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
            )
            x, _ = block(
                x,
                encoder_output=encoder_output,
                mask=causal_mask,
                deterministic=deterministic,
                capture_activations=capture_activations,
            )

            # Capture block output with proper naming
            if capture_activations:
                self.sow('intermediates', f'decoder_block_{i}', x)

        return x


class Seq2SeqTransformer(nn.Module):
    """Top-level seq2seq Transformer with activation hooks.

    This module combines the encoder and decoder into a complete seq2seq model
    that can be used for training and inference on compositional generalization
    tasks. It supports two modes of activation extraction:

    1. **Dictionary-based (backward compatible):** Returns ActivationsDict
       alongside logits for use during training.

    2. **sow-based (recommended for RGA):** Use with `mutable=['intermediates']`
       to extract all hidden states without modifying the return signature.
       This is the preferred method for Representational-Geometry Alignment
       signal extraction as it avoids memory overhead during standard training.

    Attributes:
        config: TransformerConfig object with all hyperparameters.
    """
    config: TransformerConfig

    def setup(self):
        """Initialize encoder and decoder submodules."""
        self.encoder = Encoder(
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            dropout_rate=self.config.dropout_rate,
        )
        self.decoder = Decoder(
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            dropout_rate=self.config.dropout_rate,
        )
        self.output_proj = nn.Dense(
            features=self.config.vocab_size,
            kernel_init=nn.initializers.xavier_uniform(),
        )

    def __call__(
        self,
        src: jax.Array,
        tgt: jax.Array,
        training: bool = False,
        capture_activations: bool = False,
    ) -> jax.Array:
        """Forward pass through the seq2seq transformer.

        Args:
            src: Source token indices with shape (batch, src_seq_len).
            tgt: Target token indices with shape (batch, tgt_seq_len).
            training: Whether to use training mode (apply dropout).
            capture_activations: Whether to capture intermediate activations
                via self.sow for later extraction. When True, use
                `model.apply(params, src, tgt, mutable=['intermediates'])`
                to retrieve the intermediates collection.

        Returns:
            Logits with shape (batch, tgt_seq_len, vocab_size).
        """
        # Encode
        encoder_output = self.encoder(
            src,
            deterministic=not training,
            capture_activations=capture_activations,
        )

        # Decode
        decoder_output = self.decoder(
            tgt,
            encoder_output,
            deterministic=not training,
            capture_activations=capture_activations,
        )

        # Project to vocabulary
        logits = self.output_proj(decoder_output)

        return logits


def _flatten_intermediates(nested: dict, prefix: str = "") -> Dict[str, jax.Array]:
    """Recursively flatten the intermediates structure from Flax sow calls.

    Flax's sow mechanism creates a nested dictionary structure based on module
    hierarchy. This function flattens it to a single-level dict with clear keys.

    Args:
        nested: Nested dictionary from intermediates collection.
        prefix: Prefix for keys (used in recursion).

    Returns:
        Flattened dictionary mapping layer names to activation arrays.
    """
    result = {}
    for key, value in nested.items():
        # Skip internal Flax module names (like Embed_0, TransformerBlock_0)
        # and focus on our explicit sow names
        if isinstance(value, dict):
            # Check if this dict contains our target keys
            target_keys = ['embedding', 'decoder_embedding', 'block_output']
            target_keys.extend([f'encoder_block_{i}' for i in range(20)])
            target_keys.extend([f'decoder_block_{i}' for i in range(20)])

            for subkey, subvalue in value.items():
                if subkey in target_keys:
                    # Extract the array from tuple if needed
                    if isinstance(subvalue, tuple):
                        result[subkey] = subvalue[0]
                    else:
                        result[subkey] = subvalue
                elif isinstance(subvalue, dict):
                    # Recurse for nested dicts
                    nested_result = _flatten_intermediates(subvalue, f"{key}_{subkey}")
                    result.update(nested_result)
    return result


def get_model_representations(
    params: flax.core.FrozenDict,
    model: Seq2SeqTransformer,
    src: jax.Array,
    tgt: jax.Array,
    rngs: dict | None = None,
) -> Dict[str, jax.Array]:
    """Extract all intermediate representations from the model.

    This function uses Flax's sow mechanism to capture hidden states
    from all layers in a single forward pass. It is designed for
    Representational-Geometry Alignment (RGA) signal extraction.

    Args:
        params: Model parameters (typically variables["params"]).
        model: The model instance (uninitialized or initialized).
        src: Source token indices with shape (batch, src_seq_len).
        tgt: Target token indices with shape (batch, tgt_seq_len).
        rngs: Optional RNG streams for dropout (e.g., {"dropout": key}).

    Returns:
        Dictionary mapping layer names to activation tensors.
        Keys follow the naming convention:
        - "embedding": Encoder embedding (after sqrt(d_model) scaling)
        - "encoder_block_0", "encoder_block_1", ...: Encoder layer outputs
        - "decoder_embedding": Decoder embedding (after sqrt(d_model) scaling)
        - "decoder_block_0", "decoder_block_1", ...: Decoder layer outputs

        Each activation tensor has shape (batch, seq_len, d_model).

    Example:
        >>> model = Seq2SeqTransformer(config)
        >>> variables = model.init(rng, src, tgt)
        >>> intermediates = get_model_representations(
        ...     variables["params"], model, src, tgt
        ... )
        >>> encoder_last_layer = intermediates["encoder_block_1"]
    """
    _, intermediates = model.apply(
        {"params": params},
        src,
        tgt,
        training=False,
        capture_activations=True,  # Enable activation capture
        mutable=["intermediates"],
        rngs=rngs,
    )

    # Flatten the nested intermediates structure
    raw = intermediates.get("intermediates", {})
    return _flatten_intermediates(raw)
