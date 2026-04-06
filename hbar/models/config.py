"""Configuration dataclasses for the H-Bar Transformer model."""

from flax import struct


@struct.dataclass
class TransformerConfig:
    """Configuration for the H-Bar Transformer.

    This configuration follows the exact specifications from Section 11.1 of the
    H-Bar paper for compositional generalization benchmarks (SCAN/COGS).

    Attributes:
        vocab_size: Token vocabulary size.
        max_seq_len: Maximum sequence length.
        d_model: Model dimension (Section 11.1).
        n_layers: Number of layers (Section 11.1).
        n_heads: Number of attention heads (Section 11.1).
        d_ff: Feed-forward hidden dimension.
        dropout_rate: Dropout rate.
        initializer: Weight initializer type.
    """
    vocab_size: int = struct.field(default=128)
    max_seq_len: int = struct.field(default=50)
    d_model: int = struct.field(default=128)
    n_layers: int = struct.field(default=2)
    n_heads: int = struct.field(default=4)
    d_ff: int = struct.field(default=512)
    dropout_rate: float = struct.field(default=0.1)
    initializer: str = struct.field(default="xavier_uniform")


@struct.dataclass
class ActivationsDict:
    """Dictionary of intermediate activations for RGA signal extraction.

    This dataclass captures hidden states from all encoder and decoder layers,
    enabling Representational-Geometry Alignment (RGA) analysis for H-Bar
    signal extraction (sigma_A, delta_A, alpha_A).

    Attributes:
        encoder_layers: Dictionary mapping layer names to activation tensors
            with shape (batch, seq_len, d_model).
        decoder_layers: Dictionary mapping layer names to activation tensors
            with shape (batch, seq_len, d_model).
    """
    encoder_layers: dict = struct.field(default_factory=dict)
    decoder_layers: dict = struct.field(default_factory=dict)
