"""Configuration dataclasses for the H-Bar Transformer model."""

from flax import struct


@struct.dataclass
class FusionConfig:
    """Configuration for H-Bar signal fusion (Equation 6).

    These weights control how the three operative signals (GCA, RGA, AC)
    are combined into the fused schema coherence estimate σ̃_A.

    The default weights (w_gca=0.4, w_rga=0.35, w_ac=0.25) are from the
    H-Bar paper and sum to 1.0 for normalized contribution.

    Attributes:
        w_gca: Weight for Gradient-Composition Alignment (g_A). Default: 0.4
        w_rga: Weight for Representational-Geometry Alignment (r_A). Default: 0.35
        w_ac: Weight for Augmentation Consistency (c_A). Default: 0.25
        target_sigma_critical: Threshold for Phase 2 entry (crystallization).
            Default: 0.5 (based on baseline analysis showing σ̃_A ≈ 0.2686,
            requiring approximately 2x improvement for Phase 2 entry).
    """

    w_gca: float = struct.field(default=0.4)
    w_rga: float = struct.field(default=0.35)
    w_ac: float = struct.field(default=0.25)
    target_sigma_critical: float = struct.field(default=0.5)
    kappa_alpha: float = struct.field(default=2.0)


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
    fusion_config: FusionConfig | None = struct.field(default=None)


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
