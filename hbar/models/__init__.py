"""Flax model definitions for the H-Bar Model.

This module contains Transformer and RNN architectures with
built-in support for tracking attentional fidelity (alpha_A)
and other H-Bar signals.
"""

from hbar.models.config import ActivationsDict, TransformerConfig
from hbar.models.transformer import (
    Decoder,
    Embed,
    Encoder,
    MultiHeadAttention,
    Seq2SeqTransformer,
    TransformerBlock,
)

__all__ = [
    # Config
    "TransformerConfig",
    "ActivationsDict",
    # Transformer components
    "Embed",
    "MultiHeadAttention",
    "TransformerBlock",
    "Encoder",
    "Decoder",
    "Seq2SeqTransformer",
]
