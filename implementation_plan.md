# Implementation Plan

[Overview]
Implement a Flax Linen Encoder-Decoder Transformer (2 layers, 4 heads, d_model=128) with activation hooks for RGA signal extraction, targeting SCAN/COGS compositional generalization benchmarks.

This implementation establishes the neural network backbone for the H-Bar Model V3.0+. The Transformer must be purely functional (JIT-compatible), expose intermediate hidden states via activation hooks for Representational-Geometry Alignment (RGA) signal extraction, and follow the exact specifications from Section 11.1 of the H-Bar paper. The Encoder-Decoder architecture is required because SCAN and COGS are cross-modal seq2seq mapping tasks (natural language → action sequences / logical forms).

[Types]
Define a configuration dataclass and activation tracking types using `flax.struct.dataclass` for JIT compatibility.

```python
# hbar/models/config.py
import flax.struct
import jax.numpy as jnp

class TransformerConfig(flax.struct.dataclass):
    """Configuration for the H-Bar Transformer."""
    vocab_size: int = struct.field(default=128)        # Token vocabulary size
    max_seq_len: int = struct.field(default=50)        # Maximum sequence length
    d_model: int = struct.field(default=128)           # Model dimension (Section 11.1)
    n_layers: int = struct.field(default=2)            # Number of layers (Section 11.1)
    n_heads: int = struct.field(default=4)             # Number of attention heads (Section 11.1)
    d_ff: int = struct.field(default=512)              # Feed-forward hidden dimension
    dropout_rate: float = struct.field(default=0.1)    # Dropout rate
    initializer: str = struct.field(default="xavier_uniform")  # Weight initializer

class ActivationsDict(flax.struct.dataclass):
    """Dictionary of intermediate activations for RGA signal extraction."""
    encoder_layers: dict  # {"layer_0": (batch, seq, d_model), ...}
    decoder_layers: dict  # {"layer_0": (batch, seq, d_model), ...}
```

[Files]
Create 3 new files and modify 4 existing files.

**New files:**
1. `hbar/models/config.py` — TransformerConfig and ActivationsDict dataclasses
2. `hbar/models/transformer.py` — Full Encoder-Decoder Transformer implementation with:
   - `Embed` module (token + positional embeddings)
   - `MultiHeadAttention` module
   - `TransformerBlock` module (MHA + MLP with residual connections and layer norm)
   - `Encoder` module (stack of TransformerBlocks)
   - `Decoder` module (stack of TransformerBlocks with cross-attention)
   - `Seq2SeqTransformer` top-level module returning (logits, activations_dict)
3. `tests/test_model_shapes.py` — Shape verification and JIT compatibility tests

**Modified files:**
4. `hbar/models/__init__.py` — Add exports for `config` and `transformer` modules
5. `memory-bank/activeContext.md` — Update Subtask to 1.2 (in progress), add architecture details
6. `memory-bank/progress.md` — Mark Subtask 1.2 objectives as complete
7. `memory-bank/decisionLog.md` — Add Decision 4: Encoder-Decoder Architecture with activation hooks

[Functions]
New functions in `hbar/models/transformer.py`:

- `Embed(num_embeddings, features, max_len)` — Token embedding + learned positional encoding, returns `(batch, seq_len, features)`
- `MultiHeadAttention(d_model, n_heads, dropout_rate)` — Scaled dot-product attention with multi-head projection, returns `(batch, seq_len, d_model)` and attention weights
- `TransformerBlock(d_model, n_heads, d_ff, dropout_rate)` — Self-attention + layernorm + MLP + layernorm with residual connections
- `Encoder(vocab_size, max_seq_len, d_model, n_layers, n_heads, d_ff, dropout_rate)` — Stack of encoder TransformerBlocks
- `Decoder(vocab_size, max_seq_len, d_model, n_layers, n_heads, d_ff, dropout_rate)` — Stack of decoder TransformerBlocks with self-attention and cross-attention
- `Seq2SeqTransformer(config)` — Top-level module with `__call__(src, tgt, training=False)` returning `(logits, activations_dict)` where activations_dict contains `{"encoder_layer_0": ..., "encoder_layer_1": ..., "decoder_layer_0": ..., "decoder_layer_1": ...}`

No functions removed. No existing functions modified.

[Classes]
New Flax Linen Module classes in `hbar/models/transformer.py`:

- `Embed(nn.Module)` — Embedding layer with learned positional encodings
- `MultiHeadAttention(nn.Module)` — Multi-head self-attention mechanism
- `TransformerBlock(nn.Module)` — Single Transformer layer (attention + MLP)
- `Encoder(nn.Module)` — Transformer encoder stack
- `Decoder(nn.Module)` — Transformer decoder stack with cross-attention
- `Seq2SeqTransformer(nn.Module)` — Top-level seq2seq model with activation hooks

All classes use `flax.linen.Module` pattern with `@nn.compact` decorator for functional purity.

[Dependencies]
No new dependencies required. Uses existing project stack:
- `jax`, `jax.numpy` — Core numerical operations
- `flax.linen` (as `nn`) — Neural network modules
- `flax.struct` — Dataclass utilities for JIT compatibility
- `chex` — Testing utilities (in test file)
- `pytest` — Test runner (in test file)

[Testing]
Create `tests/test_model_shapes.py` with the following test cases:

1. `test_output_logits_shape()` — Verify logits shape is `(batch, tgt_seq_len, vocab_size)`
2. `test_activations_dict_keys()` — Verify activations dict contains keys for all encoder and decoder layers
3. `test_activations_shape()` — Verify each activation tensor has shape `(batch, seq_len, d_model=128)`
4. `test_jit_compatibility()` — Verify forward pass works with `jax.jit`
5. `test_gradient_flow()` — Verify gradients flow through the model using `jax.grad`
6. `test_deterministic_dropout()` — Verify dropout is disabled when `training=False`

Test pattern follows existing `test_ode_stability.py` conventions with `pytest` class-based tests and `chex` assertions.

[Implementation Order]
Implement in the following order to minimize conflicts and enable incremental verification:

1. Create `hbar/models/config.py` — Define TransformerConfig dataclass (no dependencies)
2. Create `hbar/models/transformer.py` — Implement all modules bottom-up:
   a. `Embed` module
   b. `MultiHeadAttention` module
   c. `TransformerBlock` module
   d. `Encoder` module
   e. `Decoder` module
   f. `Seq2SeqTransformer` top-level module
3. Update `hbar/models/__init__.py` — Export new modules
4. Create `tests/test_model_shapes.py` — Write all test cases
5. Run tests with `pytest tests/test_model_shapes.py -v` — Verify implementation
6. Update `memory-bank/activeContext.md` — Reflect current state
7. Update `memory-bank/progress.md` — Mark Subtask 1.2 complete
8. Update `memory-bank/decisionLog.md` — Log architecture decision
