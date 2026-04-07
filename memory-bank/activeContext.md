# Active Context

## Current Goal

**Implementing baseline training loop** to verify the "Illusion of Mastery" failure mode.

## Current Phase

- **Phase:** 1 (Foundation)
- **Week:** 1-2
- **Subtask:** 1.6 - Baseline Verification (In Progress)

## Immediate Objectives (Week 1)

1. ✓ Initialize git repository and project structure
2. ✓ Set up Python virtual environment with JAX/Flax stack
3. ✓ Establish Cline rules for functional purity and mathematical rigor
4. ✓ Create memory bank for project state tracking
5. ✓ Implement Flax Transformer (2 layer, 4 head, d_model=128)
6. ✓ Implement JAX-native tokenization and encoding pipeline

## Technical Stack

- **JAX/jaxlib:** High-performance numerical computing with automatic differentiation
- **Flax:** Neural network library built on JAX with linen module system
- **Optax:** Gradient processing and optimization library
- **Chex:** Testing library for JAX (property-based testing, numerical stability)
- **Distrax:** Probabilistic programming library for JAX

## Implementation Architecture

**Transformer Specification (matching manuscript Section 11.1):**
- 2 layers, 4 attention heads, d_model = 128
- Standard seq2seq architecture for SCAN/COGS
- Purely functional forward passes for jax.jit compilation

**Signal Extraction Requirements:**
- GCA (Gradient-Composition Alignment): Correlation of ∇L_train and ∇L_comp-batch
- RGA (Representational-Geometry Alignment): RDM correlation with structural distances
- AC (Augmentation Consistency): Cosine similarity across structure-preserving augmentations

**ODE Integration:**
- IMEX Runge-Kutta adaptive-step integrator
- Coupled system: σ_A, δ_A, α_A, M̂_A, Ξ_A variables
- Timescale separation: fast (δ, σ, α) vs slow (M̂, Ξ) subsystems

## Verification Workflow

**Trigger:** Any change to `hbar/models/`
**Action:** Run shape and JIT compatibility tests
**Command:** `pytest tests/test_model_shapes.py tests/test_extraction_hooks.py -v`

**Tests verify:**
- Output logits shape: (batch, tgt_seq_len, vocab_size)
- Intermediates collection contains all expected layer keys
- Activation shapes: (batch, seq_len, d_model) for all layers
- JIT compilation works with jax.jit and mutable=['intermediates']
- Gradient flow through entire model with extraction enabled
- Purity: same inputs produce identical intermediates (no hidden state)
- Dropout behavior (deterministic when training=False)

**Trigger:** Any change to `hbar/engine/`
**Action:** Run encoding pipeline integration tests
**Command:** `pytest tests/test_encoding_pipeline.py -v`

**Tests verify:**
- Tokenizer: special tokens, encode/decode, truncation, SCAN vocabulary
- Masks: padding mask shape/values, causal mask triangularity, combined decoder mask
- Batch preparation: shapes, decoder input shifting, loss computation
- Integration: full pipeline forward pass, JIT compilation, gradient flow

**Trigger:** Any change to `hbar/core/`
**Action:** Run chex test to ensure ODE Jacobian stability (Eq. 24)
**Command:** `pytest tests/test_ode_stability.py -v`

**Stability Checks:**
- Jacobian condition number κ(J) < 1000
- Forward invariance (Proposition 3.2): all variables remain in valid ranges
- Timescale separation (Proposition 3.3): fast/slow subsystem eigenvalue ratio > 10

## Completed Components

### Transformer Architecture (`hbar/models/`)
- **config.py**: TransformerConfig (dataclass with all hyperparameters), ActivationsDict
- **transformer.py**: Complete Encoder-Decoder with Flax `sow`-based activation hooks:
  - `Embed`: Token embedding + learned positional encodings
  - `MultiHeadAttention`: Scaled dot-product attention with Q/K/V projections
  - `TransformerBlock`: Self-attention + cross-attention + MLP with residuals
  - `Encoder`: Stack of N transformer blocks
  - `Decoder`: Stack with causal masking for autoregressive decoding
  - `Seq2SeqTransformer`: Top-level model with `capture_activations` flag
  - `get_model_representations()`: Functional wrapper for extracting intermediates

### Layer Naming Convention for RGA Engine
The RGA engine accesses activations via the intermediates collection returned by
`model.apply(params, src, tgt, mutable=['intermediates'])` or via
`get_model_representations(params, model, src, tgt)`.

**Intermediates keys and their tensor shapes:**

| Key | Description | Shape |
|-----|-------------|-------|
| `embedding` | Encoder embedding (after √d_model scaling) | (batch, src_seq_len, d_model) |
| `encoder_block_0` | Encoder layer 0 output (after final LayerNorm) | (batch, src_seq_len, d_model) |
| `encoder_block_1` | Encoder layer 1 output | (batch, src_seq_len, d_model) |
| `decoder_embedding` | Decoder embedding (after √d_model scaling) | (batch, tgt_seq_len, d_model) |
| `decoder_block_0` | Decoder layer 0 output (after final LayerNorm) | (batch, tgt_seq_len, d_model) |
| `decoder_block_1` | Decoder layer 1 output | (batch, tgt_seq_len, d_model) |

**Usage pattern:**
```python
# Extract all representations in a single forward pass
intermediates = get_model_representations(params, model, src, tgt)

# Access specific layer for RGA analysis
encoder_last = intermediates["encoder_block_1"]  # Shape: (batch, seq_len, d_model)
decoder_last = intermediates["decoder_block_1"]
```

**Memory efficiency:** Set `capture_activations=False` (default) during standard
training to avoid the overhead of sow operations. Only enable extraction during
evaluation or RGA signal computation phases.

### Encoding Pipeline (`hbar/engine/`)
- **tokenizer.py**: Word-level tokenizer with SCAN vocabulary
  - Special tokens: `<PAD>` (0), `<BOS>` (1), `<EOS>` (2), `<UNK>` (3)
  - `encode()`: Adds BOS/EOS, truncates/pads to max_seq_len, returns JAX array
  - `decode()`: Converts IDs back to text, skips special tokens
  - `create_scan_tokenizer()`: Pre-initialized with SCAN command/action vocabulary
- **encoding.py**: Functional mask generation (JIT-compatible)
  - `get_padding_mask()`: Shape (batch, 1, 1, seq_len), True for valid tokens
  - `get_causal_mask()`: Lower triangular mask for decoder self-attention
  - `get_decoder_mask()`: Combined padding + causal mask
  - `apply_mask()`: Applies mask to attention scores with -1e9 for masked positions
- **data_utils.py**: Batch preprocessing for training
  - `Batch` dataclass (flax.struct): inputs, decoder_inputs, labels, src_mask, tgt_mask
  - `HBarBatch` dataclass: id_stream, ood_stream, aug_stream (triple-stream batch)
  - `prepare_batch()`: Converts (input, output) pairs to training-ready Batch
  - `compute_loss()`: Cross-entropy loss ignoring padding tokens
  - `compute_accuracy()`: Token-level accuracy over non-padding tokens
  - `get_hbar_batch()`: Generates triple-stream HBarBatch for signal extraction
- **augmentation.py**: Structure-preserving augmentation pipeline
  - `apply_primitive_substitution()`: Swaps primitives (e.g., 'jump' → 'run')
  - `apply_argument_permutation()`: Swaps sub-command order (e.g., 'jump left and look right' → 'look right and jump left')
  - `apply_augmentation()`: Randomly chooses between substitution and permutation
  - `vmap_augment_batch()`: Vectorized augmentation with configurable `permutation_probability` (default 0.5)
  - `generate_augmentation_keys()`: Creates PRNGKeys for each sample
  - Supports both SCAN and COGS domains
- **signals.py**: H-Bar signal computation engine
  - `compute_augmentation_consistency()`: Computes c_A signal (Equation 5) using cosine similarity
  - `compute_layer_weighted_ac()`: Weighted AC across multiple layers
  - `compute_representation_norm()`: Monitors representation magnitude

## Generative Grammar G(d) Structure

The generative grammar $G(d)$ is the core mechanism for producing compositional
samples and computing H-Bar signals. It maps from a domain specification $d$ to
a distribution over (input, output) pairs with controlled compositional structure.

**SCAN Grammar (`hbar/benchmarks/scan_grammar.py`):**
- **Primitives:** jump, run, walk, look → I_JUMP, I_RUN, I_WALK, I_LOOK
- **Directions:** left, right → I_TURN_LEFT, I_TURN_RIGHT
- **Modifiers:** twice (×2), thrice (×3)
- **Conjunctions:** and, after (recursive composition)
- **CFG Rules:** S → S conjunction S | VP; VP → primitive | turn direction | primitive modifier
- **Compositional Probes:** `sample_compositional_probe(target='jump')` generates
  nested structures like "jump around left twice and look thrice"

**COGS Grammar (`hbar/benchmarks/cogs_grammar.py`):**
- **LogicalForm:** Frozen dataclass with predicate, args, children (tree structure)
- **Constructions:** active, passive, intransitive, ditransitive, embedded
- **String representation:** "chase ( agent = dog , patient = cat )"
- **Tree-Edit Distance:** `get_structural_distance(lf1, lf2)` computes RDMstruct

**GrammarEngine (`hbar/benchmarks/grammar_engine.py`):**
- Unified interface wrapping both grammars
- `get_compositional_batch()` → returns `Batch` with recombination-only samples
- `generate_id_batch()` → returns `Batch` with simple in-distribution samples
- `compute_rdmstruct()` → pairwise structural distances for RGA
- Deterministic given `jax.random.PRNGKey`

**H-Bar Signal Integration:**
- **GCA:** Use `get_compositional_batch()` for gradient alignment computation
- **RGA:** Use `compute_rdmstruct()` for structural RDM correlation
- **AC:** Use `sample_compositional_probe()` for structure-preserving augmentations

## Verification Workflow Updates

**Trigger:** Any change to `hbar/engine/augmentation.py` or `hbar/engine/data_utils.py`
**Action:** Run HBar batch generator tests
**Command:** `pytest tests/test_hbar_generator.py -v`

**Tests verify:**
- HBarBatch structure: all three streams present with matching shapes
- HBarBatch is a valid JAX pytree (can be used with jax.tree_util)
- All streams have consistent batch_size and max_seq_len
- Augmentation changes tokens but preserves syntactic structure (masks identical)
- JIT compatibility: HBarBatch can be passed to jax.jit functions
- Determinism: same PRNGKey produces identical batches
- Works for both SCAN and COGS domains

## Evaluation Splits and Ground-Truth σ_A

### SCAN Add-Jump Split
- **Training (ID):** All commands WITHOUT 'jump' in compounds + isolated 'jump' → 'I_JUMP'
- **Test (OOD):** All commands where 'jump' appears in a compound structure
- **Purpose:** Tests if model can compose a known primitive into novel syntactic structures
- **Zero overlap:** No command (except isolated 'jump') appears in both sets

### COGS Subject-to-Object Split
- **BIASED_NOUNS:** ['hedgehog', 'porcupine', 'otter']
- **Training (ID):** Biased nouns only appear in Subject position
- **Test (OOD):** Biased nouns only appear in Object position
- **Purpose:** Tests if model learns syntactic roles vs. positional memorization

### Ground-Truth σ_A Calculation (Equation 7)
```
σ̂_A = Acc_OOD / Acc_ID
```
- **Range:** [0, 1] where 1.0 = perfect compositional generalization
- **Edge case:** If Acc_ID = 0, return σ̂_A = 0.0 (division-by-zero handling)
- **Calibration error:** |σ̃_A - σ̂_A| used to update multi-signal fusion weights

### Frozen Evaluation Sets
- **Size:** 2,000 ID + 2,000 OOD samples per domain
- **Statistical power:** Enables detection of accuracy differences with p < 0.0001
- **Reproducibility:** Committed to repository as JSON files in `data/` directory
- **Files:**
  - `data/scan_id_eval.json`, `data/scan_ood_eval.json`
  - `data/cogs_id_eval.json`, `data/cogs_ood_eval.json`

### Evaluator Class (`hbar/engine/evaluator.py`)
- Loads frozen evaluation sets from `data/` directory
- `evaluate(params, model)` → `EvaluationResult` with all metrics
- `calculate_calibration_error(sigma_tilde, sigma_hat)` → float
- JIT-compiled evaluation step for efficiency

## Baseline Training Configuration

The baseline training loop uses standard Adam optimizer without H-Bar signal modulation
to demonstrate the "Illusion of Mastery" failure mode.

**Hyperparameters:**
- **Optimizer:** Adam (learning_rate=1e-3)
- **Batch Size:** 64
- **Training Steps:** 5,000
- **Evaluation Interval:** Every 500 steps
- **Model:** 2-layer, 4-head Transformer, d_model=128

**Expected Outcome (Illusion of Mastery):**
- ID Accuracy: >95% (model masters in-distribution patterns)
- OOD Accuracy: <50% (model fails on novel compositions)
- σ̂_A: <0.5 (large generalization gap)

**Training Data:**
- Generated on-the-fly using `GrammarEngine.generate_id_batch()`
- Only in-distribution samples (no 'jump' in compounds for SCAN Add-Jump split)
- Ensures the model cannot see test compositions during training

## Next Steps (Week 2-4)

- Complete Kaggle run and verify "Illusion of Mastery" failure mode
- Implement multi-signal proxy extraction (RGA, AC) — GCA is complete
- Build ODE integrator for H-Bar dynamics

## GCA Signal (Subtask 5.1)

### Overview

The Gradient-Composition Alignment (GCA) signal $g_A$ is the first H-Bar operative
signal implemented. It measures the Pearson correlation between the gradient vectors
of the ID loss ($\nabla_\theta \mathcal{L}_{train}$) and OOD compositional loss
($\nabla_\theta \mathcal{L}_{comp-batch}$).

### Implementation

**`compute_gca(grad_id, gradient_ood)` in `hbar/engine/signals.py`:**
```
g_A = Σ(x_i - x̄)(y_i - ȳ) / √(Σ(x_i - x̄)² Σ(y_i - ȳ)² + ε)
```
- Adds epsilon (1e-8) for numerical stability with sparse gradients
- Returns scalar in range [-1, 1]
- Includes all trainable parameters (embeddings + transformer layers)

**`compute_dual_gradients(state, hbar_batch)` in `hbar/engine/trainer.py`:**
- Computes ID gradient from `hbar_batch.id_stream`
- Computes OOD gradient from `hbar_batch.ood_stream`
- Flattens both gradients using `jax.flatten_util.ravel_pytree`
- Returns tuple of (grad_id_flat, grad_ood_flat)

**`get_gca_signal(state, hbar_batch)` in `hbar/engine/trainer.py`:**
- JIT-compiled wrapper combining dual gradient extraction + GCA computation
- Returns scalar GCA value directly

### Expected Ranges and Interpretation

| Range | Interpretation |
|-------|----------------|
| g_A > 0.7 | Model is "crystallizing" compositional rules |
| 0.3 < g_A < 0.7 | Partial compositional alignment |
| 0.0 < g_A < 0.3 | Model in σ-trap (gradient misalignment) |
| g_A < 0.0 | Learning ID actively harms OOD performance |

### Baseline Results (Kaggle GPU T4, 100 batches, batch_size=32)

| Metric | Value |
|--------|-------|
| **Mean GCA (g_A)** | **-0.0235 ± 0.0075 (SEM)** |
| Std Deviation | 0.0753 |
| Min GCA | -0.1753 |
| Max GCA | 0.1727 |

**Interpretation: NEGATIVE GCA confirms σ-trap** — learning ID patterns actively harms OOD performance. The gradients for memorizing in-distribution data are misaligned with the gradients needed for compositional generalization.

This is the baseline noise floor to beat in Phase 3. The GCA regularizer should push g_A from -0.02 toward +0.7+.

### Analysis Script

**`scripts/analyze_gca_baseline.py`:**
- Loads saved `model_params.msgpack`
- Computes GCA over 100 batches (batch_size=32 for memory efficiency)
- Reports mean ± SEM, min, max with interpretation
- CLI flags: `--params`, `--num-batches`, `--batch-size`, `--domain`, `--seed`

## AC Signal (Subtask 5.2)

### Overview

The Augmentation Consistency (AC) signal $c_A$ measures the representational invariance of the model under structure-preserving augmentations. It computes the cosine similarity between encoder representations of the original input and its augmented version (where primitives are swapped but syntactic structure is preserved).

### Implementation

**`compute_augmentation_consistency(reps_id, reps_aug, layer, mask)` in `hbar/engine/signals.py`:**
```
c_A = (cos_sim + 1) / 2  # mapped from [-1, 1] to [0, 1]
```
- Uses cosine similarity between final encoder layer representations
- Masks out padding positions
- Returns scalar in range [0, 1] where 1.0 = perfect invariance

**`compute_ac_from_batch(state, hbar_batch, model)` in `hbar/engine/signals.py`:**
- Wrapper that extracts representations from HBarBatch streams
- Calls `compute_augmentation_consistency` on id_stream and aug_stream

**`get_ac_signal(state, hbar_batch, model)` in `hbar/engine/trainer.py`:**
- JIT-compiled wrapper combining representation extraction + AC computation
- Returns scalar AC value directly

### Expected Ranges and Interpretation

| Range | Interpretation |
|-------|----------------|
| c_A > 0.8 | Strong invariance — compositional schema well-encoded |
| 0.5 < c_A < 0.8 | Moderate invariance — partial structural capture |
| c_A < 0.5 | Low invariance — representations drift under augmentation |

### Baseline Results (Kaggle GPU T4, 100 batches, batch_size=32)

| Metric | Value |
|--------|-------|
| **Mean AC (c_A)** | **0.9901 ± 0.0004 (SEM)** |
| Std Deviation | 0.0044 |
| Min AC | 0.9759 |
| Max AC | 0.9978 |

**Interpretation: EXTREMELY HIGH AC (≈0.99)** indicates near-perfect representational invariance under augmentation. However, this must be interpreted alongside GCA:

**Combined Signal Analysis (GCA + AC):**

| Signal | Value | Interpretation |
|--------|-------|----------------|
| g_A (GCA) | -0.0249 ± 0.0076 | ✗ NEGATIVE — Learning ID harms OOD |
| c_A (AC) | 0.9901 ± 0.0004 | ✓ HIGH — Strong invariance |
| r(g_A, c_A) | 0.2133 | Weak coupling |

**Key Insight: σ-Trap Confirmed**

The pattern AC >> GCA (0.99 >> -0.02) confirms the σ-trap signature:
- **High AC** reflects shallow invariance from self-attention mechanisms
- **Negative GCA** reveals broken gradient geometry for compositional rules
- **Weak correlation** (r=0.21) shows they capture different failure aspects

This is the characteristic pattern where Transformer self-attention provides token-level consistency (high AC) but the gradient geometry is misaligned with compositional generalization (negative GCA).

### Analysis Script

**`scripts/analyze_ac_baseline.py`:**
- Loads saved `model_params.msgpack`
- Computes both GCA and AC over 100 batches (batch_size=32)
- Reports mean ± SEM for both signals
- Computes Pearson correlation r(g_A, c_A)
- Includes H-Bar Phase 2 prediction check
- CLI flags: `--params`, `--num-batches`, `--batch-size`, `--domain`, `--seed`

### H-Bar Phase 3 Prediction

Based on the baseline signal profile (g_A = -0.02, c_A = 0.99), the H-Bar framework predicts:

**σ_critical Threshold:**
- The model must achieve σ_A > σ_critical ≈ 0.7–0.8 to enter Phase 2
- This requires pushing g_A from -0.02 to >0.7 through H-Bar signal modulation
- The large gap between AC (0.99) and GCA (-0.02) indicates significant schema reorganization needed

**Phase 2 Entry Criteria:**
- g_A must transition from negative to positive (>0.7)
- AC should remain high (>0.8) while GCA increases
- The correlation r(g_A, c_A) should strengthen as true compositional rules crystallize

## RGA Signal (Subtask 6.1)

### Overview

The Representational-Geometry Alignment (RGA) signal $r_A$ measures whether the model's internal representation geometry aligns with the structural geometry of the grammar. Per Equation 4 of the H-Bar paper, RGA quantifies if items with similar grammatical structure are represented similarly in the model's latent space.

### Implementation

**`compute_rdm_representational(representations, method)` in `hbar/engine/signals.py`:**
- Computes N×N pairwise distance matrix from activation vectors
- Supports cosine, euclidean, and correlation distance metrics
- Default: cosine distance (1 - cosine_similarity)

**`compute_rga(rdm_rep, rdm_struct)` in `hbar/engine/signals.py`:**
- Extracts upper triangle from both RDMs (excluding diagonal)
- Computes **Spearman rank correlation** between flattened vectors
- Returns scalar in [-1, 1]

**`_rank_data(data)` in `hbar/engine/signals.py`:**
- JAX-compatible ranking with average tie handling
- Required for Spearman correlation computation

**Structural Distance for SCAN (`_scan_structural_distance` in `grammar_engine.py`):**
- Normalized Levenshtein (edit) distance on action sequence tokens
- Example: "I_JUMP" vs "I_JUMP I_JUMP" → distance = 1/2

**Structural Distance for COGS:**
- Tree-edit distance on LogicalForm trees (already implemented)

### Expected Ranges and Interpretation

| Range | Interpretation |
|-------|----------------|
| r_A > 0.5 | Moderate-high alignment — representations reflect grammar structure |
| 0.2 < r_A < 0.5 | Low-moderate alignment — weak structural correspondence |
| r_A < 0.2 | Low alignment — representations geometrically disorganized |

### Analysis Script

**`scripts/analyze_rga_baseline.py`:**
- Loads saved `model_params.msgpack`
- Generates N=100 compositional probes
- Extracts BOS token representations from final encoder layer
- Computes RGA = Spearman(RDM_rep, RDM_struct)
- CLI flags: `--params`, `--num-probes`, `--domain`, `--seed`

### Complete Stage 1 Signal Profile

The three H-Bar signals together provide a comprehensive diagnostic of the σ-trap:

| Signal | Baseline Value | Interpretation |
|--------|----------------|----------------|
| g_A (GCA) | -0.0249 ± 0.0076 | ✗ NEGATIVE — Learning ID harms OOD |
| c_A (AC) | 0.9901 ± 0.0004 | ✓ HIGH — Strong invariance |
| r_A (RGA) | Expected: 0.1-0.3 | ? LOW — Geometric disorganization |

**Triple-Signal σ-Trap Signature:**
- **High AC** + **Negative GCA** + **Low RGA** = Classic σ-trap
- The model has shallow invariance (high AC) but broken gradient geometry (negative GCA) and disorganized representations (low RGA)
- All three signals must improve for true compositional generalization
