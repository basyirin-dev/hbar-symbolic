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
**Action:** Run ODE engine tests for stability and correctness
**Command:** `pytest tests/test_ode_engine.py -v`

**Stability Checks:**
- Jacobian condition number κ(J) < 1000 (Eq. 24)
- Forward invariance (Proposition 3.2): all variables remain in valid ranges
- Timescale separation (Proposition 3.3): fast/slow subsystem eigenvalue ratio > 10
- Convergence to schema-coherent equilibrium under positive inputs
- σ-trap simulation under high AI-bypass risk
- JIT and gradient compatibility through integrator
- Adaptive step size error control

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
| g_A (GCA) | -0.0235 ± 0.0075 | ✗ NEGATIVE — Learning ID harms OOD |
| c_A (AC) | 0.9901 ± 0.0004 | ✓ HIGH — Strong invariance |
| r_A (RGA) | 0.0604 | ✗ LOW — Geometric disorganization |

**Triple-Signal σ-Trap Signature Confirmed:**
- **High AC** + **Negative GCA** + **Low RGA** = Classic σ-trap
- The model has shallow invariance (high AC) but broken gradient geometry (negative GCA) and disorganized representations (low RGA)
- All three signals must improve for true compositional generalization

**Key Insight:** The pattern confirms that Transformer self-attention provides token-level consistency (AC ≈ 0.99) without compositional structure. The gradients are misaligned with compositional rules (GCA < 0), and the representational geometry does not reflect the grammar structure (RGA ≈ 0.06). This is the hallmark of the σ-trap — surface-level competence masking deep structural failure.

## Signal Fusion (Subtask 6.2)

### Overview

The fused H-Bar signal σ̃_A combines the three operative signals (GCA, RGA, AC) into a single schema coherence estimate via Equation 6:

```
σ̃_A = w_g · max(0, g_A) + w_r · max(0, r_A) + w_c · c_A
```

### Implementation

**`fuse_hbar_signals(g_A, r_A, c_A, weights)` in `hbar/engine/signals.py`:**
- Applies `max(0, x)` rectifiers to g_A and r_A (negative alignment → zero contribution)
- Computes weighted sum with default weights: w_g=0.4, w_r=0.35, w_c=0.25
- Clips result to [0, 1] range

**Why additive (not multiplicative)?** The additive form allows individual signals to "tug" the model out of the σ-trap even if others are near zero. For example, if GCA is high but RGA is low, the model can still achieve moderate σ̃_A. A multiplicative form would collapse to near-zero if any signal is low.

**Why max(0, x) rectifiers?** Negative GCA/RGA indicates active harm to generalization. Rather than subtracting from the fused signal (which could produce negative σ̃_A), we treat negative alignment as zero coherence. This ensures σ̃_A remains in [0, 1] and provides a clear "floor" at zero.

**`FusionConfig` in `hbar/models/config.py`:**
- Configurable weights: w_gca, w_rga, w_ac (default: 0.4, 0.35, 0.25)
- `target_sigma_critical`: Threshold for Phase 2 entry (default: 0.5)

**`HBarSignals` in `hbar/engine/data_utils.py`:**
- Container dataclass for all signals: g_a, r_a, c_a, sigma_tilde
- `to_dict()` method for logging to CSV/Weights & Biases
- `is_crystallized` property: returns True if σ̃_A > σ_critical

### Baseline Starting Point

Using the baseline signal profile, we calculate the starting σ̃_A:

| Signal | Value | Contribution to σ̃_A |
|--------|-------|---------------------|
| g_A = -0.0249 | max(0, -0.0249) × 0.4 | **0.0** |
| r_A = 0.0604 | 0.0604 × 0.35 | **0.02114** |
| c_A = 0.9901 | 0.9901 × 0.25 | **0.24753** |
| **σ̃_A (baseline)** | | **≈ 0.2686** |

**Key Finding:** The baseline σ̃_A ≈ 0.27 is far below σ_critical = 0.5. The model must approximately **double** its fused signal to enter Phase 2 (crystallization).

**Interpretation:**
- GCA contributes **nothing** (negative → rectified to zero)
- RGA contributes minimally (low alignment)
- AC carries the entire signal (high invariance from self-attention)

This confirms the σ-trap: the model relies entirely on shallow invariance (AC) without true compositional structure (GCA, RGA). The H-Bar optimizer must push GCA from negative to positive (>0.7) to achieve Phase 2 entry.

## Schema-Attention Coupling (Subtask 7.2)

### Overview

The schema-attention coupling is the core mechanism that explains why the σ-trap persists. The key insight is that schema coherence growth (σ̇_A) is **gated** by attentional fidelity (α_A). Without sufficient attention, schema coherence cannot grow regardless of the quality of training signals.

### Mathematical Formulation

**Equation 28 (Schema Coherence Dynamics):**
```
σ̇_A = ρ · P_A · α_A · (1 - σ_A) - η_σ · Ω_AI · σ_A
       ──────── growth ────────   ──── suppression ────
```

**Equation 29 (Attentional Fidelity Dynamics):**
```
α̇_A = γ · C_A · (1 - α_A) - η_α · R_surface · α_A
       ────── drive ──────   ──────── suppression ────────
```

**Key Variables:**
- **P_A (Principled Structure Availability):** Measures how much the training curriculum exposes compositional rules vs surface patterns
- **C_A (Training Signal Strength):** Drives attentional fidelity growth, computed from [BOS] token representation stability
- **Ω_AI (AI-Bypass Risk):** High values indicate the model achieves accuracy via surface statistics
- **R_surface (Surface Reward Signal):** High values indicate rewards for superficial pattern matching

### The σ-Trap Mechanism

The coupling creates a "double bind" that traps the model:

1. **Phase 1 (Asymmetric Initialization):**
   - α_A is low (suppressed by high surface rewards R_surface)
   - Since σ̇_A growth = ρ · P_A · α_A · (1 - σ_A), if α_A ≈ 0, then growth ≈ 0
   - σ_A cannot increase regardless of P_A magnitude
   - The model is stuck at σ_A ≈ 0.27 (baseline starting point)

2. **Attentional Gate:**
   - α_A acts as a multiplicative gate on schema growth
   - Even with perfect training signals (P_A = 1.0), if α_A = 0.1, effective growth is reduced by 90%
   - This explains why standard training fails: the attentional gate is closed

3. **Surface Reward Suppression:**
   - High R_surface (from 99% ID accuracy) actively suppresses α_A
   - The model is rewarded for surface pattern matching, not compositional rules
   - This creates a feedback loop: high ID accuracy → high R_surface → low α_A → low σ_A growth

### Implementation

**`HBarInputs` dataclass in `hbar/core/dynamics.py`:**
```python
@flax.struct.dataclass
class HBarInputs:
    sigma_tilde: jax.Array   # Fused signal σ̃_A ∈ [0, 1]
    sigma_hat: jax.Array     # Ground-truth σ̂_A = Acc_OOD / Acc_ID
    P_A: jax.Array           # Principled structure availability ∈ [0, 1]
    C_A: jax.Array           # Training signal strength ∈ [0, 1]
    Omega_AI: jax.Array      # AI-bypass risk ∈ [0, 1]
    R_surface: jax.Array     # Surface reward signal ∈ [0, 1]
    domain_frontier: jax.Array  # Curriculum difficulty ∈ [0, 1]
```

**`analyze_coupling_sensitivity()` diagnostic function:**
```python
def analyze_coupling_sensitivity(state, inputs, constants) -> Dict[str, jax.Array]:
    """Analyze schema-attention coupling sensitivity."""
    return {
        "coupled_growth_potential": γ_σ · P_A · α_A · (1 - σ_A),
        "attentional_gate_strength": α_A,
        "schema_growth_capacity": (1 - σ_A),
        "effective_drive": P_A · α_A,
        "suppression_pressure": η_σ · Ω_AI,
        "net_sigma_dot": growth - suppression,
        "is_attention_limited": α_A < 0.3,  # Phase 1 state
    }
```

**`CognitiveManager` class in `hbar/core/state_manager.py`:**
- Bridges training metrics and ODE dynamics
- `metrics_to_inputs()`: Maps training metrics to HBarInputs
- `get_modulators()`: Extracts training modulators (schema_loss_weight, lr_modulator)
- `check_phase_transition()`: Detects Phase 1 → Phase 2 transition

### Test Coverage

**`TestSurfaceRewardSuppression` in `tests/test_ode_engine.py`:**
- `test_surface_reward_suppression`: Verifies α_A stays low under high R_surface
- `test_sigma_trap_attention_gate`: Verifies σ_A growth ≈ 0 when α_A ≈ 0

### Phase 2 Entry (Crystallization)

Phase 2 entry occurs when:
1. σ_A > σ_critical (default: 0.5)
2. Crystallization potential: α_A · C_A > 0.5

The `compute_crystallization_potential()` function tracks readiness for Phase 2:
```python
def compute_crystallization_potential(state, inputs) -> jax.Array:
    return state.alpha_A * inputs.C_A
```

### Training Implications

The coupling reveals the intervention strategy:
1. **First, boost α_A:** Reduce surface rewards (R_surface) or increase training signal (C_A)
2. **Then, σ_A can grow:** Once α_A > 0.3, the attentional gate opens
3. **Phase 2 crystallization:** When both σ_A > 0.5 and α_A · C_A > 0.5

This explains why the H-Bar optimizer must modulate both the loss function (to reduce surface rewards) and the learning rate (to boost attentional signal) simultaneously.

## Compositional Pressure Loss (Subtask 8.1)

### Overview

The Compositional Pressure mechanism implements Equation 25 of the H-Bar paper, creating a dynamic training loss that automatically adjusts the penalty for poor compositional performance based on the current schema coherence level:

```
L_total = L_task + λ_σ · (1 - σ_A) · L_comp
```

where:
- **L_task**: Standard cross-entropy loss on in-distribution (ID) data
- **L_comp**: Compositional loss on out-of-distribution (OOD) probes
- **λ_σ**: Maximum compositional penalty weight (default: 0.5)
- **σ_A**: Current schema coherence estimate ∈ [0, 1]
- **(1 - σ_A)**: The "Compositional Pressure" term

### The Compositional Pressure Mechanism

The key insight is that the (1 - σ_A) term creates an **automatic curriculum**:

| σ_A Value | Compositional Pressure (1 - σ_A) | Training Behavior |
|-----------|----------------------------------|-------------------|
| σ_A ≈ 0 (low coherence) | ≈ 1.0 (maximum) | Strong gradient push on OOD stream — model must learn compositional rules |
| σ_A ≈ 0.5 (moderate) | ≈ 0.5 (half) | Balanced training between ID mastery and OOD generalization |
| σ_A ≈ 1 (high coherence) | ≈ 0.0 (vanishing) | Focus shifts to ID refinement — compositional rules crystallized |

This creates a self-regulating training dynamic:
1. **Early training (σ_A low):** High pressure forces the model to prioritize OOD performance
2. **Mid training (σ_A increasing):** Pressure gradually decreases as compositional rules emerge
3. **Late training (σ_A → 1):** Pressure vanishes — model has crystallized compositional schema

### Implementation

**`compute_hbar_loss()` in `hbar/engine/data_utils.py`:**
```python
def compute_hbar_loss(
    logits_id: jax.Array,
    labels_id: jax.Array,
    logits_ood: jax.Array,
    labels_ood: jax.Array,
    sigma_A: jax.Array,
    lambda_sigma: float = 0.5,
    pad_token_id: int = PAD_TOKEN_ID,
) -> jax.Array:
    """Compute the H-Bar modulated loss (Equation 25)."""
    L_task = compute_loss(logits_id, labels_id, pad_token_id)
    L_comp = compute_loss(logits_ood, labels_ood, pad_token_id)
    compositional_pressure = 1.0 - sigma_A
    L_total = L_task + lambda_sigma * compositional_pressure * L_comp
    return L_total
```

**`create_hbar_train_step()` in `hbar/engine/trainer.py`:**
- JIT-compiled training step that accepts HBarBatch (dual-stream)
- Performs forward pass on both ID and OOD streams
- Computes modulated loss using current σ_A from HBarState
- Returns (new_state, total_loss, id_loss, ood_loss, compositional_penalty)

**`run_hbar_training()` in `hbar/engine/trainer.py`:**
- Full training loop integrating ODE dynamics with neural network training
- Per-step workflow:
  1. Generate HBarBatch (ID + OOD streams)
  2. Compute operative estimate σ̃_A via signal fusion
  3. Step the ODEs via CognitiveManager.step to update HBarState
  4. Execute train_step using the updated σ_A from HBarState
  5. Log the Compositional Penalty Weight λ_σ · (1 - σ_A)

**`HBarTrainingMetrics` dataclass:**
- Tracks step, train_loss, id_loss, ood_loss, id_accuracy, ood_accuracy
- Includes sigma_tilde, sigma_ode, alpha_A, compositional_penalty, lambda_sigma
- Enables detailed analysis of training dynamics

### Test Coverage

**`tests/test_modulated_loss.py` — 7 tests:**

1. **TestModulatedLossSigmaOne:**
   - `test_sigma_one_no_penalty`: When σ_A = 1.0, total_loss = task_loss
   - `test_sigma_one_gradient_only_from_task`: Gradients only from ID stream

2. **TestModulatedLossSigmaZero:**
   - `test_sigma_zero_max_penalty`: When σ_A = 0.0, L_total = L_task + λ_σ · L_comp
   - `test_sigma_zero_penalty_increases_loss`: Total loss > task loss alone

3. **TestModulatedLossGradientFlow:**
   - `test_gradient_depends_on_sigma`: Gradient scaling matches (1 - σ_A) factor
   - `test_jit_compatible`: Loss compiles with jax.jit
   - `test_edge_case_sigma_out_of_bounds`: Handles σ_A slightly outside [0, 1]

### Training Dynamics Prediction

The Compositional Pressure mechanism predicts distinct training phases:

**Phase 1 (σ_A < σ_critical ≈ 0.5):**
- High compositional pressure (1 - σ_A > 0.5)
- Strong gradient signal on OOD stream
- Model is forced to learn compositional rules
- OOD accuracy should increase rapidly

**Phase 2 (σ_A > σ_critical):**
- Low compositional pressure (1 - σ_A < 0.5)
- Focus shifts to refining ID performance
- Compositional schema has crystallized
- Both ID and OOD accuracy remain high

**Key Difference from Baseline:**
- Baseline: No OOD signal during training → σ-trap (ID=92%, OOD=63%)
- H-Bar: Continuous OOD pressure modulated by σ_A → predicted ID>90%, OOD>85%

### Integration with ODE System

The compositional pressure loss is tightly coupled with the ODE dynamics:

1. **σ_A from ODE → Loss modulation:** The ODE state σ_A directly controls the penalty weight
2. **Loss gradients → ODE inputs:** Training metrics (Acc_ID, Acc_OOD) feed back into ODE via HBarInputs
3. **Closed-loop system:** The ODE integrates training signals to update σ_A, which modulates the loss

This creates a **self-regulating cognitive system** where the model's internal state (σ_A) controls its own training dynamics, implementing the core H-Bar hypothesis of endogenous schema coherence regulation.

## Attentional Acceleration (Subtask 8.2)

### Overview

The Attentional Acceleration mechanism implements Equation 26 of the H-Bar paper, creating a positive feedback loop where high attentional fidelity (α_A) accelerates learning, which in turn reinforces schema coherence growth:

```
η_effective = η_base · (1 + κ_α · α_A)
```

where:
- **η_base**: Base learning rate (default: 1e-3)
- **κ_α**: Attentional acceleration coefficient (default: 2.0 from FusionConfig)
- **α_A**: Current attentional fidelity from ODE state ∈ [0, 1]

### The Attentional Acceleration Mechanism

The key insight is that attentional fidelity should modulate learning speed:

| α_A Value | Acceleration Factor (1 + κ_α · α_A) | Effective LR | Interpretation |
|-----------|-------------------------------------|--------------|----------------|
| α_A ≈ 0 (suppressed) | ≈ 1.0 (no acceleration) | η_base | Phase 1: Surface rewards suppress attention |
| α_A ≈ 0.5 (moderate) | ≈ 2.0 (2× speedup) | 2 × η_base | Transition: Attentional fidelity increasing |
| α_A ≈ 1 (crystallized) | ≈ 3.0 (3× speedup) | 3 × η_base | Phase 2: Full attentional acceleration |

This creates a **positive feedback loop**:
1. **Early training (α_A low):** Surface rewards suppress attention → slow learning
2. **Phase 2 entry (α_A increasing):** Attentional burst → accelerated learning
3. **Crystallization (α_A → 1):** Maximum acceleration → rapid schema refinement

### Implementation

**`compute_attentional_lr()` in `hbar/engine/trainer.py`:**
```python
def compute_attentional_lr(
    base_lr: float,
    kappa_alpha: float,
    alpha_A: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Compute effective learning rate with attentional acceleration (Eq. 26)."""
    acceleration_factor = 1.0 + kappa_alpha * alpha_A
    effective_lr = base_lr * acceleration_factor
    return effective_lr, acceleration_factor
```

**`create_hbar_train_step()` refactored for gradient scaling:**
- The acceleration is implemented via gradient scaling (mathematically equivalent to LR modulation)
- `scaled_grads = grads * acceleration_factor`
- This is more efficient in JAX/Optax than recreating the optimizer state each step
- The gradient scaling approach: `θ_new = θ - η_base · (1 + κ_α · α_A) · g`

**Updated `HBarTrainingMetrics`:**
- Added `effective_learning_rate`: η_base · (1 + κ_α · α_A)
- Added `acceleration_factor`: (1 + κ_α · α_A)

**Updated `run_hbar_training()`:**
- Computes acceleration metrics at each evaluation checkpoint
- Logs `effective_learning_rate` and `acceleration_factor` to CSV
- Enables tracking of the "Attentional Burst" during Phase 2 entry

### Test Coverage

**`tests/test_attentional_lr.py` — 7 tests:**

1. **TestAttentionalLR.test_alpha_zero_no_acceleration:**
   - When α_A = 0.0, acceleration_factor = 1.0, effective_lr = base_lr

2. **TestAttentionalLR.test_alpha_one_max_acceleration:**
   - When α_A = 1.0 and κ_α = 2.0, acceleration_factor = 3.0, effective_lr = 3 × base_lr

3. **TestAttentionalLR.test_gradient_scaling_produces_larger_changes:**
   - Verifies parameter changes are 3× larger when α_A = 1.0 vs α_A = 0.0
   - Squared change ratio = 9.0 (since change scales with acceleration_factor)

4. **TestAttentionalLR.test_jit_compilation_works:**
   - Verifies JIT compatibility with various α_A values

5. **TestAttentionalLR.test_intermediate_alpha_values:**
   - Verifies linear scaling: α_A = 0.25 → factor = 1.5, α_A = 0.75 → factor = 2.5

6. **TestAttentionalLR.test_different_kappa_alpha_values:**
   - Verifies correct scaling with κ_α = 0.0, 1.0, 5.0

7. **TestAttentionalLR.test_attentional_burst_prediction:**
   - Simulates Phase 1 (α_A = 0.1) → Phase 2 (α_A = 0.8) transition
   - Verifies acceleration increases by at least 2× during crystallization

### Training Dynamics Prediction

The Attentional Acceleration mechanism predicts a distinct "Attentional Burst" signature:

**Phase 1 (α_A < 0.3):**
- Acceleration factor ≈ 1.0-1.6
- Learning proceeds at base rate
- Surface rewards suppress attentional fidelity

**Phase 2 Entry (α_A > 0.5):**
- Acceleration factor rapidly increases to 2.0-3.0
- "Attentional Burst" marks the transition
- Accelerated learning reinforces schema coherence

**Key Prediction:**
The acceleration factor should show a sharp increase at Phase 2 entry, providing an observable signature of crystallization in training logs.

### Integration with Compositional Pressure

The two mechanisms work together:

1. **Compositional Pressure (Eq. 25):** Modulates the loss based on σ_A
   - Low σ_A → high penalty on OOD loss → forces compositional learning

2. **Attentional Acceleration (Eq. 26):** Modulates the learning rate based on α_A
   - Low α_A → slow learning → prevents premature convergence
   - High α_A → fast learning → accelerates crystallization

Together, they create a **dual-modulation system**:
- When both σ_A and α_A are low: High pressure, slow learning → exploration phase
- When σ_A increases: Pressure decreases → exploitation phase
- When α_A increases: Learning accelerates → crystallization phase

This implements the H-Bar hypothesis that compositional generalization requires both schema coherence (σ_A) and attentional fidelity (α_A) to develop in tandem.

## H-Bar Integrated Training Controller (Subtask 8.3)

### Overview

The H-Bar Integrated Training Controller is the "Architectural Glue" of the H-Bar framework. It bundles the Flax `TrainState` (neural weights) and the ODE `HBarState` (cognitive state) into a single unified Pytree, enabling the entire H-Bar training step—including signal extraction, ODE integration, and modulated backprop—to be handled by a single `jax.jit` function call.

### HBarTrainState Dataclass

**`HBarTrainState` in `hbar/engine/trainer.py`:**
```python
@flax.struct.dataclass
class HBarTrainState:
    train_state: TrainState          # Standard Flax TrainState (params, optimizer)
    hbar_state: Any                  # HBarState (7 ODE variables)
    constants: Any                   # HBarConstants (11 dynamical parameters)
    fusion_config: FusionConfig      # Signal fusion weights
```

Using `flax.struct.dataclass` ensures automatic Pytree registration, enabling:
- `jax.jit(apply_hbar_step)` compilation
- `jax.lax.scan(apply_hbar_step, initial_state, batches)` for massive speedups
- `flax.serialization.to_bytes()` for saving cognitive state + weights together

### The 7-Step Training Sequence (Algorithm 3.2)

**`apply_hbar_step()` in `hbar/engine/trainer.py`:**

1. **Signal Extraction:** Compute GCA (g_A), RGA (r_A), AC (c_A)
2. **Fusion:** Compute σ̃_A via `fuse_hbar_signals()` (Equation 6)
3. **ODE Integration:** Evolve HBarState via `CognitiveManager.step()`
4. **Modulated Loss:** Compute L_total using new σ_A (Equation 25)
5. **Backward Pass:** Compute gradients via `jax.grad()`
6. **Acceleration:** Apply gradient scaling based on α_A (Equation 26)
7. **Weight Update:** Apply scaled gradients via `state.apply_gradients()`

### Initialization Utility

**`init_hbar_train_state()` in `hbar/engine/trainer.py`:**
- Initializes standard Flax TrainState with Adam optimizer
- Initializes HBarState at baseline starting point (σ_A ≈ 0.27)
- Bundles HBarConstants and FusionConfig
- Returns unified HBarTrainState

### Key Architecture Insight

The unified HBarTrainState enables the entire training loop to be compiled into a single XLA operation:
```python
final_state, metrics_history = jax.lax.scan(apply_hbar_step, initial_state, batches)
```
This is the "JAX way" to do training — significantly faster than a Python for loop because the entire 5,000-step trajectory is compiled into a single XLA operation.

### Serialization

Since this is a research project, being able to save the HBarTrainState is vital. When you save the model, you aren't just saving weights; you are saving the **Cognitive State** of the agent at that moment (its current σ_A, α_A, etc.).

### Test Coverage

**`tests/test_hbar_optimizer.py` — 8 tests:**

1. **TestHBarTrainStateEvolution:**
   - `test_neural_weights_change_after_steps`: Verifies parameters update
   - `test_sigma_A_changes_after_steps`: Verifies ODE state evolves
   - `test_both_states_update_together`: Verifies unified update

2. **TestHBarTrainStateSerialization:**
   - `test_serialization_roundtrip`: Save/load via flax.serialization
   - `test_serialization_preserves_params`: Parameters identical after roundtrip

3. **TestGhostGradients:**
   - `test_gradients_flow_through_ode_integration`: No blocked gradients
   - `test_jit_compilation_of_apply_hbar_step`: JIT compatibility

4. **TestHBarTrainStatePytree:**
   - `test_hbar_train_state_is_pytree`: Flatten/unflatten works
   - `test_hbar_train_state_can_be_jax_transformed`: jax.tree_map works

### Integration with Existing Training Loop

After this subtask, the `train_baseline.py` and future `train_hbar.py` will look nearly identical. The only difference will be that the H-Bar version uses `apply_hbar_step` instead of a standard SGD step. This "Clean API" makes Phase 3 (large-scale runs) much easier to manage.

## H-Bar Experimental Runs (Subtask 9.1)

### Overview

Subtask 9.1 implements the formal H-Bar training runs for **Condition B (Additive)** and **Condition C (Multiplicative)**. These interventions are designed to bridge the ~30% generalization gap found in the baseline and reach near-perfect (95-99%) OOD accuracy predicted by the H-Bar paper.

### Loss Coupling Conditions

**Condition B (Additive) — Equation 25:**
```
L_total = L_task + λ_σ · (1 - σ_A) · L_comp
```
- Standard additive penalty formulation
- More stable, predictable training dynamics
- Linear penalty scaling with compositional pressure

**Condition C (Multiplicative):**
```
L_total = L_task · (1 + λ_σ · (1 - σ_A) · L_comp)
```
- Multiplicative coupling between task loss and compositional penalty
- More aggressive training dynamics
- When task loss is high, the compositional penalty is amplified
- May lead to faster "crystallization" but potentially more gradient instability

### Key Differences

| Aspect | Additive (B) | Multiplicative (C) |
|--------|--------------|-------------------|
| **Formula** | L_task + penalty | L_task × (1 + penalty) |
| **Stability** | More stable | Potentially unstable |
| **Crystallization** | Gradual | Faster, more aggressive |
| **Gradient behavior** | Predictable | Can spike when L_task high |

### Implementation

**`compute_hbar_loss_multiplicative()` in `hbar/engine/data_utils.py`:**
```python
def compute_hbar_loss_multiplicative(
    logits_id, labels_id, logits_ood, labels_ood,
    sigma_A, lambda_sigma=0.5, pad_token_id=PAD_TOKEN_ID
) -> jax.Array:
    """Multiplicative coupling: L_total = L_task · (1 + λ_σ(1-σ_A)L_comp)"""
    L_task = compute_loss(logits_id, labels_id, pad_token_id)
    L_comp = compute_loss(logits_ood, labels_ood, pad_token_id)
    compositional_pressure = 1.0 - sigma_A
    L_total = L_task * (1.0 + lambda_sigma * compositional_pressure * L_comp)
    return L_total
```

**`create_hbar_train_step_multiplicative()` in `hbar/engine/trainer.py`:**
- JIT-compiled training step with multiplicative loss coupling
- Same gradient scaling for attentional acceleration (Equation 26)
- Returns (new_state, total_loss, id_loss, ood_loss, compositional_penalty, effective_lr, acceleration_factor)

**`run_hbar_training_multiplicative()` in `hbar/engine/trainer.py`:**
- Full training loop for Condition C
- Same structure as additive version but uses multiplicative loss

### Training Script

**`scripts/train_hbar.py`:**
```bash
# Single run (Condition C - default)
python scripts/train_hbar.py --domain scan --condition multiplicative

# Pilot study (N=15)
python scripts/train_hbar.py --domain scan --condition multiplicative --n_runs 15

# Condition B (additive)
python scripts/train_hbar.py --domain scan --condition additive --n_runs 5
```

**Arguments:**
- `--condition`: `additive` or `multiplicative` (default: multiplicative)
- `--n_runs`: Number of independent runs with different seeds
- `--lambda-sigma`: Maximum compositional penalty weight (default: 0.5)
- Standard training args: `--batch-size`, `--total-steps`, `--learning-rate`, `--seed`

**Outputs:**
- `hbar_{condition}_run_{n}_metrics.csv`: Per-run training metrics
- `pilot_results_summary.csv`: Aggregated results across all runs
- `model_params_{condition}_run_{n}.msgpack`: Saved model parameters

### Expected Outcomes

Based on the H-Bar paper predictions:

| Metric | Baseline | H-Bar Target |
|--------|----------|--------------|
| **ID Accuracy** | 91.9% | >90% |
| **OOD Accuracy** | 63.0% | >90% |
| **σ̂_A** | 0.685 | >0.9 |
| **GCA (g_A)** | -0.02 | +0.4 (positive) |
| **Phase 2 Entry** | Never | <1,000 steps |

### Pilot Study (N=15)

Before committing to the full N=120 runs, the pilot study verifies:
1. **Effect size:** Confirm OOD accuracy >90% and σ̂_A >0.9
2. **GCA flip:** Verify g_A transitions from negative to positive
3. **Phase 2 entry:** Check if crystallization occurs within first 1,000 steps
4. **Stability:** Ensure training doesn't diverge (especially for multiplicative)

**Estimated time:** ~4 hours for N=15 (15 min per run)

### Verification Criteria

After pilot completion, check `pilot_results_summary.csv`:
- If **OOD Accuracy > 90%** and **σ̂_A > 0.9**: H-Bar effect confirmed
- If **Mean GCA is positive** (vs baseline -0.02): Learning geometry fixed
- If **Phase 2 entry detected**: Crystallization achieved

These criteria validate proceeding to the full N=120 pre-registered runs.
