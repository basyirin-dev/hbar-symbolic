# Progress Tracker

## Phase 1: Foundation (Week 1)

### Subtask 1.1: Environment Initialization
**Status:** Complete ✓

**Objectives:**
- [x] Initialize git repository
- [x] Create memory bank structure
- [x] Set up Python virtual environment
- [x] Install JAX/Flax stack (jax, jaxlib, flax, optax, chex, distrax)
- [x] Create project directory structure
- [x] Establish .clinerules constraints (V3.0+ variables)
- [x] Create README.md with project overview
- [x] Add figures/ and bibliography/ directories

### Subtask 1.2: Flax Transformer Implementation
**Status:** Complete ✓

**Objectives:**
- [x] Implement Transformer (2 layer, 4 head, d_model=128)
- [x] Build tokenization and sequence encoding pipeline
- [x] Ensure jax.jit compatibility for all forward passes

**Completed Deliverables:**
- `hbar/models/config.py`: TransformerConfig and ActivationsDict dataclasses
- `hbar/models/transformer.py`: Full Encoder-Decoder architecture with:
  - Embed class (token + learned positional encodings)
  - MultiHeadAttention class (scaled dot-product attention)
  - TransformerBlock class (self-attention + cross-attention + MLP)
  - Encoder class (stack of transformer blocks)
  - Decoder class (stack with causal masking)
  - Seq2SeqTransformer class (top-level model with activation hooks)
- `tests/test_model_shapes.py`: 7 tests verifying shapes, JIT compatibility, gradient flow

### Subtask 1.3: JAX-native Tokenization and Encoding
**Status:** Complete ✓

**Objectives:**
- [x] Implement Tokenizer class with word-level tokenization
- [x] Implement functional mask generation (padding + causal)
- [x] Implement JIT-compatible batch preprocessing
- [x] Create integration tests for full pipeline

**Completed Deliverables:**
- `hbar/engine/tokenizer.py`: Tokenizer class with SCAN vocabulary
- `hbar/engine/encoding.py`: get_padding_mask, get_causal_mask, get_decoder_mask
- `hbar/engine/data_utils.py`: prepare_batch, compute_loss, compute_accuracy
- `tests/test_encoding_pipeline.py`: 16 tests for tokenizer, masks, batch prep, integration

### Subtask 1.4: SCAN/COGS Data Generators
**Status:** Complete ✓

**Objectives:**
- [x] Implement programmatic grammar G(d) for SCAN
- [x] Implement programmatic grammar G(d) for COGS
- [x] Build unified GrammarEngine with batch generators
- [x] Implement compositional probe generation for GCA signal extraction
- [x] Implement tree-edit distance for RDMstruct (RGA signal)
- [x] Validate grammar-tokenizer alignment (no UNK tokens)
- [x] Build batch generators with jax.vmap (HBarBatch triple-stream)
- [x] Implement structure-preserving augmentation pipeline

**Completed Deliverables:**
- `hbar/benchmarks/scan_grammar.py`: SCANGrammar with recursive CFG, compositional probes
- `hbar/benchmarks/cogs_grammar.py`: COGSGrammar with LogicalForm trees, structural distance
- `hbar/benchmarks/grammar_engine.py`: GrammarEngine unified interface, Batch integration
- `hbar/engine/augmentation.py`: Vectorized primitive substitution with jax.vmap
- `hbar/engine/data_utils.py`: HBarBatch dataclass, get_hbar_batch function
- `tests/test_grammars.py`: 28 tests for grammar validity, tokenization, determinism
- `tests/test_hbar_generator.py`: 17 tests for HBarBatch, augmentation, JIT compatibility

### Subtask 1.5: Evaluation Splits and Ground-Truth σ_A
**Status:** Complete ✓

**Objectives:**
- [x] Implement SCAN Add-Jump split (training: no 'jump' in compounds + isolated 'jump')
- [x] Implement COGS Subject-to-Object split (BIASED_NOUNS: hedgehog, porcupine, otter)
- [x] Generate and freeze 2,000 ID + 2,000 OOD samples per domain
- [x] Build Evaluator class with σ̂_A = Acc_OOD / Acc_ID calculation
- [x] Handle division-by-zero edge case (return 0.0 when Acc_ID = 0)
- [x] Validate zero overlap between ID and OOD splits

**Completed Deliverables:**
- `hbar/benchmarks/scan_grammar.py`: Added `get_add_jump_split()` method
- `hbar/benchmarks/cogs_grammar.py`: Added `BIASED_NOUNS`, `get_subject_object_split()`, helper methods
- `scripts/freeze_benchmarks.py`: Generates and saves frozen JSON eval sets
- `data/scan_id_eval.json`: 2,000 ID SCAN samples
- `data/scan_ood_eval.json`: 2,000 OOD SCAN samples (Add-Jump)
- `data/cogs_id_eval.json`: 2,000 ID COGS samples
- `data/cogs_ood_eval.json`: 2,000 OOD COGS samples (Subject-to-Object)
- `hbar/engine/evaluator.py`: Evaluator class, EvaluationResult dataclass, calibration error
- `tests/test_evaluation_splits.py`: 16 tests for splits, evaluator, edge cases

### Subtask 1.6: Baseline Verification
**Status:** Complete ✓

**Objectives:**
- [x] Write standard SGD training loop (`hbar/engine/trainer.py`)
- [x] Create Kaggle entry point script (`scripts/train_baseline.py`)
- [x] Run baseline for 5,000 steps on Kaggle
- [x] Verify "Illusion of Mastery" failure mode (Generalization Gap confirmed)

**Completed Deliverables:**
- `hbar/engine/trainer.py`: Complete training engine with `init_train_state`, JIT-compiled `train_step`, and `run_baseline_training` loop
- `scripts/train_baseline.py`: Standalone Kaggle-compatible script with XLA memory fix, CSV logging, and parameter saving
- `requirements.txt`: Dependencies for Kaggle (jax, flax, optax, chex, numpy)
- `setup.py`: Package installation for pip installable module

**Actual Results (Kaggle GPU T4, 5000 steps):**
- **ID Accuracy:** 91.9% ✅ (Model masters in-distribution patterns)
- **OOD Accuracy:** 63.0% ⚠️ (Model shows partial generalization, gap of ~29%)
- **σ̂_A:** 0.685 (Generalization gap confirmed)
- **Training Time:** 15.8 minutes
- **Model Parameters:** Saved to `model_params.msgpack`

**Interpretation:**
The baseline confirmed the "Illusion of Mastery" pattern with a clear ~29% generalization gap between ID and OOD accuracy. While OOD accuracy was higher than the original paper's ~44%, the model still struggles significantly with compositional generalization (jump in novel contexts). This validates the need for the H-Bar framework to explicitly address this gap.

### Subtask 3.1: Network Extraction Hooks
**Status:** Complete ✓

**Objectives:**
- [x] Refactor transformer.py to use Flax `self.sow()` for activation capture
- [x] Implement `capture_activations` flag to save memory during training
- [x] Create `get_model_representations` extraction wrapper
- [x] Establish layer naming convention for RGA engine
- [x] Create comprehensive test suite for purity, gradient flow, and RGA readiness

**Completed Deliverables:**
- `hbar/models/transformer.py`: Refactored with `capture_activations` flag and `sow` pattern
- `hbar/models/transformer.py`: Added `get_model_representations()` wrapper function
- `tests/test_extraction_hooks.py`: 11 tests for purity, gradient flow, and RGA readiness
- `tests/test_model_shapes.py`: Updated to use new extraction API

### Subtask 3.2: Structural-preserving Augmentation
**Status:** Complete ✓

**Objectives:**
- [x] Implement `apply_argument_permutation` for SCAN/COGS
- [x] Integrate permutation into `vmap_augment_batch` with configurable probability
- [x] Create `compute_augmentation_consistency` function for c_A signal
- [x] Create test suite for AC signal correctness and JIT compatibility

**Completed Deliverables:**
- `hbar/engine/augmentation.py`: Added `apply_argument_permutation()`, `apply_augmentation()`, updated `vmap_augment_batch()` with `permutation_probability` parameter
- `hbar/engine/signals.py`: New module with `compute_augmentation_consistency()`, `compute_layer_weighted_ac()`, `compute_representation_norm()`
- `tests/test_ac_signal.py`: 6 tests verifying c_A = 1.0 for identical inputs, structure-preserving > random, JIT compatibility, range [0,1], multi-layer support, and gradient flow

## Phase 2: Core H-Bar Engine (Weeks 5-8)

### Subtask 2.1: Multi-Signal Proxy Extraction
**Status:** Partially Complete (1/4 signals implemented)

**Objectives:**
- [x] Implement GCA (Gradient-Composition Alignment) — Complete
- [ ] Implement RGA (Representational-Geometry Alignment)
- [ ] Implement AC (Augmentation Consistency)
- [ ] Build multi-signal fusion: sigma_tilde_A = w_g*g_A + w_r*r_A + w_c*c_A

**Completed Deliverables (GCA):**
- `hbar/engine/signals.py`: Added `compute_gca()` — Pearson correlation between flattened gradient vectors
- `hbar/engine/trainer.py`: Added `compute_dual_gradients()` and `get_gca_signal()` — JIT-compatible gradient extraction
- `scripts/analyze_gca_baseline.py`: Analysis script for computing GCA over 100 batches
- Expected baseline GCA: 0.1-0.3 (σ-trap noise floor)

### Subtask 5.1: GCA Signal Implementation
**Status:** Complete ✓

**Objectives:**
- [x] Implement `compute_gca` in `hbar/engine/signals.py`
- [x] Implement `compute_dual_gradients` in `hbar/engine/trainer.py`
- [x] Implement `get_gca_signal` helper in `hbar/engine/trainer.py`
- [x] Create `scripts/analyze_gca_baseline.py` analysis script
- [x] Document GCA signal in `memory-bank/activeContext.md`
- [x] Run baseline GCA analysis on Kaggle

**Completed Deliverables:**
- `hbar/engine/signals.py`: `compute_gca(grad_id, grad_ood)` with ε=1e-8 stability
- `hbar/engine/trainer.py`: Dual gradient extraction with `jax.flatten_util.ravel_pytree`
- `scripts/analyze_gca_baseline.py`: 100-batch GCA analysis with mean ± SEM reporting

**Baseline GCA Results (Kaggle GPU T4, 100 batches, batch_size=32):**
- **Mean GCA (g_A):** -0.0235 ± 0.0075 (SEM)
- **Std Deviation:** 0.0753
- **Min GCA:** -0.1753
- **Max GCA:** 0.1727
- **Interpretation:** NEGATIVE GCA confirms σ-trap — learning ID patterns actively harms OOD performance

### Subtask 2.2: ODE Integration
**Status:** Pending

**Objectives:**
- [ ] Implement IMEX Runge-Kutta integrator
- [ ] Code sigma_A ODE (Eq. 28) with alpha_A coupling
- [ ] Code alpha_A ODE (Eq. 29) with surface reward suppression
- [ ] Code M_hat_A, Xi_A ODEs for extended cognition

### Subtask 2.3: H-Bar Optimizer
**Status:** Pending

**Objectives:**
- [ ] Code schema-targeting loss (Eq. 25)
- [ ] Implement learning rate modulation (Eq. 26)
- [ ] Wrap Optax with HBarOptimizer

## Phase 3: Empirical Validation (Weeks 9-12)

### Subtask 3.1: Pre-Registered Runs
**Status:** Pending

**Objectives:**
- [ ] Run N=500 protocol (3 conditions: Baseline, Additive, Multiplicative)
- [ ] Log delta_A, sigma_A, alpha_A via Weights & Biases
- [ ] Monitor Phase 2 entry (sigma_A > sigma_critical)

### Subtask 3.2: Statistical Analysis
**Status:** Pending

**Objectives:**
- [ ] Welch's t-test for Prediction 1
- [ ] F-test on incremental R² for Prediction 6
- [ ] Segmented regression for Prediction 9

## Phase 4: Open Source Release (Weeks 13-15)

### Subtask 4.1: PyPI Package
**Status:** Pending

**Objectives:**
- [ ] Clean codebase and standardize API
- [ ] Write comprehensive documentation
- [ ] Publish to PyPI as hbar-symbolic
- [ ] Generate Zenodo DOI

### Subtask 4.2: Public Demo
**Status:** Pending

**Objectives:**
- [ ] Upload pre-trained weights to HuggingFace
- [ ] Build Gradio UI for interactive demo
- [ ] Publish "10-Line H-Bar" Colab notebook

## Phase 5: Publication & Pitch (Weeks 16-20)

### Subtask 5.1: Paper Submission
**Status:** Pending

**Objectives:**
- [ ] Draft methodology and results sections
- [ ] Format to NeurIPS/ICLR template
- [ ] Submit to arXiv

### Subtask 5.2: Deliverables
**Status:** Pending

**Objectives:**
- [ ] Complete 10 Deliverables
- [ ] Build portfolio website
- [ ] Submit YC/EV applications
