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
**Status:** Complete ✓

**Objectives:**
- [x] Implement GCA (Gradient-Composition Alignment) — Complete
- [x] Implement RGA (Representational-Geometry Alignment) — Complete
- [x] Implement AC (Augmentation Consistency) — Complete
- [x] Build multi-signal fusion: sigma_tilde_A = w_g*g_A + w_r*r_A + w_c*c_A

**Completed Deliverables:**
- `hbar/engine/signals.py`: Added `fuse_hbar_signals()` implementing Equation 6 with rectifier logic
- `hbar/models/config.py`: Added `FusionConfig` dataclass with configurable weights and σ_critical threshold
- `hbar/engine/data_utils.py`: Added `HBarSignals` dataclass with `to_dict()` and `is_crystallized` diagnostic
- Baseline σ̃_A ≈ 0.2686 (far below σ_critical ≈ 0.5, confirming σ-trap)

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
**Status:** Complete ✓

**Objectives:**
- [x] Implement IMEX Runge-Kutta integrator
- [x] Code sigma_A ODE (Eq. 28) with alpha_A coupling
- [x] Code alpha_A ODE (Eq. 29) with surface reward suppression
- [x] Code M_hat_A, Xi_A ODEs for extended cognition
- [x] Implement adaptive step size control
- [x] Implement boundary enforcement (forward invariance)
- [x] Create comprehensive test suite

**Completed Deliverables:**
- `hbar/core/dynamics.py`: HBarState (7 variables), HBarInputs, HBarConstants, hbar_vector_field (Eqs. 14, 28, 29, 33, 36)
- `hbar/core/integrator.py`: step_hbar_system (IMEX RK4), adaptive_step_hbar_system, enforce_boundaries, check_jacobian_condition
- `hbar/core/__init__.py`: Package exports for all core functionality
- `hbar/core/state_manager.py`: CognitiveManager class bridging training metrics and ODE dynamics
- `tests/test_ode_engine.py`: 13 tests including surface reward suppression and attentional gate tests

**ODE System Details:**
- Fast subsystem (RK4): δ_A (Gompertz growth), σ_A (gated growth − AI bypass), α_A (signal-driven − surface suppression)
- Slow subsystem (Backward Euler): M̂_A (mean-reverting calibration), Ξ_A (target-seeking, 3 components)
- Default timestep: h = 0.01
- State space: δ_A ∈ [0, 10], σ_A ∈ [0, 1], α_A ∈ [0, 1], M̂_A ∈ [0, 1], Ξ_A ∈ [0, 1]

### Subtask 7.2: Schema-Attention Coupling
**Status:** Complete ✓

**Objectives:**
- [x] Implement α_A as attentional gate in Eq. 28 (σ̇_A growth = ρ · P_A · α_A · (1 - σ_A))
- [x] Implement C_A as training signal driver in Eq. 29 (α̇_A drive = γ · C_A · (1 - α_A))
- [x] Add P_A and C_A fields to HBarInputs dataclass
- [x] Implement analyze_coupling_sensitivity diagnostic function
- [x] Implement compute_crystallization_potential function
- [x] Create CognitiveManager class for training loop integration
- [x] Add surface reward suppression tests to test suite

**Completed Deliverables:**
- `hbar/core/dynamics.py`: Updated hbar_vector_field with coupled equations
- `hbar/core/state_manager.py`: CognitiveManager with metrics_to_inputs, get_modulators, check_phase_transition
- `tests/test_ode_engine.py`: TestSurfaceRewardSuppression class with 2 tests

**Key Insight:**
The coupling σ̇_A growth = ρ · P_A · α_A · (1 - σ_A) explains the σ-trap mechanism:
when α_A is low (suppressed by surface rewards), σ_A cannot grow regardless of P_A magnitude.
This creates a "double bind" where the model is trapped in low schema coherence until
attentional fidelity is boosted through targeted intervention.

### Subtask 8.1: Compositional Pressure Loss (Eq. 25)
**Status:** Complete ✓

**Objectives:**
- [x] Implement `compute_hbar_loss` in `hbar/engine/data_utils.py`
- [x] Implement `create_hbar_train_step` in `hbar/engine/trainer.py`
- [x] Implement `run_hbar_training` loop with ODE integration
- [x] Create comprehensive test suite for modulated loss

**Completed Deliverables:**
- `hbar/engine/data_utils.py`: `compute_hbar_loss()` implementing L_total = L_task + λ_σ · (1 - σ_A) · L_comp
- `hbar/engine/trainer.py`: `create_hbar_train_step()` — JIT-compiled training step with dual-stream forward pass
- `hbar/engine/trainer.py`: `run_hbar_training()` — Full training loop integrating ODE dynamics with neural network training
- `hbar/engine/trainer.py`: `HBarTrainingMetrics` and `HBarTrainingResults` dataclasses for logging
- `tests/test_modulated_loss.py`: 7 tests verifying σ_A=1.0 (no penalty), σ_A=0.0 (max penalty), gradient flow, JIT compatibility

**Compositional Pressure Mechanism:**
The (1 - σ_A) term creates dynamic training pressure:
- When σ_A → 0 (low coherence): penalty weight → 1.0, strong gradient push on OOD stream
- When σ_A → 1 (high coherence): penalty weight → 0, focus shifts to ID refinement
- This creates an automatic curriculum that prioritizes compositional learning when needed

### Subtask 8.2: Attentional Acceleration (Eq. 26)
**Status:** Complete ✓

**Objectives:**
- [x] Add `kappa_alpha` parameter to `FusionConfig`
- [x] Implement `compute_attentional_lr` function
- [x] Refactor `create_hbar_train_step` for gradient scaling
- [x] Update `HBarTrainingMetrics` with `effective_learning_rate` and `acceleration_factor`
- [x] Update `run_hbar_training` to log acceleration metrics
- [x] Create comprehensive test suite for attentional LR

**Completed Deliverables:**
- `hbar/models/config.py`: Added `kappa_alpha: float = 2.0` to `FusionConfig`
- `hbar/engine/trainer.py`: `compute_attentional_lr(base_lr, kappa_alpha, alpha_A)` implementing η_effective = η_base · (1 + κ_α · α_A)
- `hbar/engine/trainer.py`: Refactored `create_hbar_train_step()` with gradient scaling (Equation 26)
- `hbar/engine/trainer.py`: Updated `HBarTrainingMetrics` with `effective_learning_rate` and `acceleration_factor`
- `hbar/engine/trainer.py`: Updated `run_hbar_training()` to compute and log acceleration metrics
- `tests/test_attentional_lr.py`: 7 tests verifying:
  - α_A=0.0 → no acceleration (factor=1.0)
  - α_A=1.0, κ_α=2.0 → 3× acceleration
  - Gradient scaling produces 3× larger parameter changes (9× squared change ratio)
  - JIT compilation compatibility
  - Linear scaling for intermediate α_A values
  - Correct behavior with different κ_α values
  - Attentional Burst prediction (Phase 1→2 transition)

**Attentional Acceleration Mechanism:**
The gradient scaling approach is mathematically equivalent to learning rate modulation:
```
θ_new = θ - η_base · (1 + κ_α · α_A) · g
```
This is more efficient in JAX/Optax than recreating the optimizer state each step.

**Key Predictions:**
- Phase 1: α_A low (suppressed by surface rewards) → acceleration ≈ 1.0
- Phase 2 entry: α_A increases rapidly ("Attentional Burst") → acceleration spikes to 2-3×
- This creates a positive feedback loop: high attention → faster learning → reinforced schema coherence

### Subtask 2.3: H-Bar Optimizer (Integrated Training Controller)
**Status:** Complete ✓

**Objectives:**
- [x] Code schema-targeting loss (Eq. 25)
- [x] Implement learning rate modulation (Eq. 26)
- [x] Wrap Optax with HBarOptimizer
- [x] Create unified HBarTrainState bundling TrainState + HBarState
- [x] Implement apply_hbar_step coordinating the 7-step sequence
- [x] Create comprehensive test suite for integrated controller

**Completed Deliverables:**
- `hbar/engine/trainer.py`: `HBarTrainState` dataclass bundling TrainState, HBarState, HBarConstants, FusionConfig
- `hbar/engine/trainer.py`: `init_hbar_train_state()` utility for unified state initialization
- `hbar/engine/trainer.py`: `apply_hbar_step()` implementing the full 7-step Algorithm 3.2:
  1. Signal Extraction (GCA, RGA, AC)
  2. Fusion (σ̃_A via Equation 6)
  3. ODE Integration (CognitiveManager.step)
  4. Modulated Loss (Equation 25)
  5. Backward Pass (automatic differentiation)
  6. Acceleration (gradient scaling via Equation 26)
  7. Weight Update (apply_gradients)
- `tests/test_hbar_optimizer.py`: 8 tests verifying:
  - Neural weights change after training steps
  - ODE state (sigma_A) evolves during training
  - Both TrainState and HBarState update together
  - Serialization roundtrip preserves all state
  - Parameters preserved through serialization
  - Gradients flow through ODE integration (no ghost gradients)
  - JIT compilation compatibility
  - Pytree compatibility (flatten/unflatten)

**Key Architecture Insight:**
The unified HBarTrainState enables the entire training loop to be compiled into a single XLA operation via `jax.lax.scan`:
```python
final_state, metrics_history = jax.lax.scan(apply_hbar_step, initial_state, batches)
```
This is the "JAX way" to do training — significantly faster than a Python for loop.

## Phase 3: Empirical Validation (Weeks 9-12)

### Subtask 9.1: H-Bar Experimental Runs (Conditions B & C)
**Status:** Complete ✓

**Objectives:**
- [x] Implement multiplicative loss function `compute_hbar_loss_multiplicative()`
- [x] Implement `create_hbar_train_step_multiplicative()` training step
- [x] Implement `run_hbar_training_multiplicative()` training loop
- [x] Create `scripts/train_hbar.py` with `--condition` and `--n_runs` arguments
- [x] Support both Condition B (Additive) and Condition C (Multiplicative)
- [x] Implement pilot summary aggregation to `pilot_results_summary.csv`

**Completed Deliverables:**
- `hbar/engine/data_utils.py`: Added `compute_hbar_loss_multiplicative()` implementing L_total = L_task · (1 + λ_σ(1-σ_A)L_comp)
- `hbar/engine/trainer.py`: Added multiplicative training step and loop functions
- `scripts/train_hbar.py`: Full training script supporting N independent runs with different seeds

**Usage:**
```bash
# Single run (Condition C - multiplicative)
python scripts/train_hbar.py --domain scan --condition multiplicative

# Pilot study (N=15)
python scripts/train_hbar.py --domain scan --condition multiplicative --n_runs 15

# Condition B (additive)
python scripts/train_hbar.py --domain scan --condition additive --n_runs 5
```

**Expected Outcomes:**
- OOD Accuracy should jump from ~63% (baseline) to >90%
- σ̂_A should increase from ~0.68 to >0.9
- GCA (g_A) should flip from negative (-0.02) to positive
- Phase 2 entry should occur within first 1,000 steps

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

### Subtask 5.2: AC Signal Implementation
**Status:** Complete ✓

**Objectives:**
- [x] Implement `compute_ac_from_batch()` in `hbar/engine/signals.py`
- [x] Implement `get_ac_signal()` in `hbar/engine/trainer.py`
- [x] Create `scripts/analyze_ac_baseline.py` (combined GCA + AC analysis)
- [x] Document baseline AC profile in memory bank

**Completed Deliverables:**
- `hbar/engine/signals.py`: `compute_ac_from_batch()` wrapper for AC computation
- `hbar/engine/trainer.py`: `get_ac_signal()` — extracts encoder representations from id_stream and aug_stream
- `scripts/analyze_ac_baseline.py`: Combined analysis script with GCA + AC correlation

**Baseline AC Results (Kaggle GPU T4, 100 batches, batch_size=32):**

| Metric | Value |
|--------|-------|
| **Mean AC (c_A)** | **0.9901 ± 0.0004 (SEM)** |
| Std Deviation | 0.0044 |
| Min AC | 0.9759 |
| Max AC | 0.9978 |

**Interpretation:** EXTREMELY HIGH AC (≈0.99) indicates strong representational invariance under augmentation. However, combined with negative GCA (-0.02), this confirms the σ-trap: self-attention provides token-level consistency without compositional rules.

### Subtask 6.1: RGA Signal Implementation
**Status:** Complete ✓

**Objectives:**
- [x] Implement `compute_rdm_representational()` in `hbar/engine/signals.py`
- [x] Implement `compute_rga()` with Spearman rank correlation
- [x] Implement SCAN structural distance (Levenshtein on action sequences)
- [x] Create `scripts/analyze_rga_baseline.py` analysis script
- [x] Document baseline RGA profile in memory bank

**Completed Deliverables:**
- `hbar/engine/signals.py`: `compute_rdm_representational()` (cosine/euclidean/correlation distances), `compute_rga()` (Spearman correlation on RDM upper triangles), `_rank_data()` (JAX-compatible ranking with tie handling)
- `hbar/benchmarks/grammar_engine.py`: `_scan_structural_distance()` using normalized Levenshtein distance on action sequence tokens
- `scripts/analyze_rga_baseline.py`: RGA analysis with BOS token representations from final encoder layer

**RGA Implementation Details:**
- **RDM_rep:** Pairwise cosine distances between BOS token representations (final encoder layer)
- **RDM_struct:** For SCAN — normalized Levenshtein distance on action sequences; For COGS — tree-edit distance on logical forms
- **Alignment:** Spearman rank correlation (captures monotonic relationships, not just linear)

**Baseline RGA Results (Kaggle GPU T4, 100 compositional probes, SCAN domain):**
- **Mean RGA (r_A):** 0.0604
- **Interpretation:** LOW RGA — Model's representations are geometrically disorganized relative to grammar structure

**Complete Stage 1 Signal Profile (All 3 Signals):**

| Signal | Baseline Value | Interpretation |
|--------|----------------|----------------|
| g_A (GCA) | -0.0235 ± 0.0075 (SEM) | ✗ NEGATIVE — Learning ID harms OOD |
| c_A (AC) | 0.9901 ± 0.0004 (SEM) | ✓ HIGH — Strong invariance |
| r_A (RGA) | 0.0604 | ✗ LOW — Geometric disorganization |

**Triple-Signal σ-Trap Signature Confirmed:**
The pattern **High AC (0.99) + Negative GCA (-0.02) + Low RGA (0.06)** confirms the classic σ-trap:
- **High AC**: Shallow invariance from Transformer self-attention mechanisms
- **Negative GCA**: Broken gradient geometry for compositional rules
- **Low RGA**: Representations are geometrically disorganized relative to grammar structure

### Subtask 5.2: Deliverables
**Status:** Pending

**Objectives:**
- [ ] Complete 10 Deliverables
- [ ] Build portfolio website
- [ ] Submit YC/EV applications
