# Product Context: H-Bar Model V3.0+

## Mission Statement

The H-Bar Model aims to **escape the σ-trap** in compositional generalization benchmarks like SCAN and COGS. Current neural network architectures fail to generalize systematically when faced with novel compositions of familiar elements—a fundamental limitation known as the "Illusion of Mastery."

## The σ-Trap Problem

Neural networks trained on compositional tasks exhibit a critical failure mode:

- **In-Distribution (ID) Performance:** Near-perfect accuracy on training distribution
- **Out-of-Distribution (OOD) Performance:** Catastrophic failure on novel compositions

This gap between ID and OOD performance creates an illusion that the model has learned systematic rules when it has merely memorized surface patterns.

### Proposition 4.1 (σ-Suppression)

**The primary bug we are fixing:** Standard SGD dynamics create a stable low-σ_A equilibrium (the "σ-trap"), explaining why high-ID/low-OOD agents arise systematically. The model learns surface-level statistical regularities rather than deep compositional structure, because the training objective does not penalize this failure mode. The H-Bar framework introduces explicit σ_A dynamics to break this equilibrium and drive the system toward genuine schema coherence.

## The Solution: H-Bar Phase-Structured Curriculum V3.0+

The H-Bar Model introduces a coupled dynamical system with **5 cognitive faculties**:

| Faculty | Primary Variable | Mechanism |
|---------|-----------------|-----------|
| **Learning** | `sigma_A`, `delta_A` | OOD gap as schema proxy; compositional generalization |
| **Metacognition** | `M_hat_A`, `zeta_A` | Self-model accuracy; calibration error dynamics |
| **Attention** | `alpha_A`, `CA` | Attentional fidelity to generative structure |
| **Executive Functions** | `Xi_A_P`, `Xi_A_I`, `Xi_A_F` | Planning, inhibition, cognitive flexibility |
| **Social Cognition** | `mu_AB`, `tau_A`, `Sigma_AB` | Schema legibility, theory of mind, collective field |

### Core State Variables

1. **Schema Coherence (σ_A):** Measures how well the model maintains consistent internal representations across compositional transformations
2. **Parametric Depth (δ_A):** Quantifies the hierarchical depth of learned compositional rules
3. **Attentional Fidelity (α_A):** Tracks the stability of attention mechanisms across novel compositions

### Two-Stage Estimation Protocol

- **Stage 1 (Training-time):** Multi-signal fusion proxy (GCA + RGA + AC) → `sigma_tilde_A`
- **Stage 2 (Evaluation-time):** OOD/ID accuracy ratio → `sigma_hat_A`

### Multimodal Extension

All variables extended to **Domain × Modality product space**: σ_A(d, m, t) for text, visual, auditory, sensorimotor, symbolic modalities.

### Benchmark Validity Function

V_A(B,f,t) = CI × FD × DG × R_A where:
- **CI** = Construct Isolation
- **FD** = Format Diversity
- **DG** = Difficulty Gradient
- **R_A** = Reliability

## Target Benchmarks

- **SCAN (Systematic Compositionality in Augmented Natural language):** Command-to-action translation task
- **COGS (Compositional Generalization in Sentence Processing):** Semantic parsing with complex syntactic structures

## Success Criteria

- Close the ID-OOD gap to <10% on SCAN challenging splits
- Achieve >85% OOD accuracy on COGS zero-shot generalization tests
- Demonstrate transfer to novel compositional domains beyond training distribution
- Validate Phase 2 entry inflection via segmented regression (Δβ ≥ 0.02)

## Pre-Registration Protocol

All experiments follow pre-registered statistical framework:
- Specify H0, H1, test statistic, sample size before execution
- Bonferroni correction for 8 predictions (α_corrected = 0.00625)
- Power analysis: N=500 for small effects (d=0.2), N=120 for medium effects
