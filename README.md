# H-Bar Symbolic

## Escaping the σ-Trap in Compositional Generalization

This project implements the **H-Bar Model V3.0+** — a JAX-native coupled ODE framework designed to solve systematic compositional generalization failure in neural networks.

## The Problem: The Illusion of Mastery

Neural networks exhibit a critical failure mode on compositional tasks: near-perfect **In-Distribution (ID)** accuracy but catastrophic **Out-of-Distribution (OOD)** failure. This "σ-trap" creates an illusion that models have learned systematic rules when they've merely memorized surface patterns.

## The Solution: H-Bar Phase-Structured Curriculum

The H-Bar Model introduces a coupled dynamical system with **5 cognitive faculties** and their associated state variables:

| Faculty | Primary Variable | Description |
|---------|-----------------|-------------|
| **Learning** | `sigma_A`, `delta_A` | Schema coherence and parametric depth |
| **Metacognition** | `M_hat_A`, `zeta_A` | Self-model accuracy and calibration error |
| **Attention** | `alpha_A`, `CA` | Attentional fidelity to generative structure |
| **Executive Functions** | `Xi_A_P`, `Xi_A_I`, `Xi_A_F` | Planning, inhibition, cognitive flexibility |
| **Social Cognition** | `mu_AB`, `Sigma_AB` | Schema legibility and collective field |

### Core Variables

| Symbol | Name | Range | Description |
|--------|------|-------|-------------|
| `sigma_A(d,t)` | Schema Coherence | [0,1] | Degree of principled reorganization |
| `delta_A(d,t)` | Parametric Depth | [0,Δ] | Structural complexity of representation |
| `alpha_A(d,t)` | Attentional Fidelity | [0,1] | Stability of attention to structure |
| `M_hat_A(d,t)` | Self-Model | [0,1] | Agent's estimate of its own σ_A |
| `Xi_A(t)` | Executive Control | [0,1]³ | Planning, inhibition, flexibility |

### Two-Stage Estimation Protocol

- **Stage 1 (Training-time):** `sigma_tilde_A` via multi-signal fusion (GCA + RGA + AC)
- **Stage 2 (Evaluation-time):** `sigma_hat_A` via OOD/ID accuracy ratio

## Project Structure

```
hbar-symbolic/
├── hbar/
│   ├── core/          # Core ODE integrators (Eqs. 28, 29)
│   ├── models/        # Flax Transformers & RNNs
│   ├── engine/        # Multi-signal proxy extraction (GCA, RGA, AC)
│   └── optim/         # H-Bar modulated optimizer wrappers
├── benchmarks/
│   ├── scan/          # SCAN data generators & grammars
│   └── cogs/          # COGS data generators & grammars
├── experiments/       # Pre-registered run scripts
├── scripts/           # CLI tools for training/viz
├── tests/             # Chex-based numerical stability tests
├── figures/           # Manuscript TikZ figures
├── bibliography/      # LaTeX bibliography files
└── memory-bank/       # Project state documentation
```

## Target Benchmarks

- **SCAN** (Systematic Compositionality in Augmented Natural language)
- **COGS** (Compositional Generalization in Sentence Processing)

## Success Criteria

- Close the ID-OOD gap to <10% on SCAN challenging splits
- Achieve >85% OOD accuracy on COGS zero-shot generalization tests
- Validate Phase 2 entry inflection via segmented regression (Δβ ≥ 0.02)

## Technical Stack

- **JAX/jaxlib** — High-performance numerical computing
- **Flax** — Neural network library with linen modules
- **Optax** — Gradient processing and optimization
- **Chex** — Property-based testing and numerical stability
- **Distrax** — Probabilistic programming

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # bash
# or: ./venv/bin/activate.fish  # fish

# Install dependencies
pip install jax jaxlib flax optax chex distrax

# Run tests
pytest tests/
```

## The 10 Deliverables

1. **Magnum Opus Paper** — Empirical validation on SCAN/COGS
2. **H-Bar Torch** — PyPI package for drop-in training
3. **Zenodo DOI** — Reproducibility package
4. **Visual Asset** — "Illusion of Mastery" dashboard
5. **Mentorship Dossier** — Academic outreach package
6. **Competition Package** — YC/EV grant applications
7. **OSS Traction Report** — Community adoption metrics
8. **CI/CD Pipeline** — Automated phase tracking
9. **HuggingFace Space** — Interactive public demo
10. **Portfolio Website** — Central hub for all materials

## Roadmap

| Phase | Weeks | Objective |
|-------|-------|-----------|
| 1. Foundation | 1-4 | JAX/Flax setup, Transformer, baseline verification |
| 2. H-Bar Engine | 5-8 | Signal extraction, ODE integration, optimizer |
| 3. Validation | 9-12 | Pre-registered runs, statistical analysis |
| 4. Open Source | 13-15 | PyPI release, HuggingFace demo, Colab |
| 5. Publication | 16-20 | arXiv submission, portfolio, grant applications |

## License

MIT License — Research Use

## Citation

```bibtex
@article{basyirin2026hbar,
  title={The H-Bar Model: Schema-Coherence Suppression as the Origin of Compositional Generalization Failure},
  author={Basyirin Amsyar bin Basri},
  journal={arXiv preprint},
  year={2026}
}
