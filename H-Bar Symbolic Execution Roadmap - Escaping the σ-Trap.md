*Note: This roadmap assumes the foundational theoretical paper is complete. Week 1 marks the initiation of the empirical software engineering and validation effort.*

## PHASE 1: Foundation & Sequence Architecture
**Objective:** Initialize the `hbar-symbolic` JAX repository, build a highly optimized Flax Transformer, and reproduce the standard "Illusion of Mastery" failure mode on SCAN/COGS to establish the baseline.

### Week 1: Repository Init & JAX/Flax Setup
*   **Subtask 1.1:** Initialize the `hbar-symbolic` repository. Install JAX, Flax, and Optax. Set up the project directory structure.
*   **Subtask 1.2:** Implement a standard, lightweight Flax Transformer (2 layers, 4 heads, $d_{model} = 128$) to exactly match the pre-registered experimental setup (Section 11.1).
*   **Subtask 1.3:** Build the JAX-native tokenization and sequence encoding pipeline. Ensure all forward passes are purely functional to leverage `jax.jit` compilation.

### Week 2: SCAN/COGS Data Generators
*   **Subtask 2.1:** Implement the programmatic domain generative grammar $G(d)$ for the SCAN and COGS benchmarks directly in Python/JAX.
*   **Subtask 2.2:** Build the batch generators. Use `jax.vmap` to efficiently process parallel streams of standard training batches alongside highly-structured compositional probe batches.
*   **Subtask 2.3:** Set up the formal evaluation splits (In-Distribution vs. Out-of-Distribution "Add-Primitive" splits) to serve as Stage 2 ground-truth calibration ($\hat{\sigma}_A$).

### Week 3: Activation Extraction & Augmentation Pipeline
*   **Subtask 3.1:** Create network extraction hooks. Ensure internal hidden states can be extracted at any layer without breaking JAX's functional purity.
*   **Subtask 3.2:** Build the structural-preserving augmentation pipeline (primitive substitution, argument permutation) required for the Augmentation Consistency (AC) signal.

### Week 4: The Baseline Verification ("The Illusion of Mastery")
*   **Subtask 4.1:** Write the standard SGD training loop for the baseline condition.
*   **Subtask 4.2:** Train the baseline model for 5,000 steps. 
*   **Subtask 4.3:** Validate the failure mode: Ensure the model achieves ~90-99% In-Distribution (ID) accuracy but fails catastrophically (~44.5%) on the Out-of-Distribution (OOD) compositional split. *This proves the existence of the $\sigma$-trap.*

## PHASE 2: The Core H-Bar Engine & Signal Extraction
**Objective:** Implement the multi-signal proxy estimators, code the coupled ODE system, and wrap the standard optimizer to modulate training based on Schema Coherence ($\sigma_A$).

### Week 5: Multi-Signal Proxy Extraction I (GCA & AC)
*   **Subtask 5.1:** Implement Gradient-Composition Alignment (GCA, Eq. 3): Extract $\nabla_\theta\mathcal{L}_{train}$ and $\nabla_\theta\mathcal{L}_{comp-batch}$ and compute their Pearson correlation.
*   **Subtask 5.2:** Implement Augmentation Consistency (AC, Eq. 5): Compute mean cosine similarity across structurally augmented sequence pairs inside the training step.

### Week 6: Multi-Signal Proxy Extraction II (RGA) & Fusion
*   **Subtask 6.1:** Implement Representational-Geometry Alignment (RGA, Eq. 4): Compute internal activation RDMs (Representational Dissimilarity Matrices) and correlate them against structural tree-edit distance RDMs.
*   **Subtask 6.2:** Fuse the signals into the training-time operative estimate $\tilde{\sigma}_A = w_g g_A + w_r r_A + w_c c_A$ (Eq. 6). Set default weights ($w_g = 0.4, w_r = 0.35, w_c = 0.25$).

### Week 7: ODE Integration
*   **Subtask 7.1:** Implement the Adaptive-step IMEX Runge-Kutta integrator (Algorithm 3.1) in `jax.numpy` to step the core ODE system forward alongside network training.
*   **Subtask 7.2:** Code the Schema Coherence ODE (Eq. 28) and the Attentional Fidelity ODE (Eq. 29) to track the latent cognitive variables.

### Week 8: The Modulated Optimizer & Loss Target
*   **Subtask 8.1:** Code the Schema-targeting loss function (Eq. 25): $\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_\sigma \cdot (1 - \tilde{\sigma}_A) \cdot \mathcal{L}_{comp}$.
*   **Subtask 8.2:** Implement learning rate modulation based on Attentional Fidelity: $\eta_{effective} = \eta_{base} \cdot (1 + \kappa_\alpha \cdot \alpha_A(d, t))$ (Eq. 26).
*   **Subtask 8.3:** Wrap standard Optax in a custom `HBarOptimizer` that manages ODE state tracking and loss modulation under the hood using `jax.tree_util`.

## PHASE 3: Pre-Training & Empirical Validation
**Objective:** Run the formal experiments, cross the critical bifurcation threshold, achieve structural compositional generalization, and run the statistical analyses.

### Week 9: The Pre-Registered Runs
*   **Subtask 9.1:** Initialize the formal $N=15$ and $N=120$ pre-registered training runs.
*   **Subtask 9.2:** Deploy the three conditions: Baseline (Standard SGD), H-Bar Additive, and H-Bar Multiplicative.
*   **Subtask 9.3:** Log $\delta_A$ (Parametric Depth proxy), $\tilde{\sigma}_A$ (Schema Coherence), and $\alpha_A$ (Attentional Fidelity) continuously via Weights & Biases.

### Week 10: Phase 2 Crystallization & Zero-Shot Testing
*   **Subtask 10.1:** Monitor the runs for the critical bifurcation trigger point ($\sigma_A > \sigma_{critical}$). Automatically increase the compositional batch ratio from 10% to 40% when crossed.
*   **Subtask 10.2:** Execute rigorous evaluation at 500-timestep intervals. 
*   **Subtask 10.3:** Log the targeted ~94-97% OOD structural compositionality accuracy for the H-Bar conditions, proving they successfully bridged the gap.

### Week 11: Statistical Analysis & Hypothesis Testing
*   **Subtask 11.1:** Perform the Welch's t-test to evaluate Prediction 1 (Schema Quality at Intersections).
*   **Subtask 11.2:** Run the F-test on incremental $R^2$ to evaluate Prediction 6 (Multiplicative vs. Additive dependence).
*   **Subtask 11.3:** Execute the Segmented Regression analysis (Prediction 9) to detect the exact positive kink (slope change $\Delta\beta \ge 0.02$) in the OOD accuracy trajectory, mathematically proving Phase 2 entry.

### Week 12: Visual Asset Generation (Deliverable 4)
*   **Subtask 12.1:** Export high-resolution CSVs of the coupled training dynamics ($\delta_A$, $\sigma_A$, $\alpha_A$).
*   **Subtask 12.2:** Create the "$\sigma$-Trap vs. Schema Crystallization" dashboard animation in Matplotlib or Plotly.
*   **Subtask 12.3:** Render the exact moment of bifurcation where the H-Bar network shifts from memorizing sequences to structurally recombining them.

## PHASE 4: Open Source Release & Ecosystem
**Objective:** Transition `hbar-symbolic` from a private experiment to an accessible, installable framework for AI researchers.

### Week 13: PyPI Packaging & GitHub Polish (Deliverables 2 & 3)
*   **Subtask 13.1:** Clean the `hbar-symbolic` JAX codebase. Standardize the data generation API and the `HBarOptimizer` wrapper.
*   **Subtask 13.2:** Write comprehensive documentation focusing on how researchers can extract the GCA/RGA signals from their own sequence models.
*   **Subtask 13.3:** Publish the package to PyPI (`pip install hbar-symbolic`). Generate a Zenodo DOI for strict academic reproducibility.

### Week 14: The HuggingFace Demo & Colab (Deliverable 9)
*   **Subtask 14.1:** Upload the pre-trained Baseline vs. H-Bar Flax weights to HuggingFace.
*   **Subtask 14.2:** Build a Gradio UI where users type out novel, heavily nested commands and compare the translation outputs of standard SGD vs. H-Bar.
*   **Subtask 14.3:** Publish the "10-Line H-Bar" Google Colab notebook demonstrating the pipeline.

### Week 15: CI/CD Pipeline Update (Deliverable 8)
*   **Subtask 15.1:** Configure GitHub Actions to automatically train a miniature Transformer on SCAN for 500 steps on every PR.
*   **Subtask 15.2:** Add automated statistical tests that assert the Phase 2 entry inflection occurs (using piecewise regression logic), ensuring framework updates don't break the core mathematics.

## PHASE 5: Paper Publication & Pitch Synthesis
**Objective:** Compile the theoretical ODEs and the empirical NLP results into a single Magnum Opus for arXiv, and leverage it for grants and accelerators.

### Week 16: Paper Drafting - Methodology & Results
*   **Subtask 16.1:** Draft the Implementation Architecture methodology. Detail the JAX/Flax setup and the multi-signal proxy extraction overhead.
*   **Subtask 16.2:** Format the OOD zero-shot results into formal academic tables.
*   **Subtask 16.3:** Integrate the segmented regression graphs (Figure 7) and the Multiplicative vs. Additive $R^2$ scatterplots (Figure 6) into the paper.

### Week 17: Paper Drafting - Framing & Conclusion (Deliverable 1)
*   **Subtask 17.1:** Write the Introduction. Define the $\sigma$-trap and pitch the coupled dynamical systems approach as the missing puzzle piece in modern LLM scaling.
*   **Subtask 17.2:** Write the Conclusion, validating the pre-registered predictions and outlining implications for AGI reasoning.
*   **Subtask 17.3:** Format to standard NeurIPS / ICLR LaTeX templates and submit to arXiv.

### Week 18: Portfolio Website & Dossier (Deliverables 5 & 10)
*   **Subtask 18.1:** Build the Unified Portfolio Website to host the PDF, interactive dashboard, and codebase link.
*   **Subtask 18.2:** Finalize the Mentorship & Academic Dossier. Send to target NLP and Cognitive Science labs (e.g., Lake, Linzen, DeepMind).

### Week 19: Y-Combinator / EV Pitch Preparation (Deliverable 6)
*   **Subtask 19.1:** Draft the 30-second "Founder Pitch" focusing on solving the reasoning and hallucination bottlenecks in Enterprise LLMs via structural schema crystallization.
*   **Subtask 19.2:** Finalize the Competition & Grant Package for Y-Combinator / Emergent Ventures.

### Week 20: Execution & Audit (Deliverable 7)
*   **Subtask 20.1:** Compile the initial OSS Traction Report tracking GitHub stars and PyPI downloads.
*   **Subtask 20.2:** Conduct a final audit of all 10 Deliverables.
*   **Subtask 20.3:** Execute Year-1 Planning based on feedback from the EV Grant, YC, or target academic labs.