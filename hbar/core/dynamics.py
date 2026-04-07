"""H-Bar ODE dynamics — coupled system of differential equations.

This module implements the vector field defined by Equations 14, 28, 29, 33,
and 36 of the H-Bar paper. The system models the evolution of an agent's
cognitive state during compositional learning.

All functions are purely functional and JIT-compatible.
"""

from typing import Dict

import flax.struct
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# State and parameter dataclasses
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class HBarState:
    """Full cognitive state of the H-Bar agent.

    The state vector has 7 dimensions split into two timescale groups:

    Fast subsystem (core learning variables):
        delta_A:  Parametric depth — hierarchical depth of learned rules [0, Δ_max]
        sigma_A:  Schema coherence — consistency of internal representations [0, 1]
        alpha_A:  Attentional fidelity — stability of attention mechanisms [0, 1]

    Slow subsystem (extended cognition / metacognitive variables):
        M_hat_A:  Self-model of schema coherence — agent's estimate of σ_A [0, 1]
        Xi_A_P:   Executive control: Planning — goal-directed trajectory planning [0, 1]
        Xi_A_I:   Executive control: Inhibition — suppression of surface rewards [0, 1]
        Xi_A_F:   Executive control: Flexibility — cross-domain discovery rate [0, 1]
    """

    delta_A: jax.Array
    sigma_A: jax.Array
    alpha_A: jax.Array
    M_hat_A: jax.Array
    Xi_A_P: jax.Array
    Xi_A_I: jax.Array
    Xi_A_F: jax.Array


@flax.struct.dataclass
class HBarInputs:
    """External signals fed into the ODE system at each timestep.

    These are computed by the signal extraction engine (hbar/engine/signals.py)
    and passed to the integrator at each training step.

    Attributes:
        sigma_tilde: Fused operative signal σ̃_A ∈ [0, 1] from Equation 6.
        sigma_hat: Ground-truth schema coherence σ̂_A = Acc_OOD / Acc_ID ∈ [0, 1].
        P_A: Principled structure availability ∈ [0, 1]. Measures the degree to
            which the training curriculum exposes compositional rules (vs surface
            patterns). Used in Eq. 28 as a gate for schema growth.
        C_A: Training signal strength ∈ [0, 1]. Drives attentional fidelity growth
            in Eq. 29. Computed from [BOS] token representation stability.
        Omega_AI: AI-bypass risk factor ∈ [0, 1]. High values indicate the model
            is achieving high accuracy via surface statistics rather than rules.
        R_surface: Surface reward signal ∈ [0, 1]. High values indicate the model
            is receiving rewards for superficial pattern matching.
        domain_frontier: Domain difficulty frontier Δ(d, t) ∈ [0, 1]. Represents
            the current compositional complexity of the training curriculum.
    """

    sigma_tilde: jax.Array
    sigma_hat: jax.Array
    P_A: jax.Array
    C_A: jax.Array
    Omega_AI: jax.Array
    R_surface: jax.Array
    domain_frontier: jax.Array


@flax.struct.dataclass
class HBarConstants:
    """Fixed parameters of the H-Bar ODE system.

    These constants control the timescales, capacities, and coupling strengths
    of the cognitive dynamics. Default values are based on the H-Bar paper's
    recommended settings for SCAN/COGS benchmarks.

    Fast subsystem parameters:
        r_delta: Gompertz growth rate for parametric depth (default: 0.5).
        gamma_sigma: Schema coherence growth rate (default: 0.3).
        gamma_alpha: Attentional fidelity growth rate (default: 0.2).
        mu_delta: Schema-mediated decay rate for depth (default: 0.1).
        eta_sigma: AI-bypass suppression rate (default: 0.4).
        eta_alpha: Surface reward suppression rate (default: 0.3).
        K_delta: Maximum parametric depth capacity (default: 10.0).

    Slow subsystem parameters:
        kappa_M: Self-model calibration rate (default: 0.05).
        kappa_Xi: Executive control activation rate (default: 0.1).
        lambda_Xi: Executive control decay rate (default: 0.05).

    Thresholds:
        sigma_critical: Phase 2 entry threshold (default: 0.5).
    """

    r_delta: float = flax.struct.field(default=0.5)
    gamma_sigma: float = flax.struct.field(default=0.3)
    gamma_alpha: float = flax.struct.field(default=0.2)
    mu_delta: float = flax.struct.field(default=0.1)
    eta_sigma: float = flax.struct.field(default=0.4)
    eta_alpha: float = flax.struct.field(default=0.3)
    K_delta: float = flax.struct.field(default=10.0)
    kappa_M: float = flax.struct.field(default=0.05)
    kappa_Xi: float = flax.struct.field(default=0.1)
    lambda_Xi: float = flax.struct.field(default=0.05)
    sigma_critical: float = flax.struct.field(default=0.5)


# ---------------------------------------------------------------------------
# Vector field (right-hand side of the ODE system)
# ---------------------------------------------------------------------------


def hbar_vector_field(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
) -> HBarState:
    """Compute the time derivatives of all H-Bar state variables.

    This function implements the coupled ODE system:
        δ̇_A  — Eq. 14 (Gompertz growth + schema-mediated decay)
        σ̇_A  — Eq. 28 (Gated growth − AI bypass suppression)
        α̇_A  — Eq. 29 (Signal-driven focus − surface reward suppression)
        Ṁ_A  — Eq. 36 (Mean-reverting calibration)
        Ξ̇_A  — Eq. 33 (Target-seeking trajectory control; 3 components)

    Args:
        state: Current cognitive state (7 variables).
        inputs: External signals from the training environment.
        constants: Fixed ODE parameters.

    Returns:
        HBarState containing the time derivatives [δ̇_A, σ̇_A, α̇_A,
        Ṁ_A, Ξ̇_A_P, Ξ̇_A_I, Ξ̇_A_F].
    """
    # Unpack state for readability
    delta = state.delta_A
    sigma = state.sigma_A
    alpha = state.alpha_A
    M_hat = state.M_hat_A
    Xi_P = state.Xi_A_P
    Xi_I = state.Xi_A_I
    Xi_F = state.Xi_A_F

    # Unpack inputs
    sigma_tilde = inputs.sigma_tilde
    sigma_hat = inputs.sigma_hat
    P_A = inputs.P_A           # Principled structure availability (Eq. 28)
    C_A = inputs.C_A           # Training signal strength (Eq. 29)
    Omega_AI = inputs.Omega_AI
    R_surface = inputs.R_surface
    Delta_dt = inputs.domain_frontier

    # Unpack constants
    r_d = constants.r_delta
    g_s = constants.gamma_sigma
    g_a = constants.gamma_alpha
    mu_d = constants.mu_delta
    et_s = constants.eta_sigma
    et_a = constants.eta_alpha
    K_d = constants.K_delta
    k_M = constants.kappa_M
    k_X = constants.kappa_Xi
    l_X = constants.lambda_Xi
    sigma_critical = constants.sigma_critical

    # Epsilon for numerical stability
    eps = 1e-8

    # ------------------------------------------------------------------
    # Eq. 14: δ̇_A — Gompertz growth with schema-mediated decay
    # ------------------------------------------------------------------
    # Gompertz growth: r_δ · δ · ln(K_δ / δ)
    # Prevent division by zero and log(0)
    delta_safe = jnp.maximum(delta, eps)
    gompertz = r_d * delta_safe * jnp.log(K_d / delta_safe + eps)

    # Schema-mediated decay: μ_δ · (1 - σ_A) · δ_A
    # When schema coherence is low, depth decays faster
    decay = mu_d * (1.0 - sigma) * delta_safe

    delta_dot = gompertz - decay

    # ------------------------------------------------------------------
    # Eq. 28: σ̇_A — Gated growth with AI-bypass suppression
    # ------------------------------------------------------------------
    # CRITICAL COUPLING: Growth requires α_A > 0 (attentional gate)
    # ρ · P_A · α_A · (1 - σ_A)
    # If α_A = 0, growth = 0 — schema coherence cannot form without attention
    growth_sigma = g_s * P_A * alpha * (1.0 - sigma)

    # AI-bypass suppression: η_σ · Ω_AI · σ_A
    # High AI-bypass risk actively suppresses schema coherence
    suppression_sigma = et_s * Omega_AI * sigma

    sigma_dot = growth_sigma - suppression_sigma

    # ------------------------------------------------------------------
    # Eq. 29: α̇_A — Signal-driven focus with surface reward suppression
    # ------------------------------------------------------------------
    # Signal-driven focus: γ · C_A · (1 - α_A)
    # Driven by training signal C_A (not tracking σ̃_A)
    drive_alpha = g_a * C_A * (1.0 - alpha)

    # Surface reward suppression: η_α · R_surface · α_A
    # High surface rewards suppress attentional fidelity
    suppression_alpha = et_a * R_surface * alpha

    alpha_dot = drive_alpha - suppression_alpha

    # ------------------------------------------------------------------
    # Eq. 36: Ṁ_A — Mean-reverting calibration
    # ------------------------------------------------------------------
    # M̂_A calibrates toward the ground-truth σ̂_A
    M_dot = k_M * (sigma_hat - M_hat)

    # ------------------------------------------------------------------
    # Eq. 33: Ξ̇_A — Target-seeking trajectory control (3 components)
    # ------------------------------------------------------------------
    # Planning: drives toward σ_critical target
    Xi_P_dot = k_X * jnp.maximum(0.0, sigma_critical - sigma) - l_X * Xi_P

    # Inhibition: suppresses surface-reward-driven behavior
    Xi_I_dot = k_X * Omega_AI * (1.0 - Xi_I) - l_X * Xi_I

    # Flexibility: promotes cross-domain exploration
    Xi_F_dot = k_X * Delta_dt * (1.0 - Xi_F) - l_X * Xi_F

    return HBarState(
        delta_A=delta_dot,
        sigma_A=sigma_dot,
        alpha_A=alpha_dot,
        M_hat_A=M_dot,
        Xi_A_P=Xi_P_dot,
        Xi_A_I=Xi_I_dot,
        Xi_A_F=Xi_F_dot,
    )


# ---------------------------------------------------------------------------
# Fast / slow subsystem extractors (for IMEX integration)
# ---------------------------------------------------------------------------


def fast_vector_field(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
) -> Dict[str, jax.Array]:
    """Extract derivatives of fast variables (δ_A, σ_A, α_A).

    Used by the explicit RK4 integrator for the non-stiff subsystem.

    Returns:
        Dictionary with keys 'delta_A', 'sigma_A', 'alpha_A'.
    """
    full_derivs = hbar_vector_field(state, inputs, constants)
    return {
        "delta_A": full_derivs.delta_A,
        "sigma_A": full_derivs.sigma_A,
        "alpha_A": full_derivs.alpha_A,
    }


def slow_vector_field(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
) -> Dict[str, jax.Array]:
    """Extract derivatives of slow variables (M̂_A, Ξ_A_P, Ξ_A_I, Ξ_A_F).

    Used by the implicit Backward Euler integrator for the stiff subsystem.

    Returns:
        Dictionary with keys 'M_hat_A', 'Xi_A_P', 'Xi_A_I', 'Xi_A_F'.
    """
    full_derivs = hbar_vector_field(state, inputs, constants)
    return {
        "M_hat_A": full_derivs.M_hat_A,
        "Xi_A_P": full_derivs.Xi_A_P,
        "Xi_A_I": full_derivs.Xi_A_I,
        "Xi_A_F": full_derivs.Xi_A_F,
    }


# ---------------------------------------------------------------------------
# State constructors and utilities
# ---------------------------------------------------------------------------


def init_hbar_state(
    key: jax.Array,
    sigma_tilde_init: float = 0.27,
) -> HBarState:
    """Initialize H-Bar state from the baseline starting point.

    Uses the calculated baseline σ̃_A ≈ 0.27 to set initial conditions
    that reflect the σ-trap starting state.

    Args:
        key: JAX PRNGKey for any stochastic initialization.
        sigma_tilde_init: Initial fused signal value (default: 0.27).

    Returns:
        HBarState initialized at the baseline starting point.
    """
    return HBarState(
        delta_A=jnp.array(1.0),          # Low initial depth
        sigma_A=jnp.array(sigma_tilde_init),  # Baseline σ̃_A ≈ 0.27
        alpha_A=jnp.array(0.5),          # Moderate initial attention
        M_hat_A=jnp.array(sigma_tilde_init),  # Self-model starts at baseline
        Xi_A_P=jnp.array(0.3),           # Low initial planning
        Xi_A_I=jnp.array(0.3),           # Low initial inhibition
        Xi_A_F=jnp.array(0.3),           # Low initial flexibility
    )


def state_to_array(state: HBarState) -> jax.Array:
    """Convert HBarState to a flat array for optimization/analysis.

    Order: [delta_A, sigma_A, alpha_A, M_hat_A, Xi_A_P, Xi_A_I, Xi_A_F]
    """
    return jnp.array([
        state.delta_A,
        state.sigma_A,
        state.alpha_A,
        state.M_hat_A,
        state.Xi_A_P,
        state.Xi_A_I,
        state.Xi_A_F,
    ])


def array_to_state(arr: jax.Array) -> HBarState:
    """Convert a flat array back to HBarState.

    See state_to_array for ordering.
    """
    return HBarState(
        delta_A=arr[0],
        sigma_A=arr[1],
        alpha_A=arr[2],
        M_hat_A=arr[3],
        Xi_A_P=arr[4],
        Xi_A_I=arr[5],
        Xi_A_F=arr[6],
    )


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------


def analyze_coupling_sensitivity(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
) -> Dict[str, jax.Array]:
    """Analyze the schema-attention coupling sensitivity.

    Computes diagnostic metrics that reveal how the attentional gate (α_A)
    controls schema coherence growth (σ_A). This is critical for understanding
    why σ_A stays at ~0.27 even with good GCA signals in Phase 1.

    The key insight: In Phase 1 (Asymmetric Initialization), α_A is low.
    Since σ̇_A growth = ρ · P_A · α_A · (1 - σ_A), if α_A ≈ 0, then
    σ̇_A growth ≈ 0 regardless of P_A magnitude. This explains the σ-trap.

    Args:
        state: Current cognitive state.
        inputs: External signals from the training environment.
        constants: Fixed ODE parameters.

    Returns:
        Dictionary containing:
            - coupled_growth_potential: ρ · P_A · α_A · (1 - σ_A) — the
              maximum rate at which σ_A can grow given current α_A
            - attentional_gate_strength: α_A — the current gate value
            - schema_growth_capacity: (1 - σ_A) — remaining capacity
            - effective_drive: P_A · α_A — combined principled × attentional drive
            - suppression_pressure: η_σ · Ω_AI — AI-bypass suppression force
            - net_sigma_dot: The actual σ̇_A (growth - suppression)
            - is_attention_limited: Boolean — True if α_A < 0.3 (Phase 1 state)
    """
    sigma = state.sigma_A
    alpha = state.alpha_A
    P_A = inputs.P_A
    C_A = inputs.C_A
    Omega_AI = inputs.Omega_AI
    R_surface = inputs.R_surface

    g_s = constants.gamma_sigma
    g_a = constants.gamma_alpha
    et_s = constants.eta_sigma
    et_a = constants.eta_alpha

    # Coupled growth potential: ρ · P_A · α_A · (1 - σ_A)
    # This is the maximum σ_A growth rate given current attention
    coupled_growth_potential = g_s * P_A * alpha * (1.0 - sigma)

    # Attentional gate strength
    attentional_gate_strength = alpha

    # Schema growth capacity (remaining room to 1.0)
    schema_growth_capacity = 1.0 - sigma

    # Effective drive: combined principled structure × attention
    effective_drive = P_A * alpha

    # Suppression pressure from AI-bypass
    suppression_pressure = et_s * Omega_AI

    # Net σ̇_A (actual growth minus suppression)
    net_sigma_dot = coupled_growth_potential - suppression_pressure * sigma

    # Is the system attention-limited? (Phase 1 state)
    is_attention_limited = alpha < 0.3

    return {
        "coupled_growth_potential": coupled_growth_potential,
        "attentional_gate_strength": attentional_gate_strength,
        "schema_growth_capacity": schema_growth_capacity,
        "effective_drive": effective_drive,
        "suppression_pressure": suppression_pressure,
        "net_sigma_dot": net_sigma_dot,
        "is_attention_limited": is_attention_limited,
    }


def compute_crystallization_potential(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
) -> jax.Array:
    """Compute the crystallization potential for Phase 2 entry.

    The crystallization potential is defined as the state where
    α_A · C_A > threshold, indicating the model is ready to enter
    Phase 2 (Schema Crystallization).

    Args:
        state: Current cognitive state.
        inputs: External signals from the training environment.
        constants: Fixed ODE parameters.

    Returns:
        Crystallization potential value ∈ [0, 1]. Values > 0.5 indicate
        the model is approaching Phase 2 entry.
    """
    alpha = state.alpha_A
    C_A = inputs.C_A

    # Crystallization potential: α_A · C_A
    # High when both attention and training signal are strong
    return alpha * C_A
