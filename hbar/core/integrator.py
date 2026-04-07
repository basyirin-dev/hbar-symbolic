"""IMEX Runge-Kutta integrator for the H-Bar ODE system.

This module implements Algorithm 3.1 from the H-Bar paper: an IMEX
(IMplicit-EXplicit) integrator that combines:
  - Explicit RK4 for fast, non-stiff variables (δ_A, σ_A, α_A)
  - Implicit Backward Euler for slow, stiff variables (M̂_A, Ξ_A)

The integrator includes adaptive step size control and boundary enforcement
to ensure numerical stability and forward invariance (Proposition 3.2).

All functions are purely functional and JIT-compatible.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from hbar.core.dynamics import (
    HBarState,
    HBarInputs,
    HBarConstants,
    fast_vector_field,
    slow_vector_field,
    hbar_vector_field,
)


# ---------------------------------------------------------------------------
# Explicit RK4 step for fast variables
# ---------------------------------------------------------------------------


def _rk4_fast_step(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
    h: float,
) -> HBarState:
    """Perform one explicit RK4 step for fast variables (δ, σ, α).

    The slow variables (M̂, Ξ) are held constant during this step.

    Args:
        state: Current full state.
        inputs: External signals.
        constants: ODE parameters.
        h: Step size.

    Returns:
        New HBarState with updated fast variables; slow variables unchanged.
    """
    # Helper to compute fast derivatives with slow variables frozen
    def f_fast(s: HBarState) -> dict:
        return fast_vector_field(s, inputs, constants)

    # k1
    k1 = f_fast(state)

    # k2: evaluate at state + h/2 * k1
    state_k2 = HBarState(
        delta_A=state.delta_A + 0.5 * h * k1["delta_A"],
        sigma_A=state.sigma_A + 0.5 * h * k1["sigma_A"],
        alpha_A=state.alpha_A + 0.5 * h * k1["alpha_A"],
        M_hat_A=state.M_hat_A,
        Xi_A_P=state.Xi_A_P,
        Xi_A_I=state.Xi_A_I,
        Xi_A_F=state.Xi_A_F,
    )
    k2 = f_fast(state_k2)

    # k3: evaluate at state + h/2 * k2
    state_k3 = HBarState(
        delta_A=state.delta_A + 0.5 * h * k2["delta_A"],
        sigma_A=state.sigma_A + 0.5 * h * k2["sigma_A"],
        alpha_A=state.alpha_A + 0.5 * h * k2["alpha_A"],
        M_hat_A=state.M_hat_A,
        Xi_A_P=state.Xi_A_P,
        Xi_A_I=state.Xi_A_I,
        Xi_A_F=state.Xi_A_F,
    )
    k3 = f_fast(state_k3)

    # k4: evaluate at state + h * k3
    state_k4 = HBarState(
        delta_A=state.delta_A + h * k3["delta_A"],
        sigma_A=state.sigma_A + h * k3["sigma_A"],
        alpha_A=state.alpha_A + h * k3["alpha_A"],
        M_hat_A=state.M_hat_A,
        Xi_A_P=state.Xi_A_P,
        Xi_A_I=state.Xi_A_I,
        Xi_A_F=state.Xi_A_F,
    )
    k4 = f_fast(state_k4)

    # RK4 update: y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_delta = state.delta_A + (h / 6.0) * (
        k1["delta_A"] + 2.0 * k2["delta_A"] + 2.0 * k3["delta_A"] + k4["delta_A"]
    )
    new_sigma = state.sigma_A + (h / 6.0) * (
        k1["sigma_A"] + 2.0 * k2["sigma_A"] + 2.0 * k3["sigma_A"] + k4["sigma_A"]
    )
    new_alpha = state.alpha_A + (h / 6.0) * (
        k1["alpha_A"] + 2.0 * k2["alpha_A"] + 2.0 * k3["alpha_A"] + k4["alpha_A"]
    )

    return HBarState(
        delta_A=new_delta,
        sigma_A=new_sigma,
        alpha_A=new_alpha,
        M_hat_A=state.M_hat_A,
        Xi_A_P=state.Xi_A_P,
        Xi_A_I=state.Xi_A_I,
        Xi_A_F=state.Xi_A_F,
    )


# ---------------------------------------------------------------------------
# Implicit Backward Euler step for slow variables
# ---------------------------------------------------------------------------


def _backward_euler_slow_step(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
    h: float,
    n_newton: int = 3,
) -> HBarState:
    """Perform one implicit Backward Euler step for slow variables (M̂, Ξ).

    Uses Newton's method to solve the implicit equation:
        y_{n+1} = y_n + h * f(y_{n+1})

    Since the slow subsystem is linear (Eqs. 33, 36), Newton converges
    in 1-2 iterations. We use a fixed number of iterations for JIT
    compatibility.

    Args:
        state: Current full state.
        inputs: External signals.
        constants: ODE parameters.
        h: Step size.
        n_newton: Number of Newton iterations (default: 3).

    Returns:
        New HBarState with updated slow variables; fast variables unchanged.
    """
    # Helper to compute slow derivatives
    def f_slow(s: HBarState) -> dict:
        return slow_vector_field(s, inputs, constants)

    # Newton iteration for implicit step
    # Start with explicit Euler as initial guess
    slow_derivs = f_slow(state)
    guess_M = state.M_hat_A + h * slow_derivs["M_hat_A"]
    guess_Xi_P = state.Xi_A_P + h * slow_derivs["Xi_A_P"]
    guess_Xi_I = state.Xi_A_I + h * slow_derivs["Xi_A_I"]
    guess_Xi_F = state.Xi_A_F + h * slow_derivs["Xi_A_F"]

    current_state = HBarState(
        delta_A=state.delta_A,
        sigma_A=state.sigma_A,
        alpha_A=state.alpha_A,
        M_hat_A=guess_M,
        Xi_A_P=guess_Xi_P,
        Xi_A_I=guess_Xi_I,
        Xi_A_F=guess_Xi_F,
    )

    # Newton iterations: solve F(y) = y - y_n - h*f(y) = 0
    # y_{k+1} = y_k - F(y_k) / F'(y_k)
    # For the linear slow subsystem, this converges quickly
    for _ in range(n_newton):
        slow_derivs = f_slow(current_state)
        # F(y) = y - y_n - h*f(y)
        F_M = current_state.M_hat_A - state.M_hat_A - h * slow_derivs["M_hat_A"]
        F_Xi_P = current_state.Xi_A_P - state.Xi_A_P - h * slow_derivs["Xi_A_P"]
        F_Xi_I = current_state.Xi_A_I - state.Xi_A_I - h * slow_derivs["Xi_A_I"]
        F_Xi_F = current_state.Xi_A_F - state.Xi_A_F - h * slow_derivs["Xi_A_F"]

        # Jacobian of F w.r.t. slow variables (approximated via finite differences)
        # For linear systems: F'(y) ≈ I - h*J_f where J_f is Jacobian of f
        # We use a simple secant-like update for JIT compatibility
        eps = 1e-6

        # Compute Jacobian numerically for M_hat
        perturbed = HBarState(
            delta_A=current_state.delta_A,
            sigma_A=current_state.sigma_A,
            alpha_A=current_state.alpha_A,
            M_hat_A=current_state.M_hat_A + eps,
            Xi_A_P=current_state.Xi_A_P,
            Xi_A_I=current_state.Xi_A_I,
            Xi_A_F=current_state.Xi_A_F,
        )
        perturbed_derivs = f_slow(perturbed)
        dF_dM = 1.0 - h * (perturbed_derivs["M_hat_A"] - slow_derivs["M_hat_A"]) / eps

        # Compute Jacobian for Xi_P
        perturbed = HBarState(
            delta_A=current_state.delta_A,
            sigma_A=current_state.sigma_A,
            alpha_A=current_state.alpha_A,
            M_hat_A=current_state.M_hat_A,
            Xi_A_P=current_state.Xi_A_P + eps,
            Xi_A_I=current_state.Xi_A_I,
            Xi_A_F=current_state.Xi_A_F,
        )
        perturbed_derivs = f_slow(perturbed)
        dF_dXi_P = 1.0 - h * (perturbed_derivs["Xi_A_P"] - slow_derivs["Xi_A_P"]) / eps

        # Compute Jacobian for Xi_I
        perturbed = HBarState(
            delta_A=current_state.delta_A,
            sigma_A=current_state.sigma_A,
            alpha_A=current_state.alpha_A,
            M_hat_A=current_state.M_hat_A,
            Xi_A_P=current_state.Xi_A_P,
            Xi_A_I=current_state.Xi_A_I + eps,
            Xi_A_F=current_state.Xi_A_F,
        )
        perturbed_derivs = f_slow(perturbed)
        dF_dXi_I = 1.0 - h * (perturbed_derivs["Xi_A_I"] - slow_derivs["Xi_A_I"]) / eps

        # Compute Jacobian for Xi_F
        perturbed = HBarState(
            delta_A=current_state.delta_A,
            sigma_A=current_state.sigma_A,
            alpha_A=current_state.alpha_A,
            M_hat_A=current_state.M_hat_A,
            Xi_A_P=current_state.Xi_A_P,
            Xi_A_I=current_state.Xi_A_I,
            Xi_A_F=current_state.Xi_A_F + eps,
        )
        perturbed_derivs = f_slow(perturbed)
        dF_dXi_F = 1.0 - h * (perturbed_derivs["Xi_A_F"] - slow_derivs["Xi_A_F"]) / eps

        # Newton update
        new_M = current_state.M_hat_A - F_M / jnp.maximum(jnp.abs(dF_dM), eps)
        new_Xi_P = current_state.Xi_A_P - F_Xi_P / jnp.maximum(jnp.abs(dF_dXi_P), eps)
        new_Xi_I = current_state.Xi_A_I - F_Xi_I / jnp.maximum(jnp.abs(dF_dXi_I), eps)
        new_Xi_F = current_state.Xi_A_F - F_Xi_F / jnp.maximum(jnp.abs(dF_dXi_F), eps)

        current_state = HBarState(
            delta_A=current_state.delta_A,
            sigma_A=current_state.sigma_A,
            alpha_A=current_state.alpha_A,
            M_hat_A=new_M,
            Xi_A_P=new_Xi_P,
            Xi_A_I=new_Xi_I,
            Xi_A_F=new_Xi_F,
        )

    return current_state


# ---------------------------------------------------------------------------
# Boundary enforcement (forward invariance)
# ---------------------------------------------------------------------------


def enforce_boundaries(
    state: HBarState,
    constants: HBarConstants,
) -> HBarState:
    """Project state back onto the valid state space X.

    Implements Proposition 3.2 (Forward Invariance): all state variables
    must remain in their valid ranges after each integration step.

    Valid ranges:
        δ_A ∈ [0, K_δ]   (parametric depth)
        σ_A ∈ [0, 1]     (schema coherence)
        α_A ∈ [0, 1]     (attentional fidelity)
        M̂_A ∈ [0, 1]     (self-model)
        Ξ_A ∈ [0, 1]     (executive control, all 3 components)

    Args:
        state: State after integration step.
        constants: ODE parameters (for K_δ bound).

    Returns:
        Projected state within valid bounds.
    """
    K_delta = constants.K_delta

    return HBarState(
        delta_A=jnp.clip(state.delta_A, 0.0, K_delta),
        sigma_A=jnp.clip(state.sigma_A, 0.0, 1.0),
        alpha_A=jnp.clip(state.alpha_A, 0.0, 1.0),
        M_hat_A=jnp.clip(state.M_hat_A, 0.0, 1.0),
        Xi_A_P=jnp.clip(state.Xi_A_P, 0.0, 1.0),
        Xi_A_I=jnp.clip(state.Xi_A_I, 0.0, 1.0),
        Xi_A_F=jnp.clip(state.Xi_A_F, 0.0, 1.0),
    )


# ---------------------------------------------------------------------------
# Main IMEX step function
# ---------------------------------------------------------------------------


def step_hbar_system(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
    h: float = 0.01,
) -> HBarState:
    """Perform one IMEX RK4 step of the H-Bar ODE system.

    Algorithm 3.1:
        1. Explicit RK4 for fast variables (δ_A, σ_A, α_A)
        2. Implicit Backward Euler for slow variables (M̂_A, Ξ_A)
        3. Boundary enforcement (projection onto valid state space)

    Args:
        state: Current cognitive state.
        inputs: External signals from the training environment.
        constants: Fixed ODE parameters.
        h: Step size (default: 0.01).

    Returns:
        New HBarState after one integration step.
    """
    # Step 1: Explicit RK4 for fast variables
    state_after_fast = _rk4_fast_step(state, inputs, constants, h)

    # Step 2: Implicit Backward Euler for slow variables
    state_after_slow = _backward_euler_slow_step(state_after_fast, inputs, constants, h)

    # Step 3: Enforce boundary constraints
    new_state = enforce_boundaries(state_after_slow, constants)

    return new_state


# ---------------------------------------------------------------------------
# Adaptive step size control
# ---------------------------------------------------------------------------


def estimate_step_error(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
    h: float,
) -> Tuple[HBarState, float, float]:
    """Estimate local truncation error using step doubling.

    Compares a full step (size h) with two half-steps (size h/2).
    The difference provides an estimate of the local truncation error.

    Args:
        state: Current state.
        inputs: External signals.
        constants: ODE parameters.
        h: Proposed step size.

    Returns:
        Tuple of:
            - more_accurate_state: Result from two half-steps (more accurate)
            - error: Maximum absolute difference between full and half-step results
            - h_new: Recommended step size for next iteration
    """
    # Full step with size h
    full_step = step_hbar_system(state, inputs, constants, h)

    # Two half-steps with size h/2
    half_step_1 = step_hbar_system(state, inputs, constants, h / 2.0)
    half_step_2 = step_hbar_system(half_step_1, inputs, constants, h / 2.0)

    # Error estimate: max norm of difference
    delta_err = jnp.abs(full_step.delta_A - half_step_2.delta_A)
    sigma_err = jnp.abs(full_step.sigma_A - half_step_2.sigma_A)
    alpha_err = jnp.abs(full_step.alpha_A - half_step_2.alpha_A)
    M_err = jnp.abs(full_step.M_hat_A - half_step_2.M_hat_A)
    Xi_P_err = jnp.abs(full_step.Xi_A_P - half_step_2.Xi_A_P)
    Xi_I_err = jnp.abs(full_step.Xi_A_I - half_step_2.Xi_A_I)
    Xi_F_err = jnp.abs(full_step.Xi_A_F - half_step_2.Xi_A_F)

    error = jnp.maximum(
        jnp.maximum(jnp.maximum(delta_err, sigma_err), jnp.maximum(alpha_err, M_err)),
        jnp.maximum(jnp.maximum(Xi_P_err, Xi_I_err), Xi_F_err),
    )

    # Adjust step size based on error
    tol = 1e-6
    safety = 0.9

    if error > tol:
        # Reduce step size
        h_new = h * safety * jnp.maximum(0.5, (tol / (error + 1e-30)) ** 0.25)
    elif error < tol / 10.0:
        # Increase step size
        h_new = h * safety * jnp.minimum(2.0, (tol / (error + 1e-30)) ** 0.25)
    else:
        h_new = h

    # Use the more accurate half-step result
    return half_step_2, error, h_new


def adaptive_step_hbar_system(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
    h: float = 0.01,
    tol: float = 1e-6,
) -> Tuple[HBarState, float, bool]:
    """Perform an adaptive step with error control.

    If the error exceeds the tolerance, the step is retried with a smaller
    step size. This ensures numerical accuracy while maintaining efficiency.

    Args:
        state: Current state.
        inputs: External signals.
        constants: ODE parameters.
        h: Initial proposed step size.
        tol: Error tolerance (default: 1e-6).

    Returns:
        Tuple of:
            - new_state: State after successful step
            - error: Actual error achieved
            - accepted: Whether the step was accepted on first try
    """
    # Try with current step size
    new_state, error, h_new = estimate_step_error(state, inputs, constants, h)

    if error <= tol:
        return new_state, error, True
    else:
        # Retry with smaller step size
        new_state, error, _ = estimate_step_error(state, inputs, constants, h_new)
        return new_state, error, False


# ---------------------------------------------------------------------------
# Jacobian condition number monitoring
# ---------------------------------------------------------------------------


def check_jacobian_condition(
    state: HBarState,
    inputs: HBarInputs,
    constants: HBarConstants,
    threshold: float = 1000.0,
) -> Tuple[bool, float]:
    """Check the condition number of the Jacobian (Proposition 3.5).

    A high condition number indicates numerical instability in the ODE system.
    This function is used for diagnostic monitoring during training.

    Args:
        state: Current state.
        inputs: External signals.
        constants: ODE parameters.
        threshold: Maximum acceptable condition number (default: 1000).

    Returns:
        Tuple of:
            - is_stable: True if condition number < threshold
            - cond_number: Actual condition number
    """

    def vector_field_fn(s: HBarState) -> HBarState:
        return hbar_vector_field(s, inputs, constants)

    # Compute Jacobian
    J = jax.jacrev(vector_field_fn)(state)

    # Flatten Jacobian to 2D array
    from hbar.core.dynamics import state_to_array
    state_arr = state_to_array(state)
    n = len(state_arr)

    # Compute condition number
    # For small matrices, we can use SVD-based condition number
    try:
        # Add small regularization to avoid singular matrix issues
        J_reg = J + jnp.eye(n) * 1e-10
        cond_number = jnp.linalg.cond(J_reg)
    except:
        cond_number = jnp.array(1e10)  # Assume unstable on error

    is_stable = cond_number < threshold
    return is_stable, cond_number


# ---------------------------------------------------------------------------
# Multi-step integration (for training loops)
# ---------------------------------------------------------------------------


def integrate_hbar_trajectory(
    initial_state: HBarState,
    inputs_sequence: HBarInputs,
    constants: HBarConstants,
    h: float = 0.01,
    n_steps: int = 100,
) -> HBarState:
    """Integrate the H-Bar system for multiple steps.

    Uses jax.lax.scan for efficient JIT-compiled multi-step integration.

    Args:
        initial_state: Starting state.
        inputs_sequence: Constant inputs for all steps (or use scan for varying).
        constants: ODE parameters.
        h: Step size.
        n_steps: Number of integration steps.

    Returns:
        Final state after n_steps.
    """

    def step_fn(carry, _):
        state = carry
        new_state = step_hbar_system(state, inputs_sequence, constants, h)
        return new_state, new_state

    final_state, _ = jax.lax.scan(step_fn, initial_state, None, length=n_steps)
    return final_state
