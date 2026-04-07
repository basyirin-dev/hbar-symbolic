"""H-Bar Core — ODE dynamics and integration engine.

This package provides the mathematical core of the H-Bar framework:
  - dynamics.py: Coupled ODE system (Eqs. 14, 28, 29, 33, 36)
  - integrator.py: IMEX Runge-Kutta integrator with adaptive step size

The ODE system models cognitive state evolution during compositional learning,
with 7 state variables split into fast (δ_A, σ_A, α_A) and slow (M̂_A, Ξ_A)
subsystems.

Usage:
    from hbar.core import (
        HBarState, HBarInputs, HBarConstants,
        init_hbar_state, hbar_vector_field,
        step_hbar_system, adaptive_step_hbar_system,
    )
"""

from hbar.core.dynamics import (
    HBarState,
    HBarInputs,
    HBarConstants,
    hbar_vector_field,
    fast_vector_field,
    slow_vector_field,
    init_hbar_state,
    state_to_array,
    array_to_state,
)

from hbar.core.integrator import (
    step_hbar_system,
    adaptive_step_hbar_system,
    estimate_step_error,
    enforce_boundaries,
    check_jacobian_condition,
    integrate_hbar_trajectory,
)

__all__ = [
    # State and parameters
    "HBarState",
    "HBarInputs",
    "HBarConstants",
    # Vector field
    "hbar_vector_field",
    "fast_vector_field",
    "slow_vector_field",
    # State utilities
    "init_hbar_state",
    "state_to_array",
    "array_to_state",
    # Integrator
    "step_hbar_system",
    "adaptive_step_hbar_system",
    "estimate_step_error",
    "enforce_boundaries",
    "check_jacobian_condition",
    "integrate_hbar_trajectory",
]
