"""Cognitive State Manager — bridges training metrics and H-Bar ODE dynamics.

This module provides the `CognitiveManager` class that lives alongside the
`TrainState` during training, managing the cognitive state evolution.

The manager is purely functional (using JAX pytrees) so it can be integrated
into a JIT-compiled training loop.

Usage:
    manager = CognitiveManager()
    hbar_state = manager.init_state(key)

    # In training loop:
    metrics = {"loss": loss, "gca": gca, "rga": rga, "ac": ac, ...}
    hbar_state = manager.step(hbar_state, metrics, constants, dt)
    modulators = manager.get_modulators(hbar_state)
"""

from typing import Any, Dict, Tuple

import flax.struct
import jax
import jax.numpy as jnp

from hbar.core.dynamics import (
    HBarConstants,
    HBarInputs,
    HBarState,
    init_hbar_state,
)
from hbar.core.integrator import step_hbar_system


@flax.struct.dataclass
class CognitiveManager:
    """Manages the H-Bar cognitive state during training.

    This class provides a functional interface for updating the cognitive
    state based on training metrics and extracting modulators for the
    optimizer.

    The manager is stateless — all state is passed explicitly through
    the `hbar_state` parameter, making it compatible with JAX
    transformations (jit, vmap, grad).

    Attributes:
        None (all configuration is through method parameters).
    """

    def init_state(self, key: jax.Array, sigma_tilde_init: float = 0.27) -> HBarState:
        """Initialize the H-Bar cognitive state.

        Args:
            key: JAX PRNGKey.
            sigma_tilde_init: Initial fused signal value (default: 0.27,
                the baseline σ-trap starting point).

        Returns:
            HBarState initialized at the baseline starting point.
        """
        return init_hbar_state(key, sigma_tilde_init)

    def metrics_to_inputs(
        self,
        metrics: Dict[str, jax.Array],
    ) -> HBarInputs:
        """Map raw training metrics to HBar ODE inputs.

        This function translates neural network training signals into
        the cognitive state input space defined by the H-Bar model.

        Args:
            metrics: Dictionary containing training metrics:
                - sigma_tilde: Fused signal from signal extraction engine
                - sigma_hat: Ground-truth OOD/ID ratio (at evaluation)
                - P_A: Principled structure availability (from curriculum)
                - C_A: Training signal strength (from [BOS] stability)
                - Omega_AI: AI-bypass risk (from surface vs structural accuracy gap)
                - R_surface: Surface reward (from ID accuracy)
                - domain_frontier: Curriculum difficulty level

        Returns:
            HBarInputs ready to be passed to the ODE integrator.
        """
        return HBarInputs(
            sigma_tilde=metrics.get("sigma_tilde", jnp.array(0.0)),
            sigma_hat=metrics.get("sigma_hat", jnp.array(0.0)),
            P_A=metrics.get("P_A", jnp.array(0.5)),
            C_A=metrics.get("C_A", jnp.array(0.5)),
            Omega_AI=metrics.get("Omega_AI", jnp.array(0.5)),
            R_surface=metrics.get("R_surface", jnp.array(0.5)),
            domain_frontier=metrics.get("domain_frontier", jnp.array(0.5)),
        )

    def step(
        self,
        hbar_state: HBarState,
        metrics: Dict[str, jax.Array],
        constants: HBarConstants,
        dt: float,
    ) -> HBarState:
        """Advance the cognitive state by one training step.

        This is the main update function called in the training loop.
        It maps metrics to inputs, then integrates the ODE system.

        Args:
            hbar_state: Current cognitive state.
            metrics: Training metrics from the current step.
            constants: ODE parameters.
            dt: Integration timestep (typically matches training step size).

        Returns:
            Updated HBarState after integration.
        """
        inputs = self.metrics_to_inputs(metrics)
        new_state = step_hbar_system(hbar_state, inputs, constants, dt)
        return new_state

    def get_modulators(
        self,
        hbar_state: HBarState,
    ) -> Dict[str, jax.Array]:
        """Extract training modulators from the current cognitive state.

        These modulators are used by the optimizer to adjust training
        dynamics based on the current cognitive state.

        Args:
            hbar_state: Current cognitive state.

        Returns:
            Dictionary containing:
                - sigma_A: Current schema coherence (for loss weighting)
                - alpha_A: Current attentional fidelity (for LR scaling)
                - crystallization_potential: α_A · C_A (for phase detection)
                - is_attention_limited: Whether α_A < 0.3 (Phase 1 state)
                - schema_loss_weight: (1 - σ_A) for up-weighting compositional loss
        """
        sigma = hbar_state.sigma_A
        alpha = hbar_state.alpha_A

        # Schema loss weight: up-weight compositional training when σ_A is low
        schema_loss_weight = jnp.clip(1.0 - sigma, 0.1, 1.0)

        # Learning rate modulation: scale by attentional fidelity
        lr_modulator = 1.0 + 0.5 * alpha  # ±25% modulation

        return {
            "sigma_A": sigma,
            "alpha_A": alpha,
            "schema_loss_weight": schema_loss_weight,
            "lr_modulator": lr_modulator,
        }

    def check_phase_transition(
        self,
        hbar_state: HBarState,
        inputs: HBarInputs,
        constants: HBarConstants,
    ) -> Dict[str, Any]:
        """Check for Phase 1 → Phase 2 transition (crystallization).

        Phase 2 entry occurs when:
        1. σ_A > σ_critical (schema coherence threshold)
        2. α_A · C_A > 0.5 (crystallization potential threshold)

        Args:
            hbar_state: Current cognitive state.
            inputs: Current ODE inputs.
            constants: ODE parameters.

        Returns:
            Dictionary containing:
                - current_phase: 1 or 2
                - sigma_A: Current schema coherence
                - crystallization_potential: α_A · C_A
                - phase_2_ready: Boolean indicating readiness for Phase 2
        """
        sigma = hbar_state.sigma_A
        alpha = hbar_state.alpha_A
        C_A = inputs.C_A
        sigma_critical = constants.sigma_critical

        # Crystallization potential
        cryst_potential = alpha * C_A

        # Phase 2 entry conditions
        sigma_above_threshold = sigma > sigma_critical
        cryst_above_threshold = cryst_potential > 0.5
        phase_2_ready = sigma_above_threshold & cryst_above_threshold

        current_phase = jnp.where(phase_2_ready, 2, 1)

        return {
            "current_phase": int(current_phase),
            "sigma_A": float(sigma),
            "alpha_A": float(alpha),
            "crystallization_potential": float(cryst_potential),
            "phase_2_ready": bool(phase_2_ready),
            "sigma_above_threshold": bool(sigma_above_threshold),
            "cryst_above_threshold": bool(cryst_above_threshold),
        }


# ---------------------------------------------------------------------------
# Convenience functions (functional API without class instantiation)
# ---------------------------------------------------------------------------


def create_manager() -> CognitiveManager:
    """Create a new CognitiveManager instance."""
    return CognitiveManager()


def update_cognitive_state(
    hbar_state: HBarState,
    metrics: Dict[str, jax.Array],
    constants: HBarConstants,
    dt: float,
) -> HBarState:
    """Functional convenience wrapper for CognitiveManager.step().

    Args:
        hbar_state: Current cognitive state.
        metrics: Training metrics.
        constants: ODE parameters.
        dt: Integration timestep.

    Returns:
        Updated HBarState.
    """
    manager = CognitiveManager()
    return manager.step(hbar_state, metrics, constants, dt)


def extract_modulators(
    hbar_state: HBarState,
) -> Dict[str, jax.Array]:
    """Functional convenience wrapper for CognitiveManager.get_modulators().

    Args:
        hbar_state: Current cognitive state.

    Returns:
        Dictionary of modulator values.
    """
    manager = CognitiveManager()
    return manager.get_modulators(hbar_state)
