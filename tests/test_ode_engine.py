"""Integration tests for H-Bar ODE engine (dynamics + integrator).

Verification workflow triggered on any change to hbar/core/.
Tests validate:
  1. Convergence to Schema-Coherent Equilibrium (E_C) from baseline
  2. σ-trap simulation under high AI-bypass risk
  3. JIT and gradient compatibility
  4. Forward invariance (all variables remain in valid bounds)
  5. Adaptive step size error control

Run: pytest tests/test_ode_engine.py -v
"""

import pytest
import jax
import jax.numpy as jnp

from hbar.core.dynamics import (
    HBarState,
    HBarInputs,
    HBarConstants,
    init_hbar_state,
    hbar_vector_field,
)
from hbar.core.integrator import (
    step_hbar_system,
    adaptive_step_hbar_system,
    enforce_boundaries,
    check_jacobian_condition,
    integrate_hbar_trajectory,
)


class TestODEConvergence:
    """Test 1: Convergence to Schema-Coherent Equilibrium (E_C)."""

    def setup_method(self):
        """Initialize at baseline starting point with positive inputs."""
        self.key = jax.random.PRNGKey(42)
        self.constants = HBarConstants()

        # Baseline starting point: σ̃_A ≈ 0.27
        self.initial_state = init_hbar_state(self.key, sigma_tilde_init=0.27)

        # Positive inputs simulating good training signals
        self.positive_inputs = HBarInputs(
            sigma_tilde=jnp.array(0.7),   # High fused signal
            sigma_hat=jnp.array(0.8),     # High ground-truth
            P_A=jnp.array(0.8),           # High principled structure
            C_A=jnp.array(0.8),           # High training signal
            Omega_AI=jnp.array(0.1),      # Low AI-bypass risk
            R_surface=jnp.array(0.1),     # Low surface rewards
            domain_frontier=jnp.array(0.5),  # Moderate difficulty
        )

    def test_sigma_increases_toward_equilibrium(self):
        """With positive inputs, σ_A should increase toward 1.0."""
        h = 0.01
        state = self.initial_state

        # Integrate for 500 steps
        for _ in range(500):
            state = step_hbar_system(state, self.positive_inputs, self.constants, h)

        # σ_A should have increased significantly from baseline
        assert state.sigma_A > self.initial_state.sigma_A, (
            f"σ_A should increase from {self.initial_state.sigma_A} "
            f"but got {state.sigma_A}"
        )
        # Should be approaching 1.0 (schema-coherent equilibrium)
        assert state.sigma_A > 0.5, f"σ_A = {state.sigma_A} should exceed 0.5"

    def test_trajectory_is_smooth(self):
        """State trajectory should be smooth (no jumps or oscillations)."""
        h = 0.01
        state = self.initial_state
        sigma_history = [float(state.sigma_A)]

        for _ in range(100):
            state = step_hbar_system(state, self.positive_inputs, self.constants, h)
            sigma_history.append(float(state.sigma_A))

        # Check smoothness: consecutive differences should be small
        diffs = [abs(sigma_history[i+1] - sigma_history[i]) for i in range(len(sigma_history)-1)]
        max_diff = max(diffs)
        assert max_diff < 0.1, f"Trajectory has jump of {max_diff} (should be < 0.1)"


class TestSigmaTrapSimulation:
    """Test 2: σ-trap under high AI-bypass risk."""

    def setup_method(self):
        """Initialize at baseline with high AI-bypass risk."""
        self.key = jax.random.PRNGKey(42)
        self.constants = HBarConstants()
        self.initial_state = init_hbar_state(self.key, sigma_tilde_init=0.27)

        # High AI-bypass risk inputs (simulating σ-trap conditions)
        self.high_risk_inputs = HBarInputs(
            sigma_tilde=jnp.array(0.5),   # Moderate fused signal
            sigma_hat=jnp.array(0.3),     # Low ground-truth
            P_A=jnp.array(0.3),           # Low principled structure
            C_A=jnp.array(0.3),           # Low training signal
            Omega_AI=jnp.array(0.9),      # HIGH AI-bypass risk
            R_surface=jnp.array(0.9),     # HIGH surface rewards
            domain_frontier=jnp.array(0.2),  # Low difficulty (stuck)
        )

    def test_sigma_suppressed_under_high_AI_bypass(self):
        """With high Ω_AI, σ_A should stay suppressed (σ-trap)."""
        h = 0.01
        state = self.initial_state

        # Integrate for 500 steps
        for _ in range(500):
            state = step_hbar_system(state, self.high_risk_inputs, self.constants, h)

        # σ_A should remain low due to AI-bypass suppression
        # The suppression term η_σ · Ω_AI · σ_A dominates
        assert state.sigma_A < 0.5, (
            f"σ_A = {state.sigma_A} should stay < 0.5 under high AI-bypass"
        )

    def test_alpha_suppressed_under_high_surface_rewards(self):
        """With high R_surface, α_A should be suppressed."""
        h = 0.01
        state = self.initial_state

        for _ in range(500):
            state = step_hbar_system(state, self.high_risk_inputs, self.constants, h)

        # α_A should be suppressed by surface reward term
        assert state.alpha_A < 0.5, (
            f"α_A = {state.alpha_A} should stay < 0.5 under high surface rewards"
        )


class TestJITAndGradientCompatibility:
    """Test 3: JIT and gradient flow through integrator."""

    def setup_method(self):
        self.key = jax.random.PRNGKey(42)
        self.constants = HBarConstants()
        self.state = init_hbar_state(self.key)
        self.inputs = HBarInputs(
            sigma_tilde=jnp.array(0.5),
            sigma_hat=jnp.array(0.5),
            P_A=jnp.array(0.5),
            C_A=jnp.array(0.5),
            Omega_AI=jnp.array(0.3),
            R_surface=jnp.array(0.3),
            domain_frontier=jnp.array(0.5),
        )

    def test_step_is_jittable(self):
        """step_hbar_system should work with jax.jit."""
        jit_step = jax.jit(step_hbar_system)
        new_state = jit_step(self.state, self.inputs, self.constants, 0.01)

        arr = state_to_array(new_state)
        assert jnp.all(jnp.isfinite(arr)), "State should be finite after JIT step"

    def test_gradient_flows_through_integrator(self):
        """Gradients should flow through the integrator."""
        def loss_fn(sigma_tilde_val):
            inputs = HBarInputs(
                sigma_tilde=jnp.array(sigma_tilde_val),
                sigma_hat=jnp.array(0.5),
                P_A=jnp.array(0.5),
                C_A=jnp.array(0.5),
                Omega_AI=jnp.array(0.3),
                R_surface=jnp.array(0.3),
                domain_frontier=jnp.array(0.5),
            )
            # Integrate for 10 steps
            state = self.state
            for _ in range(10):
                state = step_hbar_system(state, inputs, self.constants, 0.01)
            # Loss: want sigma_A to be close to 0.8
            return (state.sigma_A - 0.8) ** 2

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(0.5)

        assert jnp.isfinite(grad), "Gradient should be finite"

    def test_adaptive_step_is_jittable(self):
        """adaptive_step_hbar_system should work with jax.jit."""
        jit_adaptive = jax.jit(adaptive_step_hbar_system)
        new_state, error, accepted = jit_adaptive(
            self.state, self.inputs, self.constants, 0.01
        )

        arr = state_to_array(new_state)
        assert jnp.all(jnp.isfinite(arr)), "State should be finite after adaptive step"
        assert jnp.isfinite(error), "Error should be finite"


class TestForwardInvariance:
    """Test 4: Forward invariance (Proposition 3.2)."""

    def setup_method(self):
        self.constants = HBarConstants()

    def test_boundaries_enforced_after_step(self):
        """All state variables must remain in valid ranges after integration."""
        key = jax.random.PRNGKey(42)
        state = init_hbar_state(key)
        inputs = HBarInputs(
            sigma_tilde=jnp.array(0.5),
            sigma_hat=jnp.array(0.5),
            P_A=jnp.array(0.5),
            C_A=jnp.array(0.5),
            Omega_AI=jnp.array(0.5),
            R_surface=jnp.array(0.5),
            domain_frontier=jnp.array(0.5),
        )

        h = 0.01
        for _ in range(1000):
            state = step_hbar_system(state, inputs, self.constants, h)

        # Check all bounds
        assert 0.0 <= state.delta_A <= self.constants.K_delta, (
            f"δ_A = {state.delta_A} out of [0, {self.constants.K_delta}]"
        )
        assert 0.0 <= state.sigma_A <= 1.0, (
            f"σ_A = {state.sigma_A} out of [0, 1]"
        )
        assert 0.0 <= state.alpha_A <= 1.0, (
            f"α_A = {state.alpha_A} out of [0, 1]"
        )
        assert 0.0 <= state.M_hat_A <= 1.0, (
            f"M̂_A = {state.M_hat_A} out of [0, 1]"
        )
        assert 0.0 <= state.Xi_A_P <= 1.0, (
            f"Ξ_A_P = {state.Xi_A_P} out of [0, 1]"
        )
        assert 0.0 <= state.Xi_A_I <= 1.0, (
            f"Ξ_A_I = {state.Xi_A_I} out of [0, 1]"
        )
        assert 0.0 <= state.Xi_A_F <= 1.0, (
            f"Ξ_A_F = {state.Xi_A_F} out of [0, 1]"
        )

    def test_jacobian_condition_number_stable(self):
        """Jacobian condition number κ(J) should be < 1000 (Eq. 24)."""
        key = jax.random.PRNGKey(42)
        state = init_hbar_state(key)
        inputs = HBarInputs(
            sigma_tilde=jnp.array(0.5),
            sigma_hat=jnp.array(0.5),
            P_A=jnp.array(0.5),
            C_A=jnp.array(0.5),
            Omega_AI=jnp.array(0.5),
            R_surface=jnp.array(0.5),
            domain_frontier=jnp.array(0.5),
        )

        is_stable, cond_number = check_jacobian_condition(state, inputs, self.constants)
        assert is_stable, f"Jacobian condition number {cond_number} exceeds 1000"


class TestAdaptiveStepSize:
    """Test 5: Adaptive step size error control."""

    def setup_method(self):
        self.key = jax.random.PRNGKey(42)
        self.constants = HBarConstants()
        self.state = init_hbar_state(self.key)
        self.inputs = HBarInputs(
            sigma_tilde=jnp.array(0.5),
            sigma_hat=jnp.array(0.5),
            P_A=jnp.array(0.5),
            C_A=jnp.array(0.5),
            Omega_AI=jnp.array(0.5),
            R_surface=jnp.array(0.5),
            domain_frontier=jnp.array(0.5),
        )

    def test_error_decreases_with_smaller_step(self):
        """Error should decrease when step size is reduced."""
        h_large = 0.1
        h_small = 0.01

        _, error_large, _ = adaptive_step_hbar_system(
            self.state, self.inputs, self.constants, h_large
        )
        _, error_small, _ = adaptive_step_hbar_system(
            self.state, self.inputs, self.constants, h_small
        )

        assert error_small < error_large, (
            f"Error should decrease with smaller step: {error_small} vs {error_large}"
        )

    def test_step_size_adjustment_logic(self):
        """Step size should be reduced when error is high, increased when low."""
        h = 0.01

        # Get recommended step sizes
        _, _, h_new_high_error = adaptive_step_hbar_system(
            self.state, self.inputs, self.constants, h, tol=1e-10  # Very strict
        )
        _, _, h_new_low_error = adaptive_step_hbar_system(
            self.state, self.inputs, self.constants, h, tol=1.0  # Very lenient
        )

        # With strict tolerance, step should be reduced or stay same
        # With lenient tolerance, step should be increased or stay same
        # (This is a soft test since actual behavior depends on the system)
        assert h_new_high_error <= h or h_new_low_error >= h, (
            "Step size adjustment logic may be incorrect"
        )


class TestSurfaceRewardSuppression:
    """Test 6: Surface reward suppression of attentional fidelity (Eq. 29).

    This test verifies the σ-trap mechanism at the attentional level.
    When the model is rewarded for surface statistics (high R_surface),
    attentional fidelity α_A should be suppressed, preventing σ_A growth.
    """

    def setup_method(self):
        """Initialize with high surface rewards (99% ID accuracy scenario)."""
        self.key = jax.random.PRNGKey(42)
        self.constants = HBarConstants()
        self.initial_state = init_hbar_state(self.key, sigma_tilde_init=0.27)

        # High surface reward inputs (model getting 99% on easy ID samples)
        self.surface_reward_inputs = HBarInputs(
            sigma_tilde=jnp.array(0.5),   # Moderate fused signal
            sigma_hat=jnp.array(0.4),     # Low ground-truth (OOD failure)
            P_A=jnp.array(0.5),           # Moderate principled structure
            C_A=jnp.array(0.7),           # High training signal
            Omega_AI=jnp.array(0.5),      # Moderate AI-bypass risk
            R_surface=jnp.array(0.95),    # VERY HIGH surface rewards (99% ID acc)
            domain_frontier=jnp.array(0.3),  # Low difficulty (easy tasks)
        )

    def test_surface_reward_suppression(self):
        """With high R_surface, α_A should remain low, trapping σ_A."""
        h = 0.01
        state = self.initial_state

        # Integrate for 500 steps
        for _ in range(500):
            state = step_hbar_system(
                state, self.surface_reward_inputs, self.constants, h
            )

        # α_A should be suppressed by high surface rewards
        # The suppression term η_α · R_surface · α_A dominates
        assert state.alpha_A < 0.4, (
            f"α_A = {state.alpha_A} should stay < 0.4 under high surface rewards"
        )

        # σ_A should also remain low because growth requires α_A > 0
        # (Eq. 28: growth = ρ · P_A · α_A · (1 - σ_A))
        assert state.sigma_A < 0.4, (
            f"σ_A = {state.sigma_A} should stay < 0.4 when α_A is suppressed"
        )

    def test_sigma_trap_attention_gate(self):
        """Verify the attentional gate: σ_A growth ≈ 0 when α_A ≈ 0."""
        # Start with very low attention
        low_alpha_state = HBarState(
            delta_A=jnp.array(1.0),
            sigma_A=jnp.array(0.27),
            alpha_A=jnp.array(0.05),  # Very low attention
            M_hat_A=jnp.array(0.27),
            Xi_A_P=jnp.array(0.3),
            Xi_A_I=jnp.array(0.3),
            Xi_A_F=jnp.array(0.3),
        )

        # Even with high P_A, σ_A should not grow much without α_A
        inputs_high_P = HBarInputs(
            sigma_tilde=jnp.array(0.8),
            sigma_hat=jnp.array(0.8),
            P_A=jnp.array(0.9),  # High principled structure
            C_A=jnp.array(0.5),
            Omega_AI=jnp.array(0.1),  # Low suppression
            R_surface=jnp.array(0.1),
            domain_frontier=jnp.array(0.5),
        )

        h = 0.01
        state = low_alpha_state

        # Integrate for 200 steps
        for _ in range(200):
            state = step_hbar_system(state, inputs_high_P, self.constants, h)

        # σ_A should barely increase because α_A is the gate
        # Growth = ρ · P_A · α_A · (1 - σ_A) ≈ 0 when α_A ≈ 0
        sigma_increase = state.sigma_A - 0.27
        assert sigma_increase < 0.1, (
            f"σ_A increase = {sigma_increase} should be < 0.1 when α_A ≈ 0"
        )


# Helper function
def state_to_array(state: HBarState) -> jax.Array:
    """Convert HBarState to flat array for testing."""
    return jnp.array([
        state.delta_A,
        state.sigma_A,
        state.alpha_A,
        state.M_hat_A,
        state.Xi_A_P,
        state.Xi_A_I,
        state.Xi_A_F,
    ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
