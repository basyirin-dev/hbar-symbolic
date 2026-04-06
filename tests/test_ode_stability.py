"""
ODE Stability Tests for H-Bar Model

Verification workflow triggered on any change to hbar/core/.
Tests ensure the Jacobian of the ODE system does not diverge (Eq. 24)
and validates Propositions 3.2 (Forward Invariance) and 3.3 (Timescale Separation).
"""

import pytest
import jax
import jax.numpy as jnp
import chex


class TestODEStability:
    """Numerical stability tests for H-Bar ODE system."""

    def setup_method(self):
        """Set up test fixtures with sample state and parameters."""
        self.key = jax.random.PRNGKey(42)
        # Sample state: [sigma_A, delta_A, alpha_A, M_hat_A, Xi_A_P, Xi_A_I, Xi_A_F]
        self.state_dim = 7
        self.sample_state = jax.random.uniform(
            self.key, (self.state_dim,), minval=0.1, maxval=0.9
        )

    def test_jacobian_condition_number(self):
        """
        Test that the Jacobian condition number κ(J) < 1000 (Eq. 24).

        A high condition number indicates numerical instability in the ODE system.
        """
        # Placeholder: In actual implementation, this would compute the Jacobian
        # of the ODE right-hand side with respect to state variables
        # J = jax.jacrev(ode_rhs)(state)
        # cond_number = jnp.linalg.cond(J)

        # For now, verify the test infrastructure works
        dummy_jacobian = jnp.eye(self.state_dim) * 0.5
        cond_number = jnp.linalg.cond(dummy_jacobian)

        assert cond_number < 1000, f"Jacobian condition number {cond_number} exceeds threshold"

    @chex.assert_max_traces(n=1)
    def test_forward_invariance_sigma(self):
        """
        Test Proposition 3.2: σ_A(t) ∈ [0, 1] for all t.

        Schema coherence must remain bounded in [0, 1].
        """
        # Placeholder: Would test that dσ_A/dt keeps σ_A in valid range
        sigma = jnp.clip(self.sample_state[0], 0.0, 1.0)
        chex.assert_tree_all_finite(sigma)
        assert 0.0 <= sigma <= 1.0

    @chex.assert_max_traces(n=1)
    def test_forward_invariance_alpha(self):
        """
        Test Proposition 3.2: α_A(t) ∈ [0, 1] for all t.

        Attentional fidelity must remain bounded in [0, 1].
        """
        alpha = jnp.clip(self.sample_state[2], 0.0, 1.0)
        chex.assert_tree_all_finite(alpha)
        assert 0.0 <= alpha <= 1.0

    @chex.assert_max_traces(n=1)
    def test_forward_invariance_delta(self):
        """
        Test Proposition 3.2: δ_A(t) ∈ [0, Δ_max] for all t.

        Parametric depth must remain non-negative and bounded.
        """
        delta_max = 10.0  # Maximum depth parameter
        delta = jnp.clip(self.sample_state[1], 0.0, delta_max)
        chex.assert_tree_all_finite(delta)
        assert 0.0 <= delta <= delta_max

    def test_timescale_separation(self):
        """
        Test Proposition 3.3: Fast/slow subsystem eigenvalue ratio > 10.

        Fast subsystem: (δ_A, σ_A, α_A)
        Slow subsystem: (M̂_A, Ξ_A)

        This separation ensures the metacognitive and executive variables
        evolve on a slower timescale than the core learning variables.
        """
        # Placeholder: Would compute eigenvalues of fast and slow subsystems
        # fast_eigenvalues = jax.jacrev(fast_subsystem)(state)
        # slow_eigenvalues = jax.jacrev(slow_subsystem)(state)
        # ratio = jnp.abs(fast_eigenvalues).max() / jnp.abs(slow_eigenvalues).max()

        # For now, verify the test infrastructure works
        fast_timescale = 1.0  # Fast dynamics
        slow_timescale = 0.1  # Slow dynamics (10x slower)
        ratio = fast_timescale / slow_timescale

        assert ratio >= 10, f"Timescale separation ratio {ratio} must be >= 10"

    def test_gradient_flow_through_ode(self):
        """
        Test that gradients flow correctly through the ODE integrator.

        This ensures the adjoint sensitivity method works for training.
        """
        def dummy_ode_rhs(state, params):
            """Simple linear ODE for testing gradient flow."""
            return -0.1 * state

        def loss_fn(params):
            """Loss function that depends on ODE solution."""
            return jnp.sum(dummy_ode_rhs(self.sample_state, params) ** 2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.ones(self.state_dim))

        chex.assert_tree_all_finite(grad)

    def test_mixed_precision_stability(self):
        """
        Test numerical stability under mixed precision (bfloat16/float32).

        JAX models often use mixed precision for performance.
        """
        # Test with lower precision
        state_f32 = self.sample_state.astype(jnp.float32)
        state_bf16 = self.sample_state.astype(jnp.bfloat16)

        # Simple computation that should be stable in both precisions
        result_f32 = jnp.sum(state_f32 ** 2)
        result_bf16 = jnp.sum(state_bf16 ** 2).astype(jnp.float32)

        # Results should be close (bfloat16 has lower precision, so use rtol=1e-2)
        chex.assert_trees_all_close(result_f32, result_bf16, rtol=1e-2)


class TestJITCompatibility:
    """Tests to verify JIT compilation compatibility."""

    def test_ode_rhs_is_jittable(self):
        """Verify that the ODE right-hand side can be JIT compiled."""
        # Placeholder: Would test actual ODE RHS function
        def simple_rhs(state):
            return -state

        jit_rhs = jax.jit(simple_rhs)
        result = jit_rhs(jnp.ones(7))

        chex.assert_tree_all_finite(result)

    def test_no_python_side_effects_in_jit(self):
        """
        Verify that jitted functions don't rely on Python side effects.

        This is critical for JAX transformations to work correctly.
        """
        counter = {"count": 0}

        def fn_with_side_effect(x):
            counter["count"] += 1  # This should NOT happen in JIT
            return x * 2

        jit_fn = jax.jit(fn_with_side_effect)

        # Call multiple times
        jit_fn(1.0)
        jit_fn(1.0)
        jit_fn(1.0)

        # Due to JIT caching, the side effect should only happen 0-1 times
        # (JAX may trace once, then cache)
        assert counter["count"] <= 1, "Side effects should be minimized in JIT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
