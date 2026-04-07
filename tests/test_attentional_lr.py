"""Unit tests for attentional learning rate modulation (Equation 26).

This module tests the attentional acceleration mechanism from the H-Bar paper:

    η_effective = η_base · (1 + κ_α · α_A)

The tests verify:
1. No acceleration when α_A = 0.0
2. Maximum acceleration (3×) when α_A = 1.0 with κ_α = 2.0
3. Gradient scaling produces 3× larger parameter changes
4. JIT compilation works correctly with the dynamic scaling logic
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

from hbar.engine.trainer import compute_attentional_lr, TrainState


class SimpleModel(nn.Module):
    """A simple model for testing gradient scaling."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def create_test_train_state(learning_rate=1e-3):
    """Create a TrainState with a simple model for testing."""
    model = SimpleModel()
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, 2))
    params = model.init(rng, dummy_input)["params"]
    tx = optax.adam(learning_rate=learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx), model


class TestAttentionalLR:
    """Test suite for attentional learning rate modulation."""

    def test_alpha_zero_no_acceleration(self):
        """Test 1: If α_A = 0.0 (no attention), verify that η_effective = η_base.

        When attentional fidelity is zero, the acceleration factor should be 1.0,
        meaning the effective learning rate equals the base learning rate.
        """
        base_lr = 1e-3
        kappa_alpha = 2.0
        alpha_A = jnp.array(0.0)

        effective_lr, acceleration_factor = compute_attentional_lr(
            base_lr, kappa_alpha, alpha_A
        )

        # Acceleration factor should be 1.0
        assert jnp.isclose(acceleration_factor, 1.0), (
            f"Expected acceleration_factor=1.0, got {acceleration_factor}"
        )

        # Effective LR should equal base LR
        assert jnp.isclose(effective_lr, base_lr), (
            f"Expected effective_lr={base_lr}, got {effective_lr}"
        )

    def test_alpha_one_max_acceleration(self):
        """Test 2: If α_A = 1.0 and κ_α = 2.0, verify that η_effective = 3×η_base.

        When attentional fidelity is maximum (1.0) and kappa_alpha is 2.0,
        the acceleration factor should be 1 + 2.0 * 1.0 = 3.0.
        """
        base_lr = 1e-3
        kappa_alpha = 2.0
        alpha_A = jnp.array(1.0)

        effective_lr, acceleration_factor = compute_attentional_lr(
            base_lr, kappa_alpha, alpha_A
        )

        # Acceleration factor should be 3.0
        expected_factor = 3.0
        assert jnp.isclose(acceleration_factor, expected_factor), (
            f"Expected acceleration_factor={expected_factor}, got {acceleration_factor}"
        )

        # Effective LR should be 3× base LR
        expected_lr = 3.0 * base_lr
        assert jnp.isclose(effective_lr, expected_lr), (
            f"Expected effective_lr={expected_lr}, got {effective_lr}"
        )

    def test_gradient_scaling_produces_larger_changes(self):
        """Test 3: Verify that updated parameters change 3× more when α_A=1.0 vs α_A=0.0.

        This test verifies that the gradient scaling implementation correctly
        produces larger parameter updates when attentional fidelity is high.
        """
        base_lr = 1e-3
        kappa_alpha = 2.0

        # Create two identical train states
        state_zero, model = create_test_train_state(base_lr)
        state_full, _ = create_test_train_state(base_lr)

        # Create a simple loss function
        def loss_fn(params, x, y):
            pred = model.apply({"params": params}, x)
            return jnp.mean((pred - y) ** 2)

        # Create test data
        x = jnp.array([[1.0, 2.0]])
        y = jnp.array([[5.0]])

        # Compute gradients
        grads = jax.grad(loss_fn)(state_zero.params, x, y)

        # Apply gradient scaling for α_A = 0.0 (no acceleration)
        alpha_zero = jnp.array(0.0)
        _, accel_zero = compute_attentional_lr(base_lr, kappa_alpha, alpha_zero)
        scaled_grads_zero = jax.tree_util.tree_map(
            lambda g: g * accel_zero, grads
        )
        new_state_zero = state_zero.apply_gradients(grads=scaled_grads_zero)

        # Apply gradient scaling for α_A = 1.0 (max acceleration)
        alpha_full = jnp.array(1.0)
        _, accel_full = compute_attentional_lr(base_lr, kappa_alpha, alpha_full)
        scaled_grads_full = jax.tree_util.tree_map(
            lambda g: g * accel_full, grads
        )
        new_state_full = state_full.apply_gradients(grads=scaled_grads_full)

        # Compute parameter changes
        param_zero = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(lambda x: x, new_state_zero.params)
        )[0]
        param_full = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(lambda x: x, new_state_full.params)
        )[0]
        orig_param = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(lambda x: x, state_zero.params)
        )[0]

        # Calculate the magnitude of changes
        change_zero = sum(
            jnp.sum((p - o) ** 2) for p, o in zip(param_zero, orig_param)
        )
        change_full = sum(
            jnp.sum((p - o) ** 2) for p, o in zip(param_full, orig_param)
        )

        # The change with α_A=1.0 should be 3× the change with α_A=0.0
        # (since acceleration_factor goes from 1.0 to 3.0)
        ratio = change_full / change_zero
        assert jnp.isclose(ratio, 9.0, rtol=1e-5), (
            f"Expected ratio=9.0 (3²), got {ratio}"
        )

    def test_jit_compilation_works(self):
        """Test 4: Ensure jax.jit successfully compiles the step with dynamic scaling logic.

        This test verifies that the attentional acceleration mechanism is
        compatible with JAX's JIT compilation, which is critical for performance.
        """

        @jax.jit
        def jit_compute_attentional_lr(base_lr, kappa_alpha, alpha_A):
            return compute_attentional_lr(base_lr, kappa_alpha, alpha_A)

        # Test with various alpha values
        base_lr = 1e-3
        kappa_alpha = 2.0

        # Test α_A = 0.0
        eff_lr_0, accel_0 = jit_compute_attentional_lr(base_lr, kappa_alpha, jnp.array(0.0))
        assert jnp.isclose(accel_0, 1.0)
        assert jnp.isclose(eff_lr_0, base_lr)

        # Test α_A = 0.5
        eff_lr_05, accel_05 = jit_compute_attentional_lr(base_lr, kappa_alpha, jnp.array(0.5))
        assert jnp.isclose(accel_05, 2.0)
        assert jnp.isclose(eff_lr_05, 2.0 * base_lr)

        # Test α_A = 1.0
        eff_lr_1, accel_1 = jit_compute_attentional_lr(base_lr, kappa_alpha, jnp.array(1.0))
        assert jnp.isclose(accel_1, 3.0)
        assert jnp.isclose(eff_lr_1, 3.0 * base_lr)

    def test_intermediate_alpha_values(self):
        """Additional test: Verify linear scaling for intermediate α_A values.

        The acceleration should scale linearly with α_A.
        """
        base_lr = 1e-3
        kappa_alpha = 2.0

        # Test α_A = 0.25 → factor = 1.5
        eff_lr, accel = compute_attentional_lr(base_lr, kappa_alpha, jnp.array(0.25))
        assert jnp.isclose(accel, 1.5)
        assert jnp.isclose(eff_lr, 1.5 * base_lr)

        # Test α_A = 0.75 → factor = 2.5
        eff_lr, accel = compute_attentional_lr(base_lr, kappa_alpha, jnp.array(0.75))
        assert jnp.isclose(accel, 2.5)
        assert jnp.isclose(eff_lr, 2.5 * base_lr)

    def test_different_kappa_alpha_values(self):
        """Additional test: Verify correct scaling with different κ_α values."""
        base_lr = 1e-3
        alpha_A = jnp.array(1.0)

        # κ_α = 0.0 → no acceleration
        eff_lr, accel = compute_attentional_lr(base_lr, 0.0, alpha_A)
        assert jnp.isclose(accel, 1.0)

        # κ_α = 1.0 → 2× acceleration at max α_A
        eff_lr, accel = compute_attentional_lr(base_lr, 1.0, alpha_A)
        assert jnp.isclose(accel, 2.0)

        # κ_α = 5.0 → 6× acceleration at max α_A
        eff_lr, accel = compute_attentional_lr(base_lr, 5.0, alpha_A)
        assert jnp.isclose(accel, 6.0)

    def test_attentional_burst_prediction(self):
        """Test the 'Attentional Burst' prediction from the H-Bar paper.

        During Phase 1, α_A is low (suppressed by surface rewards), so acceleration ≈ 1.0.
        During Phase 2 entry (crystallization), α_A increases rapidly.
        This test simulates the transition from Phase 1 to Phase 2.
        """
        base_lr = 1e-3
        kappa_alpha = 2.0

        # Phase 1: Low attentional fidelity (suppressed)
        alpha_phase1 = jnp.array(0.1)
        _, accel_phase1 = compute_attentional_lr(base_lr, kappa_alpha, alpha_phase1)

        # Phase 2: High attentional fidelity (crystallized)
        alpha_phase2 = jnp.array(0.8)
        _, accel_phase2 = compute_attentional_lr(base_lr, kappa_alpha, alpha_phase2)

        # The acceleration should increase significantly
        assert accel_phase2 > accel_phase1
        # Phase 2 acceleration should be at least 2× Phase 1
        assert accel_phase2 / accel_phase1 >= 2.0, (
            f"Expected Phase 2 acceleration to be at least 2× Phase 1, "
            f"got {accel_phase2 / accel_phase1}"
        )
