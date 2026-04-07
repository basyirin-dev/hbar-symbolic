"""Unit tests for the H-Bar modulated loss (Equation 25).

These tests verify that the compositional pressure mechanism works correctly:
1. When σ_A = 1.0, total loss ≈ task loss (compositional penalty ≈ 0)
2. When σ_A = 0.0, compositional penalty is at maximum
3. Gradients flow correctly through the modulated loss with different σ_A values

The modulated loss follows:
    L_total = L_task + λ_σ · (1 - σ_A) · L_comp
"""

import jax
import jax.numpy as jnp
import pytest

from hbar.engine.data_utils import compute_hbar_loss, compute_loss


class TestModulatedLossSigmaOne:
    """Test that when σ_A = 1.0, the compositional penalty vanishes."""

    def test_sigma_one_no_penalty(self):
        """When σ_A = 1.0, total loss should equal task loss.

        The compositional pressure (1 - σ_A) = 0, so the OOD loss
        should not contribute to the total loss.
        """
        # Create dummy logits and labels
        batch_size = 4
        seq_len = 10
        vocab_size = 50

        logits_id = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, vocab_size))
        labels_id = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, vocab_size)

        logits_ood = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, vocab_size))
        labels_ood = jax.random.randint(jax.random.PRNGKey(3), (batch_size, seq_len), 0, vocab_size)

        # Compute with σ_A = 1.0
        sigma_A = jnp.array(1.0)
        total_loss = compute_hbar_loss(
            logits_id=logits_id,
            labels_id=labels_id,
            logits_ood=logits_ood,
            labels_ood=labels_ood,
            sigma_A=sigma_A,
            lambda_sigma=0.5,
        )

        # Compute task loss separately
        task_loss = compute_loss(logits_id, labels_id)

        # Total loss should equal task loss when σ_A = 1.0
        assert jnp.isclose(total_loss, task_loss, rtol=1e-5), (
            f"When σ_A=1.0, total_loss={total_loss:.6f} should equal task_loss={task_loss:.6f}"
        )

    def test_sigma_one_gradient_only_from_task(self):
        """When σ_A = 1.0, gradients should only come from task loss.

        The gradient of total_loss w.r.t. model parameters should be
        identical to the gradient of task_loss alone.
        """
        batch_size = 2
        seq_len = 5
        vocab_size = 20

        key = jax.random.PRNGKey(42)
        key_logits_id, key_logits_ood, key_labels = jax.random.split(key, 3)

        logits_id = jax.random.normal(key_logits_id, (batch_size, seq_len, vocab_size))
        labels_id = jax.random.randint(key_labels, (batch_size, seq_len), 0, vocab_size)
        logits_ood = jax.random.normal(key_logits_ood, (batch_size, seq_len, vocab_size))
        labels_ood = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)

        sigma_A = jnp.array(1.0)

        # Compute gradient of total loss w.r.t. logits_id
        def total_loss_fn(l_id):
            return compute_hbar_loss(
                logits_id=l_id,
                labels_id=labels_id,
                logits_ood=logits_ood,
                labels_ood=labels_ood,
                sigma_A=sigma_A,
                lambda_sigma=0.5,
            )

        def task_loss_fn(l_id):
            return compute_loss(l_id, labels_id)

        grad_total = jax.grad(total_loss_fn)(logits_id)
        grad_task = jax.grad(task_loss_fn)(logits_id)

        # Gradients should be identical
        assert jnp.allclose(grad_total, grad_task, rtol=1e-5), (
            f"When σ_A=1.0, gradients should match. "
            f"Max diff: {jnp.max(jnp.abs(grad_total - grad_task)):.8f}"
        )


class TestModulatedLossSigmaZero:
    """Test that when σ_A = 0.0, the compositional penalty is maximum."""

    def test_sigma_zero_max_penalty(self):
        """When σ_A = 0.0, total loss should include full compositional penalty.

        The compositional pressure (1 - σ_A) = 1, so the total loss should be:
            L_total = L_task + λ_σ · L_comp
        """
        batch_size = 4
        seq_len = 10
        vocab_size = 50

        logits_id = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, vocab_size))
        labels_id = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, vocab_size)

        logits_ood = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, vocab_size))
        labels_ood = jax.random.randint(jax.random.PRNGKey(3), (batch_size, seq_len), 0, vocab_size)

        # Compute with σ_A = 0.0
        sigma_A = jnp.array(0.0)
        lambda_sigma = 0.5
        total_loss = compute_hbar_loss(
            logits_id=logits_id,
            labels_id=labels_id,
            logits_ood=logits_ood,
            labels_ood=labels_ood,
            sigma_A=sigma_A,
            lambda_sigma=lambda_sigma,
        )

        # Compute individual losses
        task_loss = compute_loss(logits_id, labels_id)
        comp_loss = compute_loss(logits_ood, labels_ood)

        # Expected total loss
        expected_loss = task_loss + lambda_sigma * comp_loss

        assert jnp.isclose(total_loss, expected_loss, rtol=1e-5), (
            f"When σ_A=0.0, total_loss={total_loss:.6f} should equal "
            f"task_loss + λ_σ·comp_loss = {expected_loss:.6f}"
        )

    def test_sigma_zero_penalty_increases_loss(self):
        """When σ_A = 0.0, total loss should be higher than task loss alone.

        This verifies that the compositional penalty actually increases
        the loss, creating pressure to improve OOD performance.
        """
        batch_size = 4
        seq_len = 10
        vocab_size = 50

        logits_id = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, vocab_size))
        labels_id = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, vocab_size)

        logits_ood = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, vocab_size))
        labels_ood = jax.random.randint(jax.random.PRNGKey(3), (batch_size, seq_len), 0, vocab_size)

        task_loss = compute_loss(logits_id, labels_id)
        comp_loss = compute_loss(logits_ood, labels_ood)

        # Compute with σ_A = 0.0
        sigma_A_zero = jnp.array(0.0)
        total_loss_zero = compute_hbar_loss(
            logits_id=logits_id,
            labels_id=labels_id,
            logits_ood=logits_ood,
            labels_ood=labels_ood,
            sigma_A=sigma_A_zero,
            lambda_sigma=0.5,
        )

        # Total loss should be greater than task loss when there's OOD loss
        if comp_loss > 0:
            assert total_loss_zero > task_loss, (
                f"When σ_A=0.0, total_loss={total_loss_zero:.6f} should be > task_loss={task_loss:.6f}"
            )


class TestModulatedLossGradientFlow:
    """Test that gradients flow correctly through the modulated loss."""

    def test_gradient_depends_on_sigma(self):
        """Gradients should differ for different σ_A values.

        The gradient of total_loss w.r.t. logits_ood should be scaled
        by λ_σ · (1 - σ_A). Lower σ_A means stronger gradient on OOD stream.
        """
        batch_size = 2
        seq_len = 5
        vocab_size = 20

        key = jax.random.PRNGKey(42)
        key_logits_id, key_logits_ood, key_labels_id, key_labels_ood = jax.random.split(key, 4)

        logits_id = jax.random.normal(key_logits_id, (batch_size, seq_len, vocab_size))
        labels_id = jax.random.randint(key_labels_id, (batch_size, seq_len), 0, vocab_size)
        logits_ood = jax.random.normal(key_logits_ood, (batch_size, seq_len, vocab_size))
        labels_ood = jax.random.randint(key_labels_ood, (batch_size, seq_len), 0, vocab_size)

        lambda_sigma = 0.5

        # Compute gradient w.r.t. logits_ood for different σ_A values
        def loss_fn(l_ood, sigma):
            return compute_hbar_loss(
                logits_id=logits_id,
                labels_id=labels_id,
                logits_ood=l_ood,
                labels_ood=labels_ood,
                sigma_A=sigma,
                lambda_sigma=lambda_sigma,
            )

        # Gradient when σ_A = 0.0 (maximum pressure)
        grad_zero = jax.grad(loss_fn, argnums=0)(logits_ood, jnp.array(0.0))

        # Gradient when σ_A = 0.5 (half pressure)
        grad_half = jax.grad(loss_fn, argnums=0)(logits_ood, jnp.array(0.5))

        # Gradient when σ_A = 1.0 (no pressure)
        grad_one = jax.grad(loss_fn, argnums=0)(logits_ood, jnp.array(1.0))

        # Verify gradient scaling: grad_zero should be ~2x grad_half, grad_one should be ~0
        # The gradient from the compositional term is: λ_σ · (1 - σ_A) · ∂L_comp/∂logits_ood

        # grad_one should be zero (no compositional gradient when σ_A = 1.0)
        assert jnp.allclose(grad_one, jnp.zeros_like(grad_one), atol=1e-6), (
            f"When σ_A=1.0, gradient on OOD logits should be ~0. "
            f"Max abs grad: {jnp.max(jnp.abs(grad_one)):.8f}"
        )

        # grad_zero should be approximately 2x grad_half
        # (because (1 - 0.0) = 1.0 vs (1 - 0.5) = 0.5)
        # Only check elements where grad_half is significant
        significant_mask = jnp.abs(grad_half) > 1e-4
        num_significant = jnp.sum(significant_mask)

        # If there are significant elements, check the ratio
        if num_significant > 0:
            ratio = jnp.where(
                significant_mask,
                grad_zero / grad_half,
                jnp.ones_like(grad_zero),
            )
            expected_ratio = 2.0
            # Check only the significant elements
            significant_ratios = ratio[significant_mask]
            assert jnp.allclose(significant_ratios, expected_ratio, rtol=0.01), (
                f"Gradient ratio (σ=0 vs σ=0.5) should be ~2.0. "
                f"Got mean={jnp.mean(significant_ratios):.4f}, "
                f"std={jnp.std(significant_ratios):.6f}"
            )

    def test_jit_compatible(self):
        """The modulated loss should be JIT-compatible."""

        @jax.jit
        def jit_loss(sigma_A):
            batch_size = 2
            seq_len = 5
            vocab_size = 20
            key = jax.random.PRNGKey(0)

            logits_id = jax.random.normal(key, (batch_size, seq_len, vocab_size))
            labels_id = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
            logits_ood = jax.random.normal(key, (batch_size, seq_len, vocab_size))
            labels_ood = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)

            return compute_hbar_loss(
                logits_id=logits_id,
                labels_id=labels_id,
                logits_ood=logits_ood,
                labels_ood=labels_ood,
                sigma_A=sigma_A,
                lambda_sigma=0.5,
            )

        # Should compile and run without errors
        loss_zero = jit_loss(jnp.array(0.0))
        loss_one = jit_loss(jnp.array(1.0))

        assert loss_zero > loss_one, "JIT-compiled loss should maintain σ_A ordering"

    def test_edge_case_sigma_out_of_bounds(self):
        """The loss should handle σ_A slightly outside [0, 1] gracefully.

        While σ_A should be in [0, 1], numerical errors might push it
        slightly outside. The loss should still be well-defined.
        """
        batch_size = 2
        seq_len = 5
        vocab_size = 20

        key = jax.random.PRNGKey(0)
        logits_id = jax.random.normal(key, (batch_size, seq_len, vocab_size))
        labels_id = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
        logits_ood = jax.random.normal(key, (batch_size, seq_len, vocab_size))
        labels_ood = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)

        # Test σ_A = -0.1 (slightly negative)
        loss_neg = compute_hbar_loss(
            logits_id=logits_id,
            labels_id=labels_id,
            logits_ood=logits_ood,
            labels_ood=labels_ood,
            sigma_A=jnp.array(-0.1),
            lambda_sigma=0.5,
        )

        # Test σ_A = 1.1 (slightly above 1)
        loss_above = compute_hbar_loss(
            logits_id=logits_id,
            labels_id=labels_id,
            logits_ood=logits_ood,
            labels_ood=labels_ood,
            sigma_A=jnp.array(1.1),
            lambda_sigma=0.5,
        )

        # Both should produce finite values
        assert jnp.isfinite(loss_neg), "Loss should be finite for σ_A = -0.1"
        assert jnp.isfinite(loss_above), "Loss should be finite for σ_A = 1.1"

        # σ_A = -0.1 should give higher loss than σ_A = 0.0
        loss_zero = compute_hbar_loss(
            logits_id=logits_id,
            labels_id=labels_id,
            logits_ood=logits_ood,
            labels_ood=labels_ood,
            sigma_A=jnp.array(0.0),
            lambda_sigma=0.5,
        )
        assert loss_neg > loss_zero, "Negative σ_A should increase penalty"

        # σ_A = 1.1 should give lower loss than σ_A = 1.0
        loss_one = compute_hbar_loss(
            logits_id=logits_id,
            labels_id=labels_id,
            logits_ood=logits_ood,
            labels_ood=labels_ood,
            sigma_A=jnp.array(1.0),
            lambda_sigma=0.5,
        )
        assert loss_above < loss_one, "σ_A > 1 should decrease penalty (even negative)"
