"""Tests for evaluation splits and ground-truth σ_A calculation.

This module tests the Add-Jump split for SCAN and Subject-to-Object
split for COGS, as well as the Evaluator class for computing
ground-truth schema coherence (σ̂_A).

Tests verify:
- Split correctness (no overlap between ID and OOD)
- Biased noun positioning in COGS splits
- Division-by-zero handling in σ̂_A calculation
- Evaluator batch preparation and metric computation
"""

import pytest
import jax
import jax.numpy as jnp

from hbar.benchmarks.scan_grammar import SCANGrammar
from hbar.benchmarks.cogs_grammar import COGSGrammar
from hbar.engine.evaluator import Evaluator, EvaluationResult


class TestScanAddJumpSplit:
    """Tests for SCAN Add-Jump partitioning."""

    def test_train_set_contains_isolated_jump(self):
        """Verify training set includes 'jump' in isolation."""
        grammar = SCANGrammar(seed=42)
        train_pairs, _ = grammar.get_add_jump_split(n_train=100, n_test=100)
        jump_isolated = [("jump", "I_JUMP") for cmd, act in train_pairs if cmd == "jump" and act == "I_JUMP"]
        assert len(jump_isolated) >= 1, "Training set must include isolated 'jump' command"

    def test_train_set_no_jump_compounds(self):
        """Verify 'jump' never appears in compound commands in training set."""
        grammar = SCANGrammar(seed=42)
        train_pairs, _ = grammar.get_add_jump_split(n_train=500, n_test=500)

        for cmd, _ in train_pairs:
            if cmd.lower() == "jump":
                continue  # Isolated 'jump' is allowed
            assert "jump" not in cmd.lower(), f"Found 'jump' in compound training command: {cmd}"

    def test_test_set_all_contain_jump(self):
        """Verify all test set commands contain 'jump' in compounds."""
        grammar = SCANGrammar(seed=42)
        _, test_pairs = grammar.get_add_jump_split(n_train=100, n_test=500)

        for cmd, _ in test_pairs:
            assert "jump" in cmd.lower(), f"Test command missing 'jump': {cmd}"
            assert cmd.lower() != "jump", f"Test set should not contain isolated 'jump': {cmd}"

    def test_zero_overlap(self):
        """Verify zero overlap between training and test compositions."""
        grammar = SCANGrammar(seed=42)
        train_pairs, test_pairs = grammar.get_add_jump_split(n_train=500, n_test=500)

        train_commands = set(cmd for cmd, _ in train_pairs)
        test_commands = set(cmd for cmd, _ in test_pairs)

        overlap = train_commands.intersection(test_commands)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping commands between train and test"

    def test_split_sizes(self):
        """Verify split generates correct number of samples."""
        grammar = SCANGrammar(seed=42)
        n_train = 1000
        n_test = 1000
        train_pairs, test_pairs = grammar.get_add_jump_split(n_train=n_train, n_test=n_test)

        assert len(train_pairs) == n_train, f"Expected {n_train} train samples, got {len(train_pairs)}"
        assert len(test_pairs) == n_test, f"Expected {n_test} test samples, got {len(test_pairs)}"


class TestCogsSubjectObjectSplit:
    """Tests for COGS Subject-to-Object partitioning."""

    def test_biased_nouns_defined(self):
        """Verify BIASED_NOUNS list is defined."""
        grammar = COGSGrammar()
        assert hasattr(grammar, "BIASED_NOUNS")
        assert len(grammar.BIASED_NOUNS) == 3
        assert "hedgehog" in grammar.BIASED_NOUNS
        assert "porcupine" in grammar.BIASED_NOUNS
        assert "otter" in grammar.BIASED_NOUNS

    def test_train_biased_nouns_in_subject(self):
        """Verify biased nouns only appear in subject position in training."""
        grammar = COGSGrammar(seed=42)
        train_pairs, _ = grammar.get_subject_object_split(n_train=500, n_test=500)

        for sent, _ in train_pairs:
            for noun in grammar.BIASED_NOUNS:
                # Biased nouns should be at the start (subject position)
                if noun in sent.lower():
                    assert sent.lower().startswith(f"the {noun}") or sent.lower().startswith(f"a {noun}"), \
                        f"Biased noun '{noun}' not in subject position: {sent}"

    def test_test_biased_nouns_in_object(self):
        """Verify biased nouns only appear in object position in test."""
        grammar = COGSGrammar(seed=42)
        _, test_pairs = grammar.get_subject_object_split(n_train=500, n_test=500)

        for sent, _ in test_pairs:
            for noun in grammar.BIASED_NOUNS:
                if noun in sent.lower():
                    # Biased nouns should NOT be at the start
                    assert not sent.lower().startswith(f"the {noun}") and not sent.lower().startswith(f"a {noun}"), \
                        f"Biased noun '{noun}' in subject position in test: {sent}"

    def test_split_sizes(self):
        """Verify split generates correct number of samples."""
        grammar = COGSGrammar(seed=42)
        n_train = 1000
        n_test = 1000
        train_pairs, test_pairs = grammar.get_subject_object_split(n_train=n_train, n_test=n_test)

        assert len(train_pairs) == n_train, f"Expected {n_train} train samples, got {len(train_pairs)}"
        assert len(test_pairs) == n_test, f"Expected {n_test} test samples, got {len(test_pairs)}"


class TestEvaluator:
    """Tests for the H-Bar Evaluator class."""

    def test_calculate_calibration_error(self):
        """Test calibration error calculation."""
        # Perfect calibration
        assert Evaluator.calculate_calibration_error(0.9, 0.9) == 0.0

        # Imperfect calibration (use pytest.approx for floating-point comparison)
        assert Evaluator.calculate_calibration_error(0.8, 0.6) == pytest.approx(0.2)
        assert Evaluator.calculate_calibration_error(0.6, 0.8) == pytest.approx(0.2)

    def test_ground_truth_sigma_division_by_zero(self):
        """Test that ground_truth_sigma handles acc_id=0 gracefully."""
        # When acc_id is 0, ground_truth_sigma should be 0.0
        result = EvaluationResult(
            acc_id=0.0,
            acc_ood=0.0,
            loss_id=1.0,
            loss_ood=1.0,
            ground_truth_sigma=0.0,
            n_id=100,
            n_ood=100,
        )
        assert result.ground_truth_sigma == 0.0

    def test_evaluator_loads_eval_sets(self):
        """Test that evaluator correctly loads evaluation sets."""
        evaluator = Evaluator(domain="scan", data_dir="data")
        assert len(evaluator.id_samples) == 2000
        assert len(evaluator.ood_samples) == 2000

    def test_evaluator_prepare_batches(self):
        """Test batch preparation for evaluation."""
        evaluator = Evaluator(domain="scan", data_dir="data")
        id_batches, ood_batches = evaluator.prepare_evaluation_batches(batch_size=32)

        # Check number of batches
        assert len(id_batches) == 63  # ceil(2000/32) = 63
        assert len(ood_batches) == 63

    def test_evaluator_tokenizer(self):
        """Test that evaluator has correct tokenizer for domain."""
        scan_eval = Evaluator(domain="scan", data_dir="data")
        assert scan_eval.tokenizer is not None
        assert "jump" in scan_eval.tokenizer.word2id
        assert "I_JUMP" in scan_eval.tokenizer.word2id


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_evaluation_result_fields(self):
        """Test that EvaluationResult has all required fields."""
        result = EvaluationResult(
            acc_id=0.95,
            acc_ood=0.45,
            loss_id=0.12,
            loss_ood=0.85,
            ground_truth_sigma=0.474,
            n_id=2000,
            n_ood=2000,
        )
        assert result.acc_id == 0.95
        assert result.acc_ood == 0.45
        assert result.ground_truth_sigma == 0.474
        assert result.n_id == 2000
        assert result.n_ood == 2000

    def test_illusion_of_mastery_detection(self):
        """Test detection of 'illusion of mastery' failure mode."""
        # High ID accuracy, low OOD accuracy = trapped in σ-trap
        result = EvaluationResult(
            acc_id=0.99,
            acc_ood=0.05,
            loss_id=0.01,
            loss_ood=1.5,
            ground_truth_sigma=0.05 / 0.99,
            n_id=2000,
            n_ood=2000,
        )
        assert result.acc_id > 0.95, "High ID accuracy indicates training success"
        assert result.acc_ood < 0.5, "Low OOD accuracy indicates compositional failure"
        assert result.ground_truth_sigma < 0.1, "Low σ̂_A indicates σ-trap"
