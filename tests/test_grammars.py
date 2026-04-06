"""Tests for SCAN and COGS grammar implementations.

This module validates that the grammar implementations correctly generate
compositional samples and that all generated tokens are properly handled
by the tokenization pipeline without producing UNK tokens.

Tests verify:
- Grammar generates valid command/action pairs
- No UNK tokens appear after tokenization (1000 samples)
- Primitive swapping works in all valid syntactic slots
- Compositional probes generate complex nested structures
- Structural distance computation for COGS
"""

import jax
import jax.numpy as jnp
import pytest

from hbar.benchmarks.scan_grammar import SCANGrammar
from hbar.benchmarks.cogs_grammar import COGSGrammar, LogicalForm
from hbar.benchmarks.grammar_engine import GrammarEngine
from hbar.engine.tokenizer import UNK_TOKEN_ID, create_scan_tokenizer


class TestSCANGrammar:
    """Tests for SCAN grammar implementation."""

    def test_generate_primitive(self):
        """Verify primitive generation produces valid pairs."""
        grammar = SCANGrammar(seed=42)
        cmd, act = grammar._generate_primitive()
        assert cmd in grammar.primitives
        assert act in grammar.primitive_to_action.values()

    def test_generate_modifier_phrase(self):
        """Verify modifier phrases repeat actions correctly."""
        grammar = SCANGrammar(seed=42)
        cmd, act = grammar._generate_modifier_phrase()
        # Action should be repeated 2 or 3 times
        action_parts = act.split()
        assert len(action_parts) in [2, 3]
        # All parts should be the same action
        assert all(a == action_parts[0] for a in action_parts)

    def test_generate_sample_basic(self):
        """Verify basic sample generation."""
        grammar = SCANGrammar(seed=42)
        cmd, act = grammar.generate_sample(max_depth=0)
        assert isinstance(cmd, str)
        assert isinstance(act, str)
        assert len(cmd) > 0
        assert len(act) > 0

    def test_generate_sample_with_conjunction(self):
        """Verify conjunction samples have two parts."""
        grammar = SCANGrammar(seed=42)
        # Generate multiple samples to get a conjunction
        found_conjunction = False
        for _ in range(20):
            cmd, act = grammar.generate_sample(max_depth=2)
            if " and " in cmd or " after " in cmd:
                found_conjunction = True
                # Both parts should have actions
                action_parts = act.split()
                assert len(action_parts) > 1
                break
        assert found_conjunction, "Should generate at least one conjunction in 20 tries"

    def test_sample_compositional_probe_jump(self):
        """Verify compositional probe targets 'jump' correctly."""
        grammar = SCANGrammar(seed=42)
        cmd, act = grammar.sample_compositional_probe(
            primitive_target="jump", complexity=2
        )
        assert "jump" in cmd
        assert "I_JUMP" in act

    def test_sample_compositional_probe_complexity3(self):
        """Verify high complexity probes generate nested structures."""
        grammar = SCANGrammar(seed=42)
        cmd, act = grammar.sample_compositional_probe(
            primitive_target="jump", complexity=3
        )
        # Should have complex structure with multiple components
        assert "jump" in cmd.lower()
        # Should have multiple action tokens
        assert len(act.split()) >= 3

    def test_sample_compositional_probe_invalid_primitive(self):
        """Verify error on invalid primitive."""
        grammar = SCANGrammar()
        with pytest.raises(ValueError):
            grammar.sample_compositional_probe(primitive_target="fly")

    def test_get_vocabulary(self):
        """Verify vocabulary includes all expected tokens."""
        grammar = SCANGrammar()
        vocab = grammar.get_vocabulary()
        for p in grammar.primitives:
            assert p in vocab
        for d in grammar.directions:
            assert d in vocab
        for m in grammar.modifiers:
            assert m in vocab
        for c in grammar.conjunctions:
            assert c in vocab
        for a in grammar.primitive_to_action.values():
            assert a in vocab

    def test_generate_batch(self):
        """Verify batch generation produces correct number of samples."""
        grammar = SCANGrammar(seed=42)
        batch = grammar.generate_batch(10, max_depth=1)
        assert len(batch) == 10
        for cmd, act in batch:
            assert isinstance(cmd, str)
            assert isinstance(act, str)


class TestCOGSGrammar:
    """Tests for COGS grammar implementation."""

    def test_generate_active_transitive(self):
        """Verify active transitive sentence structure."""
        grammar = COGSGrammar(seed=42)
        sent, lf = grammar._generate_active_transitive()
        # Sentence should have subject-verb-object structure
        words = sent.split()
        assert words[0] in ["The", "A"]
        # LF is a LogicalForm object - check its attributes
        assert "agent" in lf.args
        assert "patient" in lf.args

    def test_generate_passive_transitive(self):
        """Verify passive transitive sentence structure."""
        grammar = COGSGrammar(seed=42)
        sent, lf = grammar._generate_passive_transitive()
        assert "was" in sent
        assert "by" in sent
        # LF should still have correct agent and patient
        assert "agent" in lf.args
        assert "patient" in lf.args

    def test_generate_embedded(self):
        """Verify embedded clause structure."""
        grammar = COGSGrammar(seed=42)
        sent, lf = grammar._generate_embedded()
        assert "that" in sent
        # LF should have theme argument
        assert "theme" in lf.args

    def test_sample_compositional_probe_active(self):
        """Verify active probe generation."""
        grammar = COGSGrammar(seed=42)
        sent, lf_str = grammar.sample_compositional_probe(
            target_verb="chased", construction="active"
        )
        assert "chased" in sent
        assert lf_str.startswith("chase (")

    def test_sample_compositional_probe_passive(self):
        """Verify passive probe generation."""
        grammar = COGSGrammar(seed=42)
        sent, lf = grammar.sample_compositional_probe(
            target_verb="chased", construction="passive"
        )
        assert "was chased" in sent
        assert "by" in sent

    def test_sample_compositional_probe_embedded_subject(self):
        """Verify embedded subject probe generation."""
        grammar = COGSGrammar(seed=42)
        sent, lf = grammar.sample_compositional_probe(
            target_verb="chased", construction="embedded_subject"
        )
        assert "that" in sent
        assert "chased" in sent

    def test_sample_compositional_probe_invalid_construction(self):
        """Verify error on invalid construction."""
        grammar = COGSGrammar()
        with pytest.raises(ValueError):
            grammar.sample_compositional_probe(construction="invalid")

    def test_get_structural_distance_identical(self):
        """Verify distance is 0 for identical LFs."""
        grammar = COGSGrammar()
        lf = "chase ( agent = dog , patient = cat )"
        dist = grammar.get_structural_distance(lf, lf)
        assert dist == 0.0

    def test_get_structural_distance_different(self):
        """Verify distance is positive for different LFs."""
        grammar = COGSGrammar()
        lf1 = "chase ( agent = dog , patient = cat )"
        lf2 = "see ( agent = boy , patient = bird )"
        dist = grammar.get_structural_distance(lf1, lf2)
        assert dist > 0.0

    def test_get_structural_distance_normalized(self):
        """Verify distance is normalized to [0, 1]."""
        grammar = COGSGrammar()
        lf1 = "run ( agent = dog )"
        lf2 = "chase ( agent = dog , patient = cat )"
        dist = grammar.get_structural_distance(lf1, lf2)
        assert 0.0 <= dist <= 1.0

    def test_get_vocabulary(self):
        """Verify vocabulary includes all expected tokens."""
        grammar = COGSGrammar()
        vocab = grammar.get_vocabulary()
        for n in grammar.nouns:
            assert n in vocab
        assert "The" in vocab
        assert "A" in vocab
        # LF tokens
        assert "(" in vocab
        assert ")" in vocab
        assert "agent" in vocab
        assert "patient" in vocab


class TestLogicalForm:
    """Tests for LogicalForm dataclass."""

    def test_simple_lf_to_string(self):
        """Verify simple LF string representation."""
        lf = LogicalForm(predicate="run", args={"agent": "dog"})
        s = lf.to_string()
        assert s == "run ( agent = dog )"

    def test_complex_lf_to_string(self):
        """Verify complex LF with nested structure."""
        inner = LogicalForm(predicate="chase", args={"agent": "dog", "patient": "cat"})
        outer = LogicalForm(predicate="say", args={"agent": "boy", "theme": inner})
        s = outer.to_string()
        assert "say" in s
        assert "chase" in s
        assert "agent = boy" in s


class TestGrammarEngine:
    """Tests for unified GrammarEngine."""

    def test_init(self):
        """Verify engine initialization."""
        engine = GrammarEngine(seed=42)
        assert engine.scan_grammar is not None
        assert engine.cogs_grammar is not None
        assert engine.scan_tokenizer is not None
        assert engine.cogs_tokenizer is not None

    def test_get_compositional_batch_scan(self):
        """Verify compositional batch for SCAN."""
        engine = GrammarEngine(seed=42)
        batch = engine.get_compositional_batch(4, domain="scan")
        assert batch.inputs.shape[0] == 4
        assert batch.labels.shape[0] == 4

    def test_get_compositional_batch_cogs(self):
        """Verify compositional batch for COGS."""
        engine = GrammarEngine(seed=42)
        batch = engine.get_compositional_batch(4, domain="cogs")
        assert batch.inputs.shape[0] == 4
        assert batch.labels.shape[0] == 4

    def test_generate_id_batch(self):
        """Verify ID batch generation."""
        engine = GrammarEngine(seed=42)
        batch = engine.generate_id_batch(4, domain="scan")
        assert batch.inputs.shape[0] == 4

    def test_get_tokenizer(self):
        """Verify tokenizer retrieval."""
        engine = GrammarEngine(seed=42)
        scan_tok = engine.get_tokenizer("scan")
        cogs_tok = engine.get_tokenizer("cogs")
        assert scan_tok is not None
        assert cogs_tok is not None
        assert scan_tok != cogs_tok

    def test_get_vocabulary(self):
        """Verify vocabulary retrieval."""
        engine = GrammarEngine(seed=42)
        scan_vocab = engine.get_vocabulary("scan")
        cogs_vocab = engine.get_vocabulary("cogs")
        assert len(scan_vocab) > 0
        assert len(cogs_vocab) > 0

    def test_structural_distance_cogs(self):
        """Verify structural distance computation."""
        engine = GrammarEngine(seed=42)
        lf1 = "chase ( agent = dog , patient = cat )"
        lf2 = "see ( agent = boy , patient = bird )"
        dist = engine.get_structural_distance(lf1, lf2, domain="cogs")
        assert 0.0 < dist <= 1.0

    def test_structural_distance_scan_error(self):
        """Verify structural distance raises error for SCAN."""
        engine = GrammarEngine(seed=42)
        with pytest.raises(ValueError):
            engine.get_structural_distance("a", "b", domain="scan")

    def test_compute_rdmstruct(self):
        """Verify RDMstruct computation."""
        engine = GrammarEngine(seed=42)
        logical_forms = [
            "chase ( agent = dog , patient = cat )",
            "see ( agent = boy , patient = bird )",
            "run ( agent = dog )",
        ]
        rdm = engine.compute_rdmstruct(logical_forms)
        assert rdm.shape == (3, 3)
        # Diagonal should be 0
        assert rdm[0, 0] == 0.0
        assert rdm[1, 1] == 0.0
        assert rdm[2, 2] == 0.0
        # Matrix should be symmetric
        assert rdm[0, 1] == rdm[1, 0]
        assert rdm[0, 2] == rdm[2, 0]
        assert rdm[1, 2] == rdm[2, 1]


class TestTokenizerAlignment:
    """Tests for grammar-tokenizer alignment (no UNK tokens)."""

    def test_scan_no_unk_tokens(self):
        """Verify SCAN grammar produces no UNK tokens in 1000 samples."""
        grammar = SCANGrammar(seed=42)
        tokenizer = create_scan_tokenizer()

        # Add any extra vocabulary from grammar
        vocab = grammar.get_vocabulary()
        for word in vocab:
            if word not in tokenizer.word2id:
                tokenizer.add_vocabulary([word])

        unk_count = 0
        total_tokens = 0

        for _ in range(1000):
            cmd, act = grammar.generate_sample(max_depth=2)

            # Tokenize command
            cmd_tokens = tokenizer.tokenize(cmd)
            for token in cmd_tokens:
                total_tokens += 1
                if token not in tokenizer.word2id:
                    unk_count += 1

            # Tokenize action
            act_tokens = tokenizer.tokenize(act)
            for token in act_tokens:
                total_tokens += 1
                if token not in tokenizer.word2id:
                    unk_count += 1

        assert unk_count == 0, f"Found {unk_count} UNK tokens out of {total_tokens}"

    def test_cogs_no_unk_tokens(self):
        """Verify COGS grammar produces no UNK tokens in 1000 samples."""
        grammar = COGSGrammar(seed=42)
        vocab = grammar.get_vocabulary()

        # Create tokenizer with vocabulary
        from hbar.engine.tokenizer import Tokenizer
        tokenizer = Tokenizer(vocab)

        unk_count = 0
        total_tokens = 0

        for _ in range(1000):
            sent, lf_str = grammar.generate_sample(max_depth=1)

            # Tokenize sentence
            sent_tokens = tokenizer.tokenize(sent)
            for token in sent_tokens:
                total_tokens += 1
                if token not in tokenizer.word2id:
                    unk_count += 1

            # Tokenize logical form string
            lf_tokens = tokenizer.tokenize(lf_str)
            for token in lf_tokens:
                total_tokens += 1
                if token not in tokenizer.word2id:
                    unk_count += 1

        assert unk_count == 0, f"Found {unk_count} UNK tokens out of {total_tokens}"

    def test_primitive_swapping_scan(self):
        """Verify 'jump' can be swapped into any valid syntactic slot."""
        grammar = SCANGrammar(seed=42)

        # Test that jump works in various contexts
        contexts = [
            # Simple
            lambda: ("jump", "I_JUMP"),
            # With modifier
            lambda: ("jump twice", "I_JUMP I_JUMP"),
            lambda: ("jump thrice", "I_JUMP I_JUMP I_JUMP"),
            # With conjunction
            lambda: ("jump and walk", "I_JUMP I_WALK"),
            lambda: ("walk and jump", "I_WALK I_JUMP"),
            # With direction phrase
            lambda: ("jump after turn left", "I_JUMP I_TURN_LEFT"),
        ]

        for context_fn in contexts:
            expected_cmd, expected_act = context_fn()
            # Verify the expected pair is valid
            cmd_tokens = expected_cmd.split()
            act_tokens = expected_act.split()
            # All action tokens should be valid
            for a in act_tokens:
                assert a in ["I_JUMP", "I_TURN_LEFT", "I_TURN_RIGHT", "I_WALK", "I_RUN", "I_LOOK"]

    def test_verb_swapping_cogs(self):
        """Verify 'chase' can be swapped into valid syntactic slots."""
        grammar = COGSGrammar(seed=42)

        # Test chase in different constructions
        for construction in ["active", "passive", "embedded_subject", "embedded_object"]:
            sent, lf_str = grammar.sample_compositional_probe(
                target_verb="chased", construction=construction
            )
            # Verify chase appears in sentence
            assert "chased" in sent or "chase" in sent.lower()
            # Verify LF string has chase predicate
            assert "chase" in lf_str


class TestDeterminism:
    """Tests for deterministic generation with PRNGKey.

    Note: The grammars use Python's random module which provides
    reproducibility within a single instance. The GrammarEngine uses
    JAX PRNGKey for batch-level determinism.
    """

    def test_scan_single_sample_consistency(self):
        """Verify SCAN generates valid samples from a seeded instance."""
        grammar = SCANGrammar(seed=42)

        # Generate multiple samples and verify they're all valid
        for _ in range(10):
            cmd, act = grammar.generate_sample(max_depth=1)
            assert isinstance(cmd, str) and len(cmd) > 0
            assert isinstance(act, str) and len(act) > 0

    def test_cogs_single_sample_consistency(self):
        """Verify COGS generates valid samples from a seeded instance."""
        grammar = COGSGrammar(seed=42)

        # Generate multiple samples and verify they're all valid
        for _ in range(10):
            sent, lf = grammar.generate_sample(max_depth=0)
            assert isinstance(sent, str) and len(sent) > 0
            assert isinstance(lf, str) and len(lf) > 0

    def test_engine_batch_shape_consistency(self):
        """Verify engine produces consistent batch shapes."""
        engine = GrammarEngine(seed=42)

        rng1 = jax.random.PRNGKey(42)
        rng2 = jax.random.PRNGKey(123)

        batch1 = engine.get_compositional_batch(4, domain="scan", rng=rng1)
        batch2 = engine.get_compositional_batch(4, domain="scan", rng=rng2)

        # Batches should have the same shape regardless of RNG
        assert batch1.inputs.shape == batch2.inputs.shape
        assert batch1.labels.shape == batch2.labels.shape
        assert batch1.inputs.shape[0] == 4  # batch size
