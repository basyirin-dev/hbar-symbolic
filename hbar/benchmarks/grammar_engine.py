"""Unified Grammar Engine for compositional generalization benchmarks.

This module provides a unified interface G(d) for generating compositional
data batches from SCAN and COGS grammars. The GrammarEngine is designed
to be the primary source of Out-of-Distribution (OOD) signals during
Phase 1 training and for GCA (Gradient-Composition Alignment) signal
extraction in Phase 2.

The engine supports:
- Deterministic generation given a JAX PRNGKey
- Compositional probe generation for targeted recombination testing
- Integration with the existing Batch preprocessing pipeline
- Vocabulary management for tokenizer initialization
"""

import random
from dataclasses import dataclass
from typing import List, Literal, Tuple

import jax
import jax.numpy as jnp

from hbar.benchmarks.scan_grammar import SCANGrammar
from hbar.benchmarks.cogs_grammar import COGSGrammar
from hbar.engine.data_utils import Batch, prepare_batch
from hbar.engine.tokenizer import Tokenizer, create_scan_tokenizer

Domain = Literal["scan", "cogs"]


@dataclass(frozen=True)
class GrammarOutput:
    """Output from grammar generation.

    This immutable dataclass holds the results of grammar-based sample
    generation, including both the raw pairs and metadata for analysis.

    Attributes:
        pairs: List of (input_text, output_text) tuples.
        domain: The domain ('scan' or 'cogs') these samples come from.
        is_probe: Whether these are compositional probe samples.
    """

    pairs: List[Tuple[str, str]]
    domain: str
    is_probe: bool


class GrammarEngine:
    """Unified interface for compositional grammar generation.

    This engine wraps both SCAN and COGS grammars, providing a single
    interface for generating training batches with controlled compositional
    structure. It supports deterministic generation via JAX PRNGKey for
    reproducibility in JIT-compiled training loops.

    The engine is designed to support H-Bar signal extraction:
    - GCA: Generate compositional probe batches for gradient alignment
    - RGA: Support structural distance computation via COGS logical forms
    - AC: Generate structure-preserving augmentations

    Attributes:
        scan_grammar: SCAN grammar instance.
        cogs_grammar: COGS grammar instance.
        scan_tokenizer: Tokenizer for SCAN domain.
        cogs_tokenizer: Tokenizer for COGS domain.
    """

    def __init__(self, seed: int | None = None):
        """Initialize the grammar engine.

        Args:
            seed: Optional seed for reproducibility.
        """
        self.scan_grammar = SCANGrammar(seed=seed)
        self.cogs_grammar = COGSGrammar(seed=seed)

        # Initialize tokenizers with full vocabularies
        self.scan_tokenizer = self._create_scan_tokenizer()
        self.cogs_tokenizer = self._create_cogs_tokenizer()

    def _create_scan_tokenizer(self) -> Tokenizer:
        """Create tokenizer for SCAN domain.

        Returns:
            Tokenizer initialized with SCAN vocabulary.
        """
        vocab = self.scan_grammar.get_vocabulary()
        tokenizer = create_scan_tokenizer()
        # Add any additional vocabulary from grammar
        for word in vocab:
            if word not in tokenizer.word2id:
                tokenizer.add_vocabulary([word])
        return tokenizer

    def _create_cogs_tokenizer(self) -> Tokenizer:
        """Create tokenizer for COGS domain.

        Returns:
            Tokenizer initialized with COGS vocabulary.
        """
        vocab = self.cogs_grammar.get_vocabulary()
        return Tokenizer(vocab)

    def get_compositional_batch(
        self,
        batch_size: int,
        domain: Domain = "scan",
        max_depth: int = 3,
        rng: jax.Array | None = None,
    ) -> Batch:
        """Generate a batch of recombination-only samples.

        This method generates samples that specifically test compositional
        generalization by using novel combinations of known primitives.
        These are the primary OOD samples used for GCA signal extraction.

        Args:
            batch_size: Number of samples in the batch.
            domain: The domain ('scan' or 'cogs') to generate from.
            max_depth: Maximum recursion depth for grammar expansion.
            rng: Optional JAX PRNGKey for deterministic generation.

        Returns:
            Batch object containing training-ready tensors.
        """
        pairs = self.generate_compositional_pairs(
            batch_size=batch_size,
            domain=domain,
            max_depth=max_depth,
            rng=rng,
        )

        # Select appropriate tokenizer and max sequence length
        if domain == "scan":
            tokenizer = self.scan_tokenizer
            max_seq_len = 50
        else:
            tokenizer = self.cogs_tokenizer
            max_seq_len = 80

        return prepare_batch(pairs, tokenizer, max_seq_len)

    def generate_compositional_pairs(
        self,
        batch_size: int,
        domain: Domain = "scan",
        max_depth: int = 3,
        rng: jax.Array | None = None,
    ) -> List[Tuple[str, str]]:
        """Generate compositional probe pairs.

        This method generates pairs where primitives are used in novel
        syntactic contexts, testing structural recombination ability.

        Args:
            batch_size: Number of pairs to generate.
            domain: The domain to generate from.
            max_depth: Maximum recursion depth.
            rng: Optional JAX PRNGKey.

        Returns:
            List of (input_text, output_text) tuples.
        """
        # Convert JAX RNG to Python random state for reproducibility
        if rng is not None:
            # Extract integer seed from JAX PRNGKey
            seed = int(jnp.sum(rng))
            py_rng = random.Random(seed)
        else:
            py_rng = None

        pairs = []
        if domain == "scan":
            for _ in range(batch_size):
                # Generate probes with varying complexity
                complexity = random.randint(1, 3) if py_rng is None else py_rng.randint(1, 3)
                primitives = self.scan_grammar.primitives
                target = (
                    random.choice(primitives)
                    if py_rng is None
                    else py_rng.choice(primitives)
                )
                cmd, act = self.scan_grammar.sample_compositional_probe(
                    primitive_target=target,
                    complexity=complexity,
                )
                pairs.append((cmd, act))
        else:
            for _ in range(batch_size):
                constructions = ["active", "passive", "embedded_subject", "embedded_object"]
                construction = (
                    random.choice(constructions)
                    if py_rng is None
                    else py_rng.choice(constructions)
                )
                verbs = self.cogs_grammar.verbs_transitive
                target_verb = (
                    random.choice(verbs)
                    if py_rng is None
                    else py_rng.choice(verbs)
                )
                sent, lf = self.cogs_grammar.sample_compositional_probe(
                    target_verb=target_verb,
                    construction=construction,
                )
                pairs.append((sent, lf))

        return pairs

    def generate_id_batch(
        self,
        batch_size: int,
        domain: Domain = "scan",
        rng: jax.Array | None = None,
    ) -> Batch:
        """Generate an in-distribution training batch.

        These samples follow the standard distribution of the benchmark
        and are used for regular training (not OOD testing).

        Args:
            batch_size: Number of samples in the batch.
            domain: The domain to generate from.
            rng: Optional JAX PRNGKey.

        Returns:
            Batch object containing training-ready tensors.
        """
        if rng is not None:
            # Use 2**31 - 1 to avoid integer overflow with int32
            seed = int(jnp.sum(rng) % (2**31 - 1))
            py_rng = random.Random(seed)
        else:
            py_rng = None

        if domain == "scan":
            pairs = self.scan_grammar.generate_batch(
                batch_size=batch_size,
                max_depth=1,  # Simple samples for ID
                rng=py_rng,
            )
            tokenizer = self.scan_tokenizer
            max_seq_len = 50
        else:
            pairs = self.cogs_grammar.generate_batch(
                batch_size=batch_size,
                max_depth=1,
                rng=py_rng,
            )
            tokenizer = self.cogs_tokenizer
            max_seq_len = 80

        return prepare_batch(pairs, tokenizer, max_seq_len)

    def get_tokenizer(self, domain: Domain) -> Tokenizer:
        """Get the tokenizer for a specific domain.

        Args:
            domain: The domain ('scan' or 'cogs').

        Returns:
            Tokenizer instance for the domain.
        """
        if domain == "scan":
            return self.scan_tokenizer
        else:
            return self.cogs_tokenizer

    def get_vocabulary(self, domain: Domain) -> List[str]:
        """Get the vocabulary for a specific domain.

        Args:
            domain: The domain ('scan' or 'cogs').

        Returns:
            List of vocabulary tokens.
        """
        if domain == "scan":
            return self.scan_grammar.get_vocabulary()
        else:
            return self.cogs_grammar.get_vocabulary()

    def get_structural_distance(
        self,
        lf1: str,
        lf2: str,
        domain: Domain = "cogs",
    ) -> float:
        """Compute structural distance between two logical forms.

        This computes the RDMstruct for RGA signal extraction.
        Only applicable for COGS domain (SCAN actions are flat sequences).

        Args:
            lf1: First logical form string.
            lf2: Second logical form string.
            domain: Must be 'cogs' for structural distance.

        Returns:
            Normalized structural distance in [0, 1].

        Raises:
            ValueError: If domain is not 'cogs'.
        """
        if domain != "cogs":
            raise ValueError("Structural distance is only available for COGS domain")
        return self.cogs_grammar.get_structural_distance(lf1, lf2)

    def compute_rdmstruct(
        self,
        logical_forms: List[str],
    ) -> jax.Array:
        """Compute full RDMstruct matrix for a set of logical forms.

        This computes the pairwise structural distances between all
        logical forms, producing the Representational Dissimilarity
        Matrix (structural) for RGA signal extraction.

        Args:
            logical_forms: List of logical form strings.

        Returns:
            jax.Array of shape (n, n) with pairwise distances.
        """
        n = len(logical_forms)
        rdm = jnp.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.get_structural_distance(logical_forms[i], logical_forms[j])
                rdm = rdm.at[i, j].set(dist)
                rdm = rdm.at[j, i].set(dist)

        return rdm
