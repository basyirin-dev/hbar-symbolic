"""COGS (Compositional Generalization in Sentence Processing) grammar implementation.

This module implements the generative grammar G(d) for the COGS benchmark.
COGS maps English sentences to Logical Forms (LF), requiring understanding
of syntactic structure and semantic composition.

The grammar supports:
- Active voice: 'The dog chased the cat' -> chase ( agent = dog , patient = cat )
- Passive voice: 'The cat was chased by the dog'
- Nested structures: 'The boy said that the dog chased the cat'
- Prepositional phrases and ditransitive constructions

This implementation includes tree-edit distance computation for RDMstruct
(Representational Dissimilarity Matrix structural) as required for H-Bar
RGA (Representational-Geometry Alignment) signal extraction.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class LogicalForm:
    """Represents a logical form as a tree structure.

    This immutable dataclass represents the semantic representation of
    a sentence as a tree, enabling tree-edit distance computation for
    structural similarity analysis.

    Attributes:
        predicate: The main predicate/verb of this clause.
        args: Dictionary mapping argument roles to their values.
        children: List of child LogicalForms for embedded clauses.
    """

    predicate: str
    args: dict[str, str]
    children: Tuple["LogicalForm", ...] = ()

    def to_string(self) -> str:
        """Convert the logical form to its string representation.

        Returns:
            String representation like 'chase ( agent = dog , patient = cat )'.
        """
        if not self.args and not self.children:
            return self.predicate

        parts = [self.predicate, "("]
        arg_strs = []
        for role, value in self.args.items():
            if isinstance(value, LogicalForm):
                arg_strs.append(f"{role} = {value.to_string()}")
            else:
                arg_strs.append(f"{role} = {value}")
        parts.append(" , ".join(arg_strs))

        if self.children:
            for child in self.children:
                parts.append(f" , {child.to_string()}")

        parts.append(")")
        return " ".join(parts)

    def get_tokens(self) -> List[str]:
        """Get the tokens of the logical form for tokenization.

        Returns:
            List of tokens including predicates, roles, and punctuation.
        """
        tokens = [self.predicate, "(", ","]
        for role, value in self.args.items():
            tokens.extend([role, "=", value])
        if self.children:
            for child in self.children:
                tokens.extend(child.get_tokens())
        tokens.append(")")
        return tokens


class COGSGrammar:
    """Context-free grammar for COGS benchmark.

    This class implements a recursive CFG that generates sentence-logical
    form pairs following the compositional structure of the COGS benchmark.
    The grammar supports various syntactic constructions including active,
    passive, and nested structures.

    Attributes:
        nouns: List of common nouns for subjects and objects.
        verbs_transitive: List of transitive verbs (take subject and object).
        verbs_intransitive: List of intransitive verbs (take only subject).
        verbs_ditransitive: List of ditransitive verbs (take indirect object).
        prepositions: List of prepositions for PP attachments.
    """

    # Biased nouns for Subject-to-Object split
    # These nouns appear only in subject position during training
    # and only in object position during testing
    BIASED_NOUNS: List[str] = ["hedgehog", "porcupine", "otter"]

    def __init__(self, seed: int | None = None):
        """Initialize the COGS grammar.

        Args:
            seed: Optional random seed for reproducibility.
        """
        # Noun vocabulary
        self.nouns: List[str] = [
            "dog",
            "cat",
            "boy",
            "girl",
            "man",
            "woman",
            "child",
            "teacher",
            "student",
            "bird",
        ]

        # Verb vocabulary by type
        self.verbs_transitive: List[str] = [
            "chased",
            "saw",
            "greeted",
            "liked",
            "found",
            "helped",
            "called",
            "followed",
        ]
        self.verbs_intransitive: List[str] = [
            "ran",
            "slept",
            "laughed",
            "cried",
            "arrived",
            "left",
        ]
        self.verbs_ditransitive: List[str] = [
            "gave",
            "sent",
            "showed",
            "told",
            "offered",
        ]

        # Embedding verbs
        self.verbs_embedding: List[str] = ["said", "thought", "knew", "believed"]

        # Prepositions
        self.prepositions: List[str] = ["to", "from", "with", "near", "behind"]

        if seed is not None:
            random.seed(seed)

    def _get_noun_phrase(self) -> Tuple[str, str]:
        """Generate a simple noun phrase.

        Returns:
            Tuple of (surface_form, semantic_referent).
        """
        article = random.choice(["The", "A"])
        noun = random.choice(self.nouns)
        return f"{article} {noun}", noun

    def _generate_active_transitive(self) -> Tuple[str, LogicalForm]:
        """Generate an active transitive sentence.

        Example: 'The dog chased the cat' -> chase ( agent = dog , patient = cat )

        Returns:
            Tuple of (sentence_string, logical_form).
        """
        subject_np, subject_ref = self._get_noun_phrase()
        verb = random.choice(self.verbs_transitive)
        object_np, object_ref = self._get_noun_phrase()

        # Ensure subject and object are different
        while object_ref == subject_ref:
            object_np, object_ref = self._get_noun_phrase()

        # Get base form of verb for LF
        base_verb = self._get_base_form(verb)

        sentence = f"{subject_np} {verb} {object_np}"
        lf = LogicalForm(
            predicate=base_verb,
            args={"agent": subject_ref, "patient": object_ref},
        )
        return sentence, lf

    def _generate_passive_transitive(self) -> Tuple[str, LogicalForm]:
        """Generate a passive transitive sentence.

        Example: 'The cat was chased by the dog' -> chase ( agent = dog , patient = cat )

        Returns:
            Tuple of (sentence_string, logical_form).
        """
        subject_np, subject_ref = self._get_noun_phrase()
        verb = random.choice(self.verbs_transitive)
        object_np, object_ref = self._get_noun_phrase()

        # Ensure subject and object are different
        while object_ref == subject_ref:
            object_np, object_ref = self._get_noun_phrase()

        base_verb = self._get_base_form(verb)

        sentence = f"{subject_np} was {verb} by {object_np}"
        lf = LogicalForm(
            predicate=base_verb,
            args={"agent": object_ref, "patient": subject_ref},
        )
        return sentence, lf

    def _generate_intransitive(self) -> Tuple[str, LogicalForm]:
        """Generate an intransitive sentence.

        Example: 'The dog ran' -> run ( agent = dog )

        Returns:
            Tuple of (sentence_string, logical_form).
        """
        subject_np, subject_ref = self._get_noun_phrase()
        verb = random.choice(self.verbs_intransitive)
        base_verb = self._get_base_form(verb)

        sentence = f"{subject_np} {verb}"
        lf = LogicalForm(predicate=base_verb, args={"agent": subject_ref})
        return sentence, lf

    def _generate_ditransitive(self) -> Tuple[str, LogicalForm]:
        """Generate a ditransitive sentence.

        Example: 'The boy gave the girl the book' ->
                  give ( agent = boy , patient = book , recipient = girl )

        Returns:
            Tuple of (sentence_string, logical_form).
        """
        subject_np, subject_ref = self._get_noun_phrase()
        verb = random.choice(self.verbs_ditransitive)
        io_np, io_ref = self._get_noun_phrase()

        while io_ref == subject_ref:
            io_np, io_ref = self._get_noun_phrase()

        # Use a concrete object for ditransitive
        objects = ["book", "ball", "flower", "letter", "gift"]
        obj = random.choice(objects)

        base_verb = self._get_base_form(verb)

        sentence = f"{subject_np} {verb} {io_np} the {obj}"
        lf = LogicalForm(
            predicate=base_verb,
            args={"agent": subject_ref, "recipient": io_ref, "patient": obj},
        )
        return sentence, lf

    def _generate_embedded(self) -> Tuple[str, LogicalForm]:
        """Generate a sentence with embedded clause.

        Example: 'The boy said that the dog chased the cat' ->
                  say ( agent = boy , theme = chase ( agent = dog , patient = cat ) )

        Returns:
            Tuple of (sentence_string, logical_form).
        """
        subject_np, subject_ref = self._get_noun_phrase()
        embedding_verb = random.choice(self.verbs_embedding)

        # Generate embedded clause
        embedded_sentence, embedded_lf = self._generate_simple_clause()

        sentence = f"{subject_np} {embedding_verb} that {embedded_sentence}"
        lf = LogicalForm(
            predicate=embedding_verb,
            args={"agent": subject_ref, "theme": embedded_lf},
            children=embedded_lf.children,
        )
        return sentence, lf

    def _generate_simple_clause(self) -> Tuple[str, LogicalForm]:
        """Generate a simple clause (active, passive, intransitive, or ditransitive).

        Returns:
            Tuple of (sentence_string, logical_form).
        """
        clause_type = random.choice(
            ["active", "passive", "intransitive", "ditransitive"]
        )
        if clause_type == "active":
            return self._generate_active_transitive()
        elif clause_type == "passive":
            return self._generate_passive_transitive()
        elif clause_type == "intransitive":
            return self._generate_intransitive()
        else:
            return self._generate_ditransitive()

    def generate_sample(self, max_depth: int = 2) -> Tuple[str, str]:
        """Generate a random sentence and its logical form.

        Args:
            max_depth: Maximum recursion depth for embedded clauses.

        Returns:
            Tuple of (sentence_string, logical_form_string).
        """
        if max_depth <= 0:
            # Base case: simple clause only
            sentence, lf = self._generate_simple_clause()
        elif random.random() < 0.3:
            # Embedding
            sentence, lf = self._generate_embedded()
        else:
            sentence, lf = self._generate_simple_clause()

        return sentence, lf.to_string()

    def generate_sample_with_lf(
        self, max_depth: int = 2
    ) -> Tuple[str, LogicalForm]:
        """Generate a sentence with its LogicalForm object.

        This method returns the LogicalForm object directly for tree-edit
        distance computation.

        Args:
            max_depth: Maximum recursion depth for embedded clauses.

        Returns:
            Tuple of (sentence_string, LogicalForm).
        """
        if max_depth <= 0:
            sentence, lf = self._generate_simple_clause()
        elif random.random() < 0.3:
            sentence, lf = self._generate_embedded()
        else:
            sentence, lf = self._generate_simple_clause()

        return sentence, lf

    def sample_compositional_probe(
        self,
        target_verb: str = "chased",
        construction: str = "active",
    ) -> Tuple[str, str]:
        """Generate a compositional probe targeting a specific verb/construction.

        This method generates samples where the target verb appears in
        specific syntactic contexts to test for structural recombination.

        Args:
            target_verb: The verb to target (e.g., 'chased').
            construction: The syntactic construction ('active', 'passive',
                'embedded_subject', 'embedded_object').

        Returns:
            Tuple of (sentence_string, logical_form_string).
        """
        subject_np, subject_ref = self._get_noun_phrase()
        object_np, object_ref = self._get_noun_phrase()

        while object_ref == subject_ref:
            object_np, object_ref = self._get_noun_phrase()

        base_verb = self._get_base_form(target_verb)

        if construction == "active":
            sentence = f"{subject_np} {target_verb} {object_np}"
            lf = LogicalForm(
                predicate=base_verb,
                args={"agent": subject_ref, "patient": object_ref},
            )

        elif construction == "passive":
            sentence = f"{object_np} was {target_verb} by {subject_np}"
            lf = LogicalForm(
                predicate=base_verb,
                args={"agent": subject_ref, "patient": object_ref},
            )

        elif construction == "embedded_subject":
            # 'The boy said that [subject] chased the cat'
            embedding_np, embedding_ref = self._get_noun_phrase()
            embedding_verb = random.choice(self.verbs_embedding)
            sentence = (
                f"{embedding_np} {embedding_verb} that "
                f"{subject_np} {target_verb} {object_np}"
            )
            embedded_lf = LogicalForm(
                predicate=base_verb,
                args={"agent": subject_ref, "patient": object_ref},
            )
            lf = LogicalForm(
                predicate=embedding_verb,
                args={"agent": embedding_ref, "theme": embedded_lf},
            )

        elif construction == "embedded_object":
            # 'The boy said that the cat was chased by [object]'
            embedding_np, embedding_ref = self._get_noun_phrase()
            embedding_verb = random.choice(self.verbs_embedding)
            sentence = (
                f"{embedding_np} {embedding_verb} that "
                f"{object_np} was {target_verb} by {subject_np}"
            )
            embedded_lf = LogicalForm(
                predicate=base_verb,
                args={"agent": subject_ref, "patient": object_ref},
            )
            lf = LogicalForm(
                predicate=embedding_verb,
                args={"agent": embedding_ref, "theme": embedded_lf},
            )

        else:
            raise ValueError(f"Unknown construction: {construction}")

        return sentence, lf.to_string()

    def get_structural_distance(
        self, lf1_str: str, lf2_str: str, parse: bool = True
    ) -> float:
        """Compute tree-edit distance between two logical forms.

        This computes the RDMstruct (Representational Dissimilarity Matrix
        structural) required for H-Bar RGA signal extraction (Equation 4).

        The distance is normalized to [0, 1] where 0 means identical
        structure and 1 means maximally different.

        Args:
            lf1_str: First logical form string.
            lf2_str: Second logical form string.
            parse: Whether to parse the strings into LogicalForm trees.

        Returns:
            Normalized tree-edit distance in [0, 1].
        """
        if parse:
            # Simple token-based edit distance as approximation
            tokens1 = lf1_str.replace("(", " ( ").replace(")", " ) ").split()
            tokens2 = lf2_str.replace("(", " ( ").replace(")", " ) ").split()
        else:
            tokens1 = lf1_str.split()
            tokens2 = lf2_str.split()

        # Compute Levenshtein distance
        len1, len2 = len(tokens1), len(tokens2)
        if len1 == 0 and len2 == 0:
            return 0.0

        # Use dynamic programming for edit distance
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],  # deletion
                        dp[i][j - 1],  # insertion
                        dp[i - 1][j - 1],  # substitution
                    )

        # Normalize by maximum possible distance
        max_dist = max(len1, len2)
        return dp[len1][len2] / max_dist if max_dist > 0 else 0.0

    def get_vocabulary(self) -> List[str]:
        """Get all vocabulary tokens used by the grammar.

        Returns:
            List of all unique tokens.
        """
        vocab = set()
        vocab.update(["The", "A", "the", "a"])  # Articles (capitalized and lowercase)
        vocab.update(self.nouns)
        vocab.update(self.verbs_transitive)
        vocab.update(self.verbs_intransitive)
        vocab.update(self.verbs_ditransitive)
        vocab.update(self.verbs_embedding)
        vocab.update(self.prepositions)
        vocab.update(["was", "by", "that"])  # Function words
        vocab.update(["book", "ball", "flower", "letter", "gift"])  # Objects

        # Add LF tokens
        vocab.update(["(", ")", ",", "=", "agent", "patient", "recipient", "theme"])
        # Add base forms of verbs
        for v in self.verbs_transitive + self.verbs_intransitive:
            vocab.add(self._get_base_form(v))
        for v in self.verbs_ditransitive + self.verbs_embedding:
            vocab.add(self._get_base_form(v))

        return sorted(vocab)

    @staticmethod
    def _get_base_form(verb: str) -> str:
        """Get the base form of a verb.

        Args:
            verb: Inflected verb form.

        Returns:
            Base form of the verb.
        """
        # Handle common irregular verbs and patterns
        irregular = {
            "chased": "chase",
            "gave": "give",
            "sent": "send",
            "showed": "show",
            "told": "tell",
            "offered": "offer",
            "said": "say",
            "thought": "think",
            "knew": "know",
            "believed": "believe",
            "ran": "run",
            "slept": "sleep",
            "laughed": "laugh",
            "cried": "cry",
            "arrived": "arrive",
            "left": "leave",
            "saw": "see",
            "greeted": "greet",
            "liked": "like",
            "found": "find",
            "helped": "help",
            "called": "call",
            "followed": "follow",
        }
        if verb in irregular:
            return irregular[verb]
        if verb.endswith("ed"):
            return verb[:-2]
        elif verb.endswith("d"):
            return verb[:-1]
        return verb

    def get_subject_object_split(
        self,
        n_train: int = 2000,
        n_test: int = 2000,
        rng: random.Random | None = None,
    ) -> tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Generate Subject-to-Object split for COGS benchmark.

        This implements the "Subject-to-Object" partitioning where:
        - Training set: BIASED_NOUNS only appear in Subject position
        - Test set: BIASED_NOUNS only appear in Object position

        This split tests whether the model can generalize noun roles
        across different syntactic positions.

        Args:
            n_train: Number of training samples to generate.
            n_test: Number of test samples to generate.
            rng: Optional Random instance for reproducibility.

        Returns:
            Tuple of (train_pairs, test_pairs) where each is a list of
            (sentence_string, logical_form_string) tuples.
        """
        if rng is not None:
            state = random.getstate()
            random.setstate(rng.getstate())

        train_pairs = []
        test_pairs = []

        # Generate training samples (biased nouns only in subject position)
        attempts = 0
        max_attempts = n_train * 10
        while len(train_pairs) < n_train and attempts < max_attempts:
            sample = self._generate_subject_only_sample()
            if sample is not None:
                train_pairs.append(sample)
            attempts += 1

        # Generate test samples (biased nouns only in object position)
        attempts = 0
        max_attempts = n_test * 10
        while len(test_pairs) < n_test and attempts < max_attempts:
            sample = self._generate_object_only_sample()
            if sample is not None:
                test_pairs.append(sample)
            attempts += 1

        if rng is not None:
            rng.setstate(random.getstate())
            random.setstate(state)

        return train_pairs, test_pairs

    def _generate_subject_only_sample(self) -> Tuple[str, str] | None:
        """Generate a sample where biased nouns only appear in subject position.

        Returns:
            Tuple of (sentence_string, logical_form_string) or None if invalid.
        """
        # Use biased noun as subject
        subject_noun = random.choice(self.BIASED_NOUNS)
        article = random.choice(["The", "A"])
        subject_np = f"{article} {subject_noun}"

        # Use non-biased noun as object
        object_noun = random.choice(self.nouns)
        object_article = random.choice(["the", "a"])
        object_np = f"{object_article} {object_noun}"

        # Generate with active transitive
        verb = random.choice(self.verbs_transitive)
        base_verb = self._get_base_form(verb)

        sentence = f"{subject_np} {verb} {object_np}"
        lf = LogicalForm(
            predicate=base_verb,
            args={"agent": subject_noun, "patient": object_noun},
        )
        return sentence, lf.to_string()

    def _generate_object_only_sample(self) -> Tuple[str, str] | None:
        """Generate a sample where biased nouns only appear in object position.

        Returns:
            Tuple of (sentence_string, logical_form_string) or None if invalid.
        """
        # Use non-biased noun as subject
        subject_noun = random.choice(self.nouns)
        article = random.choice(["The", "A"])
        subject_np = f"{article} {subject_noun}"

        # Use biased noun as object
        object_noun = random.choice(self.BIASED_NOUNS)
        object_article = random.choice(["the", "a"])
        object_np = f"{object_article} {object_noun}"

        # Generate with active transitive
        verb = random.choice(self.verbs_transitive)
        base_verb = self._get_base_form(verb)

        sentence = f"{subject_np} {verb} {object_np}"
        lf = LogicalForm(
            predicate=base_verb,
            args={"agent": subject_noun, "patient": object_noun},
        )
        return sentence, lf.to_string()

    def generate_batch(
        self,
        batch_size: int,
        max_depth: int = 2,
        rng: random.Random | None = None,
    ) -> List[Tuple[str, str]]:
        """Generate a batch of sentence-logical form pairs.

        Args:
            batch_size: Number of samples to generate.
            max_depth: Maximum recursion depth for each sample.
            rng: Optional Random instance for reproducibility.

        Returns:
            List of (sentence_string, logical_form_string) tuples.
        """
        if rng is not None:
            state = random.getstate()
            random.setstate(rng.getstate())

        samples = []
        for _ in range(batch_size):
            samples.append(self.generate_sample(max_depth))

        if rng is not None:
            rng.setstate(random.getstate())
            random.setstate(state)

        return samples
