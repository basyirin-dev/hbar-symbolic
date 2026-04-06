"""SCAN Context-Free Grammar implementation for compositional generalization.

This module implements the generative grammar G(d) for the SCAN benchmark
as described in Lake & Baroni (2018). It provides a recursive CFG that
generates command strings and their corresponding action sequences.

The grammar supports:
- Primitives: jump, run, walk, look
- Directions: left, right
- Modifiers: twice, thrice
- Conjunctions: and, after
- Composite operations: turn, opposite, around

This implementation enables infinite out-of-distribution sampling for
testing compositional generalization in neural networks.
"""

import random
from typing import List, Tuple


class SCANGrammar:
    """Context-free grammar for SCAN benchmark.

    This class implements a recursive CFG that generates command-action
    pairs following the compositional structure of the SCAN benchmark.
    The grammar is designed to produce both in-distribution and
    out-of-distribution samples for testing compositional generalization.

    The grammar rules follow Lake & Baroni (2018):
        S -> S conjunction S | VP
        VP -> primitive | primitive direction | primitive modifier | turn direction
        conjunction -> and | after

    Attributes:
        primitives: List of primitive actions (jump, run, walk, look).
        directions: List of direction modifiers (left, right).
        modifiers: List of repetition modifiers (twice, thrice).
        conjunctions: List of conjunction words (and, after).
    """

    def __init__(self, seed: int | None = None):
        """Initialize the SCAN grammar.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.primitives: List[str] = ["jump", "run", "walk", "look"]
        self.directions: List[str] = ["left", "right"]
        self.modifiers: List[str] = ["twice", "thrice"]
        self.conjunctions: List[str] = ["and", "after"]

        # Action mappings
        self.primitive_to_action: dict[str, str] = {
            "jump": "I_JUMP",
            "run": "I_RUN",
            "walk": "I_WALK",
            "look": "I_LOOK",
        }

        self.direction_to_action: dict[str, str] = {
            "left": "I_TURN_LEFT",
            "right": "I_TURN_RIGHT",
        }

        if seed is not None:
            random.seed(seed)

    def _generate_primitive(self) -> Tuple[str, str]:
        """Generate a primitive command and its action.

        Returns:
            Tuple of (command_string, action_string).
        """
        primitive = random.choice(self.primitives)
        action = self.primitive_to_action[primitive]
        return primitive, action

    def _generate_direction_phrase(self) -> Tuple[str, str]:
        """Generate a direction phrase (e.g., 'turn left').

        Returns:
            Tuple of (command_string, action_string).
        """
        direction = random.choice(self.directions)
        action = self.direction_to_action[direction]
        return f"turn {direction}", f"{action}"

    def _generate_modifier_phrase(self) -> Tuple[str, str]:
        """Generate a modified primitive (e.g., 'jump twice').

        Returns:
            Tuple of (command_string, action_string).
        """
        primitive, action = self._generate_primitive()
        modifier = random.choice(self.modifiers)

        # Repeat the action
        repeat_count = 2 if modifier == "twice" else 3
        repeated_action = " ".join([action] * repeat_count)

        return f"{primitive} {modifier}", repeated_action

    def _generate_simple_vp(self) -> Tuple[str, str]:
        """Generate a simple verb phrase (primitive or direction phrase).

        Returns:
            Tuple of (command_string, action_string).
        """
        # Choose between primitive and direction phrase
        if random.random() < 0.7:
            return self._generate_primitive()
        else:
            return self._generate_direction_phrase()

    def _generate_vp(self) -> Tuple[str, str]:
        """Generate a verb phrase (possibly with modifiers).

        Returns:
            Tuple of (command_string, action_string).
        """
        # Choose between simple VP and modifier phrase
        if random.random() < 0.6:
            return self._generate_simple_vp()
        else:
            return self._generate_modifier_phrase()

    def _generate_conjunction(self) -> Tuple[str, str]:
        """Generate a conjunction (and/after).

        Returns:
            Tuple of (conjunction_string, empty_action_separator).
        """
        conjunction = random.choice(self.conjunctions)
        return conjunction, ""

    def generate_sample(self, max_depth: int = 3) -> Tuple[str, str]:
        """Generate a random command and its corresponding action sequence.

        This method recursively builds a command string following the CFG
        rules. The max_depth parameter controls the complexity of generated
        commands to prevent infinite recursion.

        Args:
            max_depth: Maximum recursion depth for conjunction expansion.
                Higher values produce more complex commands.

        Returns:
            Tuple of (command_string, action_string).
        """
        if max_depth <= 0:
            # Base case: generate a simple VP
            return self._generate_vp()

        # Decide whether to use conjunction (recursive) or simple VP
        if random.random() < 0.4 and max_depth > 0:
            # Generate S -> S conjunction S
            left_cmd, left_act = self.generate_sample(max_depth - 1)
            conjunction, _ = self._generate_conjunction()
            right_cmd, right_act = self.generate_sample(max_depth - 1)

            command = f"{left_cmd} {conjunction} {right_cmd}"
            action = f"{left_act} {right_act}"
            return command, action
        else:
            # Generate simple VP
            return self._generate_vp()

    def sample_compositional_probe(
        self,
        primitive_target: str = "jump",
        complexity: int = 2,
    ) -> Tuple[str, str]:
        """Generate a compositional probe targeting a specific primitive.

        This method generates samples where the target primitive is used
        in complex, nested contexts to test for structural recombination.
        This is crucial for H-Bar GCA (Gradient-Composition Alignment)
        signal extraction.

        Examples of generated probes:
            - "jump around left twice and look thrice"
            - "jump twice after turn right"
            - "jump and walk left after jump thrice"

        Args:
            primitive_target: The primitive action to target (e.g., 'jump').
            complexity: Complexity level of the probe (1=simple, 3=complex).

        Returns:
            Tuple of (command_string, action_string).
        """
        if primitive_target not in self.primitives:
            raise ValueError(
                f"Unknown primitive: {primitive_target}. "
                f"Must be one of {self.primitives}"
            )

        target_action = self.primitive_to_action[primitive_target]

        if complexity == 1:
            # Simple: "jump twice" or "jump"
            if random.random() < 0.5:
                modifier = random.choice(self.modifiers)
                repeat_count = 2 if modifier == "twice" else 3
                return (
                    f"{primitive_target} {modifier}",
                    " ".join([target_action] * repeat_count),
                )
            else:
                return primitive_target, target_action

        elif complexity == 2:
            # Medium: "jump around left twice" or "jump twice and walk"
            probe_type = random.choice(["around", "conjunction", "direction"])

            if probe_type == "around":
                # "jump around left twice" -> I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP
                direction = random.choice(self.directions)
                dir_action = self.direction_to_action[direction]
                return (
                    f"{primitive_target} around {direction}",
                    f"{target_action} {dir_action} {target_action}",
                )

            elif probe_type == "conjunction":
                # "jump twice and [other]"
                other_cmd, other_act = self._generate_vp()
                conjunction = random.choice(self.conjunctions)
                modifier = random.choice(self.modifiers)
                repeat_count = 2 if modifier == "twice" else 3
                return (
                    f"{primitive_target} {modifier} {conjunction} {other_cmd}",
                    f"{' '.join([target_action] * repeat_count)} {other_act}",
                )

            else:
                # "jump left" is invalid, use "turn left" pattern
                # Instead: "jump after turn left"
                direction = random.choice(self.directions)
                dir_action = self.direction_to_action[direction]
                conjunction = random.choice(self.conjunctions)
                return (
                    f"{primitive_target} {conjunction} turn {direction}",
                    f"{target_action} {dir_action}",
                )

        else:
            # High complexity: nested structures
            # "jump around left twice and look thrice"
            direction = random.choice(self.directions)
            dir_action = self.direction_to_action[direction]
            modifier1 = random.choice(self.modifiers)
            repeat1 = 2 if modifier1 == "twice" else 3
            other_primitive = random.choice(
                [p for p in self.primitives if p != primitive_target]
            )
            other_action = self.primitive_to_action[other_primitive]
            modifier2 = random.choice(self.modifiers)
            repeat2 = 2 if modifier2 == "twice" else 3
            conjunction = random.choice(self.conjunctions)

            # Build complex command
            cmd_part1 = f"{primitive_target} around {direction} {modifier1}"
            cmd_part2 = f"{other_primitive} {modifier2}"
            command = f"{cmd_part1} {conjunction} {cmd_part2}"

            # Build corresponding action
            act_part1 = " ".join(
                [f"{target_action} {dir_action}"] * (repeat1 - 1) + [target_action]
            )
            act_part2 = " ".join([other_action] * repeat2)
            action = f"{act_part1} {act_part2}"

            return command, action

    def generate_batch(
        self,
        batch_size: int,
        max_depth: int = 3,
        rng: random.Random | None = None,
    ) -> List[Tuple[str, str]]:
        """Generate a batch of command-action pairs.

        Args:
            batch_size: Number of samples to generate.
            max_depth: Maximum recursion depth for each sample.
            rng: Optional Random instance for reproducibility.

        Returns:
            List of (command_string, action_string) tuples.
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

    def get_add_jump_split(
        self,
        n_train: int = 2000,
        n_test: int = 2000,
        rng: random.Random | None = None,
    ) -> tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Generate Add-Jump split for SCAN benchmark.

        This implements the "Add-Primitive" partitioning where:
        - Training set: All commands WITHOUT 'jump' in compounds, plus 'jump' alone
        - Test set: All commands where 'jump' appears in a compound structure

        This split tests whether the model can compose a known primitive ('jump')
        into novel syntactic structures (e.g., 'jump twice', 'jump left').

        Args:
            n_train: Number of training samples to generate.
            n_test: Number of test samples to generate.
            rng: Optional Random instance for reproducibility.

        Returns:
            Tuple of (train_pairs, test_pairs) where each is a list of
            (command_string, action_string) tuples.
        """
        if rng is not None:
            state = random.getstate()
            random.setstate(rng.getstate())

        train_pairs = []
        test_pairs = []

        # Add the isolated 'jump' command to training set
        train_pairs.append(("jump", "I_JUMP"))

        # Generate training samples (no 'jump' in compounds)
        attempts = 0
        max_attempts = n_train * 10
        while len(train_pairs) < n_train and attempts < max_attempts:
            cmd, act = self.generate_sample(max_depth=2)
            # Check if command contains 'jump'
            if "jump" not in cmd.lower():
                train_pairs.append((cmd, act))
            attempts += 1

        # Generate test samples (must contain 'jump' in a compound)
        attempts = 0
        max_attempts = n_test * 10
        while len(test_pairs) < n_test and attempts < max_attempts:
            cmd, act = self.sample_compositional_probe(
                primitive_target="jump",
                complexity=random.randint(1, 3),
            )
            # Ensure it's a compound (not just "jump")
            if cmd.lower() != "jump":
                test_pairs.append((cmd, act))
            attempts += 1

        if rng is not None:
            rng.setstate(random.getstate())
            random.setstate(state)

        return train_pairs, test_pairs

    def get_vocabulary(self) -> List[str]:
        """Get all vocabulary tokens used by the grammar.

        Returns:
            List of all unique tokens (commands and actions).
        """
        vocab = set()
        vocab.update(self.primitives)
        vocab.update(self.directions)
        vocab.update(self.modifiers)
        vocab.update(self.conjunctions)
        vocab.add("turn")
        vocab.add("around")
        vocab.update(self.primitive_to_action.values())
        vocab.update(self.direction_to_action.values())
        return sorted(vocab)
