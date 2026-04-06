"""JAX-native word-level tokenizer for SCAN/COGS compositional benchmarks.

This module implements a fast, word-level tokenizer designed for controlled
language benchmarks like SCAN and COGS. It uses fixed vocabularies and
produces JAX-compatible arrays for seamless integration with the Transformer.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

# Special token IDs
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3

# Special token strings
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


class Tokenizer:
    """Word-level tokenizer for SCAN/COGS benchmarks.

    This tokenizer maps words to integer IDs and vice versa, with special
    handling for padding, beginning-of-sequence, end-of-sequence, and
    unknown tokens. It produces fixed-length JAX arrays suitable for
    JIT-compiled model forward passes.

    Attributes:
        word2id: Dictionary mapping word strings to integer IDs.
        id2word: Dictionary mapping integer IDs to word strings.
        vocab_size: Total vocabulary size including special tokens.
    """

    def __init__(self, vocabulary: Sequence[str] | None = None):
        """Initialize the tokenizer with an optional vocabulary.

        Args:
            vocabulary: Optional list of words to include in the vocabulary.
                If not provided, initializes with empty vocabulary and
                only special tokens are available.
        """
        # Initialize with special tokens
        self.word2id: dict[str, int] = {
            PAD_TOKEN: PAD_TOKEN_ID,
            BOS_TOKEN: BOS_TOKEN_ID,
            EOS_TOKEN: EOS_TOKEN_ID,
            UNK_TOKEN: UNK_TOKEN_ID,
        }
        self.id2word: dict[int, str] = {v: k for k, v in self.word2id.items()}

        # Add vocabulary words if provided
        if vocabulary is not None:
            for word in vocabulary:
                if word not in self.word2id:
                    token_id = len(self.word2id)
                    self.word2id[word] = token_id
                    self.id2word[token_id] = word

        self.vocab_size = len(self.word2id)

    def add_vocabulary(self, words: Sequence[str]) -> None:
        """Add new words to the vocabulary.

        Args:
            words: Sequence of words to add to the vocabulary.
        """
        for word in words:
            if word not in self.word2id:
                token_id = len(self.word2id)
                self.word2id[word] = token_id
                self.id2word[token_id] = word
        self.vocab_size = len(self.word2id)

    def tokenize(self, text: str) -> list[str]:
        """Split text into word tokens.

        Args:
            text: Input text string to tokenize.

        Returns:
            List of word tokens.
        """
        return text.strip().split()

    def encode(
        self,
        text: str | list[str],
        max_seq_len: int,
    ) -> jax.Array:
        """Encode text to a fixed-length sequence of token IDs.

        This method tokenizes the input (if string), adds BOS and EOS tokens,
        and truncates or pads to the specified maximum sequence length.

        Args:
            text: Input text string or list of tokens.
            max_seq_len: Maximum sequence length (including BOS and EOS).

        Returns:
            JAX array of shape (max_seq_len,) containing token IDs.
        """
        # Tokenize if string
        if isinstance(text, str):
            tokens = self.tokenize(text)
        else:
            tokens = list(text)

        # Add BOS and EOS
        tokens = [BOS_TOKEN] + tokens + [EOS_TOKEN]

        # Truncate if necessary (but keep at least BOS and EOS)
        if len(tokens) > max_seq_len:
            tokens = tokens[: max_seq_len - 1] + [EOS_TOKEN]

        # Convert to IDs
        token_ids = [self.word2id.get(token, UNK_TOKEN_ID) for token in tokens]

        # Pad to max_seq_len
        padding_length = max_seq_len - len(token_ids)
        token_ids.extend([PAD_TOKEN_ID] * padding_length)

        return jnp.array(token_ids, dtype=jnp.int32)

    def encode_batch(
        self,
        texts: Sequence[str | list[str]],
        max_seq_len: int,
    ) -> jax.Array:
        """Encode a batch of texts to fixed-length sequences.

        Args:
            texts: Sequence of text strings or token lists.
            max_seq_len: Maximum sequence length for each sequence.

        Returns:
            JAX array of shape (batch_size, max_seq_len) containing token IDs.
        """
        encoded = [self.encode(text, max_seq_len) for text in texts]
        return jnp.stack(encoded, axis=0)

    def decode(
        self,
        token_ids: jax.Array,
        skip_special: bool = True,
    ) -> str:
        """Decode token IDs back to a text string.

        Args:
            token_ids: JAX array of token IDs.
            skip_special: If True, skip special tokens (BOS, EOS, PAD) in output.

        Returns:
            Decoded text string.
        """
        # Convert to numpy for iteration
        ids = np.asarray(token_ids).tolist()

        # Handle batched input
        if isinstance(ids[0], list):
            return "\n".join(self.decode(jnp.array(batch_ids), skip_special) for batch_ids in ids)

        words = []
        special_ids = {PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID}

        for token_id in ids:
            if skip_special and token_id in special_ids:
                continue
            if token_id == UNK_TOKEN_ID:
                words.append(UNK_TOKEN)
            elif token_id in self.id2word:
                words.append(self.id2word[token_id])

        return " ".join(words)

    def get_pad_token_id(self) -> int:
        """Return the padding token ID."""
        return PAD_TOKEN_ID

    def get_bos_token_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        return BOS_TOKEN_ID

    def get_eos_token_id(self) -> int:
        """Return the end-of-sequence token ID."""
        return EOS_TOKEN_ID


def create_scan_tokenizer() -> Tokenizer:
    """Create a tokenizer pre-initialized with SCAN benchmark vocabulary.

    The SCAN vocabulary includes commands (e.g., "jump", "walk", "left")
    and actions (e.g., "I_JUMP", "I_WALK", "I_TURN_LEFT").

    Returns:
        Tokenizer instance with SCAN vocabulary.
    """
    # SCAN command vocabulary
    commands = [
        "jump",
        "walk",
        "run",
        "look",
        "turn",
        "left",
        "right",
        "twice",
        "thrice",
        "after",
        "opposite",
        "around",
        "and",
    ]

    # SCAN action vocabulary
    actions = [
        "I_TURN_LEFT",
        "I_TURN_RIGHT",
        "I_LOOK",
        "I_WALK",
        "I_RUN",
        "I_JUMP",
    ]

    vocabulary = commands + actions
    return Tokenizer(vocabulary)
