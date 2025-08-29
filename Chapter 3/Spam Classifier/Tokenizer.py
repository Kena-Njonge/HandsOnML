import hashlib
from typing import Generator, Callable, Deque
from collections import deque


class BaseTokenizer:
    """Base class for all tokenizers.

    I wrote it as an implemetation excercise, sklearn provides good enough implementations
    of this though, so my implementation was jsut devloving into a clone of that, as
    I don't like wasting time, I will differ my implementation.
    """

    def tokenize(self, text: str) -> Generator[str, None, None]:
        """Tokenize the input text and yield tokens."""
        raise NotImplementedError


class SimpleTokenizer(BaseTokenizer):
    """
    A simple tokenizer that splits text into tokens based on a delimiter within a sliding window.
    """

    def __init__(
        self, window_size: int = 1000, delimiter: str = ".", stop_words: list[str] = []
    ):
        """Initialize the SimpleTokenizer with window size, delimiter, and max split options."""
        self.window_size: int = window_size
        self.delimiter: str = delimiter

    def tokenize(self, text: str) -> Generator[str, None, None]:
        """
        Tokenizes the text by sliding a window, casefolding the text, and splitting by the delimiter.
        This was originally the _token_genenrator method.
        """
        for i in range(0, len(text), self.window_size):
            window = text[i : i + self.window_size]
            for token in window.split(self.delimiter):
                yield token


class HashTokenizer(SimpleTokenizer):
    """
    A tokenizer that extends SimpleTokenizer to hash the tokens it produces.
    """

    def __init__(
        self,
        window_size: int = 1000,
        delimiter: str = ".",
        hash_function: str = "basic",
    ):
        """Initialize the HashTokenizer."""
        super().__init__(window_size, delimiter)
        self._hash_func: Callable[[str], int]
        self.set_hash_function(hash_function)

    def set_hash_function(self, name: str | None):
        """
        Set the hash function to use for the hashing of tokens.

        Questionable if this is worth it for smaller datasets, can't evaluate that yet,
        Good material to read, evulate and extend this in the future are

        Feature Hashing for Large Scale Multitask Learning” – Weinberger et al., 2009
        """
        if name == "basic" or name is None:
            self._hash_func = hash
        elif name == "sha1":
            self._hash_func = lambda x: int(hashlib.sha1(x.encode()).hexdigest(), 16)
        else:
            raise ValueError(f"Unsupported hash function: {name}")

    def hash_token(self, token: str) -> int:
        """Hashes a single token using the configured hash function."""
        return self._hash_func(token)

    def tokenize(self, text: str) -> Generator[str, None, None]:
        """
        Tokenizes the text using the parent's logic and then yields the hash of each token as a string.
        """
        for token in super().tokenize(text):
            yield str(self.hash_token(token))


class NgramTokenizer(SimpleTokenizer):
    def __init__(
        self,
        window_size: int = 1000,
        delimiter: str = ".",
        n_gram_range: tuple[int, int] = (1, 2),
    ):
        super().__init__(window_size, delimiter)
        self.n_gram_range = n_gram_range

    def tokenize(self, text: str) -> Generator[str, None, None]:
        tokens_iter = super().tokenize(text)
        min_n, max_n = self.n_gram_range
        if min_n <= 0:
            return

        # Keep a buffer of max_n size
        buffer: Deque[str] = deque(maxlen=max_n)

        # Process tokens one at a time
        for token in tokens_iter:
            buffer.append(token)
            # Once we have enough tokens, yield n-grams for each n in range
            if len(buffer) >= min_n:
                for n in range(min_n, min(len(buffer) + 1, max_n + 1)):
                    # Get last n tokens from buffer for the n-gram
                    n_gram = list(buffer)[-n:]
                    yield self.delimiter.join(n_gram)
