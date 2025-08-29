from collections import defaultdict
from typing import Literal
import numpy as np
from .Tokenizer import BaseTokenizer


class TokenVectorizer:
    """
    This class is responsible for fitting on a dataset to learn a vocabulary
    and transforming text into numerical vectors based on that vocabulary.

    I wrote it as an implemetation excercise, sklearn provides good enough implementations
    of this though, so my implementation was jsut devloving into a clone of that, as 
    I don't like wasting time, I will differ my implementation.
    """

    def __init__(self, tokenizer: BaseTokenizer, max_dims: int = 784):
        """Initialize the TokenVectorizer with a tokenizer and max dimensions for the vector."""
        self.tokenizer = tokenizer
        self.max_dims = max_dims
        self.vector_dims: list[tuple[str, int]] = []
        self.frequencies = []
        self.mapping: defaultdict[str, int] = defaultdict(int)

    def fit(self, dataset: list[str]):
        """Fit on the dataset by processing each string to build the vocabulary."""
        for i in dataset:
            self.fit_str(i)

        # The mapping now directly contains token counts, so we can get items directly.
        flattened_mapping = list(self.mapping.items())

        # Sort tokens by frequency to determine the vocabulary
        self.vector_dims = sorted(flattened_mapping, key=lambda x: x[1], reverse=True)

        # Truncate or pad the vocabulary to match max_dims
        if len(self.vector_dims) > self.max_dims:
            self.vector_dims = self.vector_dims[: self.max_dims]
        elif len(self.vector_dims) < self.max_dims:
            self.vector_dims.extend([("", 0)] * (self.max_dims - len(self.vector_dims)))

    def fit_str(self, text: str) -> None:
        """Fit the vectorizer on a single string by counting token occurrences."""
        for token in self.tokenizer.tokenize(text):
            self.mapping[token] += 1

    def transform(self, dataset: list[str]) -> np.ndarray:
        """Transform a dataset of strings into a numpy array of vectors."""
        if not self.vector_dims:
            raise ValueError(
                "The vector dimensions have not been learned yet. Please call the `fit` method before using `transform`."
            )

        result: list[list[int]] = []
        for text in dataset:
            result.append(self.transform_str(text, "uniform"))
        return np.array(result)

    def transform_str(
        self, text: str, weight: Literal["uniform", "frequency"]
    ) -> list[int]:
        """
        Transform a single string into a vector.
        - Uniform weighting: counts occurrences in the string.
        - Frequency weighting: weights by frequency in the original dataset.
        """
        if not self.vector_dims:
            raise ValueError(
                "The vector dimensions have not been learned yet, please fit the tokenizer on the data first with the Tokenizer.fit() method"
            )

        text_count: defaultdict[str, int] = defaultdict(int)
        for token in self.tokenizer.tokenize(text):
            text_count[token] += 1

        text_vector: list[int] = []
        freq = 1  # Placeholder for frequency-based weighting

        for dimension in self.vector_dims:
            # Check if the token from the vocabulary is in the current text
            if dimension and dimension[0] in text_count:
                text_vector.append(dimension[1] * freq)
            else:
                text_vector.append(0)

        return text_vector
    
