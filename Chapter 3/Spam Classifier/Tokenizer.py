import hashlib
from collections import defaultdict
from typing import Literal, Callable, Generator

import numpy as np


class Tokenizer():
    ''' For problems see the individual functions, but some general things to fix are
    
    1. What to do when we have too many dimensions passed to the function, should we jsut fill or truncate the output
    2. Return numpy arrays, refactor to use numpy, probably faster
    '''

   
    def __init__(self, window_size: int = 1000, delimiter: str='.', max_split:int = -1, max_dims:int=784, hash_input:bool=False, hash_function:str='basic') :
        '''Initialize the parameters for the tokenizer'''
        self.delimiter=delimiter
        self.window_size=window_size
        self.max_split = max_split
        self.max_dims = max_dims
        self._hash_func: Callable[[str], int] 
        self.set_hash_function(hash_function)
        self.hash_input = hash_input
        self.vector_dims: list[tuple[str,int]] = []
        
        # for the words/dimension calcualte their relative frequencies

        self.frequencies = []

        # Keyerror caught, defaults to adding the empty strign
        # just need to fix this whole approach for if we map to a vector, what a shitshow
        # I allowed myself ot lean too much into some weird coding, there should be 1 type per object
        # If you don't use a lambda when nesting the defaultdict the inner object will only be created once and 
        # thus you will always add to the same single dict for whatever key in the outer dict
        self.mapping: defaultdict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))


    def set_hash_function(self, name: str | None):
        '''
        Set the hash function for which to use for the hashing of tokens.

        Questionable if this is worth it for smaller datasets, can't evaluate that yet, 
        Good material to read, evulate and extend this in the future are

        Feature Hashing for Large Scale Multitask Learning” – Weinberger et al., 2009
        River docs, check out their spam filter
        Paul Graham – “A Plan for Spam” and “Better Bayesian Filtering” 
        Rennie et al., 2003 – “Tackling the Poor Assumptions of Naive Bayes Text Classifiers”

        '''
        if name == 'basic' or name is None:
            self._hash_func = hash
        elif name == 'sha1':
            # encode() Transform string into utf8 encoded bytes, equivalent to b string
            # Returns hash as string containing only hexadeximal digits, turn that back into a string  
            self._hash_func = lambda x: int(hashlib.sha1(x.encode()).hexdigest(), 16)
        else:
            raise ValueError(f"Unsupported hash function: {name}")
    

    def hash_token(self, token: str) -> int:
        return self._hash_func(token)


    def fit(self,dataset: list[str]):
        '''Fit on the dataset, call fit_str multiple times'''
        for i in dataset:
            self.fit_str(i)
        # further protesting, sort and select the top max_dim, assign each of them 
        # a dimension and a count, doesn't really make sense to ahve hashed tbh  
        
        # Flatten mapping for sorting
        flattened_mapping = [
            (token, sum(counts.values())) for token, counts in self.mapping.items()
        ]

        self.vector_dims = sorted(flattened_mapping, key= lambda x: x[1], reverse= True)   

        # truncate or pad list
        if len(self.vector_dims) > self.max_dims:
            self.vector_dims = self.vector_dims[:self.max_dims]
        elif len(self.vector_dims) < self.max_dims:
            # Fill with empty tuple, good idea since we can't fill with a strign or number as that may be part 
            # of our vocabulary
            # Fill with empty tuples that match the expected type
            self.vector_dims.extend([("", 0)] * (self.max_dims - len(self.vector_dims)))




    def fit_str(self,text:str)-> None:
        '''Fit the tokenizer on a single string
        
            If we use the hash function or not, we have the same type for the dict
            If we don't use the hash_function token_key is just going to be 0
        '''
        # If we have really long inputs, we probably don't want to 
        # load all that into memory, also would be nice for me to write some functions
        # with yield    
        for token in self._token_genenrator(text):
            # Use hash or default key based on `hash_input`
            token_key = self.hash_token(token) if self.hash_input else 0
            self.mapping[token][token_key] += 1


    def _token_genenrator(self, text:str) -> Generator[str,None,None]:
        '''Tokkeeeennnn Streammmming'''
        for i in range(0,len(text),self.window_size):
            window = text[i: i+self.window_size].casefold()
            for token in window.split(self.delimiter, self.max_split):
                yield token
        
    def transform(self, dataset: list[str]) -> np.ndarray:
        if not self.vector_dims:
            raise ValueError("The vector dimensions have not been learned yet. Please call the `fit` method before using `transform`.")
        
        result: list[list[int]] = []
        for text in dataset:
            result.append(self.transform_str(text, 'uniform'))
        return np.array(result)


    
    def transform_str(self, text: str, weight: Literal['uniform', 'frequency']) -> list[int]:
        '''Transforming string into vector
        
            Uniform weighting means that we weight by just the amount of times it appears in the string itself

            Frequency based weighting means that we weight by the frequency seen in the dataset,
            relative only to the other words that comprise the dimensions.

        '''
        if not self.vector_dims:
            raise ValueError('The vector dimensions have not been learned yet, please fit the tokenizer on the data first with the Tokenizer.fit() method')
        
        text_count: defaultdict[str, int] = defaultdict(int)
        for token in self._token_genenrator(text):
            text_count[token] += 1
        
        text_vector: list[int] = []

        # Addition for frequency based weighting, placeholder
        freq = 1
        if weight=='frequency':
            freq=freq
        for dimension in self.vector_dims:
            if dimension and dimension[0] in text_count:
                text_vector.append(dimension[1] * freq)
            else:
                text_vector.append(0)
        
        return text_vector

