import hashlib
from collections import defaultdict

class Tokenizer():


   
    def __init__(self, window_size: int = 1000, delimiter: str='.', max_split:int|None = None, max_dim:int=784, hash_function:str='basic') :
        '''Initialize the parameters for the tokenizer'''
        self.delimiter=delimiter
        self.window_size=window_size
        self.max_split = max_split
        self.max_dim = max_dim
        self.set_hash_function(hash_function)

        # Keyerror caught, defaults to adding the empty strign
        self.mapping = defaultdict()


    def set_hash_function(self, name: str):
        '''v
        
        '''
        if name == 'basic' or name is None:
            self._hash_func = hash
        elif name == 'sha1':
            # Transform string into utf8 encoded bytes, equivalent to b string
            # Returns hash as string containing only hexadeximal digits, turn that back into a string  
            self._hash_func = lambda x: int(hashlib.sha1(x.encode()).hexdigest(), 16)
        else:
            raise ValueError(f"Unsupported hash function: {name}")
    

    def hash_token(self, token):
        return self._hash_func(token)


    def fit(self,dataset):
        '''Fit on the dataset, call fit_str multiple times'''
        for i in dataset:
            self.fit_str(i)

        # further protesting, sort and select the top max_dim, assign each of them 
        # a dimension and a count, doesn't really make sense to ahve hashed tbh  

    def fit_str(self,text):
        '''Fit the tokenizer on a single string'''
        # If we have really long inputs, we probably don't want to 
        # load all that into memory, also would be nice for me to write some functions
        # with yield
        for token in self._token_genenrator(text):
            # Store the token as a number, update its count as 
            if token in self.mapping:
                i_int_rep = self.mapping[i].key()
                self.mapping[token][i_int_rep] += 1
            else: 
                self.mapping[token] = {self.hash_token(i): 0}

    def _token_genenrator(self, text):
        '''Tokkeeeennnn Streammmming'''
        for i in range(0,len(text),self.window_size):
            window = text[i: i+self.window_size].case_fold()
            for token in window.split(self.delimiter, self.max_split):
                yield token
        
    def transform():
        pass
