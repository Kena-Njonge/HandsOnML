import numpy as np
import math
import pandas as pd

class StandardScaler:
    '''We are trying to create a clone of the StandardScaler class in sklearn
    '''
    def __init__(self):
        self.vars = []
        self.stds = []
        self.means =[]
        self.feature_names_in_ = None
        self.x_dim = None
        self.y_dim = None
        self.n_features_in_ = None

    def fit(self,X):
        if type(X)==pd.DataFrame :
            self.feature_names_in_ = np.array(X.columns)
        X = np.array(X)
        self.x_dim = X.shape[0]
        self.y_dim = X.shape[1]
        self.n_features_in_ =  self.y_dim

        for i in range(self.y_dim):
            # np.array indexing 'for all rows, get me column i'
            x = X[:,i]
            self.means.append(np.mean(x))
            self.vars.append(np.sum((x - np.full(shape=(self.x_dim,),fill_value=(self.means[i])))**2)/self.x_dim)
            self.stds.append(math.sqrt(self.vars[i]))
        return self
    def transform(self,X):
        if type(X)==pd.DataFrame :
            if np.array(X.columns)!=self.feature_names_in_:
                raise  ValueError(f'Expected an array with columns {self.feature_names_in_} in said order')
        X = np.array(X)
        if X.shape[1]!=self.n_features_in_:
            raise ValueError(f'Expected an array of shape {('x',self.n_features_in_)} but got {X.shape}')
        Z = []
        for i in range(self.y_dim):
            x = X[:,i]
            z = (x - np.full(shape = (X.shape[0],),fill_value=self.means[i])) / np.full(shape=(X.shape[0],),fill_value=self.stds[i])
            Z.append(z)
        return np.array(Z).T
    def fit_transform(self,X):
        self = self.fit(X)
        Z = self.transform(X)
        return Z
    def inverse_transform(self,X):
        if type(X)==pd.DataFrame :
            if X.columns!=self.feature_names_in_:
                raise  ValueError(f'Expected an array with columns {self.feature_names_in_} in said order')
        X = np.array(X)
        if X.shape[1]!=self.n_features_in_:
            raise ValueError(f'Expected an array of shape {('x',self.y_dim)}')
        Z = []
        for i in range(self.y_dim):
            x = X[:,i]
            z = (x * np.full(shape=(X.shape[0],),fill_value=self.stds[i])) + np.full(shape=(X.shape[0],),fill_value=self.means[i])
            Z.append(z)
        return np.transpose(np.array(Z))
    def get_feature_names_out(self,X,input_features=None):
        if input_features:
            if np.array(input_features) == self.feature_names_in_:
                pass
            else:
                raise ValueError('The provided input features do not correspond to n_features_in')
        elif self.feature_names_in_:
            return self.feature_names_in_
        else: 
            return np.array([f'x_{i}' for i in range(X.shape[1])]) 

        

    