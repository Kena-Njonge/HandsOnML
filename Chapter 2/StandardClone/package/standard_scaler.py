import numpy as np
import math
import pandas as pd
from numpy.typing import NDArray

class StandardScaler:
    '''We are trying to create a clone of the StandardScaler class in sklearn
    '''
    def __init__(self):
        self.vars: list[float] = []
        self.stds: list[float] = []
        self.means: list[float] = []
        self.feature_names_in_: NDArray[np.str_] | None = None
        self.x_dim: int = 0
        self.y_dim: int = 0
        self.n_features_in_: int = 0

    def fit(self, data: np.ndarray | pd.DataFrame) -> "StandardScaler":
        """Fit the scaler to the data"""
        if isinstance(data, pd.DataFrame):
            self.feature_names_in_ = np.array(data.columns)
        arr_data = np.array(data)
        if arr_data.ndim != 2:
            raise ValueError("Expected 2D array, got array with shape %s" % arr_data.shape)
        
        self.x_dim = arr_data.shape[0]
        self.y_dim = arr_data.shape[1]
        self.n_features_in_ = self.y_dim

        for i in range(self.y_dim):
            # np.array indexing 'for all rows, get me column i'
            x = arr_data[:,i]
            self.means.append(float(np.mean(x)))
            self.vars.append(float(np.sum((x - np.full(shape=(self.x_dim,), fill_value=self.means[i]))**2)/self.x_dim))
            self.stds.append(float(math.sqrt(self.vars[i])))
        return self
    def transform(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            if self.feature_names_in_ is not None:
                data_cols = np.array(data.columns)
                if not (data_cols == self.feature_names_in_).all():
                    raise ValueError(f'Expected an array with columns {self.feature_names_in_} in said order')
        arr_data = np.array(data)
        if arr_data.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected an array of shape {("x", self.n_features_in_)} but got {arr_data.shape}')
        
        transformed_data: list[np.ndarray] = []
        for i in range(self.y_dim):
            x = arr_data[:,i]
            z = (x - np.full(shape=(arr_data.shape[0],), fill_value=self.means[i])) / np.full(shape=(arr_data.shape[0],), fill_value=self.stds[i])
            transformed_data.append(z)
        return np.array(transformed_data).T
    def fit_transform(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            if self.feature_names_in_ is not None:
                data_cols = np.array(data.columns)
                if not (data_cols == self.feature_names_in_).all():
                    raise ValueError(f'Expected an array with columns {self.feature_names_in_} in said order')
        arr_data = np.array(data)
        if arr_data.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected an array of shape {("x", self.y_dim)}')
        
        inverse_transformed: list[np.ndarray] = []
        for i in range(self.y_dim):
            x = arr_data[:,i]
            z = (x * np.full(shape=(arr_data.shape[0],), fill_value=self.stds[i])) + np.full(shape=(arr_data.shape[0],), fill_value=self.means[i])
            inverse_transformed.append(z)
        return np.transpose(np.array(inverse_transformed))

    def get_feature_names_out(self, data: np.ndarray | pd.DataFrame, input_features: list[str] | None = None) -> np.ndarray:
        if input_features is not None:
            feature_arr = np.array(input_features)
            if self.feature_names_in_ is not None and (feature_arr == self.feature_names_in_).all():
                return feature_arr
            else:
                raise ValueError('The provided input features do not correspond to n_features_in')
        elif self.feature_names_in_ is not None:
            return self.feature_names_in_
        else:
            arr_data = np.array(data)
            return np.array([f'x_{i}' for i in range(arr_data.shape[1])]) 

        

    