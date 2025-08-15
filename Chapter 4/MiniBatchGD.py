import pathlib
from sklearn.preprocessing import StandardScaler
import numpy as np
from copy import deepcopy
from MiniBatchSampler import MiniBatchSampler
import numpy as np # type: ignore


class MiniBatchGD:
    """Softmax Regression classifier with mini-batch gradient descent optimization.

    This implementation is designed for large datasets that don't fit in memory,
    using mini-batch training with reservoir sampling to process data from disk.
    The classifier supports both numerical and categorical targets, automatically handling
    the necessary encoding and scaling operations.

    Key Features:
        - Out-of-core learning through mini-batch processing
        - Automatic handling of categorical targets
        - Numerically stable softmax implementation
        - Feature scaling with StandardScaler
        - Early stopping with best model preservation
        - Proper handling of target columns at any position in the input csv

    Attributes:
        coef_ (np.ndarray): Learned model coefficients.
        n_features_in_ (int): Number of input features.
        feature_names_in_ (np.ndarray): Names of input features.
        path (str|pathlib.Path): Path to the dataset file.
        category_map (dict[str, int]): Mapping from categorical labels to indices.
        best_model (np.ndarray): Model coefficients with best performance.
        best_loss (np.float64): Best loss value achieved during training.
        scaler (StandardScaler|None): Scaler used for feature normalization.
    
    """
    def __init__(self, path: str | pathlib.Path) -> None:
        self.coef_: np.ndarray
        self.n_features_in_: int
        self.feature_names_in_: np.ndarray
        self.path: str | pathlib.Path = path
        self.category_map: dict[str, int] = {}  # Initialize empty dict
        self.best_model: np.ndarray
        self.best_loss: np.float64 = np.float64('inf')
        self.scaler: StandardScaler | None = None  # Store scaler for predictions
        self.loss_array: np.ndarray

    def _create_category_map(self, path: str|pathlib.Path, y_index: int) -> dict[str, int]:
        """Create a mapping from categorical values to integer indices.

        Args:
            path: Path to the CSV file containing the dataset.
            y_index: Column index of the target variable.

        Returns:
            A dictionary mapping category strings to their corresponding integer indices.
        """
        unique_categories: set[str] = set()
        with open(path, 'r', encoding="utf-8") as f:
            # Skip header
            f.readline()
            # Scan all lines to find unique categories
            for line in f:
                cat = line.strip().split(',')[y_index]
                unique_categories.add(str(cat))
        
        # Create ordered mapping
        return {cat: idx for idx, cat in enumerate(sorted(unique_categories))}

    def fit(self, y_index: int, lines: int=50, epochs: int = 100, eta: float = 0.1, gain: float = 0.1,
             validation_size: float = 0.0, early_stopping_rounds: int = 5) -> None:
        """Train the softmax regression model using mini-batch gradient descent.

        Args:
            y_index: Column index for target variable. All columns except y_index will be used as features.
                     Can be determined using the following commands:
                     - Bash/Linux: `head -n 2 file.csv`
            lines: Number of lines to sample for each mini-batch. Defaults to 50
            epochs: Number of training epochs. Defaults to 100.
            eta: Learning rate for gradient descent. Defaults to 0.1.
            gain: Standard deviation for weight initialization. Defaults to 0.1.
            validation_size: Size of validation set. Defaults to 0.0.
                            If between 0 and 1, treated as a percentage of the dataset.
                            If > 1, treated as an absolute number of samples.
                            The validation samples will be excluded from training to prevent data leakage.
            early_stopping_rounds: Number of epochs with no improvement in validation loss before
                                 stopping training. Only used if validation_size > 0. Defaults to 5.

        The method performs the following steps:
        1. Uses reservoir sampling to load mini-batches from large CSV files
        2. Automatically handles categorical target variables through ordinal encoding
        3. Uses ordinal encoding to compute one-hot encoding for use in the softmax algorithm
        4. Applies feature scaling using StandardScaler
        5. Implements numerically stable softmax regression
        6. Uses (numerically stable) cross-entropy loss for optimization
        7. Implements early stopping by saving the best model
        8. If validation_size > 0, uses a separate validation set to prevent overfitting

        Note:


            Assumes that all features are numeric, does not perform encodign for these,
            don't feel like extending it right now, may implement it in a different way in the future possibly.
            
            Should have used my own standard scaler since it is pure numpy, will save me some time, but don't feel like it right now.
        """
        def softmax(z:np.ndarray):
            # z: shape (n_samples, n_classes)
            z = np.asarray(z, dtype=np.float64)                    # (N, C)
            z = z - np.max(z, axis=1, keepdims=True)              # subtract row-wise max
            exp_z = np.exp(z)                                     # still (N, C)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)   # normalize per row

        # Initialize samplers for training and validation
        train_sampler = MiniBatchSampler()
        
        # If validation_size > 0, create a separate validation set
        validation_sampler = None
        validation_indices = []
        validation_df = None
        
        if validation_size > 0:
            # Sample validation data first
            validation_sampler = MiniBatchSampler()
            
            # For percentage-based validation size (0 < validation_size < 1),
            # we need to determine the total number of samples first
            if 0 < validation_size < 1:
                # First, count the total number of samples in the dataset
                with open(self.path, 'r', encoding="utf-8") as f:
                    # Skip header
                    f.readline()
                    total_samples = sum(1 for _ in f)
                
                # Calculate the absolute number of validation samples
                val_samples = int(total_samples * validation_size)
            else:
                # For validation_size >= 1, treat it as an absolute number
                val_samples = int(validation_size)
                
            # Sample the validation set
            validation_df = validation_sampler.sample(self.path, val_samples)
            
            # Get indices of validation samples to exclude from training
            validation_indices = validation_sampler.get_sampled_indices()
            
            # Setup training sampler to exclude validation indices
            train_sampler.exclude_indices(validation_indices)
            
        # Sample initial training data
        df = train_sampler.sample(self.path, lines)
        
        # For categorical target, scan entire dataset first to get all possible categories
        if df.dtypes.iloc[y_index] == 'object':
            self.category_map = self._create_category_map(self.path, y_index)
            # Apply mapping to first batch
            for index, cat in enumerate(df.iloc[:,y_index]):
                df.iloc[index, y_index] = self.category_map[str(cat)]
                
            # Also apply to validation set if it exists
            if validation_df is not None:
                for index, cat in enumerate(validation_df.iloc[:,y_index]):
                    validation_df.iloc[index, y_index] = self.category_map[str(cat)]
        # Scale features and target
        # I will permit myself to use the standard scaler here as I have already implemented it and don't 
        # want to create a new package, neither do I want to copy the code here
        self.scaler = StandardScaler()

        # Ok, so granted I hadn't implemented partial fit yet, 
        # And I am tired, I'll give myself this
        # Extract features, excluding the target column
        feature_cols = list(range(df.shape[1]))
        feature_cols.pop(y_index)  # Remove target column index
        
        # Scale only the feature columns
        feature_data = df.iloc[:, feature_cols].to_numpy()
        if self.scaler is not None: # type: ignore
            self.scaler.partial_fit(feature_data)
            X = self.scaler.transform(feature_data)
        else:
            X = feature_data # type: ignore
        y = df.to_numpy()[:, y_index].astype(int)
        
        # Prepare validation data if available
        X_val = None
        y_val = None
        if validation_df is not None:
            val_feature_cols = list(range(validation_df.shape[1]))
            val_feature_cols.pop(y_index)
            val_feature_data = validation_df.iloc[:, val_feature_cols].to_numpy()
            
            if self.scaler is not None: # type: ignore
                self.scaler.partial_fit(val_feature_data)
                X_val = self.scaler.transform(val_feature_data)
            else:
                X_val = val_feature_data
                
            y_val = validation_df.to_numpy()[:, y_index].astype(int)
        
        rng = np.random.default_rng()
        num_classes = len(self.category_map) if hasattr(self, 'category_map') else np.max(y) + 1
        # One-hot encoding for targets
        y_onehot = np.eye(num_classes)[y]
        
        # One-hot encoding for validation targets if available
        y_val_onehot = None
        if X_val is not None and y_val is not None:
            y_val_onehot = np.eye(num_classes)[y_val]
        
        theta = rng.normal(0, gain, size=(X.shape[1], num_classes)) # shape: (n_features, num_classes)

    
        
        # Mini-batch gradient descent for softmax regression
        best_loss: np.float64 = np.float64('inf')
        self.loss_array = np.zeros(epochs)
        
        # For early stopping
        no_improvement_count = 0
        
        for epoch in range(epochs):
            # Get a new mini-batch for this epoch
            df = train_sampler.resample()
            
            # Apply category mapping to new batch if needed
            if df.dtypes.iloc[y_index] == 'object':
                for index, cat in enumerate(df.iloc[:,y_index]):
                    df.iloc[index, y_index] = self.category_map[str(cat)]
            
            # Extract features, excluding the target column
            feature_cols = list(range(df.shape[1]))
            feature_cols.pop(y_index)
            
            # Scale only the feature columns
            feature_data = df.iloc[:, feature_cols].to_numpy()
            self.scaler.partial_fit(feature_data)
            X = self.scaler.transform(feature_data) # type: ignore
            y = df.to_numpy()[:, y_index].astype(int)
            # One-hot encoding for targets
            y_onehot = np.eye(num_classes)[y]
            
            # Forward pass: compute logits and probabilities
            logits = X @ theta  # shape: (batch_size, num_classes)
            probs = softmax(logits)  # shape: (batch_size, num_classes)
            # Compute gradient: average over batch
            grad = X.T @ (probs - y_onehot) / X.shape[0]  # shape: (n_features, num_classes)
            # Update weights
            theta = theta - eta * grad
            
            # Calculate training loss
            train_loss: np.float64 = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1)) # type: ignore
            self.loss_array[epoch] = train_loss
            
            # Calculate validation loss if validation set is available
            if X_val is not None and y_val_onehot is not None:
                val_logits = X_val @ theta
                val_probs = softmax(val_logits)
                val_loss: np.float64 = -np.mean(np.sum(y_val_onehot * np.log(val_probs + 1e-9), axis=1)) # type: ignore
                
                # Early stopping based on validation loss
                if val_loss < best_loss:
                    best_loss = val_loss # type: ignore
                    self.best_loss = best_loss  # Store best loss as class attribute
                    self.coef_ = deepcopy(theta)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if early_stopping_rounds > 0 and no_improvement_count >= early_stopping_rounds:
                        print(f"Early stopping at epoch {epoch+1}/{epochs}")
                        break
            else:
                # If no validation set, use training loss for early stopping
                if train_loss < best_loss:
                    best_loss = train_loss # type: ignore
                    self.best_loss = best_loss  # Store best loss as class attribute
                    # np.copy() only makes a shallow copy 
                    self.coef_ = deepcopy(theta)
            
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability estimates for all classes, using softmax as the underlying funcition

            For each sample, computes the probablility that it is part of a class.
        Args:
            X: Input samples of shape (n_samples, n_features).
               Should match the format of training data.

        Returns:
            Array of shape (n_samples, n_classes) containing predicted probabilities
            for each class. Each row sums to 1.

        Raises:
            ValueError: If model hasn't been fitted yet.
        """
        logits = X @ self.coef_  # shape: (n_samples, n_classes)
        z = np.asarray(logits, dtype=np.float64)
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples.

        For each sample, predicts the most likely class based on probability estimates.
        Uses argmax of predict_proba output to determine class labels.

        Args:
            X: Input samples of shape (n_samples, n_features).
               Should match the format of training data.

        Returns:
            Array of shape (n_samples,) containing predicted class labels.
            Labels correspond to original class values based on category_map.

        Raises:
            ValueError: If model hasn't been fitted yet.
        """
        if not hasattr(self, 'coef_'):
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if self.scaler is not None:
            X = self.scaler.transform(X) # type:ignore
        y_pred = np.argmax(self.predict_proba(X), axis=1)
        return y_pred
    
    def predict_categories(self, X: np.ndarray) -> list[str]:
        """Convert numeric predictions back to original category labels.

        For categorical targets, converts the numeric class predictions back to
        the original string category labels using the category_map.

        Args:
            X: Input samples of shape (n_samples, n_features).
               Should match the format of training data.

        Returns:
            List of predicted category labels corresponding to original categories.

        Raises:
            ValueError: If model was not trained on categorical data.
        """
        if not hasattr(self, 'category_map'):
            raise ValueError("Model was not trained on categorical data")
            
        y_pred = self.predict(X)
        # Reverse the category mapping
        reverse_map = {idx: cat for cat, idx in self.category_map.items()}
        return [reverse_map[idx] for idx in y_pred]
        
    def transform_categories(self, categories: list[str]) -> np.ndarray:
        """Transform string category to internal numeric indices (for the target).

        Converts categorical values to the numeric indices used internally by the model.
        This is useful for manually preprocessing data before prediction.

        Args:
            categories: List of category strings to transform. Each string should
                       match one of the categories seen during training.

        Returns:
            Array of numeric indices corresponding to the input categories.
            Shape is (n_samples,) where n_samples is len(categories).

        Raises:
            ValueError: If any category wasn't seen during training. The error message
                      will include the invalid category and list all valid options.
        """
        if not hasattr(self, 'category_map'):
            raise ValueError("Model was not trained on categorical data")
            
        try:
            return np.array([self.category_map[str(cat)] for cat in categories])
        except KeyError as e:
            unseen_category = str(e).strip("'")
            raise ValueError(
                f"Category '{unseen_category}' was not present in training data. "
                f"Available categories are: {sorted(self.category_map.keys())}"
            )
