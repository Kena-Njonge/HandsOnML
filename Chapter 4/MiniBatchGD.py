import pathlib
from sklearn.preprocessing import StandardScaler
import numpy as np
from copy import deepcopy
from MiniBatchSampler import MiniBatchSampler
import pandas as pd


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
        coef_ (np.ndarray): Learned model coefficients (best model found during training).
        n_features_in_ (int): Number of input features.
        n_samples_in_ (int): Number of input samples.
        feature_names_in_ (np.ndarray): Names of input features.
        path (str|pathlib.Path): Path to the dataset file.
        category_map (dict[str, int]): Mapping from categorical labels to indices.
        best_loss (np.float64): Best loss value achieved during training.
        scaler (StandardScaler|None): Scaler used for feature normalization.
        y_test np.ndarray: A one-hot encoding of our categories
    
    """
    def __init__(self, path: str | pathlib.Path) -> None:
        self.coef_: np.ndarray
        self.n_features_in_: int
        self.feature_names_in_: np.ndarray
        self.path: str | pathlib.Path = path
        self.category_map: dict[str, int] = {}  # Initialize empty dict
        self.best_loss: np.float64 = np.float64('inf')
        self.scaler: StandardScaler | None = None  # Store scaler for predictions
        self.train_loss_array: np.ndarray
        self.validation_loss_array: np.ndarray
        self.epochs_trained: int = 0  # Track actual number of epochs trained
        self.n_samples_in_: int
        self.X_test: np.ndarray
        self.y_test: np.ndarray
        self.test_set: pd.DataFrame

    def get_dataset_stats(self) -> dict[str, object]:
        """Return statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics including total sample count,
            number of features, and class distribution if available.
        """
        # Initialize with types that match our return type annotation
        stats: dict[str, object] = {
            "n_samples": self.n_samples_in_,
            "n_features": self.n_features_in_ if hasattr(self, 'n_features_in_') else None
        }

        # Add class distribution if we have categorical data
        if hasattr(self, 'category_map') and self.category_map:
            stats["n_classes"] = len(self.category_map)
            stats["class_names"] = sorted(self.category_map.keys())

        return stats
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
                # Save the amount of lines seen, useful for comparing mini-batch performance to batch performance if the batch size is equal to our whole dataset (and no validation size is set) 
                values = line.strip().split(',')
                if y_index < len(values):  # Ensure index is valid
                    cat = values[y_index]
                    # Always convert to string to ensure consistent types
                    unique_categories.add(str(cat).strip())

        # Create ordered mapping
        return {cat: idx for idx, cat in enumerate(sorted(unique_categories))}

    def fit(self, y_index: int, lines: int=50, epochs: int = 100, eta: float = 0.1, gain: float = 0.1,
             validation_size: float = 0.0, test_size: float = 0.2, early_stopping_rounds: int = 5, min_delta: float = 0.0, 
             min_epochs: int = 10, verbose: bool = False) -> None:
        """Train the softmax regression model using mini-batch gradient descent.

        Args:
            y_index: Column index for target variable. All columns except y_index will be used as features.
                     Can be determined using the following commands:
                     - Bash/Linux: `head -n 2 file.csv`
            lines: Number of lines to sample for each mini-batch. Defaults to 50
            epochs: Number of training epochs. Defaults to 100.
            eta: Learning rate for gradient descent. Defaults to 0.1.
            gain: Standard deviation for weight initialization. Defaults to 0.1.
            test_size: Size of the test set. Defaults to 0.2
            validation_size: Size of validation set. Defaults to 0.0.
                            If between 0 and 1, treated as a percentage of the dataset.
                            If > 1, treated as an absolute number of samples.
                            The validation samples will be excluded from training to prevent data leakage.
            early_stopping_rounds: Number of epochs with no improvement in validation loss before
                                 stopping training. Only used if validation_size > 0. Defaults to 5.
            min_delta: Minimum change in validation loss to qualify as an improvement.
                     Helps avoid stopping due to minor fluctuations. Defaults to 0.0.
            min_epochs: Minimum number of epochs to train before allowing early stopping.
                      Ensures the model has a chance to learn before stopping. Defaults to 10.
            verbose: Whether to print progress messages during training. Defaults to False.

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
        
        if test_size > 0:
            # Sample test data first
            test_sampler = MiniBatchSampler()
            
            with open(self.path, 'r', encoding="utf-8") as f:
                # Skip header
                f.readline()
                self.n_samples_in_ = sum(1 for _ in f)
            
            # Calculate the absolute number of test samples
            test_samples = int(self.n_samples_in_ * test_size)
                
            # Sample the test set
            self.test_set = test_sampler.sample(self.path, test_samples)
            
            # Get indices of test samples to exclude from training
            test_indices = test_sampler.get_sampled_indices()
            
            # Setup training and validation sampler to exclude test indices
            train_sampler.exclude_indices(test_indices)
            if validation_size>0:
                validation_sampler = MiniBatchSampler()
                validation_sampler.exclude_indices(test_indices)
                
            # Process the test set will be done after the scaler is initialized


        if validation_size > 0:            
            # For percentage-based validation size (0 < validation_size < 1),
            # we need to determine the total number of samples first
            if 0 < validation_size < 1:
                val_samples = int(self.n_samples_in_ * validation_size)
            else:
                # For validation_size >= 1, treat it as an absolute number
                val_samples = int(validation_size)
                if val_samples >= self.n_samples_in_ - self.test_set.shape[0]:
                    raise ValueError(f"""The amount of samples in the validation set cannot exceed the size of the dataset with the test_set excluded.
                                     Received val_samples = {val_samples}, but our dataset without the test set has only {self.n_samples_in_ - self.test_set.shape[0]} samples.
                                     Either decrease the size of the test set or the size of the validation set.""")
                
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
                # Make sure to convert to string and strip any whitespace
                cat_str = str(cat).strip()
                if cat_str in self.category_map:
                    df.iloc[index, y_index] = self.category_map[cat_str]
                else:
                    raise ValueError(f"Category '{cat_str}' not found in category map. Available categories: {sorted(self.category_map.keys())}")
                
            # Also apply to validation set if it exists
            if validation_df is not None:
                for index, cat in enumerate(validation_df.iloc[:,y_index]):
                    # Make sure to convert to string and strip any whitespace
                    cat_str = str(cat).strip()
                    if cat_str in self.category_map:
                        validation_df.iloc[index, y_index] = self.category_map[cat_str]
                    else:
                        raise ValueError(f"Category '{cat_str}' not found in category map. Available categories: {sorted(self.category_map.keys())}")
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
        
        # Process test set if it exists
        if hasattr(self, 'test_set'):
            # Apply category mapping to test set if needed
            if df.dtypes.iloc[y_index] == 'object':
                for index, cat in enumerate(self.test_set.iloc[:,y_index]):
                    # Make sure to convert to string and strip any whitespace
                    cat_str = str(cat).strip()
                    if cat_str in self.category_map:
                        self.test_set.iloc[index, y_index] = self.category_map[cat_str]
                    else:
                        raise ValueError(f"Category '{cat_str}' not found in category map. Available categories: {sorted(self.category_map.keys())}")
            
            # Extract and scale test features
            test_feature_cols = list(range(self.test_set.shape[1]))
            test_feature_cols.pop(y_index)
            test_feature_data = self.test_set.iloc[:, test_feature_cols].to_numpy()
            
            # Scale test features with the same scaler
            self.X_test = self.scaler.transform(test_feature_data) # type: ignore
                
            # Extract test targets
            self.y_test = self.test_set.to_numpy()[:, y_index].astype(int)
        
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

    
        
        # Initialize best model and training metrics
        best_loss: np.float64 = np.float64('inf')
        self.validation_loss_array = np.zeros(epochs)
        self.train_loss_array = np.zeros(epochs)
        
        # For early stopping
        no_improvement_count = 0
        
        # Track if we stopped early
        early_stopped = False
        
        # Training loop
        for epoch in range(epochs):
            # Get a new mini-batch for this epoch
            df = train_sampler.resample()
            
            # Apply category mapping to new batch if needed
            if df.dtypes.iloc[y_index] == 'object':
                for index, cat in enumerate(df.iloc[:,y_index]):
                    # Make sure to convert to string and strip any whitespace
                    cat_str = str(cat).strip()
                    if cat_str in self.category_map:
                        df.iloc[index, y_index] = self.category_map[cat_str]
                    else:
                        raise ValueError(f"Category '{cat_str}' not found in category map. Available categories: {sorted(self.category_map.keys())}")
            
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
            self.train_loss_array[epoch] = train_loss
            
            # Calculate validation loss if validation set is available
            if X_val is not None and y_val_onehot is not None:
                val_logits = X_val @ theta
                val_probs = softmax(val_logits)
                val_loss: np.float64 = -np.mean(np.sum(y_val_onehot * np.log(val_probs + 1e-9), axis=1)) # type: ignore
                self.validation_loss_array[epoch] = val_loss

                
                # Early stopping based on validation loss
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss # type: ignore
                    self.best_loss = best_loss  # Store best loss as class attribute
                    self.coef_ = deepcopy(theta)  # Store best model in coef_
                    no_improvement_count = 0
                    if verbose:
                        print(f"Epoch {epoch+1}/{epochs} - Validation loss improved to {best_loss:.4f}")
                else:
                    no_improvement_count += 1
                    if verbose and epoch % 5 == 0:
                        print(f"Epoch {epoch+1}/{epochs} - Validation loss: {val_loss:.4f} (no improvement for {no_improvement_count} epochs)")
                    
                    # Only apply early stopping after minimum number of epochs
                    if early_stopping_rounds > 0 and no_improvement_count >= early_stopping_rounds and epoch >= min_epochs - 1:
                        # Set the class attributes for epochs trained and final model
                        self.epochs_trained = epoch + 1
                            
                        # Trim the loss arrays to the actual number of epochs trained
                        self.train_loss_array = self.train_loss_array[:self.epochs_trained]
                        self.validation_loss_array = self.validation_loss_array[:self.epochs_trained]
                        
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}/{epochs} - Best validation loss: {best_loss:.4f}")
                        early_stopped = True
                        break
            else:
                # If no validation set, use training loss for early stopping
                if train_loss < best_loss - min_delta:
                    best_loss = train_loss # type: ignore
                    self.best_loss = best_loss  # Store best loss as class attribute
                    self.coef_ = deepcopy(theta)  # Store best model in coef_
                    no_improvement_count = 0
                    if verbose:
                        print(f"Epoch {epoch+1}/{epochs} - Training loss improved to {best_loss:.4f}")
                else:
                    no_improvement_count += 1
                    if verbose and epoch % 5 == 0:
                        print(f"Epoch {epoch+1}/{epochs} - Training loss: {train_loss:.4f} (no improvement for {no_improvement_count} epochs)")
                    
                    # Only apply early stopping after minimum number of epochs
                    if early_stopping_rounds > 0 and no_improvement_count >= early_stopping_rounds and epoch >= min_epochs - 1:
                        # Set the class attributes for epochs trained and final model
                        self.epochs_trained = epoch + 1
    
                            
                        # Trim the loss arrays to the actual number of epochs trained
                        self.train_loss_array = self.train_loss_array[:self.epochs_trained]
                        self.validation_loss_array = self.validation_loss_array[:self.epochs_trained]
                        
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}/{epochs} - Best training loss: {best_loss:.4f}")
                        early_stopped = True
                        break
        
        # If we completed all epochs without early stopping
        if not early_stopped:
            self.epochs_trained = epochs
    
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
    
    def predict_categories(self, X: np.ndarray) -> np.ndarray:
        """Convert numeric predictions back to original category labels.

        For categorical targets, converts the numeric class predictions back to
        the original string category labels using the category_map.

        Args:
            X: Input samples of shape (n_samples, n_features).
               Should match the format of training data.

        Returns:
            NumPy array of predicted category labels corresponding to original categories.

        Raises:
            ValueError: If model was not trained on categorical data.
        """
        if not hasattr(self, 'category_map'):
            raise ValueError("Model was not trained on categorical data")
            
        y_pred = self.predict(X)
        # Reverse the category mapping
        reverse_map = {idx: cat for cat, idx in self.category_map.items()}
        # Convert list of strings to numpy array of strings
        return np.array([reverse_map[idx] for idx in y_pred])
        
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
            # Make sure to convert each category to string and strip whitespace
            return np.array([self.category_map[str(cat).strip()] for cat in categories])
        except KeyError as e:
            unseen_category = str(e).strip("'")
            raise ValueError(
                f"Category '{unseen_category}' was not present in training data. "
                f"Available categories are: {sorted(self.category_map.keys())}"
            )
            
    def plot_learning_curve(self, figsize: tuple[int, int] = (10, 6), save_path: str | None = None):
        """Plot the learning curves for training and validation loss.
        
        Creates a visualization of the training process, showing both training and
        validation loss over epochs. If early stopping occurred, it marks the point
        where the best model was found.
        
        Args:
            figsize: Tuple of (width, height) for the figure size. Default is (10, 6).
            save_path: Optional path to save the figure. If None, the figure is displayed
                      but not saved. Default is None.
                      
        Returns:
            The figure object containing the plot.
            
        Raises:
            ValueError: If the model has not been trained yet.
        """
        try:
            # type: ignore
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting. Install it with 'pip install matplotlib'")
        
        if not hasattr(self, 'train_loss_array') or not hasattr(self, 'epochs_trained'):
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)  # type: ignore
        
        # Plot training loss
        epochs_range = np.arange(1, self.epochs_trained + 1)
        ax.plot(epochs_range, self.train_loss_array[:self.epochs_trained], 'b-', label='Training Loss')  # type: ignore
        
        # Plot validation loss if available
        has_validation = np.any(self.validation_loss_array[:self.epochs_trained] > 0)
        if has_validation:
            ax.plot(epochs_range, self.validation_loss_array[:self.epochs_trained], 'r-', label='Validation Loss')  # type: ignore
            
        # Find the epoch with the best model (lowest validation loss or training loss)
        if has_validation:
            best_epoch = np.argmin(self.validation_loss_array[:self.epochs_trained]) + 1
            best_loss = float(np.min(self.validation_loss_array[:self.epochs_trained]))
            loss_type = "Validation"
        else:
            best_epoch = np.argmin(self.train_loss_array[:self.epochs_trained]) + 1
            best_loss = float(np.min(self.train_loss_array[:self.epochs_trained]))
            loss_type = "Training"
            
        # Mark the best model
        ax.axvline(x=float(best_epoch), color='g', linestyle='--', alpha=0.7,  # type: ignore
                  label=f'Best Model (Epoch {best_epoch})')
        ax.plot(float(best_epoch), best_loss, 'go', markersize=8)  # type: ignore
        
        # Add annotation for best model
        ax.annotate(f'Best {loss_type} Loss: {best_loss:.4f}',  # type: ignore
                   xy=(float(best_epoch), best_loss),
                   xytext=(float(best_epoch + 1), best_loss * 1.1),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=10)
        
        # Add early stopping annotation if applicable
        if self.epochs_trained < len(self.train_loss_array):
            ax.annotate(f'Early Stopping (Epoch {self.epochs_trained})',  # type: ignore
                       xy=(float(self.epochs_trained), float(self.train_loss_array[self.epochs_trained-1])),
                       xytext=(float(self.epochs_trained - 5), float(self.train_loss_array[self.epochs_trained-1] * 0.9)),
                       arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                       fontsize=10, color='red')
        
        # Customize the plot
        ax.set_title('Learning Curves', fontsize=14)  # type: ignore
        ax.set_xlabel('Epochs', fontsize=12)  # type: ignore
        ax.set_ylabel('Loss', fontsize=12)  # type: ignore
        ax.legend(loc='upper right', fontsize=10)  # type: ignore
        ax.grid(True, linestyle='--', alpha=0.7)  # type: ignore
        
        # Set y-axis limits with some padding
        max_loss = float(max(np.max(self.train_loss_array[:self.epochs_trained]), 
                          np.max(self.validation_loss_array[:self.epochs_trained] if has_validation else [0])))
        ax.set_ylim(0, max_loss * 1.2)  # type: ignore
        
        # Ensure x-axis starts at 1
        ax.set_xlim(0.5, float(self.epochs_trained + 0.5))  # type: ignore
        
        plt.tight_layout()  # type: ignore
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)  # type: ignore
            
        return fig
