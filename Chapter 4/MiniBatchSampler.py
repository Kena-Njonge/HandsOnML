import pathlib
import numpy as np # type: ignore
import pandas as pd

class MiniBatchSampler():
    """ Mini-batch sampler for large CSV files using reservoir sampling.

    This class provides sampling from large CSV files by implementing
    a two-phase sampling strategy:
    1. Initial sampling using reservoir sampling to get a representative sample
       while counting total lines in a single pass
    2. Subsequent sampling using direct access with known line count


    Attributes:
        columns (pd.Index | None): Column names from CSV header.
        DataFrame (pd.DataFrame): Current mini-batch data.
        length (int): Total number of lines in CSV file (excluding header).
        path (str | pathlib.Path): Path to current CSV file.
        header (bool): Whether CSV has a header row.
        lines (int): Number of lines to sample in each batch.
        rng (np.random.Generator): Random number generator for sampling.
        excluded_indices (set[int]): Set of line indices that should be excluded from sampling.
                               Used to prevent validation samples from appearing in training batches.
        sampled_indices (list[int]): List of indices that were sampled in the last batch.

    Example:
        >>> sampler = MiniBatchSampler()
        >>> batch = sampler.sample('data.csv', lines=32)
        >>> # Get new batch using same parameters
        >>> next_batch = sampler.resample()
    """
    # The goal of mini-batch is to load a random subset of lines.
    # We approach the problem with a slightly modified version of reservoir sampling.
    # In the first pass, we create a sample and count the total lines simultaneously.
    # In subsequent passes, we use the known line count to efficiently seek to random lines.

    def __init__(self):
        self.columns: pd.Index|None = None
        self.DataFrame: pd.DataFrame = pd.DataFrame()
        self.length: int = 0
        self.path: str|pathlib.Path = str()
        self.header: bool = True
        self.lines: int = 0
        self.rng:np.random.Generator = np.random.default_rng(42)
        self.excluded_indices: set[int] = set()  # Indices that should not be sampled
        self.sampled_indices: list[int] = []  # Keep track of which indices were sampled

    def exclude_indices(self, indices: list[int]) -> None:
        """Set indices that should be excluded from future sampling.
        
        Args:
            indices: List of line indices (0-based) to exclude from sampling.
                   These indices correspond to data lines, not counting the header.
        """
        if self.excluded_indices:
            self.excluded_indices.update(indices)
        else:
            self.excluded_indices = set(indices)


    def init_df(self, data: list[list[str]]):
        """Initialize or update the internal DataFrame with new batch data.

        Creates a new pandas DataFrame from the provided data list using the
        column names stored in self.columns.

        Args:
            data: List of rows, where each row is a list of string values
                 corresponding to CSV columns.

        Note:
            Requires self.columns to be set before calling this method.
            This is typically handled by the sample() method.
        """
        self.DataFrame = pd.DataFrame(data, columns=self.columns)

    def sample(self, path: str|pathlib.Path, lines: int, header: bool = True) -> pd.DataFrame:
        """Sample a mini-batch from a CSV file using reservoir sampling.

        On first call, uses reservoir sampling to get an unbiased sample while
        counting total lines. Subsequent calls use efficient direct sampling
        with the known line count.

        Args:
            path: Path to the CSV file to sample from.
            lines: Number of lines to include in the mini-batch.
            header: Whether the CSV file has a header row. Defaults to True.

        Returns:
            DataFrame containing the sampled mini-batch.

        Note:
            For complex CSV parsing needs (quoted fields, escaped characters, etc.),
            consider using pd.read_csv instead of this basic implementation.
        """
        rng = self.rng
        self.path = path
        self.header = header
        self.lines = lines
        first_pass = self.DataFrame.empty or self.length == 0
        with open(path, 'r', encoding="utf-8") as f:
            # First pass: Use Reservoir Sampling for a true one-pass initial sample.
            if first_pass:
                if header:
                    columns_line = f.readline()
                    self.columns = pd.Index(columns_line.strip().split(','))
                
                reservoir: list[list[str]] = []
                self.sampled_indices = []  # Reset sampled indices
                line_count = 0
                for line in f:
                    if line_count not in self.excluded_indices:  # Skip excluded indices
                        # Fill the reservoir with the first `lines` items
                        if len(reservoir) < lines:
                            reservoir.append(line.strip().split(','))
                            self.sampled_indices.append(line_count)
                        else:
                            # For subsequent items, replace an existing item with a decreasing probability
                            j = rng.integers(0, line_count + 1)
                            if j < lines:
                                reservoir[j] = line.strip().split(',')
                                self.sampled_indices[j] = line_count
                    line_count += 1
                self.length = line_count
                self.init_df(reservoir)
                return self.DataFrame
            else:
                # We know how many lines we have, so it is more efficient to just generate random numbers
                # and go to them (in the best case). We will implement a one-pass sampling to fetch the required lines.

                # 1. Generate `lines` unique random indices to fetch from the `self.length` total data lines.
                # and filter out excluded indices
                valid_indices = np.array([i for i in range(self.length) if i not in self.excluded_indices])
                
                if len(valid_indices) < lines:
                    # Not enough lines to sample requested amount
                    indices_to_read = valid_indices.tolist()
                else:
                    # Choose random indices from available indices
                    indices_idx = rng.choice(len(valid_indices), size=lines, replace=False)
                    indices_to_read = np.sort(valid_indices[indices_idx]).tolist()
                
                self.sampled_indices = indices_to_read  # Save which indices were sampled

                new_data: list[list[str]] = []
                indices_ptr = 0

                # 2. Go to the start of the file and read the lines at the chosen indices.
                f.seek(0)
                if header:
                    f.readline()  # Skip the header row

                for i, line in enumerate(f):
                    if indices_ptr < len(indices_to_read) and i == indices_to_read[indices_ptr]:
                        # This is a line we want, so we parse it and add it to our new data
                        line_data = line.strip().split(',')
                        new_data.append(line_data)
                        indices_ptr += 1
                    
                    if indices_ptr == len(indices_to_read):
                        # If we have found all our lines, we can stop reading the file.
                        break
                
                # 3. Replace the content of the DataFrame with the new batch.
                self.init_df(new_data)
                return self.DataFrame

    def resample(self) -> pd.DataFrame:
        """Generate a new mini-batch using previously configured parameters.

        Uses the parameters from the last sample() call to generate a new
        mini-batch from the same file. This is more efficient than sample()
        for subsequent batches since it uses the known line count.

        Returns:
            DataFrame containing the new mini-batch.

        Raises:
            RuntimeError: If called before sample() or with invalid state.

        Note:
            If the file has fewer lines than the requested batch size,
            returns the entire file as a batch.
        """
        if self.length == 0:
            raise RuntimeError("Cannot resample before initial sampling")
            
        # Use sample() with same parameters but it will use the already counted length
        return self.sample(self.path, self.lines, self.header)
        
    def get_sampled_indices(self) -> list[int]:
        """Get the indices of the lines sampled in the last batch.
        
        Returns:
            List of line indices (0-based) that were sampled in the most recent batch.
            These indices correspond to data lines, not counting the header.
        """
        return self.sampled_indices.copy()
