from pathlib import Path
import numpy as np


class BaseDataset:
    """
    Simple dataset holder for deepthon pipeline.
    """

    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={len(self)})"

    @staticmethod
    def cache_dir():
        root = Path.home() / ".cache" / "deepthon_datasets"
        root.mkdir(parents=True, exist_ok=True)
        return root
