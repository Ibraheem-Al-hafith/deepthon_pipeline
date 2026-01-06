from pathlib import Path
from typing import Callable, Tuple, Any
import numpy as np
import urllib.request
import os

DATA_CACHE = Path.home() / ".deepthon" / "datasets"
DATA_CACHE.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filename: str) -> Path:
    path = DATA_CACHE / filename
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


# ---- Example dataset loaders ----

def load_numpy_file(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def load_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


# registry for plug-and-play dataset support
DATASET_LOADERS: dict[str, Callable[..., Tuple[Any, Any]]] = {
    "numpy": load_numpy_file,
    "csv": load_csv,
}
