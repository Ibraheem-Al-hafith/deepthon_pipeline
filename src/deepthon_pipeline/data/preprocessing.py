from typing import Callable, Iterable
import numpy as np

# ---- Transform implementations ----

def normalize(x: np.ndarray):
    return (x - x.mean()) / (x.std() + 1e-8)

def standardize(x: np.ndarray):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def flatten(x: np.ndarray):
    return x.reshape(len(x), -1)

# ---- Registry ----

TRANSFORM_REGISTRY: dict[str, Callable] = {
    "normalize": normalize,
    "standardize": standardize,
    "flatten": flatten,
}

# ---- Composition ----

def build_transform_pipeline(names: Iterable[str]) -> Callable:
    funcs = [TRANSFORM_REGISTRY[n] for n in names]

    def pipeline(x):
        for fn in funcs:
            x = fn(x)
        return x

    return pipeline

def apply_preprocessing(dataset, cfg):
    """Apply optional preprocessing pipeline."""
    prep_cfg = cfg["data"].get("preprocessing", None)

    if not prep_cfg:
        return dataset

    # Example placeholder — extend later
    if prep_cfg.get("normalize", False):
        dataset.X = dataset.X / 255.0

    return dataset
