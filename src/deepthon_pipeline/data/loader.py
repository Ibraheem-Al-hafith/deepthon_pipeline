from .base import *
from .registry import DATASET_REGISTRY
import numpy as np
def build_dataset(cfg):
    """
    Factory entrypoint called by the pipeline.
    """
    name = cfg.name

    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' is not registered. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    loader = DATASET_REGISTRY[name](cfg)
    loader.load()
    paths = loader.datasets_paths
    return {
        path.name:load_np_from_disk(path) for path in paths
    }

def load_np_from_disk(path: str):
    if not path:
        return
    return np.load(path)


