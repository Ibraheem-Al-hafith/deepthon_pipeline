DATASET_REGISTRY = {}


def register_dataset(name):
    """
    Decorator to register a dataset builder.
    Usage:
        @register_dataset("mnist")
        def build_mnist(cfg): ...
    """
    def wrapper(fn):
        DATASET_REGISTRY[name.lower()] = fn
        return fn
    return wrapper


def build_dataset(cfg):
    """
    Factory entrypoint called by the pipeline.
    """
    name = cfg["data"]["name"].lower()

    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' is not registered. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    return DATASET_REGISTRY[name](cfg)
