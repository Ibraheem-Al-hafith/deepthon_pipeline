from typing import Any, Tuple
from .loaders import DATASET_LOADERS
from .preprocessing import build_transform_pipeline
from .splits import split_dataset


def load_dataset_from_config(cfg) -> Tuple[Any, Any, Any]:
    ds_cfg = cfg.dataset

    loader_name = ds_cfg.loader.name
    loader_args = ds_cfg.loader.get("params", {})

    if loader_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset loader: {loader_name}")

    X, y = DATASET_LOADERS[loader_name](**loader_args)

    # ---- preprocessing ----
    transforms = ds_cfg.preprocessing.transforms or []
    pipeline = build_transform_pipeline(transforms)
    X = pipeline(X)

    # ---- split ----
    split_cfg = ds_cfg.preprocessing.split

    train, val, test = split_dataset(
        X=X,
        y=y,
        train_ratio=split_cfg.train,
        val_ratio=split_cfg.get("val", None),
        test_ratio=split_cfg.test,
        shuffle=ds_cfg.preprocessing.shuffle,
        seed=ds_cfg.seed,
    )

    return train, val, test
