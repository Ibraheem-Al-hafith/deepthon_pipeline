from deepthon.src.deepthon.utils.split import train_test_split


def split_dataset(dataset, split_cfg):
    """
    Wraps deepthon train_test_split into pipeline-friendly API.
    """
    test_size = split_cfg.get("test_size", 0.2)
    shuffle = split_cfg.get("shuffle", True)
    stratify = split_cfg.get("stratify", None)

    X_train, X_val, y_train, y_val = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        shuffle=shuffle,
        stratify=stratify,
    )

    train_ds = dataset.__class__(X_train, y_train)
    val_ds = dataset.__class__(X_val, y_val)

    return train_ds, val_ds
