from dataclasses import dataclass
from typing import Any, Tuple, Optional
from deepthon.utils.split import train_test_split


@dataclass
class DatasetSplit:
    X: Any
    y: Any


def split_dataset(
    X,
    y,
    train_ratio: float,
    val_ratio: Optional[float],
    test_ratio: float,
    shuffle: bool,
    seed: Optional[int] = None,
) -> Tuple[DatasetSplit, Optional[DatasetSplit], DatasetSplit]:

    # 1) Train + remaining split
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y,
        test_size=1 - train_ratio,
        shuffle=shuffle,
        random_state=seed,
    )

    # 2) If no validation split → remaining = test
    if val_ratio is None or val_ratio == 0:
        return (
            DatasetSplit(X_train, y_train),
            None,
            DatasetSplit(X_rem, y_rem),
        )

    # 3) Compute relative share inside remaining
    rem_total = val_ratio + test_ratio
    rel_val = val_ratio / rem_total

    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem,
        test_size=1 - rel_val,
        shuffle=shuffle,
        random_state=seed,
    )

    return (
        DatasetSplit(X_train, y_train),
        DatasetSplit(X_val, y_val),
        DatasetSplit(X_test, y_test),
    )
