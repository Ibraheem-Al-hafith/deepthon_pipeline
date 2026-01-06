from .registry import build_dataset
from .preprocessing import apply_preprocessing
from .splits import split_dataset


def build_dataset_splits(cfg):
    dataset = build_dataset(cfg)                 # download / load
    dataset = apply_preprocessing(dataset, cfg)  # preprocessing
    return split_dataset(dataset, cfg["data"]["split"])  # train / val
