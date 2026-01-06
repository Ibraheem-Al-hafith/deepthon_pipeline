from sklearn.datasets import load_breast_cancer
from .base import BaseDataset
from ..registry import register_dataset


@register_dataset("breast_cancer")
def build_breast_cancer(cfg):
    ds = load_breast_cancer()
    return BaseDataset(ds.data, ds.target)
