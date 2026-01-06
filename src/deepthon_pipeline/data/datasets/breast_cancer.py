from sklearn.datasets import load_breast_cancer
from .base import BaseDataset
from ..registry import register_dataset


@register_dataset("breast_cancer")
def build_breast_cancer(cfg):
    X,y = load_breast_cancer(return_X_y=True)
    return BaseDataset(X, y)
