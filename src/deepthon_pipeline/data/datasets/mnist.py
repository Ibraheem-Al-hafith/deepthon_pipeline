import gzip
import pickle
import urllib.request
from pathlib import Path
import numpy as np

from .base import BaseDataset
from ..registry import register_dataset


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def _download_mnist():
    cache = BaseDataset.cache_dir() / "mnist.npz"
    if not cache.exists():
        print("Downloading MNIST...")
        urllib.request.urlretrieve(MNIST_URL, cache)
    return np.load(cache)


@register_dataset("mnist")
def build_mnist(cfg):
    data = _download_mnist()

    X = data["x_train"].reshape(-1, 28 * 28).astype("float32")
    y = data["y_train"]

    return BaseDataset(X, y)

_download_mnist()