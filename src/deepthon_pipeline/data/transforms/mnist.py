import numpy as np

def transform_images(raw_bytes: bytes) -> np.ndarray:
    """
    Convert ras MNIST image bytes into a normalized NumPy array.
    Args:
        raw_btes (bytes): the original raw bytes to be transformed
    returns (np.ndarray): transformed images
    """
    # 1. Read the bytes into numpy array, skip the first 16 bytes header
    data:np.ndarray = np.frombuffer(raw_bytes, dtype=np.uint8, offset=16)

    # 2. Calculate the total number of images
    num_images: int = len(data) // (28 * 28)

    # 3. Reshape and Normalize
    images:np.ndarray = data.reshape(num_images, 28, 28)
    return images / 255.0

def transform_labels(raw_bytes: bytes) -> np.ndarray:
    """
    Convert ras MNIST labels bytes into a one hot encoded NumPy array.
    Args:
        raw_btes (bytes): the original raw bytes to be transformed
    returns (np.ndarray): transformed images
    """
    # 1. Read the bytes into numpy array, skip the first 16 bytes header
    data:np.ndarray = np.frombuffer(raw_bytes, dtype=np.uint8, offset=8)

    # 2. convert the labels into one hot encoded
    labels: np.ndarray = one_hot_numpy(data, 10)

    return labels

def one_hot_numpy(a: np.ndarray, num_classes: int) -> np.ndarray:
    """Helper function to convert numpy array into one hotted array"""
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
