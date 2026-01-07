import numpy as np
from ..utils.logging import get_logger
import requests
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.datasets import load_breast_cancer
from deepthon.utils.split import train_test_split
from .registry import register_dataset

import kagglehub
from kagglehub import KaggleDatasetAdapter

logger = get_logger(__name__)

class BaseLoader(ABC):
    """
    Base Loader class that defines the rules (methods) that every loader
    must follow but doesn't implement the specific details for a
    particular dataset.
    """
    def __init__(self, ds_cfg) -> None:
        """
        ds_cfg : the dataset configuration to be load.
        """
        self.ds_cfg = ds_cfg
        self.processed_dir = Path(ds_cfg.paths.processed_dir)
        self.marker_path = self.processed_dir / ds_cfg.paths.marker_file
        self.datasets_paths = []
    
    def is_processed(self) -> bool:
        """Helper function to check if the data is already processed"""
        return self.marker_path.exists()

    @abstractmethod
    def load(self):
        """Each specific loader must implement its own loading logic"""
        pass
    
    def download_file(self, url: str, destination: Path) -> None:
        """
        Downloads a file from a URL to a specific path.
        Args:
            url (str): the file link.
            destination (Path): the destination to save the file
        """
        if destination.exists():
            logger.info(f"File is already exist: {destination}. Skipping download")
            return
        
        try:
            logger.info(f"Downloading :-> {url} ...")
            response = requests.get(url, stream=True, timeout=5)
            response.raise_for_status() #check for http errors

            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "wb") as download:
                for chunk in response.iter_content(chunk_size=8192):
                    download.write(chunk)
            logger.info(f"Successfully downloaded file to: {destination}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            raise # Re-rise to stop the pipeline if the data is missing

@register_dataset("mnist")
class MNISTLoader(BaseLoader):
    """MNIST loader class for loading and transforming the data"""
    def load(self):
        if self.is_processed():
            self.datasets_paths = [
            self.processed_dir / p for p in ["train_x.npy","train_y.npy", "test_x.npy","test_y.npy"]
            ]
            logger.info(f"Dataset already downloaded and processed at :{self.processed_dir}")
            return
        # Define the paths for download :
        raw_img_path_train = Path(self.ds_cfg.paths.raw_dir) / self.ds_cfg.urls.train_images.split("/")[-1].split(".")[0]
        raw_labels_path_train = Path(self.ds_cfg.paths.raw_dir) / self.ds_cfg.urls.train_labels.split("/")[-1].split(".")[0]
        raw_img_path_test = Path(self.ds_cfg.paths.raw_dir) / self.ds_cfg.urls.test_images.split("/")[-1].split(".")[0]
        raw_labels_path_test = Path(self.ds_cfg.paths.raw_dir) / self.ds_cfg.urls.test_labels.split("/")[-1].split(".")[0]

        # 1. Download raw files using BaseLoader method:
        
        self.download_file(self.ds_cfg.urls.train_images, raw_img_path_train)
        self.download_file(self.ds_cfg.urls.train_labels, raw_labels_path_train)
        self.download_file(self.ds_cfg.urls.test_images, raw_img_path_test)
        self.download_file(self.ds_cfg.urls.test_labels, raw_labels_path_test)

        # 2. Transform the data:
        with open(raw_img_path_train, "rb") as transform:
            train_x = transform_images(transform.read())

        with open(raw_labels_path_train, "rb") as transform:
            train_y = transform_labels(transform.read())

        with open(raw_img_path_test, "rb") as transform:
            test_x = transform_images(transform.read())

        with open(raw_labels_path_test, "rb") as transform:
            test_y = transform_labels(transform.read())

        # 3. Save the processed data
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.processed_dir / "train_x.npy", train_x)
        np.save(self.processed_dir / "train_y.npy", train_y)
        np.save(self.processed_dir / "test_x.npy", test_x)
        np.save(self.processed_dir / "test_y.npy", test_y)
        self.datasets_paths = [
            self.processed_dir / p for p in ["train_x.npy","train_y.npy", "test_x.npy","test_y.npy"]
        ]
        # 4. Create the maker file
        self.marker_path.touch()
        logger.info(f"Ingestion complete, Marker created at {self.marker_path}")

@register_dataset("cancer")
class BREAST_CANCER_LOADER(BaseLoader):
    """brease cancer loader class"""
    def load(self):
        if self.is_processed():
            self.datasets_paths = [
            self.processed_dir / p for p in ["train_x.npy","train_y.npy", "test_x.npy","test_y.npy"]
            ]
            logger.info(f"Dataset already downloaded and processed at :{self.processed_dir}")
            return
        logger.info("Loading breast cancer dataset")
        X, y = load_breast_cancer(return_X_y=True)
        X,y = np.array(X),np.array(y)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1,stratify=y
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.processed_dir / "train_x.npy",x_train)
        np.save(self.processed_dir / "train_y.npy",y_train)
        np.save(self.processed_dir / "test_x.npy",x_test)
        np.save(self.processed_dir / "test_y.npy",y_test)
        self.datasets_paths = [
            self.processed_dir / p for p in ["train_x.npy","train_y.npy", "test_x.npy","test_y.npy"]
        ]
        self.marker_path.touch()
        logger.info(f"Ingestion complete, Marker created at {self.marker_path}")

@register_dataset("turbines")
class TURBINES_LOADER(BaseLoader):
    """brease cancer loader class"""
    def load(self):

        if self.is_processed():
            self.datasets_paths = [
            self.processed_dir / p for p in ["train_x.npy","train_y.npy", "test_x.npy","test_y.npy"]
            ]
            logger.info(f"Dataset already downloaded and processed at :{self.processed_dir}")
            return
        # Set the path to the file you'd like to load
        file_path = "Data.csv"

        # Load the latest version
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "ishank2005/wind-turbines-data-csv",
            file_path,
            # Provide any additional arguments like 
            # sql_query or pandas_kwargs. See the 
            # documenation for more information:
            # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )

        X, y = df.drop(columns="PE").values, df["PE"].values
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1,stratify=y
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.processed_dir / "train_x.npy",x_train)
        np.save(self.processed_dir / "train_y.npy",y_train)
        np.save(self.processed_dir / "test_x.npy",x_test)
        np.save(self.processed_dir / "test_y.npy",y_test)
        self.datasets_paths = [
            self.processed_dir / p for p in ["train_x.npy","train_y.npy", "test_x.npy","test_y.npy"]
        ]
        self.marker_path.touch()
        logger.info(f"Ingestion complete, Marker created at {self.marker_path}")

@register_dataset("path")
class PATH_DATASET(BaseLoader):
    def load(self):
        if not self.ds_cfg.paths.exist():
            logger.warning(f"No data set in {self.ds_cfg.paths}")
            raise
        self.datasets_paths = [self.ds_cfg.paths]


import numpy as np
import gzip

def transform_images(raw_bytes: bytes) -> np.ndarray:
    """
    Convert ras MNIST image bytes into a normalized NumPy array.
    Args:
        raw_btes (bytes): the original raw bytes to be transformed
    returns (np.ndarray): transformed images
    """
    # 0. decompress the file:
    decompressed_bytes = gzip.decompress(raw_bytes)
    # 1. Read the bytes into numpy array, skip the first 16 bytes header
    data:np.ndarray = np.frombuffer(decompressed_bytes, dtype=np.uint8, offset=16)

    # 2. Calculate the total number of images
    num_images: int = len(data) // (28 * 28)

    # 3. Reshape and Normalize
    images:np.ndarray = data.reshape(num_images, 28* 28)
    return images / 255.0

def transform_labels(raw_bytes: bytes) -> np.ndarray:
    """
    Convert ras MNIST labels bytes into a one hot encoded NumPy array.
    Args:
        raw_btes (bytes): the original raw bytes to be transformed
    returns (np.ndarray): transformed images
    """
    # 0. decompress the file:
    decompressed_bytes = gzip.decompress(raw_bytes)
    # 1. Read the bytes into numpy array, skip the first 16 bytes header
    data:np.ndarray = np.frombuffer(decompressed_bytes, dtype=np.uint8, offset=8)

    # 2. convert the labels into one hot encoded
    labels: np.ndarray = one_hot_numpy(data, 10)

    return labels

def one_hot_numpy(a: np.ndarray, num_classes: int) -> np.ndarray:
    """Helper function to convert numpy array into one hotted array"""
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])