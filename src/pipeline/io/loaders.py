import numpy as np
import logging
import requests
from pathlib import Path
from abc import ABC, abstractmethod
from src.utils.config_parser import DatasetConfig
from src.pipeline.transforms.mnist_transform import transform_images, transform_labels

logger = logging.getLogger(__name__)

class BaseLoader(ABC):
    """
    Base Loader class that defines the rules (methods) that every loader
    must follow but doesn't implement the specific details for a
    particular dataset.
    """
    def __init__(self, config: DatasetConfig) -> None:
        """
        config (DatasetConfig): the dataset configuration to be load.
        """
        self.config = config
        self.processed_dir = Path(config.paths.processed_dir)
        self.marker_path = self.processed_dir / config.paths.marker_file
    
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

class MNISTLoader(BaseLoader):
    """MNIST loader class for loading and transforming the data"""
    def load(self):
        # Define the paths for download :
        raw_img_path = Path(self.config.paths.raw_dir) / self.config.urls.images_filename
        raw_labels_path = Path(self.config.paths.raw_dir) / self.config.urls.labels_filename

        # 1. Download raw files using BaseLoader method:
        self.download_file(self.config.urls.images_url, raw_img_path)
        self.download_file(self.config.urls.labels_url, raw_labels_path)

        # 2. Transform the data:
        with open(raw_img_path, "rb") as transform:
            train_x = transform_images(transform.read())

        with open(raw_labels_path, "rb") as transform:
            train_y = transform_labels(transform.read())

        # 3. Save the processed data
        np.save(self.processed_dir / "train_x.npy", train_x)
        np.save(self.processed_dir / "train_y.npy", train_y)

        # 4. Create the maker file
        self.marker_path.touch()
        logger.info(f"Ingestion complete, Marker created at {self.marker_path}")

