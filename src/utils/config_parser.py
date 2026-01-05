from pydantic import BaseModel
from typing import Dict
import yaml
from pathlib import Path

class ParamsConfig(BaseModel):
    """
    Parameters confiurations from YAML file
    this class ensures the input and output shapes
    are integers
    """
    input_shape: int
    output_shape: int

class URLConfig(BaseModel):
    """
    URL configurations: this class is responsible for ensuring 
    that the dataset URL is valid
    """
    images_url: str
    images_filename: str
    labels_url: str
    labels_filename: str

class PathConfig(BaseModel):
    """
    Path configurations: this class is responsible for 
    ensuring the valid paths
    """
    marker_file: str
    raw_dir: str
    processed_dir: str

class DatasetConfig(BaseModel):
    """
    Data set configuration
    """
    paths: PathConfig
    urls: URLConfig
    params: ParamsConfig

class AppConfig(BaseModel):
    """
    The Root configuration class.
    Maps the 'datasets' key in YAML to a dictionary of DatasetConfig objects
    """
    datasets: Dict[str, DatasetConfig]

def load_config(config_path: Path) -> AppConfig:
    """
    Reads a YAML file and parses it into an (AppConfig object)
    Args:
        config_path (Path): the path for the configuration file.
    Returns:
        (AppConfig): dataset instancs
    """
    # 1. Load the YAML file into a standard dictionary
    with open(config_path, "r") as file:
        raw_dict = yaml.safe_load(file)
    
    # 2. Use Pydantic to validate and convert the dict to object
    return AppConfig(**raw_dict)