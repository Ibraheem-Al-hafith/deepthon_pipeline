import yaml
from pathlib import Path
from ..config.loader import load_config
from ..training.runner import ExperimentRunner


def cmd_train(config_path):
    cfg = load_config(config_path)
    runner = ExperimentRunner(cfg)
    runner.run()
