import yaml
from ..training.runner import ExperimentRunner


def cmd_train(config_path):
    cfg = yaml.safe_load(open(config_path))
    runner = ExperimentRunner(cfg)
    runner.run()
