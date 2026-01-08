import yaml
from pathlib import Path
from ..config.loader import load_config
from ..training.runner import ExperimentRunner


def cmd_train(config_path, resume: bool = False):
    cfg = load_config(config_path)
    runner = ExperimentRunner(cfg)
    runner.run(resume=resume)

# commands.py (Add this)

def cmd_test(config_path: str, checkpoint_path: str):
    """
    Run evaluation only. 
    Usage: deepthon-cli test --config my_cfg.yaml --ckpt runs/exp_1/checkpoint.pkl
    """
    cfg = load_config(config_path)
    runner = ExperimentRunner(cfg)
    
    # 1. Only build what we need for inference
    runner.build_data()
    runner.build_model()
    runner.build_optimizer() # Usually needed to init Trainer state
    runner.build_trainer()

    # 2. Run the test with the specific checkpoint
    runner.test(checkpoint_path=Path(checkpoint_path))
