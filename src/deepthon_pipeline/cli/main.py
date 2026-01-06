import argparse
from .commands import cmd_train

parser = argparse.ArgumentParser()
parser.add_argument("train", help="Run training with config")
parser.add_argument("--config", required=True)

args = parser.parse_args()
cmd_train(args.config)
