import argparse
from .commands import cmd_train, cmd_test


def main():
    parser = argparse.ArgumentParser(description="Deepthon Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train Command
    # 1. training from scratch:
    train_parser = subparsers.add_parser("train", help="Run the training pipeline")
    train_parser.add_argument("--config", required=True, help="Path to config.yaml")
    # 2. resume from checkpoint:
    train_parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint ine experiment directory")


    # Test Command
    test_parser = subparsers.add_parser("test", help="Run evaluation on a saved checkpoint")
    test_parser.add_argument("--config", required=True, help="Path to config.yaml")
    test_parser.add_argument("--ckpt", required=True, help="Path to the saved checkpoint")

    args = parser.parse_args()

    # Route to the correct function
    if args.command == "train":
        cmd_train(args.config, resume=args.resume)
    elif args.command == "test":
        cmd_test(args.config, args.ckpt)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
