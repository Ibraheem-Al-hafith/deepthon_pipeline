from pathlib import Path


def save_checkpoint(trainer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model": trainer.model.get_state(),
        "optimizer": trainer.optimizer.get_state(),
        "epoch": trainer.current_epoch,
    }

    path.write_bytes(trainer.serialize_state(state))


def load_checkpoint_if_exists(trainer, path):
    path = Path(path)
    if not path.exists():
        return False

    state = trainer.deserialize_state(path.read_bytes())

    trainer.model.load_state(state["model"])
    trainer.optimizer.load_state(state["optimizer"])
    trainer.current_epoch = state["epoch"]

    return True
