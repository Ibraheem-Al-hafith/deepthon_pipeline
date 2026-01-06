import pickle
from pathlib import Path

def save_checkpoint(trainer, path, filename="checkpoint.pkl"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = path / filename

    state = {
        "model": trainer.model.get_state(),
        "optimizer": trainer.optimizer.get_state(),
    }

    checkpoint_path.write_bytes(pickle.dumps(state))



def load_checkpoint_if_exists(trainer, path):
    path = Path(path)
    if not path.exists():
        return False

    state = pickle.loads(path.read_bytes())

    trainer.model.load_state(state["model"])
    trainer.optimizer.load_state(state["optimizer"])
    # trainer.start_epoch = state["epoch"]

    return True
