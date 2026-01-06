from pathlib import Path
import json
from datetime import datetime

from ..data.datasets import load_dataset_from_config
from ..models.builder import build_model_from_config
from ..models.registry import build_model
from ..training.checkpoints import save_checkpoint
from ..utils.logging import get_logger
from ..models.optimizer_builder import build_optimizer  # your optimizer factory

from deepthon.pipeline.trainer import Trainer
from deepthon.nn.losses import BCE

logger = get_logger(__name__)


class ExperimentRunner:
    """
    Orchestrates full training lifecycle:
        dataset → model → optimizer → trainer → metrics → checkpoint
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # create experiment directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = cfg.get("experiment", "run")
        self.exp_dir = Path("runs") / f"{name}_{ts}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment directory: {self.exp_dir}")

    # ------------------------------------------------------------------ #
    # Build components
    # ------------------------------------------------------------------ #

    def build_data(self):
        logger.info("Building dataset + preprocessing + splits...")
        self.train_data, self.val_data, self.test_data = load_dataset_from_config(self.cfg)

    def build_model(self):
        logger.info("Building model...")
        self.model = build_model_from_config(self.cfg)

    def build_optimizer(self):
        logger.info("Building optimizer...")
        optim_cfg = self.cfg.training.optimizer
        self.optimizer = build_optimizer(optim_cfg)

    def build_trainer(self):
        logger.info("Initializing Trainer...")

        tcfg = self.cfg["training"]

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_func=BCE(from_logits=False),    # deepthon provides this
            batch_size=tcfg["batch_size"],
            early_stopping=tcfg.get("early_stopping", False),
            patience=tcfg.get("patience", 5),
            metric_fn=tcfg.get("metric", None),
        )

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #

    def run(self):
        logger.info("Starting experiment...")

        self.build_data()
        self.build_model()
        self.build_optimizer()
        self.build_trainer()

        X_train, y_train = self.train_data.X,self.train_data.y
        X_val, y_val = self.val_data.X, self.val_data.y

        epochs = self.cfg["training"]["epochs"]

        self.trainer.train(
            X_train=X_train,
            y_train=y_train.reshape(-1,1),
            X_val=X_val,
            y_val=y_val.reshape(-1,1),
            epochs=epochs,
        )

        logger.info("Training complete — saving checkpoint")
        self._save_results()

    # ------------------------------------------------------------------ #
    # Artifacts & Checkpoint
    # ------------------------------------------------------------------ #

    def _save_results(self):
        history = {
            "train_losses": self.trainer.train_losses,
            "val_losses": self.trainer.val_losses,
        }

        (self.exp_dir / "history.json").write_text(json.dumps(history, indent=2))

        save_checkpoint(
            trainer=self.trainer,
            path=self.exp_dir,
        )
        
        logger.info("Checkpoint + history saved.")