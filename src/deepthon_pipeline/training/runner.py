# training/runner.py
from pathlib import Path
import json
from datetime import datetime
from typing import Optional

from ..data.loader import build_dataset
from ..data.base import DataModule,DataSplit  # Import the types for hinting
from ..models.builder import build_model_from_config
from ..training.checkpoints import save_checkpoint
from ..utils.logging import get_logger
from ..models.optimizer_builder import build_optimizer

from deepthon.pipeline.trainer import Trainer
from deepthon.nn.losses import CrossEntropy, MAE

logger = get_logger(__name__)

class ExperimentRunner:
    """
    Orchestrates full training lifecycle using structured DataModules.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dm: Optional[DataModule] = None # Placeholder for our data

        # Create experiment directory
        # ts = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        name = cfg.get("experiment", "run")
        # self.exp_dir = Path("runs") / f"{name}_{ts}"
        self.exp_dir = Path("runs") / f"{name}"
        if self.exp_dir.exists():
            logger.info(f"Experiment already exists on dir {self.exp_dir}, loading states .......")
        else:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Experiment directory created: {self.exp_dir}")

    def build_data(self):
        """Builds dataset and stores it in a structured DataModule."""
        logger.info("Building dataset...")
        
        # We assume the config specifies which dataset entry to use
        # e.g., self.cfg.datasets.mnist or self.cfg.active_dataset
        dataset_cfg = self.cfg.datasets.turbines
        self.input_dim = dataset_cfg.input_dim
        
        # This now returns a DataModule object directly
        self.dm = build_dataset(dataset_cfg)
        msg = f"Dataset Ready | Train: {self.dm.train.shape} | Val: {self.dm.val.shape}" \
        if not self.dm.test else \
        f"Dataset Ready | Train: {self.dm.train.shape} | Val: {self.dm.val.shape} | Test: {self.dm.test.shape}"
        
        logger.info(msg=msg)

    def build_model(self):
        logger.info("Building model...")
        self.model = build_model_from_config(self.cfg, self.input_dim)

    def build_optimizer(self):
        logger.info("Building optimizer...")
        self.optimizer = build_optimizer(self.cfg.training.optimizer)

    def build_trainer(self):
        logger.info("Initializing Trainer...")
        tcfg = self.cfg.training

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_func=MAE(),
            batch_size=tcfg.batch_size,
            early_stopping=tcfg.get("early_stopping", False),
            patience=tcfg.get("patience", 5),
            metric_fn=tcfg.get("metric", None),
            checkpoint_dir=self.exp_dir,
            save_every=1
        )

    def test(self, checkpoint_path: Optional[Path] = None):
        """
        Evaluate the model on the test dataset.
        If checkpoint_path is provided, it loads the model weights first.
        """
        logger.info("Starting evaluation on test set...")

        # 1. Handle standalone evaluation (optional loading)
        if checkpoint_path:
            self.trainer.load_checkpoint(checkpoint_path)
            logger.info(f"Loaded weights from {checkpoint_path}")

        # 2. Check if test data exists
        if not self.dm or not self.dm.test:
            logger.warning("No test split found in DataModule. Skipping test phase.")
            return

        # 3. Perform evaluation
        # Assuming your Trainer has an evaluate method, 
        # or we use the model directly:
        test_results,output = self.trainer.validate(
            X_val=self.dm.test.x,
            y_val=self.dm.test.y
        )
        msg = f"Test Results: {test_results} "
        if self.trainer.metric is not None:
            msg+=f"| {self.trainer.metric_fn}: {self.trainer.metric(self.dm.test.y, output)}"
        logger.info(msg)
        self._save_test_artifacts(test_results)
        return test_results

    def _save_test_artifacts(self, results):
        """Save test metrics to a separate file."""
        path = self.exp_dir / "test_results.json"
        path.write_text(json.dumps(results, indent=2))
        logger.info(f"Test artifacts saved to {path}")

    def run(self, resume: bool = False):
        """The full 'Train + Test' pipeline."""
        logger.info("Starting experiment...")
        self.build_data()
        self.build_model()
        self.build_optimizer()
        self.build_trainer()

        # --- RESUME LOGIC ---
        if resume:
            ckpt_path = self.exp_dir / "checkpoint.pkl"
            if ckpt_path.exists():
                logger.info(f"Resuming training from {ckpt_path}...")
                self.trainer.load_checkpoint(ckpt_path)
                logger.info(f"Resuming from Epoch {self.trainer.start_epoch + 1}")
            else:
                logger.warning(f"No checkpoint found at {ckpt_path}. Starting from scratch.")

        assert isinstance(self.dm, DataModule)
        # Phase 1: Training
        self.trainer.train(
            X_train=self.dm.train.x,
            y_train=self.dm.train.y,
            X_val=self.dm.val.x,
            y_val=self.dm.val.y,
            epochs=self.cfg.training.epochs,
        )

        # Phase 2: Post-Training Save
        self.trainer.save_checkpoint(filename="last_model.pkl")
        logger.info("Experiment completed.")

        # Phase 3: Immediate Testing
        self.test()