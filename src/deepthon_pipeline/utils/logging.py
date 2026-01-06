"""
Centralized logging utilities for the pipeline.

This module exposes a configurable logger factory to ensure consistent
logging behavior across pipeline components (data, models, training,
evaluation, orchestration, CLI, UI, etc.).
"""

from __future__ import annotations
import logging
from logging import Logger
from typing import Optional, Literal
from pathlib import Path
from .config import load_config

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s |"
    "%(funcName)s:%(lineno)d | %(message)s"
)

_LOGGERS: dict[str, Logger] = {}    # cache to avoid duplicate handlers

def get_logger(
        name: str = "deepthon_pipeline",
        level: Literal["DEBUG","INFO","WARNING","ERROR","CRITICAL"] = "INFO",
        log_to_file:bool = False,
        log_dir: Optional[str | Path] = None,
        fmt: str = _DEFAULT_FORMAT
    ) -> Logger:
    """
    Returns a configured logger instance.

    Args:
        name: Logger namespace (typically __name__ from caller).
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to additionally write logs to file.
        log_dir: Directory to store log files if enabled.
        fmt: Log output formatting pattern.

    Returns:
        Logger: A fully configured logger instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    logger.propagate = False # prevent duplicate logs

    formatter = logging.Formatter(fmt=fmt)

    # -------- Console Handler ---------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt = formatter)
    logger.addHandler(console_handler)

    # --------- File Handler -----------
    if log_to_file:
        log_dir = Path(log_dir or "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger

def logger_from_config(config_path: str|Path) -> Logger:
    cfg = load_config(config_path)
    logging_cfg = cfg.get("logging", {})

    return get_logger(
        name=cfg.get("name", "deepthon_pipeline"),
        level=logging_cfg.get("level", "INFO"),
        log_to_file=logging_cfg.get("to_file",False),
        log_dir=logging_cfg.get("log_dir","logs")
    )