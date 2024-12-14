import logging
from logging import Logger


def setup(name: str, level: int = logging.INFO) -> Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
