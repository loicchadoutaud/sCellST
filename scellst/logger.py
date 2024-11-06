import logging
import os
import sys
from datetime import datetime as dt


def create_logger():
    """Create a logger to store and print logs."""
    # create logger
    logger = logging.getLogger("debulk")
    logger.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # create console handler, file handler and set level to debug
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logger.level)
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


logger = create_logger()
