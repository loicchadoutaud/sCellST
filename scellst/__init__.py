import sys

from loguru import logger

logger.remove()
logger.add(
    sink=sys.stdout,
    format="<green>{time:HH:mm:ss}</green> <level>{level}</level>: <level>{message}</level>",
)
