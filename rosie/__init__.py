from rosie.log import setup_logging
setup_logging('app_all.log')

from rosie.log import logger
logger.info("Let's go, rosie!!")

from .config import (
    HOME_DIR,
    DOWNLOADS_DIR,

    BASE_DIR,
    ASSET_DIR,
    BACKEND_DIR,
    DATA_DIR,
    DOC_DIR,
    TMP_DIR,
)
