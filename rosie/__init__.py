from rosie.log import setup_logging
setup_logging('app_all.log')

# Uasge: example below ------
# from rosie.log import logger
# logger.info("Let's go, rosie!!")

from .config import (
    BASE_DIR,
    HOME_DIR,
    DOWNLOADS_DIR,

    DATA_DIR,
    DOC_DIR,
    TMP_DIR,

    ASSET_DIR,
    BACKEND_DIR,
)
