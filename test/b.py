import logging

logger = logging.getLogger(__name__)

l_config ={
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)s | %(message)s",
            "datefmt" : "%m-%d-%Y %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard"
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": True,
        }
    }
}
logging.config.dictConfig(l_config)

logger.debug('This message should go to the log file')
logger.info('So should this')
logger.warning('And this, too')
logger.error('And non-ASCII stuff, too, like Øresund and Malmö')
logger.critical("Critical")


