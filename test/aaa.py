import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format = '%(asctime)s | %(levelname)s | %(message)s',
    datefmt = '%m-%d-%Y %H:%M:%S',
    # filename='example.log',
    # encoding='utf-8',
)

logger.debug('This message should go to the log file')
logger.info('So should this')
logger.warning('And this, too')
logger.error('And non-ASCII stuff, too, like Øresund and Malmö')
logger.critical("Critical")


