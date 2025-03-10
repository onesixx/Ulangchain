# import logging
# from rosie import BASE_DIR

# logger = logging.getLogger("shinylog")
# LOG_DIR = BASE_DIR.joinpath('logs').resolve()

# def log():
#     logger = logging.getLogger("shinylog")
#     logger.setLevel(logging.DEBUG) # DEBUG, INFO, [WARNING], ERROR, CRITICAL
#     formatter = logging.Formatter(' %(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')
#     stream_handler = logging.StreamHandler()
#     file_handler = logging.FileHandler(LOG_DIR.joinpath("ufunc.log"))
#     # papertrail_handler = logging.handlers.SysLogHandler(address=('logs6.papertrailapp.com', 12345))
#     stream_handler.setFormatter(formatter)
#     file_handler.setFormatter(formatter)

#     logger.addHandler(stream_handler)
#     logger.addHandler(file_handler)
#     return logger

# logger = log()
# logger.info("Hello from ufunc.py!k")
# logger.debug("Debugging ufunc.py")

