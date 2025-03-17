import logging
# DEBUG(10) < INFO(20) < WARNING(30) < ERROR(40) < CRITICAL(50)
class DebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno <= logging.DEBUG  # DEBUG 허용

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno <= logging.INFO   # INFO 허용