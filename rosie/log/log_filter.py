import logging

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

class DebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.INFO