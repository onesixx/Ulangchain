"""
this module is for coloring log messages
"""
import logging

COLORS = { #     \033:(ESC) [:시작 94:색상 코드 m:끝
    'DEBUG':    '\033[96m',  # Cyan
    'INFO':     '\033[92m',  # Green
    'WARNING':  '\033[93m',  # Yellow
    'ERROR':    '\033[91m',  # Red
    'CRITICAL': '\033[95m' # Bold Magenta
}
# ANSI escape codes for colors
MESSAGE_COLOR = '\033[90m'  # Grey
RESET = '\033[0m'

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, RESET)
        record.levelname = f"{log_color}{record.levelname}{RESET}"

        original_message = record.msg
        record.msg = f"{MESSAGE_COLOR}{original_message}{RESET}"
        formatted_message = super().format(record)
        record.msg = original_message  # Restore original message
        return formatted_message

class ColoredFormatter2(logging.Formatter):
    # ANSI escape codes for colors
    MESSAGE_COLOR = '\033[94m' # Orange

    def format(self, record):
        log_color = COLORS.get(record.levelname, RESET)
        record.levelname = f"{log_color}{record.levelname}{RESET}"

        original_message = record.msg
        record.msg = f"{self.MESSAGE_COLOR}{original_message}{RESET}"

        return super().format(record)
