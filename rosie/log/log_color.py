"""
this module is for coloring log messages
"""
import logging

class ColoredFormatter(logging.Formatter):
    # ANSI escape codes for colors
    MESSAGE_COLOR = '\033[35m'  # Magenta
    RESET = '\033[0m'

    def format(self, record):
        original_message = record.msg
        record.msg = f"{self.MESSAGE_COLOR}{original_message}{self.RESET}"
        formatted_message = super().format(record)
        record.msg = original_message  # Restore original message
        return formatted_message

class ColoredFormatter2(logging.Formatter):
    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[94m',    # Blue    \033:(ESC) [:시작 94:색상 코드 m:끝
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[1;91m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

    def format2(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        original_message = record.msg
        record.msg = f"{log_color}{original_message}{self.RESET}"
        formatted_message = super().format(record)
        record.msg = original_message  # Restore original message
        return formatted_message
