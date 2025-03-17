# log_set <-- log_cfg <-- log_color
import logging
import logging.config
import logging.handlers

import os
import sys
from pathlib import Path
from rosie.config import BASE_DIR
import json
import atexit

# Use my own logger ,not the root logger
# and one logger per module
logger = logging.getLogger("rosie_logger")

def setup_logging(log_filename: str = 'app.log'):
    curr_dir = os.path.dirname(__file__)
    if sys.version_info >= (3, 12):
        config_file = os.path.join(curr_dir, 'log_cfg_py12.json')
        # Queue Handler for Non-blocking Logging
        queue_handler = logging.getHandlerByName("queue_handler")
        if queue_handler is not None:
            queue_handler.listener.start()
            atexit.register(queue_handler.listener.stop)
    else:
        config_file = os.path.join(curr_dir, 'log_cfg.json')
    with open(config_file) as f_in:
        config = json.load(f_in)

    LOG_DIR = BASE_DIR.joinpath('logs').resolve()
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, log_filename)
    config["handlers"]["file"]["filename"] = log_file

    # 필터 추가
    config['filters']['debugfilter']  = {'()': 'rosie.log.log_filter.DebugFilter'}
    config['filters']['infofilter'] = {'()': 'rosie.log.log_filter.InfoFilter'}

    logging.config.dictConfig(config)
