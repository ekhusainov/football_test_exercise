import logging
import logging.config

import yaml

DEFAULT_LOGGING_PATH = "configs/core_logging.conf.yml"


def setup_logging():
    "Logger from yaml config."
    with open(DEFAULT_LOGGING_PATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


