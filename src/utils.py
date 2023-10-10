import yaml
import os
import logging
import logging.config

import pandas as pd

def load_config(config_path:str, logger = logging.getLogger(__name__))->dict:
    """
    Load a yaml config file and returns a dictionnary
    """
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
    return config

def get_logger(config_path = 'logging_config.yaml')->logging.Logger:
    with open(os.path.join(config_path), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger('training')
    return logger