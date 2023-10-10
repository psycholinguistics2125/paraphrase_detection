import yaml
import os
import logging
import logging.config

import pandas as pd


import nltk
import re
nltk.download('punkt')

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


def clean_generated_text(generated_text:str)->str:
     # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(generated_text)

    # Remove the last incomplete sentence if it exists (if the last sentence ends with a period)
    if sentences and not sentences[-1].endswith('.'):
        sentences.pop()

    # Remove special characters and combine the sentences back into text with only "." between sentences
    cleaned_text = '.'.join([re.sub(r'[^a-zA-Z0-9\s]', '', sentence) for sentence in sentences])
    
    cleaned_text = cleaned_text.replace("\n", " ")

    return cleaned_text
