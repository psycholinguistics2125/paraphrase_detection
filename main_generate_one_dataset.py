# general import
import os
import logging

# specific import
from src.utils import load_config, get_logger
from src.text_generation import generate_dataset_from_config
from src.online_paraphrase_generation import generate_online_paraphrase_dataset_from_config


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Loading config...")
    config = load_config(config_path="config.yaml")
    logger.info("Config loaded")
    logger.info(f"Config : {config}")

    logger.info("Generating dataset...")
    logger.info(
        "/!\ /!\ If this is the first time you run this script, it may take a while to download the models from hugging facehub..."
    )
    online = config['generate_dataset']["online"]
    if online:
        dataset = generate_online_paraphrase_dataset_from_config(
            config=config, logger=logger, save=True, clean=True
        )
    else :
        dataset = generate_dataset_from_config(
            config=config['generate_dataset'], logger=logger, save=True, clean=True
        )
