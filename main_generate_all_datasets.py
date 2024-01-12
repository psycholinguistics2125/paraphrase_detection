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

    def generate_one_dataset(config):
        online = config['generate_dataset']["online"]
        if online:
            dataset = generate_online_paraphrase_dataset_from_config(
                config=config, logger=logger, save=True, clean=True
            )
        else :
            dataset = generate_dataset_from_config(
                config=config['generate_dataset'], logger=logger, save=True, clean=True
            )
            
    for online in [True]:
        config['generate_dataset']['online'] = online
        if online:
            for paraphrase_model in ["gpt-3.5-turbo", "ibm/qcpg-sentences"]: #text-davinci-002
                for nb_paraphrase in [1,4]:
                    for p_paraphrase in [0.1]:
                        for q_paraphrase in [0.9,0.5,0.1]:
                            for alpha_paraphrase in [0.5,0.7]:
                                config['paraphrase_dataset']['model_name'] = paraphrase_model
                                config['online_param']['nb_paraphrase_max'] = nb_paraphrase
                                config['online_param']['p_paraphrase'] = p_paraphrase
                                config['online_param']['q_paraphrase'] = q_paraphrase
                                config['online_param']['alpha'] = alpha_paraphrase
                                generate_one_dataset(config)
        
        else :
            generate_one_dataset(config)