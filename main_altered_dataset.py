""" 
Alter a datset by injecting paraphrases into the text.
use in CLI
"""

import os
import pandas as pd
import random
import yaml

from src.utils import load_config, get_logger
from src.paraphrase_generation import introduce_paraphrases_to_dataset

if __name__ == "__main__":
    # Load the config file
    config = load_config("config.yaml")["paraphrase_dataset"]

    # Load the logger
    logger = get_logger()

    # Load the dataset
    dataset_path = config["source_dataset"]
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    try : 
        dataset = pd.read_csv(dataset_path)
    except : 
        dataset = pd.read_csv(dataset_path, sep="\t")
        
    logger.info("Loaded dataset from %s", dataset_path)

    # generate the index
    dataset["index_paraphrase"] = dataset["text"].apply(
        lambda x: (random.randint(1, len(x.split(".")) - 1), len(x.split(".")) - 1)
    )
    # Apply model
    model_name = config["model_name"]
    logger.info("Loading model %s", model_name)

    logger.info(f'Paraphrasing using {config["online"]}')

    saving_folder = os.path.join(
        config["saving_folder"],"offline", f"{model_name.replace('/','-')}_{dataset_name}"
    )
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    saving_path = os.path.join(saving_folder, f"altered_dataset.csv")

    altered = introduce_paraphrases_to_dataset(
        dataset, model_name, logger=logger, save_path=saving_path
    )
    logger.info("Altered dataset saved to %s", saving_path)

    # save config in the same folder
    param_path = os.path.join(saving_folder, "dataset_model_kwargs.yaml")
    with open(param_path, "w") as file:
        documents = yaml.dump(config, file)

    logger.info(
        "Model kwargs saved to %s",
        os.path.join(saving_folder, "dataset_model_kwargs.csv"),
    )
