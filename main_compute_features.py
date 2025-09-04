import os, sys
import pandas as pd
import numpy as np
import logging
import warnings


# import from current library
from src.similarity_metrics import compute_similarity_features, add_vectors_do_dataframe
from src.semantic_density import compute_all_semantic_scores
from src.utils import split_into_sentences, build_altered_dataset_without_paraphrase, aggregate_dataset, load_config, get_logger


def add_similarity_features(data, config,col_text="text"):
    """add similarity features to the dataset

    Args:
        data (pd.DataFrame): the dataset

    Returns:
        pd.DataFrame: the dataset with the similarity features
    """
    data['sentences'] = data.apply(lambda x: split_into_sentences(x[col_text]), axis=1)
    data['num_sentences'] = data.apply(lambda x: len(x.sentences), axis=1)
    data= data[data.num_sentences > 2]

    data = compute_similarity_features(data, config=config, col_text=col_text, para=False)
    
    return data



if __name__ == "__main__" :
    warnings.filterwarnings("ignore")
    sample = False
    config = config = load_config("config.yaml")
    logger = get_logger()
    saving_path = os.path.join(config['data_isaac']['folder'],config['data_isaac']['features_file'])
    #saving_path = os.path.join(config['agg_file'].replace(".pkl","_features_2.pkl"))
    logger.info("#################")
    logger.info(f"loading the data from {config['data_isaac']['folder']}.")
    #stories = pd.read_csv(os.path.join(config['data_isaac']['folder'],config['data_isaac']['stories_file']))
    #scores = pd.read_csv(os.path.join(config['data_isaac']['folder'],config['data_isaac']['scores_file']))
    #data = pd.merge(stories, scores, on='subj')
    data = pd.read_csv(os.path.join(config['data_isaac']['folder'],config['data_isaac']['stories_file']),sep = "\t")
    #data = pd.read_pickle(config['agg_file'])
    if sample:
        data = data.sample(100)

    logger.info(f"{len(data)} stories had been loaded. ")


    logger.info("#################")
    logger.info("Adding similarity features")
    data = add_similarity_features(data, config,col_text="cleaned_story_2")
    data.to_pickle(saving_path)
    logger.info(f"Data had been saved to {saving_path}")

    logger.info("#################")
    logger.info("Adding vectors to the dataframe")
    data = add_vectors_do_dataframe(data, config)
    data.to_pickle(saving_path)
    logger.info(f"Data had been saved to {saving_path}")

    logger.info("#################")
    logger.info("Computing semantic features")
    data = data[data['sentence_sim_vectors'].apply(len)>1]
    logger.info(f"length data : {len(data)}")
    data['paragraphs_vector'] = data['sentence_sim_vectors'].apply(lambda x: np.mean(x, axis=0) if x is not None else np.zeros(768))
    data, _ = compute_all_semantic_scores(data)
    data.to_pickle(saving_path)
    logger.info(f"Data had been saved to {saving_path}")

    logger.info("#################")
    logger.info("Cleaning the data")
    data = data[data.num_sentences<20].fillna(-1)
    data.to_pickle(saving_path)
    logger.info(f"Data had been saved to {saving_path}")