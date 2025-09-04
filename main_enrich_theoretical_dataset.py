"""  
From all the dattaset generated from main_generate_all_datasets.py, we will enrich the dataset with the following steps:
- add infromation about the paraphrase type
- add semantic features
- add similarity features
- add the target (has_paraphase, had_parpahrase)
"""
import os, sys
import pandas as pd
import numpy as np
import logging
import warnings


# import from current library
from src.similarity_metrics import compute_similarity_features, add_vectors_do_dataframe
from src.semantic_density import compute_all_semantic_scores
from src.utils import split_into_sentences, build_altered_dataset_without_paraphrase, aggregate_dataset, load_config, get_logger, clean_generated_data



def main_compute_and_save_data(config, save=True, sample=True,model_name_list = ['gpt2',"pythia"],paraphrase_model_list = ["gpt-3.5-turbo"], logger = logging.getLogger(__name__),use_roc_data = False):
    
    # load the data
    data = aggregate_dataset(model_name_list, config, paraphrase_model_list)

    roc_data = pd.read_csv(os.path.join(config['data_folder'],"corpus/ROCStories_spring2016.csv"),sep=",")
    altered_without_para = build_altered_dataset_without_paraphrase(data)
    
    if sample:
        data = data.sample(100)
        roc_data = roc_data.sample(100)
        altered_without_para = altered_without_para.sample(100)
    else : 
        roc_data = roc_data.sample(3000)
    
   
    
    # enrich Roc data
    roc_data['text'] = roc_data['sentence1'] + " " + roc_data['sentence2'] + " " + roc_data['sentence3'] + " " + roc_data['sentence4'] + " " + roc_data['sentence5']
    roc_data['altered_text'] = roc_data['text']
    roc_data['clean_paraphrase_index'] = None
    
    # add the paraphrase type
    roc_data['clean_paraphase_type'] = 0
    altered_without_para['clean_paraphase_type'] = 0
    
    # had_parahrase features
    roc_data['had_paraphrase'] = False
    altered_without_para["had_paraphrase"] = True
    data['had_paraphrase'] = False
    
    # had the paraphrase features
    altered_without_para['source'] = altered_without_para.llm_model
    data['source'] = data.llm_model
    roc_data["source"] = "roc_story"

    if use_roc_data==False:
        roc_data = pd.DataFrame(columns = data.columns)
        dataset_list = [data,altered_without_para]
    else :
        dataset_list = [data,roc_data,altered_without_para]

    # remove empty sentences
    for dataset in dataset_list:
        print(dataset.columns)
        dataset['sentences'] = dataset.apply(lambda x: split_into_sentences(x.altered_text), axis=1)
        dataset['num_sentences'] = dataset.apply(lambda x: len(x.sentences), axis=1)
        dataset["clean_text"] = dataset.apply(lambda x: x.altered_text if type(x.text) != str else x.text, axis=1)
        dataset = dataset[dataset.num_sentences > 2]
        dataset = dataset[dataset.num_sentences < 20]
        dataset = dataset.reset_index(drop=True)

    logger.info("#################")
    logger.info(f'length data : {len(data)}')
    logger.info(f'length roc_data : {len(roc_data)}')
    logger.info(f'length altered_without_para : {len(altered_without_para)}')

    # had the similarity features
    logger.info("### ROC DATA ###")
    if use_roc_data:
        roc_data = compute_similarity_features(roc_data, config, col_text="text", para=False)
    logger.info("### ALTERED WITHOUT PARA ###")
    altered_without_para = compute_similarity_features(altered_without_para, config, col_text="altered_text", para=False)
    logger.info("### ALTERED DATA WITH PARA ###")
    data = compute_similarity_features(data, config, col_text='altered_text', para=True)
    
    # concatenate all the data
    data = pd.concat(dataset_list)
    
    
    if save : 
        data.to_pickle(config['agg_file'])
    return data



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    config = config = load_config("config.yaml")
    logger = get_logger()

   

    paraphrase_model_list = ["gpt-3.5-turbo"]
    model_name_list = ["gpt2","pythia"]
    sample = False
    roc_data = False
    add_vectors = False
   
    config['agg_file'] = os.path.join(config['data_folder'],f"corpus/20240503_aggregated_database_{'_'.join(paraphrase_model_list)}_{'_'.join(model_name_list)}.pkl")
    
    logger.info("#################")
    logger.info(f"Computing and saving data for paraphrase model {paraphrase_model_list} and model {model_name_list}")

    # logger.info("#################")
    # logger.info("Computing similarly features")
    # data = main_compute_and_save_data(config, 
    #                                  save=True, 
    #                                  sample=sample, 
    #                                  model_name_list = model_name_list,
    #                                  paraphrase_model_list = paraphrase_model_list,
    #                                  logger = logger,
    #                                  use_roc_data = roc_data)

    data = pd.read_pickle(config['agg_file'])
    data = data[data["sentence_sim_vectors"].apply(type)==np.ndarray]
    logger.info(f"length data : {len(data)}")
    logger.info("#################")
    if add_vectors :
        logger.info("Adding vectors to the dataframe")
        data = add_vectors_do_dataframe(data, config)
   
        data.to_pickle(config['agg_file'])

        data = pd.read_pickle(config['agg_file'])
    logger.info("#################")
    logger.info("Computing semantic features")
    data = data[data['sentence_sim_vectors'].apply(len)>1]
    logger.info(f"length data : {len(data)}")
    data['paragraphs_vector'] = data['sentence_sim_vectors'].apply(lambda x: np.mean(x, axis=0) if x is not None else np.zeros(768))
    data, _ = compute_all_semantic_scores(data)
    data.to_pickle(config['agg_file'])

    logger.info("#################")
    logger.info("Cleaning the data")
    data = data[data.num_sentences<20].fillna(-1)
    data = clean_generated_data(data)
    data.to_pickle(config['agg_file'])

