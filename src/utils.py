import yaml
import os
import logging
import logging.config

import pandas as pd
import numpy as np


import nltk
import re

import ast

nltk.download("punkt")


def load_config(config_path: str, logger=logging.getLogger(__name__)) -> dict:
    """
    Load a yaml config file and returns a dictionnary
    """
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
    return config


def get_logger(config_path="logging_config.yaml") -> logging.Logger:
    with open(os.path.join(config_path), "r") as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger("training")
    return logger


def clean_generated_text(generated_text: str) -> str:
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(generated_text)

    # Remove the last incomplete sentence if it exists (if the last sentence ends with a period)
    if sentences and not sentences[-1].endswith("."):
        sentences.pop()

    # Remove special characters and combine the sentences back into text with only "." between sentences
    cleaned_text = ".".join(
        [re.sub(r"[^a-zA-Z0-9\s]", "", sentence) for sentence in sentences]
    )

    cleaned_text = cleaned_text.replace("\n", " ")

    return cleaned_text



import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


def replace_multiple_periods(text):
    # Use a regular expression to replace consecutive periods with a single period
    cleaned_text = re.sub(r'\.+', '.', text)
    return cleaned_text

def generate_random_index_for_paraphrase_location(list_length, alpha=2.0, type = "close"):
    """
    Generate a random index based on a discrete heavy-tailed distribution (Pareto distribution).
    
    Parameters:
    - last_index: The index of the last selected sentence.
    - list_length: The total number of sentences in the list.
    - alpha: The shape parameter of the Pareto distribution (default is 2.0).
    - type: close or distant
    
    Returns:
    A randomly selected index.
    """
    probabilities = np.array([1 / (i + 1)**alpha for i in range(list_length)])
    probabilities /= np.sum(probabilities)
    
    if type == "close":
        probabilities = probabilities[::-1] # inverse the heavy tailed distribution according to the type of paraphrase location
    
    return np.random.choice(np.arange(list_length), p=probabilities)



def string_to_list(input_string):
    try:
        # Using ast.literal_eval to safely evaluate the string
        result_list = ast.literal_eval(input_string)
        
        # Convert the result to a list
        if isinstance(result_list, tuple):
            result_list = list(result_list)
            
        return result_list
    except (SyntaxError, ValueError) as e:
        # Handle exceptions if the string is not a valid representation of a tuple
        #print(f"Error: {e}")
        return []

def flatten_list(input_list):
    """
    Flatten a list of lists.

    Parameters:
    - input_list: List of lists to be flattened.

    Returns:
    - Flattened list.
    """
    flattened = []
    for element in input_list:
        if isinstance(element, list):
            flattened.extend(flatten_list(element))
        else:
            flattened.append(element)
    return flattened




def aggregate_dataset(model_name_list, config):
    agg_dataset = pd.DataFrame()
    for model_name in model_name_list: 
        data_folder =os.path.join(config['data_folder'], "altered_corpus")
        #loading offline corpus
        data = pd.read_csv(os.path.join(config['data_folder'],"corpus",f"offline_{model_name}_dataset.csv"),sep="\t")
        # initiate the columns
        data['llm_model'] = model_name
        data['paraphrase_model'] = None
        data['altered_text'] = data['text']
        data['nb_paraphrase_max'] = 0
        data['paraphase_type'] = ['absent']*len(data)
        for paraphrase_model in ["gpt-3.5-turbo","ibm-qcpg-sentences"]: #text-davinci-002
            for nb_paraphrase in [1]:
                for p_paraphrase in [0.1]:
                    for q_paraphrase in [0.9,0.5,0.1]:
                        for alpha_paraphrase in [0.5,0.7]:
                            alpha_str = str(alpha_paraphrase).replace(".","-")
                            saving_folder = os.path.join(data_folder,f"online_{model_name}_paraphrase-{paraphrase_model}_n{nb_paraphrase}_p{p_paraphrase}_q{q_paraphrase}_alpha{str(alpha_str)}")
                            try : 
                                new_data = pd.read_csv(os.path.join(saving_folder,"altered_dataset.csv"),sep="\t")
                                new_data['paraphrase_model'] = paraphrase_model
                                new_data['nb_paraphrase_max'] = nb_paraphrase
                                new_data['p_paraphrase'] = p_paraphrase
                                new_data['q_paraphrase'] = q_paraphrase
                                new_data['alpha_paraphrase'] = alpha_paraphrase
                                new_data['llm_model'] = model_name
                                data = pd.concat([data,new_data])
                            except :
                                continue
        data = data.reset_index(drop=True)
        agg_dataset = pd.concat([agg_dataset,data])
    
    agg_dataset = agg_dataset.reset_index(drop=True)
    agg_dataset['clean_paraphase_type'] = agg_dataset['paraphase_type'].apply(lambda x: clean_type(x))
    agg_dataset = agg_dataset[~agg_dataset["altered_text"].str.contains("www|http")]
    agg_dataset['clean_paraphrase_index'] = agg_dataset['index_paraphrase'].apply(lambda x: string_to_list(x))
    agg_dataset['one_paraphrase'] = agg_dataset['clean_paraphrase_index'].apply(lambda x: len(x) == 1)
    agg_dataset['first_paraphrase'] = agg_dataset.apply(lambda x: is_first(x.clean_paraphrase_index) if x.one_paraphrase else None,axis =1)
    agg_dataset['consecutive_paraphrase'] = agg_dataset.apply(lambda x: is_consecustive(x.clean_paraphrase_index) if x.one_paraphrase else None,axis =1)
    
    return agg_dataset


def clean_type(x):
    if x == 'absent' or x =="[]":
        return 0
    elif x == "['close']":
        return 1
    elif x == "['distant']":
        return 2
    else :
        return x
    

def is_first(x):
    if len(x) == 0:
        return None
    else :
        if x[0] == 1:
            return 1
        else :
            return 0

def is_consecustive(x):
    if len(x) == 0:
        return None
    else :
        if x[1] - x[0] == 1:
            return 1
        else :
            return 0

