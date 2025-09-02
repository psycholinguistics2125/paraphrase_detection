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




def aggregate_dataset(model_name_list, config, paraphrase_model_list = ["gpt-3.5-turbo","ibm-qcpg-sentences"]):
    agg_dataset = pd.DataFrame()
    for model_name in model_name_list: 
        data_folder =os.path.join(config['data_folder'], "altered_corpus")
        #loading offline corpus
        data = pd.read_csv(os.path.join(config['data_folder'],"corpus",f"offline_{model_name}_dataset.csv"),sep="\t")
        for str_temp in ['_temperature_2-3',"_temperature_4-5"]:
           
            new_data = pd.read_csv(os.path.join(config['data_folder'],"corpus",f"offline_{model_name}{str_temp}_dataset.csv"),sep="\t")
            data = pd.concat([data,new_data])
      
        # initiate the columns
        data['llm_model'] = model_name
        data['paraphrase_model'] = None
        data['altered_text'] = data['text']
        data['nb_paraphrase_max'] = 0
        data['paraphase_type'] = ['absent']*len(data)
        for str_temp in ['','_temperature_2-3',"_temperature_4-5"]:
            for paraphrase_model in paraphrase_model_list : #text-davinci-002
                for nb_paraphrase in [1]:
                    for p_paraphrase in [0.1]:
                        for q_paraphrase in [0.9,0.5,0.1]:
                            for alpha_paraphrase in [0.5,0.7]:
                                alpha_str = str(alpha_paraphrase).replace(".","-")
                                saving_folder = os.path.join(data_folder,f"online_{model_name}_paraphrase-{paraphrase_model}_n{nb_paraphrase}_p{p_paraphrase}_q{q_paraphrase}_alpha{str(alpha_str)}{str_temp}")
                                try : 
                                    """if model_name == "phi" :
                                        try :
                                            new_data = rewrite_data_from_phi(os.path.join(saving_folder,"altered_dataset.csv"))
                                        except: 
                                            new_data = pd.read_csv(os.path.join(saving_folder,"altered_dataset.csv"),sep="\t")"""
                                    new_data = pd.read_csv(os.path.join(saving_folder,"altered_dataset.csv"),sep="\t")
                                    new_data['paraphrase_model'] = paraphrase_model
                                    new_data['nb_paraphrase_max'] = nb_paraphrase
                                    new_data['p_paraphrase'] = p_paraphrase
                                    new_data['q_paraphrase'] = q_paraphrase
                                    new_data['alpha_paraphrase'] = alpha_paraphrase
                                    new_data['llm_model'] = model_name
                                    data = pd.concat([data,new_data])
                                    #print(f"{saving_folder} loaded")
                                except :
                                    continue
        data = data.reset_index(drop=True)
        agg_dataset = pd.concat([agg_dataset,data])
    
    agg_dataset = agg_dataset.reset_index(drop=True)
    agg_dataset['clean_paraphase_type'] = agg_dataset['paraphase_type'].apply(lambda x: clean_type(x))
    agg_dataset = agg_dataset[agg_dataset["altered_text"].str.contains("www|http")==False]
    agg_dataset['clean_paraphrase_index'] = agg_dataset['index_paraphrase'].apply(lambda x: string_to_list(x))
    agg_dataset['one_paraphrase'] = agg_dataset['clean_paraphrase_index'].apply(lambda x: len(x) == 2)
    agg_dataset['first_paraphrase'] = agg_dataset.apply(lambda x: is_first(x.clean_paraphrase_index) if x.one_paraphrase else None,axis =1)
    agg_dataset['consecutive_paraphrase'] = agg_dataset.apply(lambda x: is_consecustive(x.clean_paraphrase_index) if x.one_paraphrase else None,axis =1)
    
    return agg_dataset



    
def remove_paraphrase(x):
    sentences = x.sentences
    index_para = x.clean_paraphrase_index
    # on ne traite que els cas ou il y a effectivement eu une paraphrase
    if len(index_para)>0:
        if len(index_para)==2 :
            try : 
                new_sentences = [sentences[i] for i in range(len(sentences)) if i != index_para[1]]
                new_text = " ".join(new_sentences)
            except :
                print(sentences)
                print(index_para)
        else : 
            new_text = None

        return new_text
    
    else : 
        return None

def build_altered_dataset_without_paraphrase(data):
    
    data['sentences'] = data.apply(lambda x: split_into_sentences(x.altered_text), axis=1)
    
    old_para = data[['prompt', 'generated_text', 'model_name', 'temperature', 'num_beams',
       'text', 'llm_model', 'paraphrase_model','sentences', "consecutive_paraphrase","first_paraphrase","one_paraphrase",
       'nb_paraphrase_max', 'paraphase_type','p_paraphrase', 'q_paraphrase', 'alpha_paraphrase',"clean_paraphrase_index"]]
    
    old_para['altered_text'] = old_para.apply(lambda x: remove_paraphrase(x), axis=1)
    #old_para["clean_paraphrase_index"] = 0
    old_para = old_para[old_para.altered_text.notnull()]

    #old_para['one_paraphrase'] = 0
    #old_para['first_paraphrase'] = 0
    #old_para['consecutive_paraphrase']=0
    return old_para.reset_index(drop=True)
    

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

def clean_type(x):
    if x == 'absent' or x =="[]":
        return 0
    elif x == "['close']":
        return 1
    elif x == "['distant']":
        return 2
    else :
        return x
    

def rewrite_data_from_phi(path) :
    f = open(path,"r").readlines()
    prompt = []
    generated_text = []
    model_name = []
    temperature = []
    num_beams = []
    index_paraphrase = []
    paraphrase_type = []
    altered_text = []
    for elt in f : 
        if len(elt.split("\t"))==8 :
            prompt.append(elt.split("\t")[0])
            generated_text.append(elt.split("\t")[1])
            model_name.append(elt.split("\t")[2])
            temperature.append(elt.split("\t")[3])
            num_beams.append(elt.split("\t")[4])
            index_paraphrase.append(elt.split("\t")[5])
            paraphrase_type.append(elt.split("\t")[6])
            altered_text.append(elt.split("\t")[7].replace("endoftextIllustration",""))

    d = pd.DataFrame({"prompt":prompt,"generated_text":generated_text,"model_name":model_name,"temperature":temperature,"num_beams":num_beams,"index_paraphrase":index_paraphrase,"paraphrase_type":paraphrase_type,"altered_text":altered_text}).drop(0)

    d.to_csv(path, sep = '\t', index = False)
    return d


def has_address(text):
    pattern = r'\b\d{1,6}\s+(?:\w+\s*){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Square|Sq|Trail|Trl|Circle|Cir)(?:\s+\w+)?(?:,\s+\w+)?(?:,\s+\d{5}(?:-\d{4})?)?\b'
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        return 1
    else : 
        return 0 

def has_mail(text) :
    pattern = pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return 1
    else : 
        return 0 

def has_phone(text):
    pattern = r'\b(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|(\d)\1{9})\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return 1
    else : 
        return 0 



def detect_list(input_string):
    pattern = r"(\.\s?([0-9\-–*]{1})\.?\s?){1,2}"
    match = re.search(pattern, input_string, flags=re.IGNORECASE)
    if match:
        #print(match[0])
        return True
    else:
        return False


def paraphrase_is_short(clean_paraphrase_index,sentences) :
    if type(clean_paraphrase_index)== list  :
        if len(clean_paraphrase_index)==0:
            return 0 
        else :
            i_para = clean_paraphrase_index[0]
            try :
                para = sentences[i_para-1]
            except : 
                return 0
            if len(para.split(" "))<5:
                return 1
            else :
                return 0
    else :
        return 0

def paraphrase_happened_at_the_end(clean_paraphrase_index, num_sentences) :
    if clean_paraphrase_index != None :
        if len(clean_paraphrase_index)==2 :
            if clean_paraphrase_index[0] >= num_sentences-2 :
                return 1
            else :
                return 0
        else : 
            return 0
    else :
        return 0

def paraphrase_happened_at_the_end(clean_paraphrase_index, num_sentences) :
    if clean_paraphrase_index != None :
        if len(clean_paraphrase_index)==2 :
            if clean_paraphrase_index[0] >= num_sentences-2 :
                return 1
            else :
                return 0
        else : 
            return 0
    else :
        return 0

def clean_generated_data(data) :
    print(f"Data contains at the begining : {len(data)}")
    # filter by number of sentences
    new_data = data[data.num_sentences>5]
    new_data = new_data[new_data.num_sentences<20]
    print(f"Filtering data with more than 20 sentences and less than 5 sentences. {len(new_data)} samples left")

    # filter by content
    new_data['has_phone'] = new_data.apply(lambda x: has_phone(x.altered_text), axis=1)
    new_data = new_data[new_data.has_phone==0]
    print(f"Filtering data with phone numbers. {len(new_data)} samples left")

    new_data['has_address'] = new_data.apply(lambda x: has_address(x.altered_text), axis=1)
    new_data = new_data[new_data.has_address==0]
    print(f"Filtering data with addresses. {len(new_data)} samples left")

    new_data['has_mail'] = new_data.apply(lambda x: has_mail(x.altered_text), axis=1)
    new_data = new_data[new_data.has_mail==0]
    print(f"Filtering data with mail addresses. {len(new_data)} samples left")

    new_data['has_list'] = new_data.apply(lambda x: detect_list(x.altered_text), axis=1)
    new_data = new_data[new_data.has_list==0]
    print(f"Filtering data with list. {len(new_data)} samples left")

    # filter on paraphrasing condition if it happeened at the end or on to short paraphrase
    new_data['paraphrase_is_short'] = new_data.apply(lambda x: paraphrase_is_short(x.clean_paraphrase_index,x.sentences), axis=1)
    new_data = new_data[new_data.paraphrase_is_short==0]
    print(f"Filtering data with short paraphrase. {len(new_data)} samples left")

    new_data['paraphrase_at_end'] = new_data.apply(lambda x: paraphrase_happened_at_the_end(x.clean_paraphrase_index, x.num_sentences), axis=1)
    new_data = new_data[new_data.paraphrase_at_end==0]
    print(f"Filtering data with paraphrase at the end. {len(new_data)} samples left")

    # adding new columns for analysis
    new_data['has_paraphrase'] = new_data['clean_paraphase_type'].apply(lambda x: 1 if x != 0 else 0)

    prompt_list =  ["Most people start the day by", 
        "Today I am feeling", 
        "The thing I like most in the world is",
        "When I was a little kid", 
        "I had a terrifying dream last night in which",
        "I worry a lot about"]

    new_data = new_data[new_data.prompt.isin(prompt_list)]
    new_data['prompt_cat'] = new_data['prompt'].astype('category').cat.codes

    try :
        # labele paraphrase that have huge lexical repetition
        jaccard_median = new_data[new_data.lemma_para_vs_others_jaccard_sentences_similarity>0]['lemma_para_vs_others_jaccard_sentences_similarity'].median()
        new_data['has_lexical_repetition'] = new_data['lemma_para_vs_others_jaccard_sentences_similarity'].apply(lambda x: 1 if  x > jaccard_median else  0)
    except :
        pass

    return new_data.reset_index()


def encode_temperature(temperature):
    if temperature <2 :
        return 0
    elif temperature <4 :
        return 1
    else : 
        return 2
    

def encode_temperature_str(temperature):
    if temperature <2 :
        return "low_temperature"
    elif temperature <4 :
        return "medium_temperature"
    else : 
        return "high_temperature"

def encode_paraphrase(paraphrase):
    if paraphrase ==0 :
        return "no_paraphrase"
    else : 
        return "paraphrase"
    

def load_and_clean_data(config):
    """
    """
    data = pd.read_pickle(config['agg_file'])
    data = clean_generated_data(data)
    data['temperature_enc'] = data['temperature'].apply(encode_temperature_str)
    data['paraphrase_enc'] = data['has_paraphrase'].apply(encode_paraphrase)
    data['combined_label'] = data.apply(lambda x: f"{x['temperature_enc']}_with_{x['paraphrase_enc']}", axis = 1)
    data["cat_label"] = data['combined_label'].astype('category').cat.codes
    data['random'] = np.random.rand(data.shape[0])

    return data