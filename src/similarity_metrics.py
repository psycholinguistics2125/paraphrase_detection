

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd

import os
import spacy
from src.utils import split_into_sentences
import textdistance

from src.utils import  aggregate_dataset

def compute_wmdistance(token_list_1, token_list_2, model):

    try:
        distance = model.wmdistance(token_list_1, token_list_2)
        return distance
    except Exception as e:
        print(f"Error calculating WMD: {e}")
        return 0

def load_model(config):
    model_folder = config['similarity_param']['model_folder']
    if config['similarity_param']['model_name'] == 'w2v':
        return  KeyedVectors.load(os.path.join(model_folder,'wv-kv-300.kv'))
    elif config['similarity_param']['model_name'] == 'glove':
        return  KeyedVectors.load(os.path.join(model_folder,'glove-kv-300.kv'))
    elif config['similarity_param']['model_name'] == 'fast_text':
        return  KeyedVectors.load(os.path.join(model_folder,'fast_text.kv'))
    elif config['similarity_param']['model_name']== 'sentence_sim':
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    elif config['similarity_param']['model_name'] == 'lemma':
        return spacy.load("en_core_web_sm")
    else:
        raise ValueError("model not supported")


class Encoder():
    def __init__(self,config):
        self.model = load_model(config)
        self.config = config
            
    def encode(self,s1) -> float:
        if self.config['similarity_param']['model_name'] in ["glove","w2v","fast_text"]:
            token_list =  word_tokenize(s1.lower())
            # Filter out words that are not in the Word2Vec model's vocabulary
            words_in_vocab = [word for word in token_list if word in self.model.key_to_index]

            #compute the embedings
            word_embeddings = [self.model[word] for word in words_in_vocab]

            # Calculate the average vector
            avg_vector = np.mean(word_embeddings, axis=0)

            return avg_vector
        elif self.config['similarity_param']['model_name'] == 'lemma':
            doc = self.model(s1)
            lemmatized_tokens = [token.lemma_.lower() for token in doc]
            return lemmatized_tokens

        else:
            return self.model.encode(s1)
        
    def preprocess(self,s1):
        
        token_list =  word_tokenize(s1.lower())
        # Filter out words that are not in the Word2Vec model's vocabulary
        words_in_vocab = [word for word in token_list if word in self.model.key_to_index]

        return words_in_vocab
        

class SimCalc():
    def __init__(self):
        pass

    def __call__(self,v1,v2) -> float:
        return(np.sum(np.multiply(v1,v2))/(norm(v1)*norm(v2)))




def compute_consecutive_sentences_similarity(text, encoder,sim_type='cosine'):
   
    sentences = split_into_sentences(text)
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    sim = SimCalc()
    if len(sentences) == 0:
        return 0
    if len(sentences) == 1:
        return 1
    
    if sim_type == 'cosine':
        result = []
        sentences_vectors = [encoder.encode(sentence) for sentence in sentences]
        for i in range(len(sentences_vectors)-1):
            result.append(sim(sentences_vectors[i], sentences_vectors[i+1]))
    
        return result
    elif sim_type == 'wmd':
        result = []
        for i in range(len(sentences)-1):
            wms = max(0,1-encoder.model.wmdistance(encoder.preprocess(sentences[i]), encoder.preprocess(sentences[i+1]))/2)
            result.append(wms)
        
        return result
    
    elif sim_type =="jaccard":
        result = []
        for i in range(len(sentences)-1):
            s1 = encoder.encode(sentences[i])
            s2 = encoder.encode(sentences[i+1])
            result.append(textdistance.jaccard.similarity(s1,s2))
        
        return result

    else :
        return []

def compute_all_sentences_similarity(text, encoder,sim_type='cosine'):
   
    sentences = split_into_sentences(text)
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    sim = SimCalc()
    if len(sentences) == 0:
        return 0
    if len(sentences) == 1:
        return 1
    if sim_type == 'cosine':
        result = []
        sentences_vectors = [encoder.encode(sentence) for sentence in sentences]
        for i in range(len(sentences_vectors)):
            for j in range(i+1, len(sentences_vectors)):
                result.append(sim(sentences_vectors[i], sentences_vectors[j]))
    
        return result
    
    elif sim_type == 'wmd':
        result = []
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                wms = max(0,1-encoder.model.wmdistance(encoder.preprocess(sentences[i]), encoder.preprocess(sentences[j]))/2)
                result.append(wms)
    
        return result
    
    elif sim_type =="jaccard":
        result = []
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                s1 = encoder.encode(sentences[i])
                s2 = encoder.encode(sentences[j])
                result.append(textdistance.jaccard.similarity(s1,s2))
        
        return result
    
    else :
        return []

def compute_paraphrase_similarity_vs_rest(text, para_index,encoder,sim_type='cosine'):
    sentences = split_into_sentences(text)
    sentences = [s.strip() for s in sentences]
    sim = SimCalc()
    if len(para_index) == 0:
        return 0
    else :
        if sim_type == 'cosine':
            try:
                para_sentences = [sentences[para_index[0]], sentences[para_index[1]] ]
                para_sentences_vectors = [encoder.encode(sentence) for sentence in para_sentences]
                para_sim = sim(para_sentences_vectors[0], para_sentences_vectors[1])
                others_sim = []
                others_sentences_vectors = [encoder.encode(sentence) for sentence in sentences if sentence not in [para_sentences[-1]]]
                for i in range(len(others_sentences_vectors)):
                    for j in range(i+1, len(others_sentences_vectors)):
                        others_sim.append(sim(others_sentences_vectors[i], others_sentences_vectors[j]))
            
                return para_sim - np.mean(others_sim)
            except:
                return 0
        
        elif sim_type == 'wmd':
            try:
                para_sentences = [sentences[para_index[0]], sentences[para_index[1]] ]
                para_wms = max(0,1-encoder.model.wmdistance(encoder.preprocess(sentences[0]), encoder.preprocess(sentences[1]))/2)
               
                others_sim = []
                for i in range(len(sentences)):
                    for j in range(i+1, len(sentences)):
                        if i not in para_index and j not in para_index:
                            wms = max(0,1-encoder.model.wmdistance(encoder.preprocess(sentences[i]), encoder.preprocess(sentences[j]))/2)
                            others_sim.append(wms)
            
                return para_wms - np.mean(others_sim)
            except:
                return 0
        elif sim_type =="jaccard":
            try:
                para_sentences = [sentences[para_index[0]], sentences[para_index[1]] ]
                para_sim = textdistance.jaccard.similarity(encoder.encode(sentences[0]), encoder.encode(sentences[1]))
                others_sim = []
                for i in range(len(sentences)):
                    for j in range(i+1, len(sentences)):
                        if i not in para_index and j not in para_index:
                            s1 = encoder.encode(sentences[i])
                            s2 = encoder.encode(sentences[j])
                            others_sim.append(textdistance.jaccard.similarity(s1,s2))
            
                return para_sim - np.mean(others_sim)
            except:
                return 0
        else :
            return 0

        




def compute_similarity_features(data, config, col_text="text", para = True):
   
    for model_name in ['w2v',"fast_text","glove","sentence_sim","lemma"]:
        config['similarity_param']['model_name'] = model_name
        # load the proper encoder 
        encoder = Encoder(config)
        if model_name == 'lemma':
            if para:
                data[model_name + '_para_vs_others_jaccard_sentences_similarity'] = data.apply(lambda x: compute_paraphrase_similarity_vs_rest(x[col_text], x.clean_paraphrase_index,encoder,sim_type="jaccard"),axis =1)
            data[model_name + '_consecutive_jaccard_sentences_similarity'] = data[col_text].apply(lambda x: compute_consecutive_sentences_similarity(x, encoder,sim_type='jaccard'))
            data[model_name + '_all_jaccard_sentences_similarity'] = data[col_text].apply(lambda x: compute_all_sentences_similarity(x, encoder,sim_type='jaccard'))

            for sim_type in ['consecutive_jaccard_sentences_similarity', 'all_jaccard_sentences_similarity']:
                data[model_name + '_' + sim_type + '_mean'] = data[model_name + '_' + sim_type].apply(lambda x: np.mean(x))
                data[model_name + '_' + sim_type + '_median'] = data[model_name + '_' + sim_type].apply(lambda x: np.median(x))
                data[model_name + '_' + sim_type + '_std'] = data[model_name + '_' + sim_type].apply(lambda x: np.std(x))
                data[model_name + '_' + sim_type + '_min'] = data[model_name + '_' + sim_type].apply(lambda x: np.min(x))
                data[model_name + '_' + sim_type + '_max'] = data[model_name + '_' + sim_type].apply(lambda x: np.max(x))
        else :
            if para :
                data[model_name + '_para_vs_others_cosine_sentences_similarity'] = data.apply(lambda x: compute_paraphrase_similarity_vs_rest(x[col_text], x.clean_paraphrase_index,encoder),axis =1)
            data[model_name + '_consecutive_cosine_sentences_similarity'] = data[col_text].apply(lambda x: compute_consecutive_sentences_similarity(x, encoder))
            data[model_name + '_all_cosine_sentences_similarity'] = data[col_text].apply(lambda x: compute_all_sentences_similarity(x, encoder))
    
            if model_name in ['w2v',"fast_text","glove"] :
                if para:
                    data[model_name + '_para_vs_others_wmd_sentences_similarity'] = data.apply(lambda x: compute_paraphrase_similarity_vs_rest(x[col_text], x.clean_paraphrase_index,encoder,sim_type="wmd"),axis =1)
                data[model_name + '_consecutive_wmd_sentences_similarity'] = data[col_text].apply(lambda x: compute_consecutive_sentences_similarity(x, encoder,sim_type='wmd'))
                data[model_name + '_all_wmd_sentences_similarity'] = data[col_text].apply(lambda x: compute_all_sentences_similarity(x, encoder,sim_type='wmd'))

        
            # add features for each cols mean median and std
            for sim_type in ['consecutive_cosine_sentences_similarity', 'all_cosine_sentences_similarity',"all_wmd_sentences_similarity","consecutive_wmd_sentences_similarity"]:
                if "wmd" in sim_type:
                    if model_name == 'sentence_sim':
                        continue
                data[model_name + '_' + sim_type + '_mean'] = data[model_name + '_' + sim_type].apply(lambda x: np.mean(x))
                data[model_name + '_' + sim_type + '_median'] = data[model_name + '_' + sim_type].apply(lambda x: np.median(x))
                data[model_name + '_' + sim_type + '_std'] = data[model_name + '_' + sim_type].apply(lambda x: np.std(x))
                data[model_name + '_' + sim_type + '_min'] = data[model_name + '_' + sim_type].apply(lambda x: np.min(x))
                data[model_name + '_' + sim_type + '_max'] = data[model_name + '_' + sim_type].apply(lambda x: np.max(x))
        

    return data



def main_compute_and_save_data(config, save=True, sample=True):
    model_name_list = ['gpt2',"pythia"]

    data = aggregate_dataset(model_name_list, config)
    roc_data = pd.read_csv("/home/robin/Code_repo/psycholinguistic2125/paraphrase_detection/data/corpus/ROCStories_spring2016.csv",sep=",")
    if sample:
        data = data.sample(100)
        roc_data = roc_data.sample(100)
    else : 
        roc_data = roc_data.sample(3000)
    
    roc_data['text'] = roc_data['sentence1'] + " " + roc_data['sentence2'] + " " + roc_data['sentence3'] + " " + roc_data['sentence4'] + " " + roc_data['sentence5']
    roc_data['clean_paraphase_type'] = 0
    roc_data['clean_paraphrase_index'] = None

    roc_data = compute_similarity_features(roc_data, config, col_text="text", para=False)
    data = compute_similarity_features(data, config, col_text='altered_text', para=True)

    data['source'] = data.llm_model
    roc_data["source"] = "roc_story"

    data = pd.concat([data,roc_data])

    data["clean_text"] = data.apply(lambda x: x.altered_text if type(x.text) != str else x.text, axis=1)
    data['sentences'] = data.apply(lambda x: split_into_sentences(x.clean_text), axis=1)
    data['num_sentences'] = data.apply(lambda x: len(x.sentences), axis=1)
    data = data[data.num_sentences > 1]
    data = data.reset_index(drop=True)
    if save : 
        data.to_pickle(config['agg_file'])
    return data



    


""" def compute_similarity_features_roc(data, config, col_text="text"):
    #data['clean_paraphrase_index'] = data['index_paraphrase'].apply(lambda x: string_to_list(x))
    for model_name in ['w2v',"fast_text","glove","sentence_sim"]:
        config['similarity_param']['model_name'] = model_name
        # load the proper encoder 
        encoder = Encoder(config)
        
        data[model_name + '_consecutive_cosine_sentences_similarity'] = data[col_text].apply(lambda x: compute_consecutive_sentences_similarity(x, encoder))
        data[model_name + '_all_cosine_sentences_similarity'] = data[col_text].apply(lambda x: compute_all_sentences_similarity(x, encoder))
    
        if model_name != 'sentence_sim':
            data[model_name + '_consecutive_wmd_sentences_similarity'] = data[col_text].apply(lambda x: compute_consecutive_sentences_similarity(x, encoder,sim_type='wmd'))
            data[model_name + '_all_wmd_sentences_similarity'] = data[col_text].apply(lambda x: compute_all_sentences_similarity(x, encoder,sim_type='wmd'))

        
        # add features for each cols mean median and std
        for sim_type in ['consecutive_cosine_sentences_similarity', 'all_cosine_sentences_similarity',"all_wmd_sentences_similarity","consecutive_wmd_sentences_similarity"]:
            if "wmd" in sim_type:
                if model_name == 'sentence_sim':
                    continue
            data[model_name + '_' + sim_type + '_mean'] = data[model_name + '_' + sim_type].apply(lambda x: np.mean(x))
            data[model_name + '_' + sim_type + '_median'] = data[model_name + '_' + sim_type].apply(lambda x: np.median(x))
            data[model_name + '_' + sim_type + '_std'] = data[model_name + '_' + sim_type].apply(lambda x: np.std(x))
            data[model_name + '_' + sim_type + '_min'] = data[model_name + '_' + sim_type].apply(lambda x: np.min(x))
            data[model_name + '_' + sim_type + '_max'] = data[model_name + '_' + sim_type].apply(lambda x: np.max(x))
        

    return data"""