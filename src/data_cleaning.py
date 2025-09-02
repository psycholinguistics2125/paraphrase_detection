from deepmultilingualpunctuation import PunctuationModel
import spacy
import contextualSpellCheck
import string

from spello.model import SpellCorrectionModel 
import tqdm

import language_tool_python

from spellchecker import SpellChecker

"""nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)"""


def correct_text(text):
    spell = SpellChecker()
    word_list = text.split(" ")
    corrected = []
    for word in word_list :
        correction = str(spell.correction(str(word)))
        #print(correction)
        corrected.append(correction)


def clean_generale_population_text(text_list:list):
    # load the ressources
    model = PunctuationModel()
    tool = language_tool_python.LanguageToolPublicAPI('en-US')


    #spell correction
    #sp = SpellCorrectionModel(language='en')
    #sp.load("../models/spello_en_large.pkl")


    
    cleaned_text_list = []
    for i,text in enumerate(text_list):
        try :# remove punctuation
            text = str(text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            # spell correction
            #text = sp.spell_correct(text)['spell_corrected_text']
            #text = correct_text(text)
            #doc = nlp(text)
            #text = doc._.outcome_spellCheck
            text  = tool.correct(text)
            # Restore punctuation
            text = model.restore_punctuation(text)
        except Exception as e:
            text = str(text)
            print(f'error in cleaning text {i} because of {e} ')
            print(text)
        if i%100 == 0:
            print(i)
        cleaned_text_list.append(text)
    return cleaned_text_list


def add_average_data(data):
    all_cosine_sentences_similarity_mean = data.filter(regex = '(sentence_sim|glove|w2v|fast_text)(.*)(all_cosine_sentences_similarity_mean)').columns.tolist()
    all_cosine_sentences_similarity_std = data.filter(regex = '(sentence_sim|glove|w2v|fast_text)(.*)(all_cosine_sentences_similarity_std)').columns.tolist()
    all_wmd_sentences_similarity_mean = data.filter(regex = '(sentence_sim|glove|w2v|fast_text)(.*)(all_wmd_sentences_similarity_mean)').columns.tolist()
    all_wmd_sentences_similarity_std = data.filter(regex = '(sentence_sim|glove|w2v|fast_text)(.*)(all_wmd_sentences_similarity_std)').columns.tolist()

    consecutive_cosine_sentences_similarity_mean = data.filter(regex = '(sentence_sim|glove|w2v|fast_text)(.*)(consecutive_cosine_sentences_similarity_mean)').columns.tolist()
    consecutive_cosine_sentences_similarity_std = data.filter(regex = '(sentence_sim|glove|w2v|fast_text)(.*)(consecutive_cosine_sentences_similarity_std)').columns.tolist()
    consecutive_wmd_sentences_similarity_mean = data.filter(regex = '(sentence_sim|glove|w2v|fast_text)(.*)(consecutive_wmd_sentences_similarity_mean)').columns.tolist()

    data['all_cosine_sentences_similarity_mean'] = data[all_cosine_sentences_similarity_mean].mean(axis = 1)
    data['all_cosine_sentences_similarity_mean_std'] = data[all_cosine_sentences_similarity_mean].std(axis = 1)
    data['all_cosine_ci'] = 1.96 * (data['all_cosine_sentences_similarity_mean_std'] / (len(all_cosine_sentences_similarity_mean) ** 0.5))


    data['all_cosine_sentences_similarity_std'] = data[all_cosine_sentences_similarity_std].mean(axis = 1)
    data['all_cosine_sentences_similarity_std_std'] = data[all_cosine_sentences_similarity_std].std(axis = 1)
    data['all_cosine_ci_std'] = 1.96 * (data['all_cosine_sentences_similarity_std_std'] / (len(all_cosine_sentences_similarity_std) ** 0.5))


    data['consecutive_cosine_sentences_similarity_mean'] = data[consecutive_cosine_sentences_similarity_mean].mean(axis = 1)
    data['consecutive_cosine_sentences_similarity_mean_std'] = data[consecutive_cosine_sentences_similarity_mean].std(axis = 1)
    data['consecutive_cosine_ci'] = 1.96 * (data['consecutive_cosine_sentences_similarity_mean_std'] / (len(consecutive_cosine_sentences_similarity_mean) ** 0.5))

    data['consecutive_cosine_sentences_similarity_std'] = data[consecutive_cosine_sentences_similarity_std].mean(axis = 1)
    data['consecutive_cosine_sentences_similarity_std_std'] = data[consecutive_cosine_sentences_similarity_std].std(axis = 1)
    data['consecutive_cosine_ci_std'] = 1.96 * (data['consecutive_cosine_sentences_similarity_std_std'] / (len(consecutive_cosine_sentences_similarity_std) ** 0.5))

    data["consecutive_wmd_sentences_similarity_mean"] = data[consecutive_wmd_sentences_similarity_mean].mean(axis = 1)
    data["consecutive_wmd_sentences_similarity_mean_std"] = data[consecutive_wmd_sentences_similarity_mean].std(axis = 1)
    data["consecutive_wmd_ci"] = 1.96 * (data["consecutive_wmd_sentences_similarity_mean_std"] / (len(consecutive_wmd_sentences_similarity_mean) ** 0.5))
    

    data['all_wmd_sentences_similarity_mean'] = data[all_wmd_sentences_similarity_mean].mean(axis = 1)
    data['all_wmd_sentences_similarity_mean_std'] = data[all_wmd_sentences_similarity_mean].std(axis = 1)
    data['all_wmd_ci'] = 1.96 * (data['all_wmd_sentences_similarity_mean_std'] / (len(all_wmd_sentences_similarity_mean) ** 0.5))


    data['all_wmd_sentences_similarity_std'] = data[all_wmd_sentences_similarity_std].mean(axis = 1)
    data['all_wmd_sentences_similarity_std_std'] = data[all_wmd_sentences_similarity_std].std(axis = 1)
    data['all_wmd_ci_std'] = 1.96 * (data['all_wmd_sentences_similarity_std_std'] / (len(all_wmd_sentences_similarity_std) ** 0.5))



    Structural_density = [
                        'cluster_density_score_HDBSCAN', 
                        'cluster_reverse_silhouette_score_HDBSCAN',
                        'cluster_density_score_MeanShift', 
                        'cluster_reverse_silhouette_score_MeanShift',
                        'reduction_score_PCA_explained_variance', 
                        'reduction_score_PCA_prop_of_components',]

    Semantic_contribution = ['regression_coef_density_score_Lasso',
    'regression_error_score_Lasso',
    'reduction_score_Lasso',
    ]

    data['narrative_speed_ci'] = 0

    data['structural_density'] = data[Structural_density].mean(axis = 1)
    data['structural_density_std'] = data[Structural_density].std(axis = 1)
    data['structural_density_ci'] = 1.96 * (data['structural_density_std'] / (len(Structural_density) ** 0.5))

    data['semantic_contribution'] = data[Semantic_contribution].mean(axis = 1)
    data['semantic_contribution_std'] = data[Semantic_contribution].std(axis = 1)
    data['semantic_contribution_ci'] = 1.96 * (data['semantic_contribution_std'] / (len(Semantic_contribution) ** 0.5))

    silhouette_based = ['cluster_reverse_silhouette_score_MeanShift', 'cluster_reverse_silhouette_score_HDBSCAN']
    cluster_density_based = ['cluster_density_score_MeanShift', "cluster_density_score_HDBSCAN" ]
    pca_based = ['reduction_score_PCA_explained_variance', 'reduction_score_PCA_prop_of_components']
    lasso_based = ['regression_coef_density_score_Lasso', 'regression_error_score_Lasso', 'reduction_score_Lasso']

    data['silhouette_based'] = data[silhouette_based].mean(axis = 1)
    data['silhouette_based_std'] = data[silhouette_based].std(axis = 1)
    data['silhouette_based_ci'] = 1.96 * (data['silhouette_based_std'] / (len(silhouette_based) ** 0.5))

    data['cluster_density_based'] = data[cluster_density_based].mean(axis = 1)
    data['cluster_density_based_std'] = data[cluster_density_based].std(axis = 1)
    data['cluster_density_based_ci'] = 1.96 * (data['cluster_density_based_std'] / (len(cluster_density_based) ** 0.5))

    data['pca_based'] = data[pca_based].mean(axis = 1)
    data['pca_based_std'] = data[pca_based].std(axis = 1)
    data['pca_based_ci'] = 1.96 * (data['pca_based_std'] / (len(pca_based) ** 0.5))

    data['lasso_based'] = data[lasso_based].mean(axis = 1)
    data['lasso_based_std'] = data[lasso_based].std(axis = 1)
    data['lasso_based_ci'] = 1.96 * (data['lasso_based_std'] / (len(lasso_based) ** 0.5))


    narrative_progression_features = [
    'narrative_speed_score'
]

    # Global Topic Structure
    global_topic_structure_features = [
        'cluster_density_score_MeanShift',
        'cluster_reverse_silhouette_score_MeanShift'
    ]

    # Local Topic Structure
    local_topic_structure_features = [
        'cluster_density_score_HDBSCAN',
        'cluster_reverse_silhouette_score_HDBSCAN'
    ]

    # Semantic Complexity
    semantic_complexity_features = [
        'reduction_score_PCA_explained_variance',
        'regression_error_score_Lasso'
    ]

    data['global_topic_structure'] = data[global_topic_structure_features].mean(axis = 1)
    data["global_topic_structure_ci"] = 1.96 * (data[global_topic_structure_features].std(axis = 1) / (len(global_topic_structure_features) ** 0.5))

    data['local_topic_structure'] = data[local_topic_structure_features].mean(axis = 1)
    data["local_topic_structure_ci"] = 1.96 * (data[local_topic_structure_features].std(axis = 1) / (len(local_topic_structure_features) ** 0.5))

    data['semantic_complexity'] = data[semantic_complexity_features].mean(axis = 1)
    data["semantic_complexity_ci"] = 1.96 * (data[semantic_complexity_features].std(axis = 1) / (len(semantic_complexity_features) ** 0.5))

    return data