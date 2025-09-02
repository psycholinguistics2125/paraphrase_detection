
# Standard imports
import os
import sys
import random

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning tools and models
from sklearn.model_selection import (cross_val_score, StratifiedKFold, StratifiedGroupKFold,
                                     train_test_split, KFold, GroupShuffleSplit)
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report,
                             f1_score, )
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

# SHAP for model interpretation
import shap


#proportion score matching
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, fontsize = 25, num_to_label = None):
    """
    Plot the confusion matrix with options for normalization and customization of the colormap.

    Parameters:
    cm (array-like): Confusion matrix to be plotted.
    classes (list): List of class labels to be used in the x and y axes of the matrix.
    normalize (bool): Whether to normalize the confusion matrix.
    title (str): Title of the confusion matrix plot.
    cmap (matplotlib.colors.Colormap): Colormap to be used for plotting.

    Returns:
    None
    """
    sns.set(style="whitegrid", color_codes=True)
    plt.figure(figsize=(10, 8))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"
    
    classes_name = [num_to_label[i] for i in range(len(num_to_label))]
    heatmap = sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=True, xticklabels=classes_name, yticklabels=classes_name,
                          annot_kws={"size": 30})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=fontsize, rotation=15, ha='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=fontsize)
    plt.xlabel('Predicted labels',fontsize=fontsize)
    plt.ylabel('True labels',fontsize=fontsize)
    plt.title(title,)
    plt.show()

def plot_importances(result, feature_names):
    """
    Plot permutation importances of features.

    Parameters:
    result (object): Result object from permutation importance computation.
    feature_names (list): Names of the features corresponding to the importances.

    Returns:
    None
    """
    sorted_importances_idx = result.importances_mean.argsort()[-15:]
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=feature_names[sorted_importances_idx],
    )

    ax = importances.plot.box(vert=False, whis=10,figsize=(10, 8))
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    plt.yticks(fontsize =17)
    ax.figure.tight_layout()
    plt.show()

def train_and_evaluate_rf(features_dict, best_parameters_dict, data,y_name = "cat_label", num_to_label = None, plot = True, plot_error_type =True, seed = 42):
    """
    Train and evaluate a RandomForest classifier using provided features and parameters.

    Parameters:
    features_dict (dict): Dictionary of feature lists keyed by feature set name.
    best_parameters_dict (dict): Dictionary of parameter sets keyed by feature set name.
    data (DataFrame): The dataset containing features and target.

    Returns:
    None
    """
    df_results = pd.DataFrame(columns=["features_name","accuracy","f1_score","confusion_matrix"])
    i = 0
    for features_name, features in features_dict.items():
        #features.append('random')  # Assuming purposeful inclusion for experiment control
        parameters = best_parameters_dict[features_name]
        print("##" * 20)
        clf = RandomForestClassifier(**parameters,random_state=seed)
        print(f"Training with {features_name}")
        X = data[features]
        y = data[y_name]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_index, test_index = next(gss.split(data, groups=data['prompt_cat']))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:", clf.score(X_test, y_test))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

        cm = confusion_matrix(y_test, y_pred, labels = list(num_to_label.keys()))
        

        result_dict = {"features_name":features_name,"accuracy":clf.score(X_test, y_test),"f1_score":f1_score(y_test, y_pred, average='weighted'),"confusion_matrix":cm}
        df_results.loc[i] = pd.Series(result_dict)

        result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        
        if plot:
            plot_confusion_matrix(cm, classes= list(num_to_label.keys()), normalize=True, num_to_label= num_to_label)


        if plot : 
            plot_importances(result, X.columns)

        if plot_error_type :
            from_cm_to_error_plot(cm, num_to_label,y_test, plot = True)

        i+=1

    return df_results    



# Function to extract binary labels and predictions for each context
def extract_binary_labels_and_predictions(cm, context, label_encoded_dict):
    if context == "low_temperature":
        paraphrase_idx = label_encoded_dict['low_temperature_with_paraphrase']
        no_paraphrase_idx = label_encoded_dict['low_temperature_with_no_paraphrase']
    elif context == "medium_temperature":
        paraphrase_idx = label_encoded_dict['medium_temperature_with_paraphrase']
        no_paraphrase_idx = label_encoded_dict['medium_temperature_with_no_paraphrase']
    elif context == "high_temperature":
        paraphrase_idx = label_encoded_dict['high_temperature_with_paraphrase']
        no_paraphrase_idx = label_encoded_dict['high_temperature_with_no_paraphrase']

    y_true = []
    y_scores = []
    
    # Process each row in the confusion matrix
    for i in range(cm.shape[0]):
        if i == paraphrase_idx:
            # True paraphrase
            y_true.extend([1] * cm[i, i])  # True positives
            y_true.extend([0] * cm[i, no_paraphrase_idx])  # False positives
            y_scores.extend([1] * (cm[i, i] + cm[i, no_paraphrase_idx]))
        elif i == no_paraphrase_idx:
            # True no paraphrase
            y_true.extend([0] * cm[i, i])  # True negatives
            y_true.extend([1] * cm[i, paraphrase_idx])  # False negatives
            y_scores.extend([0] * (cm[i, i] + cm[i, paraphrase_idx]))
        else:
            # Process off-diagonal elements where paraphrase or no paraphrase are misclassified
            y_true.extend([1] * cm[i, paraphrase_idx])
            y_true.extend([1] * cm[i, no_paraphrase_idx])
            y_scores.extend([0] * cm[i, paraphrase_idx])
            y_scores.extend([0] * cm[i, no_paraphrase_idx])
    
    return y_true, y_scores

def calculate_auc_score(cm,contexts= ["low_temperature", "medium_temperature", "high_temperature"],label_encoded_dict = None):
# Calculate AUC for each context
    auc_scores = {}

    for context in contexts:
        y_true, y_scores = extract_binary_labels_and_predictions(cm, context, label_encoded_dict)
        auc = roc_auc_score(y_true, y_scores)
        auc_scores[context] = auc

    return auc_scores

def train_and_evaluate_xgb(features_dict, data, y_name = "cat_label", num_to_label = None, plot = True,compute_context_auc = False, plot_error_type = True):
    """
    Train and evaluate a RandomForest classifier using provided features and parameters.

    Parameters:
    features_dict (dict): Dictionary of feature lists keyed by feature set name.
    best_parameters_dict (dict): Dictionary of parameter sets keyed by feature set name.
    data (DataFrame): The dataset containing features and target.

    Returns:
    None
    """
    df_results = pd.DataFrame(columns=["features_name","accuracy","f1_score","confusion_matrix","auc_scores"])
    i = 0
    for features_name, features in features_dict.items():
        #features.append('random')  # Assuming purposeful inclusion for experiment control
        #parameters = best_parameters_dict[features_name]
        print("##" * 20)
        clf = XGBClassifier()
        print(f"Training with {features_name}")
        X = data[list(set(features))]
        y = data[y_name]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
        train_index, test_index = next(gss.split(data, groups=data['prompt_cat']))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:", clf.score(X_test, y_test))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
        print(classification_report(y_test, y_pred,target_names= list(num_to_label.values())))

        cm = confusion_matrix(y_test, y_pred, labels = list(num_to_label.keys()))
        result = {"features_name":features_name,"accuracy":clf.score(X_test, y_test),"f1_score":f1_score(y_test, y_pred, average='weighted'),"confusion_matrix":cm}
        if compute_context_auc:
            result["auc_scores"] = calculate_auc_score(cm)
            
        df_results.loc[i] = pd.Series(result)
        
        if plot:
            plot_confusion_matrix(cm, classes= list(num_to_label.keys()), normalize=True, num_to_label= num_to_label)

        result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        if plot : 
            plot_importances(result, X.columns)

        if plot_error_type :
            from_cm_to_error_plot(cm, num_to_label,y_test, plot = True)
        i+=1

    if compute_context_auc:
        r = pd.concat([df_results,df_results['auc_scores'].apply(pd.Series)],axis = 1)
        df_melted = r.melt(id_vars=['features_name'], 
                                    value_vars=['low_temperature', 'medium_temperature', 'high_temperature'], 
                                    var_name='temperature', 
                                    value_name='max_auc')

        # Plot using seaborn
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_melted, hue='features_name', y='max_auc', x='temperature',)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=12)
        plt.xlabel('Temperature', fontsize=14)
        plt.ylabel('Max AUC', fontsize=14)
        plt.title('Max AUC vs Temperature for Different Features', fontsize=16)
        plt.show()

    return df_results


def parse_label(label):
    parts = label.split('_')
    temperature = parts[0]  # 'low', 'medium', 'high'
    paraphrase = 'with_paraphrase' if 'with_paraphrase' in label else 'with_no_paraphrase'
    return temperature, paraphrase


def from_cm_to_error_plot(cm, num_to_label, y_test, plot=True):
    import matplotlib.pyplot as plt
    from collections import Counter

    labels = list(num_to_label.keys())
    # Build a mapping from index to (temperature, paraphrase)
    index_to_label_components = {}
    label_num_to_components = {}
    for idx, label_num in enumerate(labels):
        label_name = num_to_label[label_num]
        temperature, paraphrase = parse_label(label_name)
        index_to_label_components[idx] = (temperature, paraphrase)
        label_num_to_components[label_num] = (temperature, paraphrase)

    # Mapping temperatures to numerical values for comparison
    temperature_order = {'low': 1, 'medium': 2, 'high': 3}

    error_paraphrase_only = 0
    error_temperature_only = 0
    error_fail_dissociate = 0
    total_samples = len(y_test)  # Total number of samples

    # Count true class counts
    true_class_counts = Counter(y_test)

    # Process the confusion matrix to count errors
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            if i == j:
                continue  # Correct classification
            else:
                true_temp_str, true_para = index_to_label_components[i]
                pred_temp_str, pred_para = index_to_label_components[j]
                true_temp = temperature_order[true_temp_str]
                pred_temp = temperature_order[pred_temp_str]
                # Error categories
                if true_temp == pred_temp and true_para != pred_para:
                    error_paraphrase_only += count
                elif true_temp != pred_temp and true_para == pred_para:
                    error_temperature_only += count
                elif ((true_para == 'with_no_paraphrase' and pred_para == 'with_paraphrase' and pred_temp > true_temp) or
                      (true_para == 'with_paraphrase' and pred_para == 'with_no_paraphrase' and true_temp > pred_temp)):
                    error_fail_dissociate += count
                else:
                    continue  # Other errors not counted in the specified categories

    # Compute possible misclassifications for each category

    # Error_paraphrase_only
    possible_misclassifications_paraphrase_only = 0
    for label_num_i in labels:
        true_class_count = true_class_counts.get(label_num_i, 0)
        true_temp, true_para = label_num_to_components[label_num_i]
        # Find other labels with same temp and different paraphrase
        other_paraphrases = [p for p in ['with_paraphrase', 'with_no_paraphrase'] if p != true_para]
        other_label_nums = [label_num_j for label_num_j in labels
                            if label_num_j != label_num_i and
                            label_num_to_components[label_num_j][0] == true_temp and
                            label_num_to_components[label_num_j][1] in other_paraphrases]
        num_possible_misclassifications = len(other_label_nums)
        possible_misclassifications_paraphrase_only += true_class_count * num_possible_misclassifications

    # Error_temperature_only
    possible_misclassifications_temperature_only = 0
    for label_num_i in labels:
        true_class_count = true_class_counts.get(label_num_i, 0)
        true_temp, true_para = label_num_to_components[label_num_i]
        # Find other labels with different temp and same paraphrase
        other_temps = [t for t in ['low', 'medium', 'high'] if t != true_temp]
        other_label_nums = [label_num_j for label_num_j in labels
                            if label_num_j != label_num_i and
                            label_num_to_components[label_num_j][0] in other_temps and
                            label_num_to_components[label_num_j][1] == true_para]
        num_possible_misclassifications = len(other_label_nums)
        possible_misclassifications_temperature_only += true_class_count * num_possible_misclassifications

    # Error_fail_dissociate
    possible_misclassifications_fail_dissociate = 0
    for label_num_i in labels:
        true_class_count = true_class_counts.get(label_num_i, 0)
        true_temp, true_para = label_num_to_components[label_num_i]
        true_temp_value = temperature_order[true_temp]
        if true_para == 'with_no_paraphrase':
            # Possible misclassifications to higher temperature with paraphrase
            higher_temps = [t for t, v in temperature_order.items() if v > true_temp_value]
            other_label_nums = [label_num_j for label_num_j in labels
                                if label_num_j != label_num_i and
                                label_num_to_components[label_num_j][0] in higher_temps and
                                label_num_to_components[label_num_j][1] == 'with_paraphrase']
            num_possible_misclassifications = len(other_label_nums)
            possible_misclassifications_fail_dissociate += true_class_count * num_possible_misclassifications
        elif true_para == 'with_paraphrase':
            # Possible misclassifications to lower temperature with no paraphrase
            lower_temps = [t for t, v in temperature_order.items() if v < true_temp_value]
            other_label_nums = [label_num_j for label_num_j in labels
                                if label_num_j != label_num_i and
                                label_num_to_components[label_num_j][0] in lower_temps and
                                label_num_to_components[label_num_j][1] == 'with_no_paraphrase']
            num_possible_misclassifications = len(other_label_nums)
            possible_misclassifications_fail_dissociate += true_class_count * num_possible_misclassifications

    # Adjusted error rates
    if possible_misclassifications_paraphrase_only > 0:
        error_paraphrase_only_rate = (error_paraphrase_only / possible_misclassifications_paraphrase_only) * 100
    else:
        error_paraphrase_only_rate = 0

    if possible_misclassifications_temperature_only > 0:
        error_temperature_only_rate = (error_temperature_only / possible_misclassifications_temperature_only) * 100
    else:
        error_temperature_only_rate = 0

    if possible_misclassifications_fail_dissociate > 0:
        error_fail_dissociate_rate = (error_fail_dissociate / possible_misclassifications_fail_dissociate) * 100
    else:
        error_fail_dissociate_rate = 0

    # Create the bar plot
    categories = [
        'Fail to detect \n repetitiveness \n(paraphrase)',
        'Fail to detect \n  derailment level \n(temperature)',
        'Fail to dissociate \nrepetitiveness \n and derailment'
    ]
    counts = [error_paraphrase_only, error_temperature_only, error_fail_dissociate]
    rates = [error_paraphrase_only_rate, error_temperature_only_rate, error_fail_dissociate_rate]
    colors = ['#ffcc99', '#dda0dd', '#ff9999']  # Light orange, light purple, light red

    print("possible_misclassifications_paraphrase_only",possible_misclassifications_paraphrase_only)
    print("error_paraphrase_only",error_paraphrase_only)
    
    print("possible_misclassifications_temperature_only",possible_misclassifications_temperature_only)
    print("error_temperature_only",error_temperature_only)
    
    print("possible_misclassifications_fail_dissociate",possible_misclassifications_fail_dissociate)
    print("error_fail_dissociate",error_fail_dissociate)
    if plot:
        plt.rcParams.update({'font.size': 12})  # Increase global font size

        # Create a figure with adjusted size
        plt.figure(figsize=(6, 4))
        plt.bar(categories, rates, color=colors)
        for index, value in enumerate(rates):
            plt.text(index, value + 0.5, f"{value:.2f}%", ha='center', fontsize=12)
        plt.xlabel('Error Types', fontsize=12)
        plt.ylabel('Adjusted Error Rate (%)', fontsize=12)
        plt.title('Adjusted Error Rates by Category', fontsize=14)
        plt.ylim(0, max(rates) + 5)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.tight_layout()

        plt.show()

    return categories, rates




def from_cm_to_error_plot_old(cm, num_to_label, plot = True):
    labels = list(num_to_label.keys())
    # Build a mapping from index to (temperature, paraphrase)
    index_to_label_components = {}
    for idx, label_num in enumerate(labels):
        label_name = num_to_label[label_num]
        index_to_label_components[idx] = parse_label(label_name)

    error_paraphrase_only = 0
    error_temperature_only = 0
    error_both = 0
    total_errors = 0

    # Process the confusion matrix
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            if i == j:
                continue  # Correct classification
            else:
                total_errors += count  # Count total errors
                true_temp, true_para = index_to_label_components[i]
                pred_temp, pred_para = index_to_label_components[j]
                if true_temp == pred_temp and true_para != pred_para:
                    error_paraphrase_only += count
                elif true_temp != pred_temp and true_para == pred_para:
                    error_temperature_only += count
                elif true_temp != pred_temp and true_para != pred_para:
                    error_both += count

    if total_errors > 0:
        error_paraphrase_only_pct = (error_paraphrase_only / total_errors) * 100
        error_temperature_only_pct = (error_temperature_only / total_errors) * 100
        error_both_pct = (error_both / total_errors) * 100
    else:
        error_paraphrase_only_pct = 0
        error_temperature_only_pct = 0
        error_both_pct = 0

    # Create the bar plot
    categories = [' Fail to detect repetitiveness  \n (paraphrase)', 'Fail to detect incoherence level \n (temperature)', 'Fail to detect both \n (temperature and paraphrase)']
    counts = [error_paraphrase_only, error_temperature_only, error_both]
    percents = [error_paraphrase_only_pct, error_temperature_only_pct, error_both_pct]
    colors = ['#add8e6', '#6495ed', '#00008b'] 

    if plot : 
        plt.rcParams.update({'font.size': 12})  # Increase global font size

        # Create a figure with reduced size
        plt.figure(figsize=(6, 4)) 
        plt.bar(categories, counts, color=colors,)
        for index, value in enumerate(counts):
            plt.text(index, value + 2, f"{value:.0f}", ha='center', fontsize=12)
        plt.xlabel('Percentage of Total Errors (%)')
        plt.ylabel('Error Types')
        plt.title('Normalized Distribution of Error Types')
        #plt.ylim(0, 80)

        plt.xticks(rotation=45, ha='right',fontsize = 14)
        plt.tight_layout()

        plt.show()

    return categories, counts

def make_treatment_1(x):
    if x in ['low_temperature_with_no_paraphrase']:
        return 0
    elif x in ['high_temperature_with_paraphrase', 'medium_temperature_with_paraphrase']:
        return 1
    else :  return -1




def from_treatment_name_to_index(dataset, treatment_name, target_list = ['average_mean_sim']):
    dataset = dataset[dataset[treatment_name ]!= -1]
    df_to_match = dataset[target_list + [treatment_name,'index']]

    psm = PsmPy(df_to_match, treatment=treatment_name, indx="index",exclude = [])
    psm.logistic_ps(balance = False)
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=0.2)#0.001)
    index_to_keep = psm.df_matched['index'].tolist()

    return index_to_keep




def filter_dataset(dataset, filter_name, target_list = ['average_mean_sim']):
    ### Use propoension score matchinf or simple filter for dataset
    if "medium_temperature" in filter_name :
        a = len(dataset)
        dataset = dataset[dataset.temperature_enc!="medium_temperature"]
        b = len(dataset)
        print(f"Number of samples before and after filter: {a} -> {b}")
    elif "high_temperature" in filter_name :
        a = len(dataset)
        dataset = dataset[dataset.temperature_enc!="high_temperature"]
        b = len(dataset)
        print(f"Number of samples before and after filter: {a} -> {b}")
    elif filter_name == "psm":
        a = len(dataset)
        dataset['treatment_1'] = dataset['combined_label'].apply(make_treatment_1)
        dataset['index'] = dataset.index
        index_to_keep_1 = from_treatment_name_to_index(dataset, "treatment_1", target_list = target_list)
        #index_to_keep_2 = from_treatment_name_to_index(dataset, "treatment_1", target_list = ['fast_text_all_wmd_sentences_similarity_mean',
        #        'fast_text_all_cosine_sentences_similarity_mean'])
        index_to_keep =  index_to_keep_1 #list(set(index_to_keep_1+ index_to_keep_2))
        dataset = dataset[dataset.index.isin(index_to_keep)].reset_index(drop=True)
        b = len(dataset)
        print(f"Number of samples before and after PSM: {a} -> {b}")

    return dataset


def resample_dataset(df, target_name = "cat_label"):
    X = df.drop(target_name, axis=1)
    y = df[target_name]

    # Apply SMOTE
    smote = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Create a balanced DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_name] = y_resampled

    return df_resampled