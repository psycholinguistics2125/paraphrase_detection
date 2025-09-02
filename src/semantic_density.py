""" 
robin quillivic -- jannuary 2024
"""


import os
import logging
import numpy as np
import pandas as pd


#sklearn
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge, Lasso, Ridge, ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MeanShift, OPTICS, HDBSCAN

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, KernelPCA
from umap import UMAP
from umap import validation


from scipy.interpolate import CubicSpline
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


### Embedings trajectory (Experimentale)

def compute_mean_dtw_distance(sentence_embeddings):
    dtw_distances = []
    for i in range(len(sentence_embeddings)):
        for j in range(i+1, len(sentence_embeddings)):
            distance, _ = fastdtw(sentence_embeddings[i], sentence_embeddings[j])
            dtw_distances.append(distance)
    return np.mean(dtw_distances)


def embeddings_trajectory_score(embeddings:np.array): # can be use for sentence or word embeddings
    # Compute pairwise distances between word embeddings
    distances = cosine_distances(embeddings)
    # Compute cumulative distance along the trajectory
    cumulative_distances = np.cumsum(np.diag(distances, k=1))
    # Compute Dynamic Time Warping (DTW) distance
    dtw_distance, _ = fastdtw(embeddings, cumulative_distances)
    # Normalize DTW distance by trajectory length
    trajectory_score = dtw_distance / len(embeddings)
    
    return trajectory_score


def narrative_speed_score(embeddings:np.array): # can be use for sentence or word embeddings
    # Compute pairwise distances between word embeddings
    distances = cosine_distances(embeddings)
    cumulative_distances = np.cumsum(np.diag(distances, k=1))
    if len(cumulative_distances) <=2:
        return 0 
    # sum of gradient
    else :
        speed = np.sum(np.abs(np.gradient(cumulative_distances))) / len(cumulative_distances)
  
    
    return speed * 2


### Dimensionality Reduction

reduction_methods = ['PCA'] #UMAP


def dimensionality_reduction_score(embeddings, reduction_method='PCA', score_method='unexplained_variance_ratio', explained_variance_threshold=0.5):
    # Convert embeddings to numpy array
    X = np.array(embeddings).T # we consider the words as features and the dimensions as samples
    n = len(embeddings)-1

    if reduction_method in ['PCA']:
       
        pca = PCA(n)
        X_reduced = pca.fit_transform(X)

        if score_method == 'unexplained_variance_ratio':
            score = 1-pca.explained_variance_ratio_[:2].sum()
        elif score_method == 'prop_of_components':
            cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components_threshold = np.argmax(cumulative_explained_variance >= explained_variance_threshold) + 1
            score = n_components_threshold / len(pca.components_)
        else:
            raise ValueError("Invalid score_method. Choose 'explained_variance_ratio' or 'number_of_components'.")
    
    elif reduction_method == 'UMAP':
        umap = UMAP()
        X_reduced = umap.fit_transform(X)
        if score_method != 'umap_trustworthiness':
            raise ValueError("UMAP does not support score_method other than 'explained_variance_ratio'.")
        score = np.mean(validation.trustworthiness_vector(X, X_reduced, max_k=n))
        
    
    else:
        raise ValueError("Invalid reduction_method. Choose 'PCA', 'KernelPCA', or 'UMAP'.")
    
 
    
    return score




### Clustering Score
cluster_methods = ['HDBSCAN',"MeanShift"] #"MeanShift", 'OPTICS'
def cluster_density_score(embeddings, method='Agglomerative'):
    # Convert embeddings to numpy array
    X = np.array(embeddings)

    # Perform clustering based on the specified method


    if method == 'OPTICS':
        clustering = OPTICS(min_samples=2)  
    elif method == 'HDBSCAN':
        clustering = HDBSCAN(min_cluster_size=2)  
    elif method == 'MeanShift':
        clustering = MeanShift()  # Example: Mean Shift Clustering
    else:
        raise ValueError("Invalid method. Choose from 'Agglomerative', 'DBSCAN', 'OPTICS', 'HDBSCAN', or 'MeanShift'.")
    try : 
        cluster_labels = clustering.fit_predict(X)

        # Compute the silhouette score
        silhouette = 1 - silhouette_score(X, cluster_labels) #* len(embeddings)

        # Compute the ratio of the number of clusters to the number of sentences
        cluster_density = len(set(cluster_labels)) / len(X)
    except:
        cluster_density = 0
        silhouette = 0
    return cluster_density, silhouette




### Regression SCore
regression_methods = ['Lasso', ] #'Ridge', 'ElasticNet']

def regression_coef_density_score(sentence_vectors, paragraph_vector, method='BayesianRidge', alpha = 0.01):
    # Convert sentence vectors and paragraph vector to numpy arrays
    X = np.array(sentence_vectors)
    y = np.array(paragraph_vector)

    # Initialize regression model based on the specified method
    if method == 'BayesianRidge':
        regression_model = BayesianRidge()
    elif method == 'Lasso':
        regression_model = Lasso(alpha=alpha)  # Adjust alpha as needed
    elif method == 'Ridge':
        regression_model = Ridge(alpha=alpha)  # Adjust alpha as needed
    elif method == 'ElasticNet':
        regression_model = ElasticNet(alpha=alpha, l1_ratio=0.5)  # Adjust alpha and l1_ratio as needed
    else:
        raise ValueError("Invalid method. Choose from 'BayesianRidge', 'Lasso', 'Ridge', or 'ElasticNet'.")

    # Fit regression model
    regression_model.fit(X.T, y)
    difficulty = 1 - regression_model.score(X.T, y)
    # Compute the density of non-zero coefficients
    density = np.count_nonzero(regression_model.coef_) / len(regression_model.coef_)
    #print(regression_model.coef_)

    return density, difficulty





### Feature Selection
features_methods =  ['Lasso'] # 'SGD']

def cross_validation_performance(X, y, estimator, n_features_to_select, cv=5, scoring='neg_mean_squared_error'):
    scores = []
    for n_features in n_features_to_select:
        rfe = RFE(estimator, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)
        score = cross_val_score(estimator, X_selected, y, cv=cv, scoring=scoring)
        scores.append(score.mean())
    return np.array(scores)

def compute_density_RFE_selection_score(sentence_embedings, paragraph_vector, estimator_name='Lasso') :
    X = sentence_embedings.T
    y = paragraph_vector
    n = len(sentence_embedings)
    if estimator_name == 'Lasso':
        estimator = Lasso(alpha=0.01)
    elif estimator_name == 'RandomForest':
        estimator = RandomForestRegressor(n_estimators = n)
    elif estimator_name == 'SGD':
        estimator = SGDRegressor(max_iter=1000, tol=1e-3)
    else:
        raise ValueError("Invalid estimator_name. Choose 'Lasso', 'RandomForest', or 'SGD'.")
    try : 
        n_features_to_select = [n-k for k in range(n-1,1,-1)]  # Different numbers of features to select
        cv_scores = cross_validation_performance(X, y, estimator, n_features_to_select)
        best_n_features = n_features_to_select[np.argmax(cv_scores)]
        score = best_n_features/n
    except Exception as e:
        score = 0
        print(e)
    return score


def compute_all_scores(data):
    features_list = []
    #embeddings_trajectory_score
    #data['mean_dtw_scores'] = data['sentence_sim_vectors'].apply(lambda x: compute_mean_dtw_distance(x)) / data['sentence_sim_vectors'].apply(lambda x: len(x))
    #features_list.append('mean_dtw_scores')
    
    data['trajectory_score'] = data['sentence_sim_vectors'].apply(lambda x: embeddings_trajectory_score(x)) / data['sentence_sim_vectors'].apply(lambda x: len(x))
    features_list.append('trajectory_score')

    #dimensionality_reduction_score
    data['reduction_score_'+"PCA_unexplained_variance"] = data['sentence_sim_vectors'].apply(lambda x: dimensionality_reduction_score(x, reduction_method="PCA",score_method="unexplained_variance_ratio"))
    features_list.append('reduction_score_'+"PCA_unexplained_variance")
    data['reduction_score_'+"PCA_prop_of_components"] = data['sentence_sim_vectors'].apply(lambda x: dimensionality_reduction_score(x, reduction_method="PCA",score_method="prop_of_components"))
    features_list.append('reduction_score_'+"PCA_prop_of_components")
    #data['reduction_score_UMAP_trustworthiness'] = data['sentence_sim_vectors'].apply(lambda x: dimensionality_reduction_score(x, reduction_method="UMAP",score_method="umap_trustworthiness"))
    # features_list.append('reduction_score_UMAP_trustworthiness')

    #cluster_density_score
    for cluster_method in cluster_methods:
        data['cluster_density_score_'+cluster_method], data['cluster_reverse_silhouette_score_'+cluster_method] = \
            zip(*data['sentence_sim_vectors'].apply(lambda x: cluster_density_score(x, method=cluster_method)))
        features_list.append('cluster_density_score_'+cluster_method)
        features_list.append('cluster_reverse_silhouette_score_'+cluster_method)

    #regression_coef_density_score
    for regression_method in regression_methods:
        data['regression_coef_density_score_'+regression_method], data['regression_error_score_'+regression_method] = \
            zip(*data.apply(lambda x: regression_coef_density_score(x.sentence_sim_vectors,x.paragraphs_vector, method=regression_method, alpha = 0.01),axis=1))
        features_list.append('regression_coef_density_score_'+regression_method)
        features_list.append('regression_error_score_'+regression_method)
    
    #compute_density_RFE_selection_score
    for rfe_method in features_methods:
        data['reduction_score_RFE_'+rfe_method] = data.apply(lambda x: compute_density_RFE_selection_score(x.sentence_sim_vectors,x.paragraphs_vector, estimator_name = rfe_method), axis=1)
        features_list.append('reduction_score_RFE_'+rfe_method)

    return data, features_list


def compute_all_semantic_scores(data):
    features_list = []
    #embeddings_trajectory_score
    #data['mean_dtw_scores'] = data['sentence_sim_vectors'].apply(lambda x: compute_mean_dtw_distance(x)) / data['sentence_sim_vectors'].apply(lambda x: len(x))
    #features_list.append('mean_dtw_scores')
    
    data['narrative_speed_score'] = data['sentence_sim_vectors'].apply(lambda x: narrative_speed_score(x)) 
    data['narrative_speed_score_norm'] = data['sentence_sim_vectors'].apply(lambda x: narrative_speed_score(x)) / data['sentence_sim_vectors'].apply(lambda x: len(x))
    features_list.append('trajectory_score')

    #dimensionality_reduction_score
    data['reduction_score_'+"PCA_explained_variance"] = data['sentence_sim_vectors'].apply(lambda x: dimensionality_reduction_score(x, reduction_method="PCA",score_method="unexplained_variance_ratio"))
    features_list.append('reduction_score_'+"PCA_unexplained_variance")
    data['reduction_score_'+"PCA_prop_of_components"] = data['sentence_sim_vectors'].apply(lambda x: dimensionality_reduction_score(x, reduction_method="PCA",score_method="prop_of_components"))
    features_list.append('reduction_score_'+"PCA_prop_of_components")
    #data['reduction_score_UMAP_trustworthiness'] = data['sentence_sim_vectors'].apply(lambda x: dimensionality_reduction_score(x, reduction_method="UMAP",score_method="umap_trustworthiness"))
    # features_list.append('reduction_score_UMAP_trustworthiness')

    #cluster_density_score
    for cluster_method in cluster_methods:
        data['cluster_density_score_'+cluster_method], data['cluster_reverse_silhouette_score_'+cluster_method] = \
            zip(*data['sentence_sim_vectors'].apply(lambda x: cluster_density_score(x, method=cluster_method)))
        features_list.append('cluster_density_score_'+cluster_method)
        features_list.append('cluster_reverse_silhouette_score_'+cluster_method)

    #regression_coef_density_score
    for regression_method in regression_methods:
        data['regression_coef_density_score_'+regression_method], data['regression_error_score_'+regression_method] = \
            zip(*data.apply(lambda x: regression_coef_density_score(x.sentence_sim_vectors,x.paragraphs_vector, method=regression_method, alpha = 0.01),axis=1))
        features_list.append('regression_coef_density_score_'+regression_method)
        features_list.append('regression_error_score_'+regression_method)
    
    #compute_density_RFE_selection_score
    for rfe_method in features_methods:
        data['reduction_score_'+rfe_method] = data.apply(lambda x: compute_density_RFE_selection_score(x.sentence_sim_vectors,x.paragraphs_vector, estimator_name = rfe_method), axis=1)
        features_list.append('reduction_score_'+rfe_method)

    return data, features_list