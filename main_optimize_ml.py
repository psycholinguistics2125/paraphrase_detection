import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from src.utils import encode_temperature_str, encode_paraphrase, load_config,  load_and_clean_data

def perform_grid_search(data, features_list, target):
    """
    Perform grid search to find the best parameters for a RandomForestClassifier
    targeting a weighted F1 score using StratifiedGroupKFold.

    Parameters:
        data (pd.DataFrame): The dataset containing features and target.
        features_list (list): List of feature names to be used in the model.
        target (str): Name of the target column in the data.

    Returns:
        best_model (GridSearchCV): The GridSearchCV object fitted with the best parameters.
    """
    # Ensure the features and target exist in the DataFrame
    if not set(features_list).issubset(data.columns) or target not in data.columns:
        raise ValueError("Some features or target are not in the DataFrame")

    # Prepare features and target
    X = data[features_list]
    y = data[target]
    groups = data['prompt_cat']

    # Setup the StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    # Define the model
    clf = RandomForestClassifier(random_state=42)

    # Parameters grid
    param_grid = {
        'n_estimators': [50,100,200,300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1, 2,4],
        'bootstrap': [True, False]
    }

    # Define scorer
    f1_scorer = make_scorer(f1_score, average='weighted')

    # Setup the GridSearchCV
    grid_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions =param_grid,
        scoring=f1_scorer,
        cv=sgkf,
        n_iter=200,
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    grid_search.fit(X, y, groups=groups)

    # Return the fitted GridSearchCV object
    results_df = pd.DataFrame(grid_search.cv_results_)
    return grid_search, results_df







# Example usage
if __name__ == "__main__":
    # Load data (example path, replace with actual data loading method)
    config = load_config("config.yaml")
    data = load_and_clean_data(config)
    
    # Define the features and target
    
    density_features = data.filter(regex='score').columns.tolist()
    mean_sim_metrics = data.filter(regex = '(sentence_sim|w2v|glove|fast_text)(.*)(mean)').columns.tolist()
    std_sim_metrics = data.filter(regex = '(sentence_sim|w2v|glove|fast_text)(.*)(std)').columns.tolist()
    sim_features = mean_sim_metrics + std_sim_metrics
    all_features = density_features + sim_features

    features_dict = {
        "sim_features": sim_features,
        "density_features": density_features,
        "mean_sim_metrics": mean_sim_metrics,
        "std_sim_metrics": std_sim_metrics,
        "all_features": all_features
    }

    target = "cat_label"
    
    for features_name, features_list in features_dict.items():
        print(f"Features: {features_name}")
        # Perform grid search
        best_model, results_df = perform_grid_search(data, features_list, target)
        results_df.to_csv(f"{features_name}_random_search_results_2.csv", index=False)
        
        # Best model details
        print("Best Parameters:", best_model.best_params_)
        print("Best F1 Score (weighted):", best_model.best_score_)
    