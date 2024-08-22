from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Ensure these imports match your project structure
from src.utils.imports import *
from src.utils.utils import logger_nonseq, generate_path_ns
from src.pre_processing.pre_process_nonseq import prep, missing_madras
from models.nonseq_models import NonModel
from utils.plots import nested_auccurve_nonseq

# Define model and its parameters -> Paper used Random Forest but any scikit-learn model should work
non_sequ_modelparameter = {
    'rf': {
        'model': RandomForestClassifier(class_weight='balanced'),
        'param_grid': [{'n_estimators': [3, 5, 10, 25],
                        'min_samples_split': [5, 15, 25, 50],
                        'max_depth': [3, 5, 10, 25, 50],
                        'bootstrap': [True, False]}]
    },
}


def model_training(x: np.ndarray,
                   y: np.ndarray,
                   desc: str,
                   name: str,
                   x_columns: pd.Index,
                   scaler_object: Optional[StandardScaler] = StandardScaler(),
                   path: str = '',
                   scoring: str = 'roc_auc',
                   k: int = 5,
                   reps: int = 5) -> None:
    """
    Train and evaluate the model using k-fold cross-validation and grid search.

    :param x: Features for training, as a NumPy array.
    :param y: Labels for training, as a NumPy array.
    :param desc: Description or name of the model run (used for saving results).
    :param name: Name of the model that corresponds to key in non_sequ_modelparameter
    :param x_columns: Columns of the feature data.
    :param scaler_object: Scaler object to normalize features, default is StandardScaler.
    :param path: Directory path to save the models and results. Needs to be adapted to your project structure.
    :param scoring: Scoring metric for model evaluation. Default is 'roc_auc'.
    :param k: Number of folds for cross-validation. Default is 5.
    :param reps: Number of repetitions for cross-validation with different seeds. Default is 5.
    :return: None
    """
    # Initiate CV
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Cam be changed, where fixed to ensure same test sets between sequential and non-sequential models
    seeds = [1, 11, 111, 42, 66]

    results_all = {}
    inner_cvs = []

    for j in range(reps):
        results = {}
        model_info = non_sequ_modelparameter[name]
        model_info['param_grid'][0]['random_state'] = [seeds[j]]
        mod = NonModel(model_info['model'], name)

        x = np.array(x)
        y = np.array(y).ravel()

        for i, (train, test) in enumerate(cv.split(x, y)):
            # Preprocess the data with missing value handling here to avoid target leak
            x_train, x_test = missing_madras(x[train], x[test], x_columns)

            logging.info('{} Distribution Training {}'.format(i, pd.DataFrame(y[train]).value_counts().to_string()))
            logging.info('{} Distribution Test {}'.format(i, pd.DataFrame(y[test]).value_counts().to_string()))

            # Perform grid search and fit the model
            mod.gridfit(x_train, y[train], model_info['param_grid'], scaler_object, scoring=scoring)

            # Save the trained model
            model_file = f"{path}/models/{desc}_{i}_{name}.pkl"
            with open(model_file, 'wb') as file:
                pickle.dump(mod, file)

            # Save model results for each fold
            results[i] = mod.model_results_cv()
            results[i]['y_true'] = y[test]
            results[i]['y_proba'] = mod.grid_search.predict_proba(x_test)[:, 1]
            results[i]['y_pred'] = mod.grid_search.predict(x_test)
            results[i]['parameters'] = mod.best_params
            results[i]['test_index'] = test

        inner_cvs.append([results[i]['best_cv_score'] for i in range(k)])
        results_all[j] = results

    # Plot AUC curves
    outer_aucs = nested_auccurve_nonseq(results_all, desc, name, k, reps)
    outer_mean_auc = np.mean(outer_aucs)

    # Save all results
    results_file = f"{path}/results/{desc}_Results.pkl"
    with open(results_file, 'wb') as handle:
        pickle.dump(results_all, handle)

    logging.info(f"{name} model: mean outer CV {outer_mean_auc:.2f}, mean inner CV {np.mean(inner_cvs):.2f}")


def run_experiment(PATH: str,
                   name: str,
                   data_version: str,
                   ys: List[str],
                   xs: List[str],
                   mouse: Optional[str] = None,
                   mouse_feat: Optional[List[str]] = None,
                   mouse_only: bool = False) -> None:
    """
    Run the experiment for training and evaluating models on the given dataset.

    :param PATH: Path to the data directory. Needs to be adapted to your project structure.
    :param name: Name of the experiment (used for saving results).
    :param data_version: Version or identifier for the dataset file.
    :param ys: List of target variables.
    :param xs: List of features to be used in the model.
    :param mouse: Optional path to mouse-related feature data.
    :param mouse_feat: List of specific mouse-related features to include.
    :param mouse_only: Boolean indicating if only mouse features should be used.
    :return: None
    """
    # Check if there is already a result folder for this name, if not generate
    path = f"../results/non_sequential/{name}"
    generate_path_ns(path)
    logging.info(f'Data Version - {data_version}')

    # Include mouse features if specified
    if mouse:
        xs.extend([x for x in mouse_feat if x not in xs])

    # Prepare the data for modeling
    x, y = prep(PATH, data_version, ys, xs, mouse=mouse, mouse_only=mouse_only)

    logging.info(f'Features included: {str(x.columns)}')

    # Train the model
    model_training(x, y, name, x_columns=x.columns, scaler_object=StandardScaler(),
                   path=path, scoring='roc_auc', k=5)


if __name__ == '__main__':
    # Define paths and parameters for the experiment
    PATH = '/path/to/data/directory/'  # Adapt this path to your project structure
    data_version = 'dataset_version.csv'
    ys = ['dropout_mod']

    # Initialize the logger
    logger = logger_nonseq('experiment_log')

    # Run baseline model -> Adapt Features to project
    baseline_features = ['MADRS_sum_PRE', 'MADRS_sum_SCREEN', 'Treatment_norm', 'age', 'female']
    run_experiment(PATH, 'baseline', data_version, ys, xs=baseline_features)

    # Run an experiment with selected mouse features (example with 10 features)
    mouse_selected_10 = ['participant_id', 'speed_avg', 'angle_change_mean', 'acute_angles',
                         'obtuse_angles', 'jitter', 'pause_time_total', 'moved_dist',
                         'number_dp', 'pauses_no', 'scroll_speed_mean']

    run_experiment(PATH, '10mouse_selected', data_version, ys,
                   xs=baseline_features, mouse='features_user_all.csv',
                   mouse_feat=mouse_selected_10)

    # Run an experiment with a smaller subset of mouse features (example with 3 features)
    mouse_selected_3 = ['participant_id', 'speed_avg', 'pause_time_total', 'scroll_speed_mean']

    run_experiment(PATH, '3mouse_selected', data_version, ys,
                   xs=baseline_features, mouse='features_user_all.csv',
                   mouse_feat=mouse_selected_3)

    # Plot NN results from JSON file (path needs to be adapted)
    #path_to_json = '/path/to/results/pred_probabilities_all.json'  # Adapt this path to your project structure
    #graph_nns(path_to_json, 5, 'temporal_spatial', 'experiment_name')