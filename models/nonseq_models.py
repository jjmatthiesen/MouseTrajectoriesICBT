from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

class NonModel:
    """
    A utility class for working with scikit-learn models, providing
    grid search with cross-validation functionality and easy integration with pipelines.
    """
    def __init__(self, model: BaseEstimator, model_name: str, parameters: Optional[Dict[str, Union[int, float, str]]] = None) -> None:
        """
        Initializes the NonModel class with the specified model, model name, and optional parameters.

        :param model: The scikit-learn model instance (e.g., LogisticRegression, RandomForestClassifier).
        :param model_name: A string representing the name of the model, used as an identifier in pipelines.
        :param parameters: An optional dictionary of parameters to set for the model during initialization.
        """
        self.name = model_name
        self.parameters = parameters
        self.model = model

        # Set the model parameters if provided
        if parameters:
            model.set_parameters(**self.parameters)

    def gridfit(self,
                x_train: np.ndarray,
                y_train: np.ndarray,
                param_grid: List[Dict[str, Union[int, float, str]]],
                scaler_object: Optional[BaseEstimator] = None,
                scoring: str = 'roc_auc',
                k: int = 5) -> None:
        """
        Runs a grid search with k-fold cross-validation to find the best parameters for the model.

        :param x_train: The training data features, as a NumPy array.
        :param y_train: The training data labels, as a NumPy array.
        :param param_grid: A list of dictionaries representing the grid of parameters to search over.
        :param scaler_object: An optional scaling object (e.g., StandardScaler, MinMaxScaler) for preprocessing.
        :param scoring: The scoring metric to use for evaluating the model during cross-validation. Default is 'roc_auc'.
        :param k: The number of folds for cross-validation. Default is 5.
        :return: None
        """
        # Reminder: Adding "passthrough" allows to skip parts of the pipeline
        # Prefix parameter names with the model name for compatibility with the pipeline
        param_grid = [{self.name + '__' + key: value for key, value in params.items()} for params in param_grid]

        # Create a pipeline with the scaler (if provided) and the model
        self.pipeline_all = Pipeline([('scaler', scaler_object), (self.name, self.model)])

        # Initialize the GridSearchCV object with the pipeline and parameters
        self.grid_search = GridSearchCV(self.pipeline_all,
                                        param_grid=param_grid,
                                        n_jobs=-1,  # Use all available processors
                                        cv=k,  # Number of folds for cross-validation
                                        scoring=scoring)  # Scoring metric

        # Fit the model using grid search
        try:
            self.grid_search.fit(x_train, np.array(y_train).ravel())
        except Exception as e:
            print(f"Error during grid search fitting: {e}")

        # Store the best parameters and score from the grid search
        self.best_params = self.grid_search.best_params_
        self.best_score_ = self.grid_search.best_score_

    def model_results_cv(self) -> Dict[str, Union[float, Dict[str, Union[int, float, str]]]]:
        """
        Returns the results of the grid search, including the best cross-validation score and the corresponding parameters.

        :return: A dictionary containing the best cross-validation score and the best parameters.
        """

        return {
            'best_cv_score': self.best_score_,  # Best cross-validation score
            'parameters': self.best_params  # Best parameters found during the grid search
        }
