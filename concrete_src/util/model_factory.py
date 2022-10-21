import yaml, os, sys
import numpy as np
from pyexpat import model
from cmath import log
from typing import List, Tuple

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score

from concrete_src.exception import ConcreteException
from concrete_src.logger import logging
from concrete_src.util.util import read_yaml_file
from concrete_src.constant import *
from concrete_src.entity.model_entity import *
from concrete_src.util.model_factory_util import *


def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, base_accuracy:float=0.6) -> MetricInfoArtifact:
    """
    Description:
    This function compare multiple regression models and return the best model

    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature

    return
    It returns a named tuple    
    MetricInfoArtifact = namedtuple("MetricInfo",
                                ["model_name", "model_object", "train_rmse", "train_accuracy",
                                 "model_accuracy", "index_number"])

    """
    try:
        
    
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)  #getting model name based on model object
            logging.info(f"{'>>'*10}Started evaluating model: [{type(model).__name__}] {'<<'*10}")
            
            #Getting prediction for training dataset
            y_train_pred = model.predict(X_train)

            #Calculating r squared score on training testing dataset
            train_acc = r2_score(y_train, y_train_pred)
            
            #Calculating mean squared error on training testing dataset
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

            # Model accuracy using kfold cv
            scores = cross_val_score(model, X_train, y_train, cv = 3, n_jobs = 2, scoring = 'r2')
            model_accuracy = scores.mean()
            
            #logging all important metric
            logging.info(f"Train Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t{model_accuracy}")

            logging.info(f" Loss ->Train root mean squared error: [{train_rmse}]")


            #if model accuracy is greater than base accuracy we will accept that model as accepted model
            if model_accuracy >= base_accuracy:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                        model_object=model,
                                                        train_rmse=train_rmse,
                                                        train_accuracy=train_acc,
                                                        model_accuracy=model_accuracy,
                                                        index_number=index_number)

                logging.info(f"Acceptable model found: {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy than base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise ConcreteException(e, sys) from e

class ModelFactory:
    def __init__(self, model_config_path: str = None,):
        try:
            self.model_config: dict = read_yaml_file(model_config_path)

            self.grid_search_class_name: str = self.model_config[MODEL_CONFIG_GRID_SEARCH_KEY][MODEL_CONFIG_CLASS_KEY]
            self.grid_search_cv_module: str = self.model_config[MODEL_CONFIG_GRID_SEARCH_KEY][MODEL_CONFIG_MODULE_KEY]
            self.grid_search_param_data: dict = dict(self.model_config[MODEL_CONFIG_GRID_SEARCH_KEY][MODEL_CONFIG_PARAM_KEY])
            self.supplied_models: dict = dict(self.model_config[MODEL_CONFIG_SUPPLIED_MODELS])
            # all the accepted models(> base accuracy) with best configurations of parameters
            self.grid_searched_best_model_list = None
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def execute_grid_search_operation(self, supplied_model: ModelInitializationDetail, input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        Function will perform paramter search operation and will return you the best optimistic model with best paramter.
        
        Input:
        supplied_model: Model Initialization Detail
        input_feature: your all input features
        output_feature: Target/Dependent features
        
        return: 
        Function will return the best model
        """
        try:
            # instantiating GridSearchCV class     
           
            grid_search_cv_ref = class_for_name(module_name=self.grid_search_cv_module, class_name=self.grid_search_class_name)
            grid_search_cv = grid_search_cv_ref(estimator=supplied_model.model,
                                                param_grid=supplied_model.param_grid)
            grid_search_cv = update_property_of_class(grid_search_cv, self.grid_search_param_data)

            
            message = f"Training {type(supplied_model.model).__name__} Started."
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=supplied_model.model_serial_number,
                                                             model=supplied_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
                                                             )
            logging.info(f"Best score: {grid_search_cv.best_score_}, Best Parameters: {grid_search_cv.best_params_}")
            message = f"Training {type(supplied_model.model).__name__} Ended."
            logging.info(message)
            return grid_searched_best_model
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_supplied_model_details(self) -> List[ModelInitializationDetail]:
        """
        This function will return a list of supplied model details.
        return List[Model_Initialization_Detail]
        """
        try:
            supplied_model_list = []
            logging.info("Getting supplied model details")
            for model_index in self.supplied_models.keys():
                model_initialization_config = self.supplied_models[model_index]
                model_obj_ref = class_for_name(module_name=model_initialization_config[MODEL_CONFIG_MODULE_KEY],
                                                            class_name=model_initialization_config[MODEL_CONFIG_CLASS_KEY]
                                            )
                model = model_obj_ref()
                logging.debug(f"Model: {model}")
                if MODEL_CONFIG_PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[MODEL_CONFIG_PARAM_KEY])
                    model = update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)

                param_grid = model_initialization_config[MODEL_CONFIG_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODEL_CONFIG_MODULE_KEY]}.{model_initialization_config[MODEL_CONFIG_CLASS_KEY]}"

                model_initialization_config = ModelInitializationDetail(model_serial_number=model_index,
                                                                     model=model,
                                                                     param_grid=param_grid,
                                                                     model_name=model_name
                                                                     )

                supplied_model_list.append(model_initialization_config)

            self.supplied_models_details = supplied_model_list
            logging.info(f"Supplied Models: {self.supplied_models_details}")
            return self.supplied_models_details
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def initiate_best_parameter_search_for_supplied_models(self,
                                                              supplied_model_list: List[ModelInitializationDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:

        try:
            self.grid_searched_best_model_list = []
            for supplied_model in supplied_model_list:
                grid_searched_best_model = self.execute_grid_search_operation(
                    supplied_model=supplied_model,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise ConcreteException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.6
                                                          ) -> ClusterModelDetails:
        try:
            best_model = None
            logging.info(f"Looking for best model based on score")
            for grid_searched_best_model in grid_searched_best_model_list:
                logging.info(f"Model: {type(grid_searched_best_model.model).__name__}-> Score: {grid_searched_best_model.best_score}")
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.debug(f"Acceptable model found")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            cluster_model_details = ClusterModelDetails(best_model = best_model, \
                                    grid_searched_best_model_list = grid_searched_best_model_list)
            return cluster_model_details
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_best_model(self, X, y,base_accuracy=0.6) -> ClusterModelDetails:
        """
        Find the best model by performing grid search on the models supplied.
        """
        try:    
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_supplied_models(
                supplied_model_list=self.supplied_models_details,
                input_feature=X,
                output_feature=y
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise ConcreteException(e, sys)