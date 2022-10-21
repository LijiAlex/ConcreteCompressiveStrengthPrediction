from concrete_src.entity.config_entity import ModelTrainerConfig
from concrete_src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from concrete_src.util.model_factory import ModelFactory
from concrete_src.exception import ConcreteException
from concrete_src.logger import logging
from concrete_src.constant import *
from concrete_src.util.util import *
from concrete_src.entity.model_entity import *
from concrete_src.util.model_factory import evaluate_regression_model

import os, sys
from typing import List
import pandas as pd

class ConcreteStrengthEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        Train Model constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

class ModelTrainer:
    #loading transformed training  datset
    #reading model config file 
    #getting best model on training datset
    #evaludation models on both training & testing datset -->model object
    #loading preprocessing pbject
    #custom model object by combining both preprocessing obj and model obj
    #saving custom model object
    #return model_trainer_artifact

    def __init__(self, model_trainer_config : ModelTrainerConfig,
     data_transformation_artifact : DataTransformationArtifact):
        try:
            logging.info(f"{'*'*20}Model Trainer log started{'*'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_df = pd.read_csv(transformed_train_file_path)

            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)  

            logging.info("Initializing models from config file")
            model_factory.get_supplied_model_details()             
            
            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy: {base_accuracy}")

            all_clusters = list(train_df['cluster'].unique())
            trained_models_file_path = list()
            train_rmse_list = list()
            train_accuracy_list = list()
            model_accuracy_list = list()

            for cluster in all_clusters:
                cluster_data=train_df[train_df['cluster']==cluster] # filter the data for one cluster
                # Prepare the feature and Label columns
                cluster_features = cluster_data[train_df.columns[:-2]]
                cluster_label= cluster_data[train_df.columns[-1]]
                #getting the best model for each of the clusters  
                logging.info(f"{'#'*20}Training for cluster {cluster}{'#'*20}")  
                best_model, grid_searched_best_model_list = model_factory.get_best_model(X=cluster_features,y=cluster_label,base_accuracy=base_accuracy)
                logging.info(f"Best model for cluster {cluster}: {best_model.model} with score {best_model.best_score}")
                                      
                model_list = [grid_searched_model.best_model for grid_searched_model in grid_searched_best_model_list ]
                logging.info(f"Evaluating all trained model on training dataset by Kfold cv")
                metric_info:MetricInfoArtifact = evaluate_regression_model(model_list=model_list,X_train=cluster_features,y_train=cluster_label,base_accuracy=base_accuracy)

                logging.info(f"Best found model: {metric_info.model_name}")
                
                preprocessing_obj=  load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
                model_object = metric_info.model_object


                trained_models_path=self.model_trainer_config.trained_models_path
                trained_model_file_name = 'model_cluster'+str(cluster)+'.pkl'
                trained_model_file_path = os.path.join(trained_models_path, trained_model_file_name)
                trained_models_file_path.append(trained_model_file_path)
                concrete_strength_estimator_model = ConcreteStrengthEstimatorModel(preprocessing_object=preprocessing_obj,trained_model_object=model_object)
                logging.info(f"Saving model at path: {trained_model_file_path}")
                save_object(file_path=trained_model_file_path,obj=concrete_strength_estimator_model)
                train_rmse_list.append(metric_info.train_rmse)
                train_accuracy_list.append(metric_info.train_accuracy)
                model_accuracy_list.append(metric_info.model_accuracy)

            model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
                clusters = all_clusters,
                trained_models_file_path=trained_models_file_path,
                train_rmse=train_rmse_list,
                train_accuracy=train_accuracy_list,
                model_accuracy=model_accuracy_list          
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact                
        except Exception as e:
            raise ConcreteException(e,sys) from e

    def __del__(self):
        """
        Acts as destructor. Called before all references to the class object are deleted.
        """
        logging.info(f"{'*' *25} Model Trainer log completed {'*' *25}")