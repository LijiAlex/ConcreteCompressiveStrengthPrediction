import os
import sys
import numpy as np
from typing import List

from concrete_src.constant import *
from concrete_src.entity.model_entity import BestModel
from concrete_src.logger import logging
from concrete_src.exception import ConcreteException
from concrete_src.entity.config_entity import ModelEvaluationConfig
from concrete_src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from concrete_src.util.util import create_list, get_cluster, write_yaml_file, read_yaml_file, load_object, load_data
from concrete_src.util.model_factory import evaluate_regression_model

class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"\n{'*' * 30}Model Evaluation log started.{'*' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_evaluation_files_path = create_list(len(self.model_trainer_artifact.clusters))
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_previous_best_models(self)->List[BestModel]:
        """
        Returns previous best models and populates evaluation report names for each cluster.
        Return:
        Dictionary with key as cluster number and value as previous best model path
        """
        try:
            logging.info("Get previous best models")
            previous_models = create_list(len(self.model_trainer_artifact.clusters))
            
            for cluster in self.model_trainer_artifact.clusters:
                model_evaluation_file_path = os.path.join(
                    self.model_evaluation_config.model_evaluation_files_folder,
                    self.model_evaluation_config.model_evaluation_file_prefix+str(cluster)+".yaml"
                )
                self.model_evaluation_files_path[cluster] = model_evaluation_file_path
                logging.debug("Get previous best model:cluster {cluster} from {model_evaluation_file_path}")
                if not os.path.exists(model_evaluation_file_path):
                    write_yaml_file(file_path=model_evaluation_file_path)
                    previous_models[cluster]=None
                    logging.info("No previous model found")
                    continue
                model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

                model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

                if BEST_MODEL_KEY not in model_eval_file_content:
                    previous_models[cluster]=None
                    logging.info("No previous model found")
                    continue

                model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
                previous_models[cluster]=model
                logging.info(f"cluster {cluster} : {model}")
            return previous_models
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def update_evaluation_report(self,cluster:int, evaluated_model_path:str,
                                                                    is_model_accepted:bool):
        try:
            eval_file_path = self.model_evaluation_files_path[cluster]
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            current_eval_content = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: evaluated_model_path,
                }
            }

            if previous_best_model is not None:
                previous_model = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: previous_model}
                    current_eval_content.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(previous_model)

            model_eval_content.update(current_eval_content)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise ConcreteException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_models_file_path = self.model_trainer_artifact.trained_models_file_path
            trained_model_objects = create_list(len(self.model_trainer_artifact.clusters))
            logging.info("Get currently trained model objects")
            for model_file in trained_models_file_path:                
                cluster = get_cluster(model_file)
                trained_model_objects[cluster] = load_object(file_path=model_file)
                logging.info(f"Cluster {cluster}: {trained_model_objects[cluster]}")

            previous_models = self.get_previous_best_models()

            train_file_path = self.data_ingestion_artifact.train_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            train_dataframe = load_data(file_path=train_file_path,
                                                           schema_file_path=schema_file_path
                                                           )              

            evaluated_model_paths = create_list(len(self.model_trainer_artifact.clusters))
            is_models_accepted = create_list(len(self.model_trainer_artifact.clusters))

            for cluster in self.model_trainer_artifact.clusters:
                logging.info(f"{'>'*10}Cluster:{cluster} model evaluation{'>'*10}")                
                if previous_models[cluster] is None:
                    logging.info("Not found any existing model.")
                    evaluated_model_paths[cluster] = trained_models_file_path[cluster]
                    is_models_accepted[cluster] = True
                    self.update_evaluation_report(cluster= cluster, evaluated_model_path=trained_models_file_path[cluster],
                                                                    is_model_accepted=True)
                    logging.info(f"Currently trained model accepted. {trained_models_file_path[cluster]}")
                    continue

                previous_model = previous_models[cluster]
                trained_model = trained_model_objects[cluster]

                both_models = [previous_model, trained_model]

                logging.info(f"Comparing previous model {previous_model} & current model {trained_model}")
                metric_info_artifact = evaluate_regression_model(model_list=both_models,
                                                                X_train=train_dataframe[train_dataframe.columns[:-1]],
                                                                y_train=train_dataframe[train_dataframe.columns[-1]],
                                                                flag = 1
                                                                )
                logging.info(f"Model evaluation completed. Model metric artifact: {metric_info_artifact}")

                if metric_info_artifact is None:
                    evaluated_model_paths[cluster] = trained_models_file_path[cluster]
                    is_models_accepted[cluster] = False
                    logging.info(f"Current model rejected. {trained_models_file_path[cluster]}")
                    continue

                if metric_info_artifact.index_number == 1:
                    evaluated_model_paths[cluster] = trained_models_file_path[cluster]
                    is_models_accepted[cluster] = True
                    self.update_evaluation_report(cluster= cluster, evaluated_model_path=trained_models_file_path[cluster],
                                                                    is_model_accepted=True)
                    logging.info(f"Current model accepted. {trained_models_file_path[cluster]} ")

                else:
                    logging.info("Trained model is no better than existing model hence not accepting trained model")
                    evaluated_model_paths[cluster] = trained_models_file_path[cluster]
                    is_models_accepted[cluster] = False

            model_evaluation_artifact = ModelEvaluationArtifact(clusters = self.model_trainer_artifact.clusters, is_models_accepted = is_models_accepted, 
                                        evaluated_model_paths = evaluated_model_paths)
            logging.info(f"Model EvaluationArtifact {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20}\n")