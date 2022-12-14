import os, sys
import uuid
from threading import Thread

from concrete_src.config.configuartion import Configuration
from concrete_src.exception import ConcreteException
from concrete_src.logger import logging
from concrete_src.entity.artifact_entity import *
from concrete_src.component.data_ingestion import DataIngestion
from concrete_src.component.data_validation import DataValidation
from concrete_src.component.data_transformation import DataTransformation
from concrete_src.component.model_trainer import ModelTrainer
from concrete_src.component.model_evaluation import ModelEvaluation
from concrete_src.component.model_pusher import ModelPusher
from concrete_src.entity.experiment import ExperimentDetails, Experiment


class Pipeline(Thread):

    def __init__(self, config: Configuration ) -> None:
        try:            
            self.config = config
            self.experiment = Experiment(config.training_pipeline_config.artifact_dir, config.time_stamp)            
            super().__init__(daemon=False, name="pipeline")
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def run_pipeline(self):
        try:
            if self.experiment.current_experiment!=None and self.experiment.current_experiment.running_status:
                msg:str = "Pipeline already running"
                logging.info(msg)
                return self.experiment

            logging.info(f"{'*'*20}Pipeline starting{'*'*20}\n")
            self.experiment.start_experiment("Pipeline has been started.")
            logging.info(f"Pipeline experiment: {self.experiment.current_experiment}")

            self.experiment.save_experiment()

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
            if True in model_evaluation_artifact.is_models_accepted:
                model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
                logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            else:
                logging.info("Trained models rejected. Models not pushed")
            logging.info("Pipeline completed.")

            msg = "Pipeline has been completed."

            self.experiment.stop_experiment(msg = msg, is_model_accepted = model_evaluation_artifact.is_models_accepted,
             model_accuracy = model_trainer_artifact.model_accuracy)
            logging.info(f"Pipeline experiment: {self.experiment.current_experiment}")
            self.experiment.save_experiment()
        except Exception as e:
                raise ConcreteException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact)-> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact
                                             )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact
                                  ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )            
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise ConcreteException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_train_config(),
                                         data_transformation_artifact=data_transformation_artifact
                                         )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               data_validation_artifact: DataValidationArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            model_eval = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact)
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.config.get_model_pusher_config(),
                model_evaluation_artifact=model_eval_artifact
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise ConcreteException(e, sys) from e


    def __del__(self):
        """
        Acts as destructor. Called before all references to the class object are deleted.
        """
        logging.info(f"{'*' *25} Pipeline log completed {'*' *25}\n")
