import os, sys

from concrete_src.config.configuartion import Configuration
from concrete_src.exception import ConcreteException
from concrete_src.logger import logging
from concrete_src.entity.artifact_entity import *
from concrete_src.component.data_ingestion import DataIngestion
from concrete_src.component.data_validation import DataValidation
from concrete_src.component.data_transformation import DataTransformation
from concrete_src.component.model_trainer import ModelTrainer

class Pipeline:
    def __init__(self, config: Configuration ) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            self.config = config
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def run_pipeline(self):
        try:
                logging.info(f"{'*'*20}Pipeline starting{'*'*20}\n")
                data_ingestion_artifact = self.start_data_ingestion()
                data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
                data_transformation_artifact = self.start_data_transformation(
                    data_ingestion_artifact=data_ingestion_artifact,
                    data_validation_artifact=data_validation_artifact
                )
                model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

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


    def __del__(self):
        """
        Acts as destructor. Called before all references to the class object are deleted.
        """
        logging.info(f"{'*' *25} Pipeline completed {'*' *25}\n")
