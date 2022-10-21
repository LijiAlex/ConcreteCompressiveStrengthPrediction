import sys

from concrete_src.constant import *
from concrete_src.util.util import read_yaml_file
from concrete_src.exception import ConcreteException
from concrete_src.logger import logging
from concrete_src.entity.config_entity import *

class Configuration:

    def __init__(self, 
    config_file_path:str = CONFIG_FILE_PATH, 
    current_time_stamp:str = CURRENT_TIME_STAMP
    ) -> None:
        """
        Read the configuration file.
        Get training pipeline configurations.
        """
        try:
            self.config_info = read_yaml_file(file_path = config_file_path)
            logging.info(f"Read configuration file: {config_file_path}")
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = current_time_stamp
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_training_pipeline_config(self)->TrainingPipeLineConfig:
        """
        Get pipeline configuration from config file.
        Returns:
        TrainingPipeLineConfig
        """
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR,
            training_pipeline_config[TRAINING_PIPELINE_NAME_KEY], 
            training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            )            
            training_pipeline_config = TrainingPipeLineConfig(artifact_dir = artifact_dir) 
            logging.info(f"TrainingPipelineConfig: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            """
            Get data ingestion configuration
            Returns:
            DataIngestionConfig
            """
            data_ingestion_config_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            data_ingestion_artifact_dir = os.path.join(self.training_pipeline_config.artifact_dir, DATA_INGESTION_ARTIFACT_DIR, self.time_stamp) 
            dataset_name = data_ingestion_config_info[DATASET_NAME]
            dataset_filename = data_ingestion_config_info[DATASET_FILENAME]
            raw_data = os.path.join(data_ingestion_artifact_dir,data_ingestion_config_info[DATA_INGESTION_RAW_DATA_DIR])
            injested_data_dir = os.path.join(data_ingestion_artifact_dir, data_ingestion_config_info[DATA_INGESTION_DIR_NAME_KEY])
            train_dir = os.path.join(injested_data_dir, data_ingestion_config_info[DATA_INGESTION_TRAIN_DIR])

            data_ingestion_config_info = DataIngestionConfig(
                dataset_name = dataset_name, 
                dataset_filename = dataset_filename,
                raw_data_dir=raw_data,
                ingested_train_dir=train_dir
                )
                
            logging.info(f"DataIngestionConfig: {data_ingestion_config_info}")
            return data_ingestion_config_info
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_data_validation_config(self)->DataValidationConfig:
        """
        Reads data validation configuration
        Returns:
            DataValidationConfig
        """
        try:
            data_validation_config_info = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            data_validation_schema_file_name = data_validation_config_info[DATA_VALIDATION_SCHEMA_FILE_NAME]
            file_path = os.path.join(ROOT_DIR, data_validation_config_info[DATA_VALIDATION_SCHEMA_DIR], data_validation_schema_file_name)
            data_validation_config_info = DataValidationConfig(
                schema_file_path = file_path) 
            logging.info(f"DataValidationConfig: {data_validation_config_info}")
            return data_validation_config_info
        except Exception as e:
            raise ConcreteException(e, sys)  from e

    def get_data_transformation_config(self)->DataTransformationConfig:
        """
        Reads data transformation configuration
        Returns:
            DataTransformationConfig
        """
        try:
            data_transformation_config_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            data_transformation_artifact_dir = os.path.join(self.training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_ARTIFACT_DIR, self.time_stamp) 
            transformed_dir = os.path.join(data_transformation_artifact_dir, TRANFORMED_DIR)
            transformed_train_dir = os.path.join(transformed_dir, data_transformation_config_info[TRANSFORMED_TRAIN_DIR])
            preprocessing_dir = os.path.join(data_transformation_artifact_dir, PREPROCESSING_DIR)
            preprocessed_file_name = os.path.join(preprocessing_dir, data_transformation_config_info[PREPROCESSED_OBJECT_FILE_NAME] )

            data_transformation_config_info = DataTransformationConfig(
                transformed_train_dir = transformed_train_dir, 
                preprocessed_object_file_path = preprocessed_file_name
                )
            logging.info(f"DataTransformationConfig: {data_transformation_config_info}")
            return data_transformation_config_info
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def get_model_train_config(self)->ModelTrainerConfig:
        """
        Reads model trainer configuration details
        Returns:
            ModelTrainerConfig
        """
        try:
            model_train_config_info = self.config_info[MODEL_TRAINER_CONFIG_INFO]
            artifact_dir = self.training_pipeline_config.artifact_dir

            model_trainer_artifact_dir=os.path.join(
                artifact_dir,
                MODEL_TRAINER_ARTIFACT_DIR,
                self.time_stamp
            )
            trained_models_path = os.path.join(model_trainer_artifact_dir, TRAINED_MODEL_DIR, self.time_stamp)
            base_accuracy = model_train_config_info[BASE_ACCURACY]

            model_config_file_path = os.path.join(model_train_config_info[MODEL_CONFIG_DIR],
            model_train_config_info[MODEL_CONFIG_FILE_NAME]
            )

            model_train_config_info = ModelTrainerConfig(
                trained_models_path = trained_models_path, 
                base_accuracy = base_accuracy,
                model_config_file_path = model_config_file_path
                )
            logging.info(f"ModelTrainingConfig: {model_train_config_info}")
            return model_train_config_info
        except Exception as e:
            raise ConcreteException(e, sys) from e
