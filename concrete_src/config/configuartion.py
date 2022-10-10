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
            data_ingestion_artifact_dir = os.path.join(self.get_training_pipeline_config().artifact_dir, DATA_INGESTION_ARTIFACT_DIR, self.time_stamp) 
            dataset_name = data_ingestion_config_info[DATASET_NAME]
            dataset_filename = data_ingestion_config_info[DATASET_FILENAME]
            raw_data = os.path.join(data_ingestion_artifact_dir,data_ingestion_config_info[DATA_INGESTION_RAW_DATA_DIR])
            
            data_ingestion_config_info = DataIngestionConfig(
                dataset_name = dataset_name, 
                dataset_filename = dataset_filename,
                raw_data_dir=raw_data
                )
                
            logging.info(f"DataIngestionConfig: {data_ingestion_config_info}")
            return data_ingestion_config_info
        except Exception as e:
            raise ConcreteException(e, sys) from e