import os, sys

from concrete_src.config.configuartion import Configuration
from concrete_src.exception import ConcreteException
from concrete_src.logger import logging
from concrete_src.entity.artifact_entity import *
from concrete_src.component.data_ingestion import DataIngestion

class Pipeline:
    def __init__(self, config: Configuration ) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            self.config = config
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def run_pipeline(self):
        try:
                logging.info(f"{'*'*20}Pipeline starting{'*'*20}")
                data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
                raise ConcreteException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
            try:
                data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
                return data_ingestion.initiate_data_ingestion()
            except Exception as e:
                raise ConcreteException(e, sys) from e