from cmath import inf
import sys, os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from concrete_src.entity.config_entity import DataIngestionConfig 
from concrete_src.entity.artifact_entity import DataIngestionArtifact
from concrete_src.exception import ConcreteException
from concrete_src.logger import logging

class DataIngestion:

    def __init__(self, data_ingestion_config:DataIngestionConfig) -> None:
        try:
            logging.info(f"{'*'*20}Data Ingestion started{'*'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def download_data(self)->DataIngestionArtifact:
        try:
            """
            Downloads the data from kaggle into raw data dir
            """

            # authenticate kaggle api with kaggle.json present in .kaggle
            api = KaggleApi()
            api.authenticate()
          
            
            # folder location to download file
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            # create folder
            os.makedirs(raw_data_dir, exist_ok=True)
            
            dataset_name = self.data_ingestion_config.dataset_name
            dataset_filename = self.data_ingestion_config.dataset_filename

            logging.info(f"Downloading file [{dataset_filename}] from [{dataset_name}] into [{raw_data_dir}]")
            api.dataset_download_file(dataset = dataset_name, file_name = dataset_filename, path = raw_data_dir)
            raw_data_file_path = os.path.join(raw_data_dir,dataset_filename)
            logging.info(f"Download completed. File {raw_data_file_path} downloaded successfully")
            
            data_ingestion_artifact = DataIngestionArtifact(
                raw_data_file_path = raw_data_file_path,
                is_ingested = True, 
                message = "Data ingested successfully")
            logging.info(f"DataIngestionArtifact: [{data_ingestion_artifact}]")
            
            return data_ingestion_artifact
            
        except Exception as e:
            raise ConcreteException(e, sys) from e
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        """
        Initiate data download
        """
        try:
            raw_data_file_path = self.download_data()
            return raw_data_file_path
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def __del__(self):
        """
        Acts as destructor. Called before all references to the class object are deleted.
        """
        logging.info(f"{'*' *25} Data Ingestion completed {'*' *25}\n")


