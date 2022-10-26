from concrete_src.logger import logging
from concrete_src.exception import ConcreteException
from concrete_src.config.configuartion import Configuration
from concrete_src.entity.config_entity import DataValidationConfig
from concrete_src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from concrete_src.constant import *
from concrete_src.util.util import read_yaml_file

import sys, os
import pandas as pd
import numpy as np
from collections import OrderedDict


class DataValidation:
    def __init__( self, data_validation_config: DataValidationConfig, 
        data_ingestion_artifact:DataIngestionArtifact)-> None:
        """
        Initialtes the inputs to this componnent
        Inputs:
        data_validation_config: DataValidationConfig
        data_ingestion_artifact:DataIngestionArtifact
        """
        try:
            logging.info(f"\n{'*'*20}Data Validation{'*'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def is_input_file_exists(self):
        try:
            """
            Checks if input file exists
            """
            logging.debug(f"Checking if input file exists")
            is_file_exists = False

            is_file_exists = os.path.exists(self.data_ingestion_artifact.train_file_path)

            logging.info(f"Input File [{self.data_ingestion_artifact.train_file_path}] exists? {is_file_exists}")

            if not (is_file_exists):
                raise Exception(f"Input file not available")
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def validate_dataset_schema(self):
        try:
            """
            Cross checks the input data with the schema file
            1. Number of columns
            2. Column names
            3. Data types
            """      
            schema_file_path = os.path.join(self.data_validation_config.schema_file_path)   
            logging.info(f"Reading schema file: {schema_file_path}")    
            schema_info = read_yaml_file(file_path = schema_file_path)
            schema_columns = OrderedDict(sorted(schema_info["columns"].items()))
            logging.debug(f"Schema Info: {schema_columns}")
            input_data = pd.read_csv(self.data_ingestion_artifact.train_file_path)  
            logging.debug(f"Data Columns: {input_data.columns.sort_values()}")         
            if len(schema_columns) == input_data.shape[1]:
                logging.debug(f"Validated no. of columns: {input_data.shape[1]}")
                logging.debug("Validating columns and types")
                for data_col, schema_col in zip(input_data.columns.sort_values(), schema_columns.keys()):
                    if str.strip(data_col)==str.strip(schema_col):
                        if not(np.issubdtype(input_data[data_col].dtype, np.dtype(schema_columns[schema_col]))):
                            raise Exception(f"Schema Column {schema_col}: {np.dtype(schema_columns[schema_col])} not same as DataColumn {data_col}: {input_data[data_col].dtype}")
                    else:
                        raise Exception(f"Schema Column {schema_col} not same as DataColumn {data_col}")                 
            else:
                raise Exception(f"No. of columns not matching") 
            logging.info("Validated columns and types")
        except Exception as e:
            raise ConcreteException(e, sys) from e

    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            """
            Start data validation
            """
            self.is_input_file_exists()
            self.validate_dataset_schema()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path = self.data_validation_config.schema_file_path, 
                is_validated = True,
                message = "Data Validation Performed Successfully"
            )
            logging.info(f"data_validation_artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ConcreteException(e, sys) from e
        
    def __del__(self):
        """
        Acts as destructor. Called before all references to the class object are deleted.
        """
        logging.info(f"{'*' *25} Data Validation log completed {'*' *25}\n")
