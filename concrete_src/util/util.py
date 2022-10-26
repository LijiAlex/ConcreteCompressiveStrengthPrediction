from typing import List
import yaml, sys, os, dill
import numpy as np
import pandas as pd

from concrete_src.exception import ConcreteException
from concrete_src.constant import *

def read_yaml_file(file_path:str)->dict:
    """
    Reads YAML file and returns contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ConcreteException(e,sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise ConcreteException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ConcreteException(e, sys) from e

def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise ConcreteException(e,sys) from e


def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise ConcreteException(e,sys) from e

def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise ConcreteException(e,sys)

def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    """
    Reads csv file and checks if it is as per schema
    file_path: Path of input file
    schema_file_path: Path of schema file
    """
    try:
        datatset_schema = read_yaml_file(schema_file_path)

        schema = datatset_schema[SCHEMA_FILE_COLUMNS_KEY]

        dataframe = pd.read_csv(file_path)

        error_messgae = ""


        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_messgae = f"{error_messgae} \nColumn: [{column}] is not in the schema."
        if len(error_messgae) > 0:
            raise Exception(error_messgae)
        return dataframe

    except Exception as e:
        raise ConcreteException(e,sys) from e

def get_cluster(file_path: str)->int:
    """
    Returns the cluster associated with the model file.
    Model file format: 'model_cluster'+<cluster>+'.pkl'    
    """
    try:
        file_name = os.path.basename(file_path)
        return int(file_name[-5:-4])
        
    except Exception as e:
        raise ConcreteException(e,sys) from e

def create_list(size:int)->List:
    """
    Creates an empty list of size specified.
    Input:
    size: size of list
    """
    return [None]*size