import yaml, sys

from concrete_src.exception import ConcreteException

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