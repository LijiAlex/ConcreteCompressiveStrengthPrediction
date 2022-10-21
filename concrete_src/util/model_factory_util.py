import importlib, sys

from concrete_src.logger import logging
from concrete_src.exception import ConcreteException

def class_for_name(module_name:str, class_name:str):
    """
    Load the class mentioned in the name supplied.
    Input:
    module_name: Module in which class is present
    class_name: Name of the class
    """
    try:
        # load the module, will raise ImportError if module cannot be loaded
        module = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        logging.debug(f"Executing command: from {module} import {class_name}")
        class_ref = getattr(module, class_name)
        return class_ref
    except Exception as e:
        raise ConcreteException(e, sys) from e

def update_property_of_class(instance_ref:object, property_data: dict):
    """
    Add attribute values to the class object
    Input:
    instance_ref : object
    property_data: attribute

    """
    try:
        if not isinstance(property_data, dict):
            raise Exception(f"property_data parameter required to dictionary: {property_data}")
        for key, value in property_data.items():
            logging.debug(f"Add property {str(instance_ref)}.{key}={value}")
            setattr(instance_ref, key, value)
        return instance_ref
    except Exception as e:
        raise ConcreteException(e, sys) from e