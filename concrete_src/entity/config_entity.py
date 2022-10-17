from collections import namedtuple 

TrainingPipeLineConfig = namedtuple("TrainingPipeLineConfig", ["artifact_dir"])

DataIngestionConfig = namedtuple("DataIngestionConfig", ["dataset_name", "dataset_filename","raw_data_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path"])
