from collections import namedtuple 

TrainingPipeLineConfig = namedtuple("TrainingPipeLineConfig", ["artifact_dir"])

DataIngestionConfig = namedtuple("DataIngestionConfig",
["dataset_name", "dataset_filename","raw_data_dir", "ingested_train_dir", "ingested_test_dir"])