from collections import namedtuple 

TrainingPipeLineConfig = namedtuple("TrainingPipeLineConfig", ["artifact_dir"])

DataIngestionConfig = namedtuple("DataIngestionConfig", ["dataset_name", "dataset_filename","raw_data_dir", "ingested_train_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_train_dir", "preprocessed_object_file_path"])

ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_models_path", "base_accuracy", "model_config_file_path"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_evaluation_files_folder", "model_evaluation_file_prefix", "time_stamp"])

ModelPusherConfig = namedtuple("ModelPushConfig", ["export_dir_path"])