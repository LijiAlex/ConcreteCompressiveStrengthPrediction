import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

# common config
ROOT_DIR = os.getcwd() # to get current working directory
CURRENT_TIME_STAMP = get_current_time_stamp()

# config.yaml
CONFIG_DIR = 'config'
CONFIG_FILE = 'config.yaml'
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE)

# training pipeline config
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"

# data ingestion config
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATASET_NAME = "dataset_name"
DATASET_FILENAME = "dataset_file_name"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR = "raw_data_dir"
DATA_INGESTION_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR = "ingested_train_dir"

# data validation config
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_DIR = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME = "schema_file_name"

# data transformation config
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
TRANFORMED_DIR = "transformed_dir"
TRANSFORMED_TRAIN_DIR = "transformed_train_dir"
PREPROCESSING_DIR = "preprocessing_dir"
PREPROCESSED_OBJECT_FILE_NAME = "preprocessed_object_file_name"

# model trainer config
MODEL_TRAINER_ARTIFACT_DIR = "model_trainer"
MODEL_TRAINER_CONFIG_INFO = "model_trainer_config"
TRAINED_MODEL_DIR = "trained_model"
TRIANED_MODEL_FILENAME = "model_file_name"
BASE_ACCURACY = "base_accuracy"
MODEL_CONFIG_DIR = "model_config_dir"
MODEL_CONFIG_FILE_NAME = "model_config_file_name"

# model evaluation config
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_DIR = "model_evaluation"
MODEL_EVALUATION_FILE_NAME = "model_evaluation_file_name"

# model_evaluation.yaml
BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"

# model pusher config
MODEL_PUSH_CONFIG_INFO_KEY = "model_pusher_config"
MODEL_EXPORT_DIR = "model_export_dir"

# schema.yaml
SCHEMA_FILE_COLUMNS_KEY = "columns"
SCHEMA_FILE_TARGET_COLUMNS = "target_column"

# experiment.yaml
EXPERIMENT_DIR_NAME="experiment"
EXPERIMENT_FILE_NAME="experiment.csv"
