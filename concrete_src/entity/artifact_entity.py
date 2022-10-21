from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact", [ "train_file_path", "is_ingested", "message" ])

DataValidationArtifact = namedtuple("DataValidationArtifact", ["schema_file_path", "is_validated", "message" ])

DataTransformationArtifact = namedtuple("DataTransformationArtifact", ["transformed_train_file_path", "preprocessed_object_file_path", "is_transformed", "message" ])

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", ["is_trained", "message", "clusters" ,"trained_models_file_path",
                                                           "train_rmse", "train_accuracy", "model_accuracy"])
