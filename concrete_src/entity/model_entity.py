from collections import namedtuple

ModelInitializationDetail = namedtuple("ModelInitializationDetail",
                                    ["model_serial_number", "model", "param_grid", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score",
                                                             ])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])

ClusterModelDetails = namedtuple("ClusterModelDetails", ["best_model", "grid_searched_best_model_list"])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "train_accuracy",
                                 "model_accuracy", "index_number"])