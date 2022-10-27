from concrete_src.logger import logging
from concrete_src.exception import ConcreteException
from concrete_src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact 
from concrete_src.entity.config_entity import ModelPusherConfig
from concrete_src.util.util import create_list

import os, sys
import shutil


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact
                 ):
        try:
            logging.info(f"\n{'>>' * 30}Model Pusher log started.{'<<' * 30} ")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise ConcreteException(e, sys) from e

    def export_model(self) -> ModelPusherArtifact:
        try:
            evaluated_model_file_paths = self.model_evaluation_artifact.evaluated_model_paths
            export_dir = self.model_pusher_config.export_dir_path
            is_models_pushed = create_list(len(self.model_evaluation_artifact.clusters))
            export_model_file_paths = create_list(len(self.model_evaluation_artifact.clusters))
            for cluster in self.model_evaluation_artifact.clusters:
                if self.model_evaluation_artifact.is_models_accepted[cluster]:
                    model_file_name = os.path.basename(evaluated_model_file_paths[cluster])
                    export_model_file_path = os.path.join(export_dir, model_file_name)
                    logging.debug(f"Exporting model file: [{export_model_file_path}]")
                    os.makedirs(export_dir, exist_ok=True)

                    shutil.copy(src=evaluated_model_file_paths[cluster], dst=export_model_file_path)
                    logging.info(
                        f"Cluster{cluster} Trained model: {evaluated_model_file_paths[cluster]} is copied in export dir:[{export_model_file_path}]")
                    is_models_pushed[cluster] = True
                    export_model_file_paths[cluster] = export_model_file_path
                else:
                    is_models_pushed[cluster] = False
                    logging.info(f"cluster{cluster} trained model rejected and hence not pushed.")

            model_pusher_artifact = ModelPusherArtifact(is_models_pushed = is_models_pushed,
                                                        export_model_file_paths = export_model_file_paths
                                                        )
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            return model_pusher_artifact
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise ConcreteException(e, sys) from e

    def __del__(self):
        logging.info(f"\n{'>>' * 20}Model Pusher log completed.{'<<' * 20} ")