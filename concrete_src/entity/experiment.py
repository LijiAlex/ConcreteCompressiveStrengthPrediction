from collections import namedtuple
from datetime import datetime
import os, sys
import pandas as pd
import uuid

from concrete_src.exception import ConcreteException
from concrete_src.constant import *

ExperimentDetails = namedtuple("ExperimentDetails", ["experiment_id", "initialization_timestamp", "artifact_time_stamp",
                                       "running_status", "start_time", "stop_time", "execution_time", "message",
                                       "experiment_file_path", "accuracy", "is_model_accepted"])

class Experiment:
    def __init__(self, artifact_dir, timestamp):
        self.current_experiment = None
        os.makedirs(artifact_dir, exist_ok=True)
        self.experiment_file_path=os.path.join(artifact_dir,EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
        self.timestamp = timestamp

    def get_experiment_details(self):
        return self.current_experiment

    def start_experiment(self, msg):
        experiment_id = str(uuid.uuid4())
        self.current_experiment = ExperimentDetails(experiment_id=experiment_id,
                                             initialization_timestamp=self.timestamp,
                                             artifact_time_stamp=self.timestamp,
                                             running_status=True,
                                             start_time=datetime.now(),
                                             stop_time=None,
                                             execution_time=None,
                                             experiment_file_path=self.experiment_file_path,
                                             is_model_accepted=None,
                                             message= msg,
                                             accuracy=None,
                                             )

    def stop_experiment(self, msg, is_model_accepted, model_accuracy):
        stop_time = datetime.now()
        self.current_experiment = ExperimentDetails(experiment_id=self.current_experiment.experiment_id,
                                             initialization_timestamp=self.timestamp,
                                             artifact_time_stamp=self.timestamp,
                                             running_status=False,
                                             start_time=self.current_experiment.start_time,
                                             stop_time=stop_time,
                                             execution_time=stop_time - self.current_experiment.start_time,
                                             message=msg,
                                             experiment_file_path=self.experiment_file_path,
                                             is_model_accepted=is_model_accepted,
                                             accuracy=model_accuracy
                                             )

    def save_experiment(self):
        try:
            if self.current_experiment.experiment_id is not None:
                experiment_dict = self.current_experiment._asdict()
                experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}

                experiment_dict.update({
                    "created_time_stamp": [datetime.now()],
                    "experiment_file_path": [os.path.basename(self.current_experiment.experiment_file_path)]})

                experiment_report = pd.DataFrame(experiment_dict)
                
                os.makedirs(os.path.dirname(self.experiment_file_path) , exist_ok=True)
                if os.path.exists(self.experiment_file_path):
                    experiment_report.to_csv(self.experiment_file_path, index=False, header=False, mode="a")
                else:
                    experiment_report.to_csv(self.experiment_file_path, mode="w", index=False, header=True)
            else:
                print("Please start experiment")
        except Exception as e:
            raise ConcreteException(e, sys) from e
