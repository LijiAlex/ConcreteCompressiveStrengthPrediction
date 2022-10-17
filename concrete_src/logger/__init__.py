import logging
from datetime import datetime
import os
from time import asctime

LOG_DIR = "development_logs"
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

os.makedirs(LOG_DIR, exist_ok= True)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    filemode= "a",
    format='[%(asctime)s] %(filename)s:%(lineno)d - %(levelname)s -%(message)s',
    level= logging.INFO
)


