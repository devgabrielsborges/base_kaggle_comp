import os

import mlflow
from dotenv import load_dotenv

load_dotenv()


def init_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    competition = os.getenv("KAGGLE_COMPETITION_NAME", "default")
    target = os.getenv("TARGET_COLUMN", "target")
    experiment_name = f"{competition}/{target}"

    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()