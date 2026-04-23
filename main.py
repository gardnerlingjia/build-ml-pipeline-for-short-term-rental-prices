import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config', config_path='.')  # Adding version_base for Python 3.13 compatibility
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                uri=os.path.join(get_original_cwd(), config["main"]["components_repository"], "get_data"),
                entry_point="main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": config["etl"]["artifact_name"],
                    "artifact_type": config["etl"]["artifact_type"],
                    "artifact_description": config["etl"]["artifact_description"],
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_name": config["basic_cleaning"]["input_name"],
                    "output_name": config["basic_cleaning"]["output_name"],
                    "output_type": config["basic_cleaning"]["output_type"],
                    "output_description": config["basic_cleaning"]["output_description"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )


        if "data_check" in active_steps:
            mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": config["data_check"]["csv"],
                    "ref": config["data_check"]["ref"],
                    "kl_threshold": str(config["data_check"]["kl_threshold"]),
                    "min_price": str(config["etl"]["min_price"]),
                    "max_price": str(config["etl"]["max_price"]),
                },
            )

        if "data_split" in active_steps:
            mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "train_val_test_split"),
                "main",
                parameters={
                    "input": config["data_split"]["input"],
                    "test_size": str(config["modeling"]["test_size"]),
                    "random_seed": str(config["modeling"]["random_seed"]),
                    "stratify_by": config["data_split"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": config["data_split"]["trainval_artifact"],
                    "val_size": str(config["modeling"]["val_size"]),
                    "random_seed": str(config["modeling"]["random_seed"]),
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": str(config["modeling"]["max_tfidf_features"]),
                    "output_artifact": config["random_forest"]["output_artifact"],
                },
            )


        if "test_regression_model" in active_steps:

            mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "test_regression_model"),
                "main",
                parameters={
                    "mlflow_model": config["test_regression_model"]["mlflow_model"],
                    "test_dataset": config["test_regression_model"]["test_dataset"],
                },
            )

if __name__ == "__main__":
    go()
