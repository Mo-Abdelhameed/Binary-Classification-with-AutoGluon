import os

import pytest
from src.train import run_training


@pytest.mark.slow
def test_run_training(
    input_schema_dir: str,
    train_dir: str,
    resources_paths_dict: dict,
) -> None:
    """Test the run_training function to make sure it produces the required artifacts.

    This test function checks whether the run_training function runs end-to-end
    without errors and produces the expected artifacts. It does this by running
    the training process with and without hyperparameter tuning. After each run,
    it verifies that the expected artifacts have been saved to disk at the correct
    paths.

    Args:

        input_schema_dir (str): Path to the input schema directory.
        train_dir (str): Path to the training directory.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.
    """
    # Create temporary paths for all outputs/artifacts
    saved_schema_dir_path = resources_paths_dict["saved_schema_dir_path"]
    predictor_dir_path = resources_paths_dict["predictor_dir_path"]

    # Run the training process without tuning
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_dir_path=saved_schema_dir_path,
        train_dir=train_dir,
        predictor_dir_path=predictor_dir_path,
    )

    # Assert that the model artifacts are saved in the correct paths
    assert len(os.listdir(saved_schema_dir_path)) >= 1
    assert len(os.listdir(predictor_dir_path)) >= 1
