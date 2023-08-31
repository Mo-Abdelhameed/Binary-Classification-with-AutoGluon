import json
import os
import time
import tracemalloc
from typing import Dict, Tuple, Union

import pandas as pd
import pytest

from predict import run_batch_predictions
from tests.performance_tests.performance_test_helpers import (
    delete_dir_if_exists,
    delete_file_if_exists,
    generate_schema_and_data,
    set_seeds_for_data_gen,
    store_results_to_csv,
)
from train import run_training

DATASET_ROWS_LIST = [200, 2000]  # update to [200, 2000, 20000] for full test
DATASET_COLUMNS_LIST = [2, 20]  # update to [2, 20, 200] for full test
DATASET_SIZES = [
    (rows, columns) for rows in DATASET_ROWS_LIST for columns in DATASET_COLUMNS_LIST
]


def store_schema_and_data(
    tmpdir, schema_dict: Dict, sample_data: pd.DataFrame
) -> Tuple:
    """Stores the schema and data to tmpdir for testing.

    Args:
        tmpdir: Temporary directory for storing the schema and data provided by pytest.
        schema_dict (Dict): The generated schema.
        sample_data (pd.DataFrame): The generated data.

    Returns:
        Tuple: Contains the input_schema_dir and train_dir.
    """
    # save schema to dir in tmpdir from where the trainer will read it
    input_schema_dir_path = str(tmpdir.join("input_schema"))
    delete_dir_if_exists(input_schema_dir_path)
    input_schema_dir = tmpdir.mkdir("input_schema")
    schema_file_path = input_schema_dir.join("schema.json")
    with open(schema_file_path, "w", encoding="utf-8") as file:
        json.dump(schema_dict, file)

    # save data to dir in tmpdir from where trainer will read it
    train_dir_path = str(tmpdir.join("train"))
    delete_dir_if_exists(train_dir_path)
    train_dir = tmpdir.mkdir("train")
    train_data_file_path = train_dir.join("train.csv")
    sample_data.to_csv(train_data_file_path, index=False)

    return input_schema_dir, train_dir


def run_training_and_record(
    tmpdir,
    input_schema_dir: str,
    train_dir: str,
) -> Dict[str, Union[str, float]]:
    """Run training process and record the memory usage and execution time.

    Args:
        tmpdir: Temporary directory provided by pytest.
        input_schema_dir (str): Path to input schema directory.
        train_dir (str): Path to training data directory.

    Returns:
        Dict[str, Union[str, float]]: Dictionary containing paths of output files and
        execution metrics (time and memory).
    """
    # Create temporary paths for output files/artifacts
    saved_schema_dir_path = str(tmpdir.join("saved_schema"))
    predictor_dir_path = str(tmpdir.join("predictor"))

    # Start recording
    start_time = time.perf_counter()
    tracemalloc.start()

    # Run the training process with tuning
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_dir_path=saved_schema_dir_path,
        train_dir=train_dir,
        predictor_dir_path=predictor_dir_path,
    )

    # Stop recording
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()

    # Calculate memory in MB
    memory = peak / 10**6

    tracemalloc.stop()

    result_dict = {
        "saved_schema_dir_path": saved_schema_dir_path,
        "predictor_dir_path": predictor_dir_path,
        "training_time": end_time - start_time,
        "training_memory": memory,
    }

    return result_dict


def run_prediction_and_record(
    tmpdir,
    predictor_dir_path: str,
    test_dir: str,
) -> Tuple[str, float, float]:
    """Run prediction process and record the memory usage and execution time.

    Args:
        tmpdir: Temporary directory.
        predictor_dir_path (str): Path to predictor file.
        test_dir (str): Path to testing data directory.

    Returns:
        Tuple[str, float, float]: Tuple containing the path of prediction file,
        execution time and memory usage.
    """

    # Create temporary paths for prediction
    predictions_file_path = str(tmpdir.join("predictions.csv"))

    # Start recording - for prediction
    start_time = time.perf_counter()
    tracemalloc.start()

    # Run the prediction process
    run_batch_predictions(
        test_dir=test_dir,  # re-using the train data for prediction
        predictor_dir=predictor_dir_path,
        predictions_file_path=predictions_file_path,
        return_proba=True
    )

    # Stop recording
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()

    # Calculate memory in MB
    memory = peak / 10**6

    tracemalloc.stop()

    return predictions_file_path, end_time - start_time, memory


@pytest.mark.slow
def test_train_predict_performance(
    tmpdir,
    train_predict_perf_results_path: str,
):
    """Test the training and prediction workflows while recording performance.

    This function runs the training and prediction workflows with various dataset sizes
    and records the memory usage and execution time. The function also asserts that the
    model artifacts and predictions are saved in the correct paths and that the number
    of rows in the predictions matches the number of rows in the test data.

    Args:
        tmpdir: Temporary directory.
        train_predict_perf_results_path (str): Path to the file where training and
        prediction performance results will be stored.
    """
    # If the results file already exists, delete it
    delete_file_if_exists(train_predict_perf_results_path)

    for num_rows, num_features in DATASET_SIZES:

        # set seeds for data reproducibility
        set_seeds_for_data_gen()

        # Generate sample data
        schema_dict, sample_data = generate_schema_and_data(num_rows, num_features)

        # save schema dict and train data from where the training workflow will
        # read them
        input_schema_dir, train_dir = store_schema_and_data(
            tmpdir, schema_dict, sample_data
        )

        # run training workflow and record metrics
        results = run_training_and_record(
            tmpdir,
            input_schema_dir,
            train_dir,
        )

        # Assert that the model artifacts are saved in the correct paths
        assert len(os.listdir(results["saved_schema_dir_path"])) >= 1
        assert len(os.listdir(results["predictor_dir_path"])) >= 1

        # Store training workflow performance metrics
        store_results_to_csv(
            train_predict_perf_results_path,
            ("task", "num_rows", "num_features", "exec_time_secs", "memory_usage_mb"),
            (
                "train",
                num_rows,
                num_features,
                round(results["training_time"], 4),
                round(results["training_memory"], 4),
            ),
        )

        # run prediction workflow and record metrics
        (
            predictions_file_path,
            prediction_time,
            prediction_memory,
        ) = run_prediction_and_record(
            tmpdir,
            results["predictor_dir_path"],
            train_dir,
        )

        # Assert that the predictions file is saved in the correct path
        assert os.path.isfile(predictions_file_path)

        # Load predictions and validate the format
        predictions_df = pd.read_csv(predictions_file_path)

        # Assert that the number of rows in the predictions matches the number of rows
        # in sample_data
        assert len(predictions_df) == len(sample_data)

        # Store prediction workflow performance metrics
        store_results_to_csv(
            train_predict_perf_results_path,
            ("task", "num_rows", "num_features", "exec_time_secs", "memory_usage_mb"),
            (
                "batch_prediction",
                num_rows,
                num_features,
                round(prediction_time, 4),
                round(prediction_memory, 4),
            ),
        )
