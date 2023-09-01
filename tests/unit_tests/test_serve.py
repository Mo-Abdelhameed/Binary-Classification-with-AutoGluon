import json
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serve import create_app
from src.serve_utils import get_model_resources
from src.train import run_training
from src.config import paths
from src.utils import read_json_as_dict


@pytest.fixture
def train_dir(sample_train_data, tmpdir, train_data_file_name):
    """Fixture to create and save a sample DataFrame for testing"""
    train_data_dir = tmpdir.mkdir("train")
    train_data_file_path = train_data_dir.join(train_data_file_name)
    sample_train_data.to_csv(train_data_file_path, index=False)
    return str(train_data_dir)


@pytest.fixture
def sample_data():
    """Fixture to create a larger sample DataFrame for testing"""
    np.random.seed(0)
    N = 100
    data = pd.DataFrame(
        {
            "id": range(1, N + 1),
            "numeric_feature_1": np.random.randint(1, 100, size=N),
            "numeric_feature_2": np.random.normal(0, 1, size=N),
            "categorical_feature_1": np.random.choice(["A", "B", "C"], size=N),
            "categorical_feature_2": np.random.choice(
                ["P", "Q", "R", "S", "T"], size=N
            ),
            "target_field": np.random.choice(["A", "B"], size=N),
        }
    )
    return data


@pytest.fixture
def sample_train_data(sample_data):
    """Fixture to create a larger sample DataFrame for testing"""
    N_train = int(len(sample_data) * 0.8)
    return sample_data.head(N_train)


@pytest.fixture
def sample_test_data(sample_data):
    """Fixture to create a larger sample DataFrame for testing"""
    N_test = int(len(sample_data) * 0.2)
    return sample_data.tail(N_test)


@pytest.fixture
def input_schema_file_name():
    return "schema.json"


@pytest.fixture
def train_data_file_name():
    return "train.csv"


@pytest.fixture
def input_schema_dir(schema_dict, tmpdir, input_schema_file_name):
    """Fixture to create and save a sample schema for testing"""
    schema_dir = tmpdir.mkdir("input_schema")
    schema_file_path = schema_dir.join(input_schema_file_name)
    with open(schema_file_path, "w") as file:
        json.dump(schema_dict, file)
    return str(schema_dir)


@pytest.fixture
def config_file_paths_dict():
    return read_json_as_dict(paths.MODEL_CONFIG_FILE_PATH)


@pytest.fixture
def saved_schema_dir():
    return paths.SAVED_SCHEMA_DIR_PATH


@pytest.fixture
def predictor_dir(tmpdir):
    return tmpdir.mkdir('predictor')



@pytest.fixture
def app(
        input_schema_dir: str,
        train_dir: str,
        saved_schema_dir: str,
        predictor_dir: str,
        config_file_paths_dict: dict,
):
    """
    Define a fixture for the test app.

    Args:
        input_schema_dir (str): Directory path to the input data schema.
        train_dir (str): Directory path to the training data.
        saved_schema_dir (str): Directory path to the saved schema.
        predictor_dir (str): Directory path to the predictor model.
        config_file_paths_dict (dict): Dictionary containing the paths to the
            configuration files.
    """

    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_dir_path=saved_schema_dir,
        train_dir=train_dir,
        predictor_dir_path=predictor_dir,
    )

    # create model resources dictionary
    model_resources = get_model_resources()

    # create test app
    return TestClient(create_app(model_resources))


def test_ping(app):
    """Test the /ping endpoint."""
    response = app.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "Pong!"}


@patch("src.serve.transform_req_data_and_make_predictions")
def test_infer_endpoint(mock_transform_and_predict, app, sample_request_data):
    """
    Test the infer endpoint.

    The function creates a mock request and sets the expected return value of the
    mock_transform_and_predict function.
    It then sends a POST request to the "/infer" endpoint with the mock request data.
    The function asserts that the response status code is 200 and the JSON response
    matches the expected output.
    Additionally, it checks if the mock_transform_and_predict function was called with
    the correct arguments.

    Args:
        mock_transform_and_predict (MagicMock): A mock of the
            transform_req_data_and_make_predictions function.
        app (TestClient): The TestClient fastapi app

    """
    # Define what your mock should return
    mock_transform_and_predict.return_value = pd.DataFrame(), {
        "status": "success",
        "predictions": [],
    }
    response = app.post("/infer", data=json.dumps(sample_request_data))

    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"status": "success", "predictions": []}
    # You can add more assertions to check if the function was called with the
    # correct arguments
    mock_transform_and_predict.assert_called()
