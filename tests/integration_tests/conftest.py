import os
import pandas as pd
import numpy as np
import json
import pytest

from src.schema.data_schema import BinaryClassificationSchema
from src.serve_utils import get_model_resources
from src.train import run_training
from fastapi.testclient import TestClient
from src.serve import create_app


@pytest.fixture
def schema_dict():
    """Fixture to create a sample schema for testing"""
    valid_schema = {
        "title": "test dataset",
        "description": "test dataset",
        "modelCategory": "binary_classification",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "id", "description": "unique identifier."},
        "target": {
            "name": "target_field",
            "description": "some target desc.",
            "classes": ["A", "B"],
        },
        "features": [
            {
                "name": "numeric_feature_1",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 50,
                "nullable": True,
            },
            {
                "name": "numeric_feature_2",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 0.5,
                "nullable": False,
            },
            {
                "name": "categorical_feature_1",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["A", "B", "C"],
                "nullable": True,
            },
            {
                "name": "categorical_feature_2",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["P", "Q", "R", "S", "T"],
                "nullable": False,
            },
        ],
    }
    return valid_schema


@pytest.fixture
def test_data_file_name():
    return "test.csv"


@pytest.fixture
def test_dir(sample_test_data, tmpdir, test_data_file_name):
    """Fixture to create and save a sample DataFrame for testing"""
    test_data_dir = tmpdir.mkdir("test")
    test_data_file_path = test_data_dir.join(test_data_file_name)
    sample_test_data.to_csv(test_data_file_path, index=False)
    return str(test_data_dir)


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
def schema_dict():
    """Fixture to create a sample schema for testing"""
    valid_schema = {
        "title": "test dataset",
        "description": "test dataset",
        "modelCategory": "binary_classification",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "id", "description": "unique identifier."},
        "target": {
            "name": "target_field",
            "description": "some target desc.",
            "classes": ["A", "B"],
        },
        "features": [
            {
                "name": "numeric_feature_1",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 50,
                "nullable": True,
            },
            {
                "name": "numeric_feature_2",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 0.5,
                "nullable": False,
            },
            {
                "name": "categorical_feature_1",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["A", "B", "C"],
                "nullable": True,
            },
            {
                "name": "categorical_feature_2",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["P", "Q", "R", "S", "T"],
                "nullable": False,
            },
        ],
    }
    return valid_schema


@pytest.fixture
def schema_provider(schema_dict):
    """Fixture to create a sample schema for testing"""
    return BinaryClassificationSchema(schema_dict)


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
def performance_test_results_dir_path():
    tests_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir_path = os.path.join(tests_dir_path, "test_results", "performance_tests")
    return results_dir_path


@pytest.fixture
def train_predict_perf_results_path(performance_test_results_dir_path):
    file_path = os.path.join(
        performance_test_results_dir_path, "train_predict_performance_results.csv"
    )
    return str(file_path)


@pytest.fixture
def inference_apis_perf_results_path(performance_test_results_dir_path):
    file_path = os.path.join(
        performance_test_results_dir_path, "inference_api_performance_results.csv"
    )
    return str(file_path)


@pytest.fixture
def docker_img_build_perf_results_path(performance_test_results_dir_path):
    file_path = os.path.join(
        performance_test_results_dir_path, "docker_img_build_performance_results.csv"
    )
    return str(file_path)


@pytest.fixture
def resources_paths_dict(test_resources_dir_path):
    """Define a fixture for the paths to the test model resources."""
    return {
        "saved_schema_dir_path": os.path.join(test_resources_dir_path, "schema"),
        "predictor_dir_path": os.path.join(test_resources_dir_path, "predictor"),
    }


@pytest.fixture
def test_resources_dir_path(tmpdir):
    """Define a fixture for the path to the test_resources directory."""
    tmpdir.mkdir("test_resources")
    test_resources_path = os.path.join(tmpdir, "test_resources")
    return test_resources_path


@pytest.fixture
def sample_request_data(schema_dict):
    # Define a fixture for test request data
    sample_dict = {
        # made up id for this test
        schema_dict["id"]["name"]: "42",
    }
    for feature in schema_dict["features"]:
        if feature["dataType"] == "NUMERIC":
            sample_dict[feature["name"]] = feature["example"]
        elif feature["dataType"] == "CATEGORICAL":
            sample_dict[feature["name"]] = feature["categories"][0]
    return {"instances": [{**sample_dict}]}


@pytest.fixture
def sample_response_data(schema_dict):
    # Define a fixture for expected response
    return {
        "status": "success",
        "message": "",
        "timestamp": "...varies...",
        "requestId": "...varies...",
        "targetClasses": schema_dict["target"]["classes"],
        "targetDescription": schema_dict["target"]["description"],
        "predictions": [
            {
                "sampleId": "42",
                # unknown because we don't know the predicted class
                "predictedClass": "unknown",
                # predicted probabilities are made up for this test
                "predictedProbabilities": [0.5, 0.5],
            }
        ],
    }


@pytest.fixture
def app(
    input_schema_dir,
    train_dir,
    resources_paths_dict: dict,
):
    """
    Define a fixture for the test app.

    Args:
        input_schema_dir (str): Directory path to the input data schema.
        train_dir (str): Directory path to the training data.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.
    """

    # Create temporary paths for all outputs/artifacts
    saved_schema_dir_path = resources_paths_dict["saved_schema_dir_path"]
    predictor_dir_path = resources_paths_dict["predictor_dir_path"]

    # Run the training process
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_dir_path=saved_schema_dir_path,
        train_dir=train_dir,
        predictor_dir_path=predictor_dir_path,
    )

    # create model resources dictionary
    model_resources = get_model_resources(**resources_paths_dict)

    # create test app
    return TestClient(create_app(model_resources))
