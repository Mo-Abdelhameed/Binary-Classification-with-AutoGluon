import os
import numpy as np
import pandas as pd
import pytest
from autogluon.core import TabularDataset
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from Classifier import Classifier
from schema.data_schema import BinaryClassificationSchema
from config.paths import PREDICTIONS_FILE_PATH, PREDICTOR_DIR_PATH


@pytest.fixture
def schema_dict():
    """Define schema as dict"""
    schema_dict = {
        "title": "Exclusive-Or dataset",
        "description": "Synthetically generated 5-dimensional Exclusive-Or (XOR) dataset.",
        "modelCategory": "binary_classification",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {
            "name": "id",
            "description": "unique identifier"
        },
        "target": {
            "name": "target",
            "description": "binary class (0 or 1)",
            "classes": [
                "0",
                "1"
            ]
        },
        "features": [
            {
                "name": "x1",
                "description": "feature 1 of synthetic XOR data",
                "dataType": "NUMERIC",
                "example": 0.8179,
            },
            {
                "name": "x2",
                "description": "feature 2 of synthetic XOR data",
                "dataType": "NUMERIC",
                "example": 0.551,
            },
            {
                "name": "x3",
                "description": "feature 3 of synthetic XOR data",
                "dataType": "NUMERIC",
                "example": 0.4198,
            },
            {
                "name": "x4",
                "description": "feature 4 of synthetic XOR data",
                "dataType": "NUMERIC",
                "example": 0.0987,
            },
            {
                "name": "x5",
                "description": "feature 5 of synthetic XOR data",
                "dataType": "NUMERIC",
                "example": 0.811,
            }
        ],
    }
    return schema_dict


@pytest.fixture
def schema_provider(schema_dict):
    """Define schema as BinaryClassificationSchema"""
    return BinaryClassificationSchema(schema_dict)


@pytest.fixture
def xor_data(schema_provider):
    """Define the synthetic dataset fixture"""

    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x1': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'x2': [3, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        'x3': [4, 1, 0, 1, 2, 6, 7, 75, 66, 57],
        'x4': [77, 1, 0, 1, 2, 3, 84, 5, 6, 37],
        'x5': [66, 145, 131, 1, 82, 3, 54, 445, 62, 71],
        'target': [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    }
    return TabularDataset(pd.DataFrame(data))


@pytest.fixture
def synthetic_data():
    """Define the synthetic dataset fixture"""
    x, y = make_classification(n_samples=100, n_features=5, random_state=42)
    x = pd.DataFrame(x, columns=[f"feature_{i+1}" for i in range(x.shape[1])])
    y = pd.Series(y, name="target")
    train_x, train_y = x[:80], y[:80]
    test_x, test_y = x[80:], y[80:]
    return train_x, train_y, test_x, test_y


@pytest.fixture
def classifier(xor_data, schema_provider):
    """Define classifier fixture"""
    classifier = Classifier(xor_data, schema_provider)
    return classifier


def test_train_predict_model(xor_data, schema_provider):
    """
    Test if the classifier is created, trained and makes predictions.
    """
    classifier = Classifier(xor_data, schema_provider)
    classifier.train()
    assert os.path.exists(PREDICTIONS_FILE_PATH)

    predictions = classifier.predict(xor_data)
    assert predictions.shape == (xor_data.shape[0],)
    assert np.array_equal(predictions, predictions.astype(bool))

    proba_predictions = classifier.predict_proba(xor_data)
    assert proba_predictions.shape == (xor_data.shape[0], 2)

    classifier.save(PREDICTOR_DIR_PATH)
    assert os.path.exists(os.path.join(PREDICTOR_DIR_PATH, 'predictor.joblib'))


def test_save_load_model(tmpdir, classifier, xor_data):
    """
    Test if the save and load methods work correctly.
    """
    # Specify the file path
    model_dir_path = tmpdir.mkdir("model")
    classifier.train()
    # Save the model
    classifier.save(model_dir_path)

    # Load the model
    loaded_clf = Classifier.load(model_dir_path)

    # Test predictions
    predictions = loaded_clf.predict(xor_data)
    assert np.array_equal(predictions, classifier.predict(xor_data))

    proba_predictions = loaded_clf.predict_proba(xor_data)
    assert np.array_equal(proba_predictions, classifier.predict_proba(xor_data))


def test_classifier_str_representation(classifier):
    """
    Test the `__str__` method of the `Classifier` class.

    The test asserts that the string representation of a `Classifier` instance is
    correctly formatted and includes the model name.

    Args:
        classifier (Classifier): An instance of the `Classifier` class.

    Raises:
        AssertionError: If the string representation of `classifier` does not
        match the expected format.
    """
    classifier_str = str(classifier)

    assert classifier.model_name in classifier_str


def test_predict_with_model(xor_data, schema_provider):
    """
    Test that the 'predict_with_model' function returns predictions of correct size
    and type.
    """
    classifier = Classifier(xor_data, schema_provider)
    classifier.train()
    predictions = Classifier.predict_with_model(classifier, xor_data, return_proba=True)

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape[0] == xor_data.shape[0]


def test_save_predictor_model(tmpdir, xor_data, schema_provider):
    """
    Test that the 'save_predictor_model' function correctly saves a Classifier instance
    to disk.
    """
    model_dir_path = os.path.join(tmpdir, "model")
    classifier = Classifier(xor_data, schema_provider)
    classifier.train()
    Classifier.save_predictor_model(classifier, model_dir_path)
    assert os.path.exists(model_dir_path)
    assert len(os.listdir(model_dir_path)) >= 1


def test_untrained_save_predictor_model_fails(tmpdir, classifier):
    """
    Test that the 'save_predictor_model' function correctly raises  NotFittedError
    when saving an untrained classifier to disk.
    """
    with pytest.raises(NotFittedError):
        model_dir_path = os.path.join(tmpdir, "model")
        Classifier.save_predictor_model(classifier, model_dir_path)
