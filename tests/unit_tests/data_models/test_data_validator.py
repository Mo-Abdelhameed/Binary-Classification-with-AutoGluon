from typing import Any

import pandas as pd
import pytest

from schema.data_schema import BinaryClassificationSchema
from src.data_models.data_validator import validate_data


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
                "nullable": False
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
def sample_train_data():
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x1': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'x2': [3, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        'x3': [4, 1, 0, 1, 2, 6, 7, 75, 66, 57],
        'x4': [77, 1, 0, 1, 2, 3, 84, 5, 6, 37],
        'x5': [66, 145, 131, 1, 82, 3, 54, 445, 62, 71],
        'target': [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data():
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x1': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'x2': [3, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        'x3': [4, 1, 0, 1, 2, 6, 7, 75, 66, 57],
        'x4': [77, 1, 0, 1, 2, 3, 84, 5, 6, 37],
        'x5': [66, 145, 131, 1, 82, 3, 54, 445, 62, 71],
        'target': [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)


def test_validate_data_correct_train_data(
    schema_provider: Any,
    sample_train_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with correct train data.

    The test ensures that when the input DataFrame is correctly formatted according
    to the schema and is used for training, no error is raised, and the returned
    DataFrame is identical to the input DataFrame.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                    which encapsulates the
                                                    data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame formatted
                                            correctly according to the schema.
    """
    try:
        result_train_data = validate_data(sample_train_data, schema_provider, True)
        # check if train DataFrame is unchanged
        pd.testing.assert_frame_equal(result_train_data, sample_train_data)
    except AssertionError as exc:
        pytest.fail(
            f"Returned DataFrame is not identical to the input DataFrame: {exc}"
        )


def test_validate_data_correct_test_data(
    schema_provider: Any,
    sample_test_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with correct test data.

    The test ensures that when the input DataFrame is correctly formatted according
    to the schema and is used for testing, no error is raised, and the returned
    DataFrame is identical to the input DataFrame.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                    which encapsulates the data
                                                    schema.
        sample_test_data (pd.DataFrame): A sample testing DataFrame formatted
                                        correctly according to the schema.
    """
    try:
        result_test_data = validate_data(sample_test_data, schema_provider, False)
        # check if test DataFrame is unchanged
        pd.testing.assert_frame_equal(result_test_data, sample_test_data)
    except AssertionError as exc:
        pytest.fail(
            f"Returned DataFrame is not identical to the input DataFrame: {exc}"
        )


def test_validate_data_missing_feature_column_train_data(
    schema_provider: Any,
    sample_train_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with missing feature column in train data.

    The test ensures that when a required feature column (according to the schema)
    is missing from the input DataFrame used for training, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame with a missing
                                          feature column.
    """
    missing_feature_data = sample_train_data.drop(columns=["x1"])
    with pytest.raises(ValueError):
        validate_data(missing_feature_data, schema_provider, True)


def test_validate_data_missing_feature_column_test_data(
    schema_provider: Any, sample_test_data: pd.DataFrame
):
    """
    Test the `validate_data` function with missing feature column in test data.

    The test ensures that when a required feature column (according to the schema)
    is missing from the input DataFrame used for testing, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_test_data (pd.DataFrame): A sample testing DataFrame with a missing
                                         feature column.
    """
    missing_feature_data = sample_test_data.drop(columns=["x1"])
    with pytest.raises(ValueError):
        validate_data(missing_feature_data, schema_provider, False)


def test_validate_data_missing_id_column_train_data(
    schema_provider: Any, sample_train_data: pd.DataFrame
):
    """
    Test the `validate_data` function with missing id column in train data.

    The test ensures that when the ID column (according to the schema) is missing
    from the input DataFrame used for training, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame with a missing
                                          id column.
    """
    missing_id_data = sample_train_data.drop(columns=["id"])
    with pytest.raises(ValueError):
        validate_data(missing_id_data, schema_provider, True)


def test_validate_data_missing_id_column_test_data(
    schema_provider: Any,
    sample_test_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with missing id column in test data.

    The test ensures that when the ID column (according to the schema) is missing
    from the input DataFrame used for testing, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_test_data (pd.DataFrame): A sample testing DataFrame with a missing
                                         id column.
    """
    missing_id_data = sample_test_data.drop(columns=["id"])
    with pytest.raises(ValueError):
        validate_data(missing_id_data, schema_provider, False)


def test_validate_data_missing_target_column_train_data(
    schema_provider: Any,
    sample_train_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with missing target column in train data.

    The test ensures that when the target column (according to the schema) is
    missing from the input DataFrame used for training, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame with a missing
                                          target column.
    """
    missing_target_data = sample_train_data.drop(columns=["target"])
    with pytest.raises(ValueError):
        validate_data(missing_target_data, schema_provider, True)


def test_validate_data_duplicate_ids_train_data(
    schema_provider: Any,
    sample_train_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with duplicate IDs in train data.

    The test ensures that when the ID column (according to the schema) contains
    duplicate values in the input DataFrame used for training, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame with
                                          duplicate IDs.
    """
    duplicate_id_data = sample_train_data.copy()
    duplicate_id_data = duplicate_id_data.append(
        duplicate_id_data.iloc[0], ignore_index=True
    )
    with pytest.raises(ValueError):
        validate_data(duplicate_id_data, schema_provider, True)


def test_validate_data_non_nullable_feature_contains_null_values(
    schema_provider: Any,
    sample_train_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with a non-nullable feature containing null
    values.

    The test ensures that when a non-nullable feature in the input DataFrame contains
    null values, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame with a
                                          non-nullable feature containing null values.
    """
    null_feature_data = sample_train_data.copy()
    null_feature_data.loc[0, "x1"] = None
    with pytest.raises(ValueError):
        validate_data(null_feature_data, schema_provider, True)


def test_validate_data_numeric_feature_contains_non_numeric_value(
    schema_provider: Any,
    sample_train_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with a numeric feature containing non-numeric
    values.

    The test ensures that when a numeric feature in the input DataFrame contains
    non-numeric values, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame with a
                                        numeric feature containing non-numeric values.
    """
    non_numeric_feature_data = sample_train_data.copy()
    non_numeric_feature_data.loc[0, "x1"] = "non-numeric"
    with pytest.raises(ValueError):
        validate_data(non_numeric_feature_data, schema_provider, True)


def test_validate_data_all_target_classes_present_in_target_column(
    schema_provider: Any,
    sample_train_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with the training data's target column not
    containing all target classes.

    The test ensures that when the training data's target column does not contain all
    target classes specified in the schema, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame with target
                                        column missing one or more target classes.
    """
    incomplete_target_class_data = sample_train_data.copy()
    incomplete_target_class_data[
        schema_provider.target
    ] = schema_provider.target_classes[0]

    with pytest.raises(ValueError):
        validate_data(incomplete_target_class_data, schema_provider, True)


def test_validate_data_unexpected_target_classes(
    schema_provider: Any,
    sample_train_data: pd.DataFrame,
):
    """
    Test the `validate_data` function with unexpected target classes in the target
    column.

    The test ensures that when the target column in the input DataFrame contains
    classes not defined in the schema, a ValueError is raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance
                                                which encapsulates the data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame with an
                                          unexpected target class.
    """
    unexpected_target_class_data = sample_train_data.copy()
    unexpected_target_class_data.loc[0, schema_provider.target] = "unexpected_class"

    with pytest.raises(ValueError):
        validate_data(unexpected_target_class_data, schema_provider, True)