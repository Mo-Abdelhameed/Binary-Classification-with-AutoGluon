from config import paths
from logger import get_logger
from Classifier import Classifier
from schema.data_schema import load_json_data_schema
from utils import read_csv_in_directory, save_dataframe_as_csv, ResourceTracker

logger = get_logger(task_name="predict")


def run_batch_predictions(
        test_dir: str = paths.TEST_DIR,
        predictor_dir: str = paths.PREDICTOR_DIR_PATH,
        predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
        return_proba=False
) -> None:
    """
        Run batch predictions on test data, save the predicted probabilities to a CSV file.

        This function reads test data from the specified directory,
        loads the preprocessing pipeline and pre-trained predictor model,
        transforms the test data using the pipeline,
        makes predictions using the trained predictor model,
        adds ids into the predictions dataframe,
        and saves the predictions as a CSV file.

        Args:
            test_dir (str): Path to the directory containing test data
            predictor_dir (str): Path to the directory containing the predictor model.
            predictions_file_path (str): Path in which to store the prediction file.
            return_proba (bool): If true, outputs the probabilities of the target classes.
        """
    with ResourceTracker(logger, monitoring_interval=0.1):
        x_test = read_csv_in_directory(test_dir)
        model = Classifier.load(predictor_dir)
        data_schema = load_json_data_schema(paths.INPUT_SCHEMA_DIR)
        ids = x_test[data_schema.id]
        logger.info("Making predictions...")
        predictions_df = Classifier.predict_with_model(model, x_test, return_proba=return_proba)
    if return_proba:
        predictions_df.insert(0, data_schema.id, ids)
    else:
        predictions_df[data_schema.id] = ids
    logger.info("Saving predictions...")
    save_dataframe_as_csv(
        dataframe=predictions_df, file_path=predictions_file_path
    )

    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions(return_proba=True)
