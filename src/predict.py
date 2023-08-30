from config import paths
from logger import get_logger
from Classifier import Classifier
from utils import read_csv_in_directory, save_dataframe_as_csv

logger = get_logger(task_name="predict")


def run_batch_predictions() -> None:
    """
        Run batch predictions on test data, save the predicted probabilities to a CSV file.

        This function reads test data from the specified directory,
        loads the preprocessing pipeline and pre-trained predictor model,
        transforms the test data using the pipeline,
        makes predictions using the trained predictor model,
        adds ids into the predictions dataframe,
        and saves the predictions as a CSV file.
        """
    x_test = read_csv_in_directory(paths.TEST_DIR)
    model = Classifier.load(paths.PREDICTOR_DIR_PATH)

    logger.info("Making predictions...")
    predictions_df = Classifier.predict_with_model(model, x_test, return_proba=True)

    logger.info("Saving predictions...")
    save_dataframe_as_csv(
        dataframe=predictions_df, file_path=paths.PREDICTIONS_FILE_PATH
    )

    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions()
