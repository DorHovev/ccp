from datetime import datetime
import time # For job duration calculation
import os
from batch_processor import config
from batch_processor.   database import DatabaseManager
from batch_processor.preprocessing import DataPreprocessor
from batch_processor.prediction import ModelPredictor
from batch_processor.monitoring import (
    logger, push_metrics_to_gateway, record_error,
    BATCH_JOB_LAST_SUCCESS_TIMESTAMP, BATCH_JOB_DURATION_SECONDS,
    ROWS_AFTER_PREPROCESSING_TOTAL, JOB_START_TIMESTAMP, FILES_PROCESSED_TOTAL, ROWS_PROCESSED_PER_FILE, CUSTOM_ERROR_TYPE_TOTAL
)

class BatchPredictionJob:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.preprocessor = DataPreprocessor()
        self.predictor = ModelPredictor()
        self.start_time = None

    def _load_and_prepare_data(self, reprocess_all=False):
        """Loads data from CSVs, saves to DB, then fetches and preprocesses for prediction."""
        total_new_rows_loaded = 0
        files_processed = 0
        rows_per_file = {}
        self.db_manager.create_tables_if_not_exist()
        
        for csv_file in config.INPUT_CSV_FILES:
            if not os.path.exists(csv_file):
                logger.warning(f"Input CSV file {csv_file} not found in batch_processor directory. Skipping.")
                record_error("csv_file_missing_at_job_run", f"File: {csv_file}")
                CUSTOM_ERROR_TYPE_TOTAL.labels(custom_error_type="csv_file_missing").inc()
                continue
            
            inserted, skipped, missing_id, conversion_errors = self.db_manager.load_csv_to_db(csv_file)
            total_new_rows_loaded += inserted
            files_processed += 1
            rows_per_file[csv_file] = inserted
            ROWS_PROCESSED_PER_FILE.labels(csv_file=os.path.basename(csv_file)).set(inserted)
            if missing_id > 0:
                logger.warning(f"{missing_id} rows skipped from {csv_file} due to missing customerID.")
                # Metric already incremented in db_manager.load_csv_to_db
            if conversion_errors > 0:
                logger.warning(f"{conversion_errors} rows from {csv_file} had data conversion issues.")
        
        FILES_PROCESSED_TOTAL.set(files_processed)
        logger.info(f"Total new rows loaded from all CSVs: {total_new_rows_loaded}")

        # Fetch and preprocess data; pass reprocess_all flag
        unprocessed_df = self.db_manager.fetch_data_for_preprocessing(reprocess_all=reprocess_all)

        if unprocessed_df is None or unprocessed_df.empty:
            logger.info("No new or unprocessed data found in the database for prediction.")
            if total_new_rows_loaded > 0:
                 logger.warning("CSV data was loaded, but no corresponding new data found for preprocessing. Check IDs or logic.")
                 record_error("data_pipeline_discrepancy", "New CSV data loaded but not found for preprocessing")
            return None, None # No data to process

        customer_ids = unprocessed_df['customerid'].tolist()
        # Pass only the necessary raw features for preprocessing
        # The preprocessor will select and create the final model features
        features_to_preprocess = unprocessed_df[config.RAW_FEATURES_FOR_PREPROCESSING].drop(columns=['customerid'])
        
        processed_df = self.preprocessor.preprocess(features_to_preprocess)
        
        if not processed_df.empty:
            ROWS_AFTER_PREPROCESSING_TOTAL.set(len(processed_df))
        else:
            ROWS_AFTER_PREPROCESSING_TOTAL.set(0)
            if not unprocessed_df.empty(): # We had data, but preprocessing resulted in empty df
                 logger.warning("Preprocessing resulted in an empty DataFrame. Check preprocessing logic.")
                 record_error("preprocessing_empty_output", "Data existed but preprocessing yielded no rows")

        return processed_df, customer_ids

    def run(self, reprocess_all=False):
        self.start_time = datetime.now()
        JOB_START_TIMESTAMP.set(time.time())
        logger.info(f"Batch prediction job started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        processed_df = None # Initialize to handle early exit
        try:
            processed_df, customer_ids = self._load_and_prepare_data(reprocess_all=reprocess_all)

            if processed_df is None or processed_df.empty or not customer_ids:
                logger.info("No data to predict after loading and preprocessing. Job will conclude.")
                BATCH_JOB_LAST_SUCCESS_TIMESTAMP.set(time.time()) # Considered a success if no data
            else:
                logger.info(f"Proceeding to make predictions for {len(customer_ids)} customers.")
                predictions, probabilities = self.predictor.predict(processed_df)

                if predictions and probabilities and len(predictions) == len(customer_ids):
                    self.db_manager.persist_predictions(customer_ids, processed_df, predictions, probabilities)
                    BATCH_JOB_LAST_SUCCESS_TIMESTAMP.set(time.time())
                    logger.info("Batch job completed successfully.")
                elif not predictions and not probabilities and self.predictor.model is None:
                    # This case means model loading failed, error already recorded by ModelPredictor
                    logger.critical("Model was not loaded. Cannot persist predictions.")
                    # Error already recorded by ModelPredictor, no need to call record_error here again
                else:
                    logger.error("Prediction step did not return expected results or lengths mismatch. Cannot persist.")
                    record_error("prediction_output_mismatch", f"Preds: {len(predictions)}, IDs: {len(customer_ids)}")

        except ConnectionError as ce:
            logger.critical(f"Database connection error during job execution: {ce}")
            record_error("job_db_connection_error", str(ce))
            CUSTOM_ERROR_TYPE_TOTAL.labels(custom_error_type="db_connection_error").inc()
        except Exception as e:
            logger.critical(f"Unhandled exception during batch job execution: {e}", exc_info=True)
            record_error("job_unhandled_exception", str(e))
            CUSTOM_ERROR_TYPE_TOTAL.labels(custom_error_type="unhandled_exception").inc()
        finally:
            if self.start_time:
                job_duration = (datetime.now() - self.start_time).total_seconds()
                BATCH_JOB_DURATION_SECONDS.observe(job_duration)
                logger.info(f"Batch job finished. Duration: {job_duration:.2f} seconds.")
            push_metrics_to_gateway() 