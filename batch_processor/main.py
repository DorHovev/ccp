import pandas as pd
import joblib
import psycopg2
import os
from datetime import datetime
import logging # For better logging
from batch_processor.job import BatchPredictionJob
from batch_processor.monitoring import logger, push_metrics_to_gateway, record_error, BATCH_JOB_LAST_SUCCESS_TIMESTAMP, BATCH_JOB_DURATION_SECONDS
import sys

# --- Logging Setup --- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database Configuration --- #
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/mlops_db")
MODEL_PATH = os.getenv("MODEL_PATH", "./churn_model.pickle") # Default path for the model
INPUT_CSV_FILES_STR = os.getenv("INPUT_CSV_FILES", "database_input.csv,database_input.csv,database_input2.csv")
INPUT_CSV_FILES = [f.strip() for f in INPUT_CSV_FILES_STR.split(',') if f.strip()]

# --- Helper Functions --- #
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DB_URL)
        logger.info("Successfully connected to the database.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def create_tables_if_not_exist(conn):
    """Creates necessary tables if they don't already exist."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_data (
                id SERIAL PRIMARY KEY,
                csv_row_index INTEGER, -- Original index from CSV, can be part of a composite key with data_source
                customerID VARCHAR(255) NOT NULL, -- Assuming customerID is a key identifier
                gender VARCHAR(10),
                SeniorCitizen INTEGER, -- 0 or 1
                Partner VARCHAR(3), -- Yes/No
                Dependents VARCHAR(3), -- Yes/No
                tenure INTEGER,
                PhoneService VARCHAR(3), -- Yes/No
                MultipleLines VARCHAR(20), -- Yes/No/No phone service
                InternetService VARCHAR(20), -- DSL/Fiber optic/No
                OnlineSecurity VARCHAR(20), -- Yes/No/No internet service
                OnlineBackup VARCHAR(20),
                DeviceProtection VARCHAR(20),
                TechSupport VARCHAR(20),
                StreamingTV VARCHAR(20),
                StreamingMovies VARCHAR(20),
                Contract VARCHAR(20), -- Month-to-month/One year/Two year
                PaperlessBilling VARCHAR(3), -- Yes/No
                PaymentMethod VARCHAR(50),
                MonthlyCharges FLOAT,
                TotalCharges FLOAT, -- This might have spaces or be empty
                data_source VARCHAR(255), -- To track which CSV file it came from
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (customerID) -- Ensuring customerID is unique across all sources
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS churn_predictions (
                id SERIAL PRIMARY KEY,
                customerID VARCHAR(255) REFERENCES customer_data(customerID), -- Foreign Key
                prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                TotalCharges FLOAT,
                "Contract_Month-to-month" INTEGER, -- Quoted because of hyphen
                "Contract_One year" INTEGER,
                "Contract_Two year" INTEGER,
                PhoneService INTEGER,
                tenure INTEGER,
                churn_prediction INTEGER, -- 0 for No Churn, 1 for Churn
                churn_probability_churn FLOAT,
                churn_probability_no_churn FLOAT,
                model_version VARCHAR(50)
            );
        """)
        conn.commit()
        logger.info("Tables checked/created successfully.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        conn.rollback()
    finally:
        cursor.close()

def load_csv_to_db(conn, csv_filepath):
    """Loads data from a CSV file into the customer_data table, avoiding duplicates based on customerID."""
    cursor = conn.cursor()
    rows_inserted_count = 0
    rows_skipped_duplicates_count = 0
    rows_with_missing_customerid_count = 0
    rows_failed_conversion_count = 0
    try:
        try:
            df = pd.read_csv(csv_filepath)
        except FileNotFoundError:
            logger.error(f"CSV file not found at {csv_filepath}. Skipping.")
            return 0, 0, 0, 0 # inserted, skipped, missing_id, failed_conversion
        except pd.errors.EmptyDataError:
            logger.warning(f"CSV file {csv_filepath} is empty. Skipping.")
            return 0, 0, 0, 0
        except Exception as e:
            logger.error(f"Failed to read CSV {csv_filepath}: {e}")
            return 0, 0, 0, 0

        df.rename(columns=lambda x: x.strip(), inplace=True)
        # Simplified column name cleaning for this context, assuming fairly clean headers
        # In a more complex scenario, explicit mapping might be better.

        if 'customerID' not in df.columns:
            logger.critical(f"Critical Error: 'customerID' column not found in {csv_filepath}. Cannot process this file.")
            return 0, 0, 0, 0

        for index, row in df.iterrows():
            customer_id = row.get('customerID')
            if pd.isna(customer_id) or str(customer_id).strip() == "":
                logger.warning(f"Skipping row {index+2} from {csv_filepath} due to missing or empty customerID.")
                rows_with_missing_customerid_count += 1
                continue

            cursor.execute("SELECT customerID FROM customer_data WHERE customerID = %s", (str(customer_id),))
            if cursor.fetchone():
                rows_skipped_duplicates_count += 1
                continue
            
            total_charges_val = row.get('TotalCharges')
            if isinstance(total_charges_val, str):
                total_charges_val = total_charges_val.strip()
                if total_charges_val == "":
                    total_charges_val = None 
                else:
                    try:
                        total_charges_val = float(total_charges_val)
                    except ValueError:
                        logger.warning(f"Could not convert TotalCharges '{total_charges_val}' to float for customer {customer_id}. Setting to NULL.")
                        total_charges_val = None
                        rows_failed_conversion_count +=1
            elif pd.isna(total_charges_val):
                total_charges_val = None

            try:
                # Ensure all fields from the CSV that match the table are included
                # Using .get() with default None for safety, though DB schema might not allow all NULLs
                values = (
                    row.get('Unnamed: 0', index), # Assuming the first unnamed column is an index from CSV
                    str(customer_id),
                    row.get('gender'), 
                    int(row.get('SeniorCitizen', 0)) if pd.notna(row.get('SeniorCitizen')) else None,
                    row.get('Partner'), 
                    row.get('Dependents'), 
                    int(row.get('tenure', 0)) if pd.notna(row.get('tenure')) else None,
                    row.get('PhoneService'), 
                    row.get('MultipleLines'), 
                    row.get('InternetService'),
                    row.get('OnlineSecurity'), 
                    row.get('OnlineBackup'), 
                    row.get('DeviceProtection'),
                    row.get('TechSupport'), 
                    row.get('StreamingTV'), 
                    row.get('StreamingMovies'),
                    row.get('Contract'), 
                    row.get('PaperlessBilling'), 
                    row.get('PaymentMethod'),
                    float(row.get('MonthlyCharges', 0.0)) if pd.notna(row.get('MonthlyCharges')) else None,
                    total_charges_val, 
                    os.path.basename(csv_filepath)
                )
                
                sql_insert_query = """
                    INSERT INTO customer_data (
                        csv_row_index, customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
                        PhoneService, MultipleLines, InternetService, OnlineSecurity,
                        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                        StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                        MonthlyCharges, TotalCharges, data_source
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql_insert_query, values)
                rows_inserted_count += 1
            except psycopg2.Error as db_err:
                logger.error(f"Database error inserting row {index+2} (customerID: {customer_id}) from {csv_filepath}: {db_err}")
                conn.rollback() 
            except Exception as e:
                logger.error(f"Generic error inserting row {index+2} (customerID: {customer_id}) from {csv_filepath}: {e}")
                conn.rollback()
        conn.commit()
        logger.info(f"From {csv_filepath}: Loaded {rows_inserted_count} new rows. Skipped {rows_skipped_duplicates_count} duplicates. Skipped {rows_with_missing_customerid_count} for missing ID. Failed conversion for {rows_failed_conversion_count} TotalCharges.")
        return rows_inserted_count, rows_skipped_duplicates_count, rows_with_missing_customerid_count, rows_failed_conversion_count
    except Exception as e:
        logger.error(f"Critical error during CSV loading for {csv_filepath}: {e}")
        conn.rollback()
        return 0, 0, 0, 0

def preprocess_data_from_df(df):
    """Preprocesses the DataFrame for the model based on Jupyter notebook logic."""
    print(f"Starting preprocessing for a DataFrame with {df.shape[0]} rows.")
    processed_df = df.copy()

    # Fill NaN for TotalCharges (from notebook: 2279 or mean)
    # For simplicity, let's use the provided mean if available, otherwise a placeholder.
    # In a real scenario, this mean might be calculated or configurable.
    tc_mean_fill = 2279.0
    if 'TotalCharges' in processed_df.columns:
        # Handle cases where TotalCharges might be an object type due to spaces
        if processed_df['TotalCharges'].dtype == 'object':
            processed_df['TotalCharges'] = processed_df['TotalCharges'].str.replace(' ', str(tc_mean_fill), regex=False).astype(float)
        processed_df['TotalCharges'] = processed_df['TotalCharges'].fillna(tc_mean_fill).astype(float)
    else:
        print("Warning: 'TotalCharges' column not found for preprocessing.")
        # Potentially add a default or raise an error depending on model sensitivity
        processed_df['TotalCharges'] = tc_mean_fill 

    # Handle PhoneService (from notebook: fillna 'No', then map Yes/No to 1/0)
    if 'PhoneService' in processed_df.columns:
        processed_df['PhoneService'] = processed_df['PhoneService'].fillna('No')
        processed_df['PhoneService'] = processed_df['PhoneService'].map({'Yes': 1, 'No': 0}).fillna(0) # FillNA for unexpected values
    else:
        print("Warning: 'PhoneService' column not found. Defaulting to 0 (No).")
        processed_df['PhoneService'] = 0

    # Handle Contract (from notebook: dropna, then one-hot encode)
    # For batch, we need to ensure all contract types are present or handled.
    # The API directly receives these one-hot encoded features.
    # For batch, we'll create them based on the 'Contract' column.
    if 'Contract' in processed_df.columns:
        processed_df['Contract'] = processed_df['Contract'].fillna('Unknown') # Handle nulls before one-hot
        contract_dummies = pd.get_dummies(processed_df['Contract'], prefix='Contract').astype(int)
        processed_df = pd.concat([processed_df, contract_dummies], axis=1)
        # Ensure all required contract columns exist, filling with 0 if not
        for col in ['Contract_Month-to-month', 'Contract_One year', 'Contract_Two year']:
            if col not in processed_df.columns:
                processed_df[col] = 0
    else:
        print("Warning: 'Contract' column not found. Defaulting contract type columns to 0.")
        processed_df['Contract_Month-to-month'] = 0
        processed_df['Contract_One year'] = 0
        processed_df['Contract_Two year'] = 0

    # Tenure: fillna with mean (from notebook)
    if 'tenure' in processed_df.columns:
        if processed_df['tenure'].isnull().any():
            tenure_mean = processed_df['tenure'].mean()
            processed_df['tenure'] = processed_df['tenure'].fillna(tenure_mean).astype(int) # or float
    else:
        print("Warning: 'tenure' column not found. Defaulting to 0 or a pre-defined mean.")
        processed_df['tenure'] = 0 # Or some other sensible default like 32 (overall mean)

    # Select and order columns for the model
    # API: TotalCharges, Month-to-month, One year, Two year, PhoneService, tenure
    # These are the direct inputs the API expects. For batch, we need to map them.
    # Corrected column names from one-hot encoding for Contract
    model_columns = [
        'TotalCharges',
        'Contract_Month-to-month', # After one-hot encoding, it becomes Contract_Month-to-month
        'Contract_One year',
        'Contract_Two year',
        'PhoneService',
        'tenure'
    ]

    # Ensure all model columns are present, fill with 0 or a sensible default if not
    for col in model_columns:
        if col not in processed_df.columns:
            print(f"Warning: Expected model column '{col}' not found after preprocessing. Adding it with default 0.")
            processed_df[col] = 0 # Or another default
            
    return processed_df[model_columns]

def persist_predictions(conn, customer_ids, original_features_df, predictions, probabilities, model_version="1.0"):
    """Persists prediction results to the database."""
    cursor = conn.cursor()
    try:
        for i, cust_id in enumerate(customer_ids):
            # Extract original features used for prediction for logging
            # This assumes original_features_df is aligned with predictions and customer_ids
            total_charges = original_features_df.iloc[i]['TotalCharges']
            c_m2m = original_features_df.iloc[i]['Contract_Month-to-month']
            c_1y = original_features_df.iloc[i]['Contract_One year']
            c_2y = original_features_df.iloc[i]['Contract_Two year']
            phone_service = original_features_df.iloc[i]['PhoneService']
            tenure = original_features_df.iloc[i]['tenure']

            cursor.execute("""
                INSERT INTO churn_predictions (
                    customerID, TotalCharges, 
                    Contract_Month_to_month, Contract_One_year, Contract_Two_year, 
                    PhoneService, tenure, 
                    churn_prediction, churn_probability_churn, churn_probability_no_churn, model_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                cust_id, total_charges, 
                c_m2m, c_1y, c_2y, 
                phone_service, tenure, 
                int(predictions[i]), 
                float(probabilities[i][1]), # Churn probability
                float(probabilities[i][0]), # No Churn probability
                model_version
            ))
        conn.commit()
        print(f"Successfully persisted {len(predictions)} predictions.")
    except Exception as e:
        print(f"Error persisting predictions: {e}")
        conn.rollback()
    finally:
        cursor.close()

# --- Data Preprocessing --- #
def preprocess_data_from_db(conn):
    """Retrieves data from DB, preprocesses it, and returns features and customer IDs."""
    try:
        # Fetch all columns needed for preprocessing and linking
        # Note: The Jupyter notebook uses specific column names after some processing.
        # We need to replicate that processing here or ensure the query gets the right base columns.
        # Based on the notebook: TotalCharges, Contract, PhoneService, tenure are key.
        query = """
            SELECT customerID, "TotalCharges", "Contract", "PhoneService", "tenure"
            FROM customer_data 
            -- Add a WHERE clause here to select only new/unprocessed customers if you implement that logic
            -- e.g., WHERE customerID NOT IN (SELECT DISTINCT customerID FROM churn_predictions)
        """
        df = pd.read_sql_query(query, conn)
        print(f"Retrieved {df.shape[0]} rows from customer_data for preprocessing.")

        if df.empty:
            print("No data retrieved from database for preprocessing.")
            return None, None

        customer_ids = df['customerID'].tolist()
        
        # The preprocess_data_from_df function expects a DataFrame with at least these columns.
        # It will create the one-hot encoded contract columns and map PhoneService.
        # It will also select the final model_columns.
        processed_df = preprocess_data_from_df(df[['TotalCharges', 'Contract', 'PhoneService', 'tenure']])
        
        return processed_df, customer_ids
    except Exception as e:
        print(f"Error in preprocess_data_from_db: {e}")
        return None, None

if __name__ == "__main__":
    logger.info("Starting batch processor main script...")
    # Check for --reprocess-all argument
    reprocess_all = False
    if "--reprocess-all" in sys.argv:
        reprocess_all = True
        logger.info("Reprocessing ALL data in the database (ignoring churn_predictions table).")
    job = BatchPredictionJob()
    try:
        job.run(reprocess_all=reprocess_all)
    except Exception as e:
        # This is a top-level catch for unexpected errors during job.run() setup or critical failures
        logger.critical(f"Critical failure in main execution block: {e}", exc_info=True)
        record_error("main_script_critical_failure", str(e))
        # Attempt to push any gathered metrics even on critical failure before job.run() or if it crashes badly
        if job.start_time: # If job.run() at least started
             job_duration = (datetime.now() - job.start_time).total_seconds()
             BATCH_JOB_DURATION_SECONDS.observe(job_duration)
        push_metrics_to_gateway()
    logger.info("Batch processor main script finished.")

    # --- TODO: Add Monitoring --- #
    # - Emit metrics (e.g., number of rows processed, errors, time taken)
    #   to Prometheus (e.g., via a Pushgateway or by exposing a metrics endpoint if this were a long-running service)
    # - Implement logging to a centralized system
    # --- TODO: Add Alerting --- #
    # - Set up alerts for job failure, data quality issues, etc.
    # --- TODO: Error Handling & Data Quality --- #
    # - More robust checks for missing data, null columns, schema mismatches. 