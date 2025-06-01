import pandas as pd
import os
import config
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, TIMESTAMP, select, text
from sqlalchemy.exc import SQLAlchemyError
from monitoring import (
    logger, record_error, 
    ROWS_LOADED_FROM_CSV_TOTAL, ROWS_SKIPPED_DUPLICATES_TOTAL, 
    DATA_CONVERSION_ERRORS_TOTAL, ROWS_MISSING_CUSTOMERID_TOTAL,
    ROWS_FETCHED_FOR_PREPROCESSING_TOTAL, PREDICTIONS_PERSISTED_TOTAL
)

class DatabaseManager:
    def __init__(self, db_url=None):
        self.db_url = db_url or config.DATABASE_URL
        self.engine = create_engine(self.db_url)
        self.metadata = MetaData()

        # Define tables (schema)
        self.customer_data = Table(
            'customer_data', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('csv_row_index', Integer),
            Column('customerid', String(255), unique=True, nullable=False),
            Column('gender', String(10)),
            Column('seniorcitizen', Integer),
            Column('partner', String(3)),
            Column('dependents', String(3)),
            Column('tenure', Integer),
            Column('phoneservice', String(3)),
            Column('multiplelines', String(20)),
            Column('internetservice', String(20)),
            Column('onlinesecurity', String(20)),
            Column('onlinebackup', String(20)),
            Column('deviceprotection', String(20)),
            Column('techsupport', String(20)),
            Column('streamingtv', String(20)),
            Column('streamingmovies', String(20)),
            Column('contract', String(20)),
            Column('paperlessbilling', String(3)),
            Column('paymentmethod', String(50)),
            Column('monthlycharges', Float),
            Column('totalcharges', Float),
            Column('data_source', String(255)),
            Column('loaded_at', TIMESTAMP)
        )
        self.churn_predictions = Table(
            'churn_predictions', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('customerid', String(255)),
            Column('prediction_timestamp', TIMESTAMP),
            Column('totalcharges', Float),
            Column('contract_month_to_month', Integer),
            Column('contract_one_year', Integer),
            Column('contract_two_year', Integer),
            Column('phoneservice', Integer),
            Column('tenure', Integer),
            Column('churn_prediction', Integer),
            Column('churn_probability_churn', Float),
            Column('churn_probability_no_churn', Float),
            Column('model_version', String(50))
        )

    def create_tables_if_not_exist(self):
        try:
            self.metadata.create_all(self.engine)
            logger.info("Database tables checked/created successfully (SQLAlchemy).")
        except SQLAlchemyError as e:
                logger.error(f"Error creating database tables: {e}")
                record_error("db_schema_creation", f"SQLAlchemy Error: {e}")

    def load_csv_to_db(self, csv_filepath):
        rows_inserted = 0
        rows_skipped_duplicates = 0
        rows_missing_id = 0
        rows_conversion_error = 0

        try:
            df = pd.read_csv(csv_filepath)
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_filepath}")
            record_error("csv_load_file_not_found", f"File: {os.path.basename(csv_filepath)}")
            return rows_inserted, rows_skipped_duplicates, rows_missing_id, rows_conversion_error
        except pd.errors.EmptyDataError:
            logger.warning(f"CSV file is empty: {csv_filepath}")
            return rows_inserted, rows_skipped_duplicates, rows_missing_id, rows_conversion_error
        except Exception as e:
            logger.error(f"Error reading CSV {csv_filepath}: {e}")
            record_error("csv_read_error", f"File: {os.path.basename(csv_filepath)}, Error: {e}")
            return rows_inserted, rows_skipped_duplicates, rows_missing_id, rows_conversion_error

        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        df.rename(columns={'unnamed: 0': 'csv_row_index'}, inplace=True)

        if 'customerid' not in df.columns:
            logger.critical(f"'customerid' column missing in {csv_filepath}. Skipping this file.")
            record_error("csv_format_error", f"File: {os.path.basename(csv_filepath)}, Missing customerid column")
            return rows_inserted, rows_skipped_duplicates, rows_missing_id, rows_conversion_error

        with self.engine.begin() as conn:
            for idx, row in df.iterrows():
                customer_id = str(row.get('customerid', '')).strip()
                if not customer_id:
                    logger.warning(f"Row {idx+2} in {csv_filepath}: customerid is missing or empty. Skipping.")
                    rows_missing_id += 1
                    DATA_CONVERSION_ERRORS_TOTAL.labels(csv_file=os.path.basename(csv_filepath), column_name='customerid_missing').inc()
                    continue

                # Check for duplicates
                result = conn.execute(
                    select(self.customer_data.c.customerid).where(self.customer_data.c.customerid == customer_id)
                ).fetchone()
                if result:
                    rows_skipped_duplicates += 1
                    continue

                try:
                    insert_values = {
                        'csv_row_index': row.get('csv_row_index', idx),
                        'customerid': customer_id,
                        'gender': row.get('gender'),
                        'seniorcitizen': int(row.get('seniorcitizen', 0)) if pd.notna(row.get('seniorcitizen')) else None,
                        'partner': row.get('partner'),
                        'dependents': row.get('dependents'),
                        'tenure': int(row.get('tenure', 0)) if pd.notna(row.get('tenure')) else None,
                        'phoneservice': row.get('phoneservice'),
                        'multiplelines': row.get('multiplelines'),
                        'internetservice': row.get('internetservice'),
                        'onlinesecurity': row.get('onlinesecurity'),
                        'onlinebackup': row.get('onlinebackup'),
                        'deviceprotection': row.get('deviceprotection'),
                        'techsupport': row.get('techsupport'),
                        'streamingtv': row.get('streamingtv'),
                        'streamingmovies': row.get('streamingmovies'),
                        'contract': row.get('contract'),
                        'paperlessbilling': row.get('paperlessbilling'),
                        'paymentmethod': row.get('paymentmethod'),
                        'monthlycharges': float(row.get('monthlycharges', 0.0)) if pd.notna(row.get('monthlycharges')) else None,
                        'totalcharges': float(row.get('totalcharges', 0.0)) if pd.notna(row.get('totalcharges')) else None,
                        'data_source': os.path.basename(csv_filepath)
                    }
                    conn.execute(self.customer_data.insert().values(**insert_values))
                    rows_inserted += 1
                except Exception as e:
                    logger.error(f"Row {idx+2} in {csv_filepath} (customerid: {customer_id}): Unexpected error - {e}. Skipping row.")
                    rows_conversion_error += 1
                    record_error("unknown_insert_row_error", f"File: {os.path.basename(csv_filepath)}, Customer: {customer_id}, Error: {e}")
            
            logger.info(f"Processed {os.path.basename(csv_filepath)}: Inserted: {rows_inserted}, Skipped Duplicates: {rows_skipped_duplicates}, Skipped Missing ID: {rows_missing_id}, Conversion Errors: {rows_conversion_error}")
            if rows_inserted > 0:
                ROWS_LOADED_FROM_CSV_TOTAL.labels(csv_file=os.path.basename(csv_filepath)).inc(rows_inserted)
            if rows_skipped_duplicates > 0:
                 ROWS_SKIPPED_DUPLICATES_TOTAL.labels(csv_file=os.path.basename(csv_filepath)).inc(rows_skipped_duplicates)
            if rows_missing_id > 0:
                ROWS_MISSING_CUSTOMERID_TOTAL.labels(csv_file=os.path.basename(csv_filepath)).inc(rows_missing_id)

        return rows_inserted, rows_skipped_duplicates, rows_missing_id, rows_conversion_error

    def fetch_data_for_preprocessing(self):
        try:
            with self.engine.connect() as conn:
                query = text(f'''
                    SELECT customerid, totalcharges, contract, phoneservice, tenure
                    FROM customer_data
                    WHERE NOT EXISTS (
                        SELECT 1 FROM churn_predictions WHERE churn_predictions.customerid = customer_data.customerid
                    )
                ''')
                df = pd.read_sql_query(query, conn)
                logger.info(f"Fetched {len(df)} new rows from database for preprocessing.")
                if not df.empty:
                    ROWS_FETCHED_FOR_PREPROCESSING_TOTAL.set(len(df))
                else:
                    ROWS_FETCHED_FOR_PREPROCESSING_TOTAL.set(0)
                return df
        except Exception as e:
            logger.error(f"Unexpected error fetching data for preprocessing: {e}")
            record_error("db_fetch_unexpected_error", f"Error: {e}")
            return pd.DataFrame()

    def persist_predictions(self, customer_ids, features_df, predictions, probabilities, model_version="1.0"):
        """Persists prediction results to the churn_predictions table."""
        def to_python_type(val):
            if hasattr(val, 'item'):
                return val.item()
            return val

        rows_to_insert = []
        for i, customer_id in enumerate(customer_ids):
            current_features = features_df.iloc[i]
            insert_values = {
                'customerid': to_python_type(customer_id),
                'totalcharges': to_python_type(current_features.get('TotalCharges', current_features.get('totalcharges'))),
                'contract_month_to_month': to_python_type(current_features.get('Month-to-month')),
                'contract_one_year': to_python_type(current_features.get('One year')),
                'contract_two_year': to_python_type(current_features.get('Two year')),
                'phoneservice': to_python_type(current_features.get('PhoneService', current_features.get('phoneservice'))),
                'tenure': to_python_type(current_features.get('tenure')),
                'churn_prediction': int(predictions[i]),
                'churn_probability_churn': float(probabilities[i][1]),
                'churn_probability_no_churn': float(probabilities[i][0]),
                'model_version': str(model_version)
            }
            rows_to_insert.append(insert_values)
        with self.engine.begin() as conn:
            conn.execute(self.churn_predictions.insert(), rows_to_insert)
        logger.info(f"Successfully persisted {len(predictions)} predictions.")