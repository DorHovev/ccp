import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
from batch_processor.preprocessing import DataPreprocessor
from batch_processor.database import DatabaseManager
import tempfile

# --- Preprocessing Tests ---
def test_preprocessing_handles_missing_columns():
    df = pd.DataFrame({"totalcharges": [100, None], "tenure": [1, 2]})
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess(df)
    # Should fill missing columns with defaults
    assert not processed.isnull().any().any()
    assert set(["TotalCharges", "Month-to-month", "One year", "Two year", "PhoneService", "tenure"]).issubset(processed.columns)

def test_preprocessing_output_shape():
    df = pd.DataFrame({
        "TotalCharges": [100, 200],
        "contract": ["Month-to-month", "One year"],
        "phoneservice": ["Yes", "No"],
        "tenure": [1, 2]
    })
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess(df)
    assert processed.shape[0] == 2
    assert processed.shape[1] == 6

# --- DatabaseManager Tests (using SQLite for isolation) ---
def test_create_tables_and_load_csv(tmp_path):
    # Use SQLite in-memory DB for testing
    db_url = f"sqlite:///{tmp_path}/test.db"
    dbm = DatabaseManager(db_url=db_url)
    dbm.create_tables_if_not_exist()
    # Create a temp CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        "customerid": ["A1", "A2"],
        "totalcharges": [100, 200],
        "contract": ["Month-to-month", "One year"],
        "phoneservice": ["Yes", "No"],
        "tenure": [1, 2]
    })
    df.to_csv(csv_path, index=False)
    inserted, skipped, missing, conv = dbm.load_csv_to_db(str(csv_path))
    assert inserted == 2
    assert skipped == 0
    assert missing == 0
    assert conv == 0
    # Fetch data for preprocessing
    fetched = dbm.fetch_data_for_preprocessing()
    assert fetched.shape[0] == 2
    assert "customerid" in fetched.columns 