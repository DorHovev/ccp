import pandas as pd
from batch_processor import config
from batch_processor.monitoring import logger, record_error, DATA_CONVERSION_ERRORS_TOTAL

class DataPreprocessor:
    def __init__(self):
        self.total_charges_fill = config.TOTAL_CHARGES_DEFAULT_FILL
        self.tenure_fill = config.TENURE_DEFAULT_MEAN_FILL 
        self.model_columns = config.MODEL_COLUMNS_ORDERED

    def _map_input_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map/rename input DataFrame columns to the model's expected names."""
        column_mapping = {
            'totalcharges': 'TotalCharges',
            'phoneservice': 'PhoneService',
            'contract': 'Contract',
            'tenure': 'tenure',  # This is already correct, but include for clarity
            # Add any others as needed
        }
        # Only rename columns that exist in the DataFrame
        columns_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}
        return df.rename(columns=columns_to_rename)

    def _handle_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'TotalCharges' in df.columns:
            # Convert to numeric, coercing errors. This turns unconvertible strings into NaN.
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
            
            # Log rows where conversion failed before filling NaN
            failed_conversion_mask = df['TotalCharges'].isnull() & df['TotalCharges'].notna() # original was notna but failed conversion
            if failed_conversion_mask.any():
                logger.warning(f"{failed_conversion_mask.sum()} rows had TotalCharges conversion issues.")
                DATA_CONVERSION_ERRORS_TOTAL.labels(csv_file='N/A_db_source', column_name='TotalCharges').inc(failed_conversion_mask.sum())

            df['TotalCharges'] = df['TotalCharges'].fillna(self.total_charges_fill)
            logger.info(f"'TotalCharges' processed. Filled NaNs with {self.total_charges_fill}")
        else:
            logger.warning("'TotalCharges' column not found. Creating with default fill value.")
            df['TotalCharges'] = self.total_charges_fill
            record_error("preprocessing_missing_column", "TotalCharges column absent, filled with default")
        return df

    def _handle_phone_service(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle both 'phoneservice' and 'PhoneService'
        col = None
        if 'phoneservice' in df.columns:
            col = 'phoneservice'
        elif 'PhoneService' in df.columns:
            col = 'PhoneService'
        if col:
            df[col] = df[col].fillna('No')
            unknown_values = df[~df[col].isin(['Yes', 'No'])][col].unique()
            if len(unknown_values) > 0:
                logger.warning(f"Unknown values found in {col}: {unknown_values}. These will become 0.")
                record_error("preprocessing_unknown_value", f"{col} contained: {unknown_values}")
            df['PhoneService'] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
            if col != 'PhoneService':
                df.drop(columns=[col], inplace=True)
            logger.info(f"'{col}' processed. Mapped Yes/No to 1/0, NaNs/Unknowns to 0.")
        else:
            logger.warning("'PhoneService' column not found. Creating with default 0 (No).")
            df['PhoneService'] = 0
            record_error("preprocessing_missing_column", "PhoneService column absent, filled with 0")
        return df

    def _handle_contract(self, df: pd.DataFrame) -> pd.DataFrame:
        required_contract_cols = ['Month-to-month', 'One year', 'Two year']
        if 'Contract' in df.columns:
            null_mask = df['Contract'].isnull()
            count = null_mask.sum()
            if count > 0:
                logger.warning(f"Dropping {count} rows due to NaN in 'Contract'.")
                for idx, row in df[null_mask].iterrows():
                    logger.error(f"Dropping row {idx} due to missing value in 'Contract'")
                    record_error("preprocessing_missing_important_column", f"Row {idx} dropped: missing contract")
                df = df[~null_mask]
            try:
                contract_dummies = pd.get_dummies(df['Contract'], prefix='contract', dtype=int)
                df = pd.concat([df, contract_dummies], axis=1)
                # Rename columns to match config/model's expected feature names
                df.rename(columns={
                    'contract_Month-to-month': 'Month-to-month',
                    'contract_One year': 'One year',
                    'contract_Two year': 'Two year'
                }, inplace=True)
                logger.info("'Contract' one-hot encoded and columns renamed for model compatibility.")
            except Exception as e:
                logger.error(f"Error during one-hot encoding of contract: {e}. Filling required columns with 0.")
                record_error("preprocessing_onehot_error", f"Contract column: {e}")
                for col in required_contract_cols:
                    df[col] = 0 # Fallback
        else:
            logger.warning("'Contract' column not found. Dropping all rows due to missing contract column.")
            record_error("preprocessing_missing_column", "Contract column absent, all rows dropped")
            return df.iloc[0:0]  # Return empty DataFrame
        # Ensure all required one-hot encoded columns are present
        for col in required_contract_cols:
            if col not in df.columns:
                logger.warning(f"Missing expected contract column '{col}' after one-hot encoding. Adding it with 0.")
                df[col] = 0
        return df

    def _handle_tenure(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'tenure' in df.columns:
            null_mask = df['tenure'].isnull()
            count = null_mask.sum()
            if count > 0:
                logger.warning(f"Dropping {count} rows due to NaN in 'tenure'.")
                for idx, row in df[null_mask].iterrows():
                    logger.error(f"Dropping row {idx} due to missing value in 'tenure'")
                    record_error("preprocessing_missing_important_column", f"Row {idx} dropped: missing tenure")
                df = df[~null_mask]
            try:
                df['tenure'] = df['tenure'].astype(int)
            except ValueError as e:
                logger.error(f"Could not convert tenure to int after dropping NaNs: {e}. Attempting float then int.")
                record_error("preprocessing_type_error", f"Tenure conversion: {e}")
                df['tenure'] = df['tenure'].astype(float).astype(int)
            logger.info("'tenure' processed.")
        else:
            logger.warning("'tenure' column not found. Dropping all rows due to missing tenure column.")
            record_error("preprocessing_missing_column", "tenure column absent, all rows dropped")
            return df.iloc[0:0]  # Return empty DataFrame
        return df

    def _ensure_model_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all required model columns are present and in the correct order."""
        missing_cols = []
        for col in self.model_columns:
            if col not in df.columns:
                logger.warning(f"Expected model column '{col}' not found. Adding it with default 0.")
                df[col] = 0 # Defaulting missing features to 0
                missing_cols.append(col)
        if missing_cols:
            record_error("preprocessing_missing_model_features", f"Missing: {', '.join(missing_cols)}")
        
        # Reorder to match model's expected input
        try:
            df = df[self.model_columns]
        except KeyError as e:
            logger.critical(f"Critical error: Could not reorder columns for model. Missing: {e}. Batch may fail.")
            record_error("preprocessing_column_reorder_failed", f"Missing key(s): {e}")
            # Depending on strictness, might raise an error or return df as is hoping for the best
            # For now, we try to return what we have, but this is a severe issue.
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all preprocessing steps to the input DataFrame."""
        logger.info(f"Starting preprocessing for a DataFrame with {df.shape[0]} rows and columns: {df.columns.tolist()}")
        
        if df.empty:
            logger.warning("Input DataFrame for preprocessing is empty.")
            return pd.DataFrame(columns=self.model_columns) # Return empty df with expected columns

        processed_df = df.copy()
        processed_df = self._map_input_columns(processed_df)


        # Always apply all handlers to ensure correct types
        processed_df = self._handle_total_charges(processed_df)
        processed_df = self._handle_phone_service(processed_df)
        processed_df = self._handle_contract(processed_df) # This adds the one-hot encoded columns
        processed_df = self._handle_tenure(processed_df)

        # Add any other feature engineering steps from the notebook here
        # For example, if there were other mappings or transformations on other columns.
        # The current implementation focuses on the 6 features for the model.

        processed_df = self._ensure_model_columns(processed_df)
        
        logger.info(f"Preprocessing complete. Output DataFrame shape: {processed_df.shape}, Columns: {processed_df.columns.tolist()}")
        logger.info(f"Preprocessing output DataFrame head:\n{processed_df.head()}")
        logger.info(f"Preprocessing output DataFrame dtypes:\n{processed_df.dtypes}")

        # Final NaN check and fill
        if processed_df.isnull().any().any():
            logger.warning("NaN values detected in processed DataFrame. Filling with 0.")
            logger.warning(f"NaN locations:\n{processed_df.isnull().sum()}")
            processed_df = processed_df.fillna(0)
        return processed_df 