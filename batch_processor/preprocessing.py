import pandas as pd
from . import config
from batch_processor.monitoring import logger, record_error, DATA_CONVERSION_ERRORS_TOTAL

class DataPreprocessor:
    def __init__(self):
        self.total_charges_fill = config.TOTAL_CHARGES_DEFAULT_FILL
        self.tenure_fill = config.TENURE_DEFAULT_MEAN_FILL 
        self.model_columns = config.MODEL_COLUMNS_ORDERED

    def _handle_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'totalcharges' in df.columns:
            # Convert to numeric, coercing errors. This turns unconvertible strings into NaN.
            df['totalcharges'] = pd.to_numeric(df['totalcharges'].astype(str).str.strip(), errors='coerce')
            
            # Log rows where conversion failed before filling NaN
            failed_conversion_mask = df['totalcharges'].isnull() & df['totalcharges'].notna() # original was notna but failed conversion
            if failed_conversion_mask.any():
                logger.warning(f"{failed_conversion_mask.sum()} rows had TotalCharges conversion issues.")
                DATA_CONVERSION_ERRORS_TOTAL.labels(csv_file='N/A_db_source', column_name='totalcharges').inc(failed_conversion_mask.sum())

            df['totalcharges'] = df['totalcharges'].fillna(self.total_charges_fill)
            logger.info(f"'totalcharges' processed. Filled NaNs with {self.total_charges_fill}")
        else:
            logger.warning("'totalcharges' column not found. Creating with default fill value.")
            df['totalcharges'] = self.total_charges_fill
            record_error("preprocessing_missing_column", "totalcharges column absent, filled with default")
        return df

    def _handle_phone_service(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'phoneservice' in df.columns:
            df['phoneservice'] = df['phoneservice'].fillna('No')
            # Log values not in map before mapping
            unknown_values = df[~df['phoneservice'].isin(['Yes', 'No'])]['phoneservice'].unique()
            if len(unknown_values) > 0:
                logger.warning(f"Unknown values found in phoneservice: {unknown_values}. These will become 0.")
                record_error("preprocessing_unknown_value", f"phoneservice contained: {unknown_values}")
            df['phoneservice'] = df['phoneservice'].map({'Yes': 1, 'No': 0}).fillna(0)
            logger.info("'phoneservice' processed. Mapped Yes/No to 1/0, NaNs/Unknowns to 0.")
        else:
            logger.warning("'phoneservice' column not found. Creating with default 0 (No).")
            df['phoneservice'] = 0
            record_error("preprocessing_missing_column", "phoneservice column absent, filled with 0")
        return df

    def _handle_contract(self, df: pd.DataFrame) -> pd.DataFrame:
        required_contract_cols = ['Month-to-month', 'One year', 'Two year']
        if 'contract' in df.columns:
            df['contract'] = df['contract'].fillna('Unknown') # Handle NaNs before get_dummies
            try:
                contract_dummies = pd.get_dummies(df['contract'], prefix='contract', dtype=int)
                df = pd.concat([df, contract_dummies], axis=1)
                # Rename columns to match model's expected feature names
                df.rename(columns={
                    'contract_Month-to-month': 'Month-to-month',
                    'contract_One year': 'One year',
                    'contract_Two year': 'Two year'
                }, inplace=True)
                logger.info("'contract' one-hot encoded and columns renamed for model compatibility.")
            except Exception as e:
                logger.error(f"Error during one-hot encoding of contract: {e}. Filling required columns with 0.")
                record_error("preprocessing_onehot_error", f"contract column: {e}")
                for col in required_contract_cols:
                    df[col] = 0 # Fallback
        else:
            logger.warning("'contract' column not found. Creating dummy contract columns with 0.")
            record_error("preprocessing_missing_column", "contract column absent, dummy columns created with 0")
            for col in required_contract_cols:
                df[col] = 0
        
        # Ensure all required one-hot encoded columns are present
        for col in required_contract_cols:
            if col not in df.columns:
                logger.warning(f"Missing expected contract column '{col}' after one-hot encoding. Adding it with 0.")
                df[col] = 0
        return df

    def _handle_tenure(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'tenure' in df.columns:
            if df['tenure'].isnull().any():
                # Use configured default mean, or calculate if not available (though less ideal for consistency)
                fill_value = self.tenure_fill
                logger.info(f"Filling NaNs in 'tenure' with {fill_value}.")
                df['tenure'] = df['tenure'].fillna(fill_value)
            try:
                df['tenure'] = df['tenure'].astype(int)
            except ValueError as e:
                logger.error(f"Could not convert tenure to int after fillna: {e}. Attempting float then int.")
                record_error("preprocessing_type_error", f"Tenure conversion: {e}")
                df['tenure'] = df['tenure'].astype(float).astype(int) # If it was float string
            logger.info("'tenure' processed.")
        else:
            logger.warning(f"'tenure' column not found. Creating with default fill value {self.tenure_fill}.")
            df['tenure'] = int(self.tenure_fill)
            record_error("preprocessing_missing_column", f"Tenure column absent, filled with {self.tenure_fill}")
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

        processed_df = self._handle_total_charges(processed_df)
        processed_df = self._handle_phone_service(processed_df)
        processed_df = self._handle_contract(processed_df) # This adds the one-hot encoded columns
        processed_df = self._handle_tenure(processed_df)

        # Add any other feature engineering steps from the notebook here
        # For example, if there were other mappings or transformations on other columns.
        # The current implementation focuses on the 6 features for the model.

        processed_df = self._ensure_model_columns(processed_df)
        
        logger.info(f"Preprocessing complete. Output DataFrame shape: {processed_df.shape}, Columns: {processed_df.columns.tolist()}")
        return processed_df 