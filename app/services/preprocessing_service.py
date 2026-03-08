import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple
import os
import logging

logger = logging.getLogger(__name__)

# Import transformers from src
from src.features.transformers import (
    MissingValueImputer,
    ColumnExcluder,
    FeatureGenerator
)

class PreprocessingService:
    """
    Service for preprocessing prediction data using same pipeline as training.
    Uses transformers from src.features.transformers:
    - ColumnExcluder: Exclude specified columns
    - MissingValueImputer: Handle missing values
    - FeatureGenerator: Create engineered features
    """
    
    def __init__(self, encoders: Dict = None, exclude_columns: list = None):
        """
        Initialize preprocessing service.
        
        Args:
            encoders: Dictionary of fitted LabelEncoders for categorical columns
            exclude_columns: List of columns to exclude from processing
        """
        self.encoders = encoders
        self.exclude_columns = exclude_columns or [
            "customer_id", "age", "monthly_charges", "Date",
            "avg_monthly_value", "support_call_ratio", "payment_risk"
        ]
        
        # Create preprocessing pipeline using src transformers
        self.pipeline = Pipeline(
            steps=[
                ("column_excluder", ColumnExcluder(exclude_columns=self.exclude_columns)),
                ("imputer", MissingValueImputer()),
                ("feature_generator", FeatureGenerator())
            ]
        )
        
        logger.info(f"PreprocessingService initialized with exclude_columns: {self.exclude_columns}")
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess prediction data:
        1. Exclude specified columns
        2. Handle missing values (imputation)
        3. Generate engineered features
        4. Encode categorical variables
        
        Args:
            data: Input DataFrame with customer features
        
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        df = data.copy()
        
        logger.debug(f"Input shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # Apply feature engineering pipeline
        df = self.pipeline.fit_transform(df)
        
        logger.debug(f"After pipeline shape: {df.shape}")
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            logger.debug(f"Encoding categorical columns: {categorical_cols}")
            df = self._encode_categorical(df, categorical_cols)
        
        logger.debug(f"Output shape: {df.shape}, columns: {df.columns.tolist()}")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """
        Encode categorical variables using stored encoders or create new ones.
        
        Args:
            df: DataFrame to encode
            categorical_cols: List of categorical column names
        
        Returns:
            DataFrame with encoded categorical variables
        """
        for cat_col in categorical_cols:
            if self.encoders and cat_col in self.encoders:
                # Use stored encoder from training
                le = self.encoders[cat_col]
                try:
                    df[cat_col] = le.transform(df[cat_col].astype(str))
                    logger.debug(f"✓ Encoded '{cat_col}' using stored encoder")
                except ValueError as e:
                    logger.warning(f"Stored encoder for '{cat_col}' failed: {e}. Creating new encoder.")
                    le = LabelEncoder()
                    df[cat_col] = le.fit_transform(df[cat_col].astype(str))
            else:
                # Create new encoder if not available
                le = LabelEncoder()
                df[cat_col] = le.fit_transform(df[cat_col].astype(str))
                logger.debug(f"Created new encoder for '{cat_col}'")
        
        return df


