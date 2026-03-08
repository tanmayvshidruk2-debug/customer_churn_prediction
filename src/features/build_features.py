import os
import yaml
import pandas as pd

from sklearn.pipeline import Pipeline

from src.logger.logger import get_logger
from src.schema.data_schema import DataConfig
from src.features.transformers import (
    MissingValueImputer,
    ColumnExcluder,
    FeatureGenerator
)

logger = get_logger(__name__)


class FeatureBuilder:
    """
    Feature Engineering Pipeline

    Responsibilities:
    - Exclude specified columns
    - Apply transformations
    - Generate features
    - Save processed dataset
    """

    def __init__(self, config, exclude_columns=None):

        self.config = DataConfig(**config["data"])
        self.exclude_columns = exclude_columns or []

        os.makedirs(
            os.path.dirname(self.config.processed_path),
            exist_ok=True
        )

        self.pipeline = Pipeline(
            steps=[
                ("column_excluder", ColumnExcluder(exclude_columns=self.exclude_columns)),
                ("imputer", MissingValueImputer()),
                ("feature_generator", FeatureGenerator())
            ]
        )

        logger.info("FeatureBuilder initialized")

    def build(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Starting feature engineering")
        
        if self.exclude_columns:
            logger.info(f"Excluding columns: {self.exclude_columns}")

        processed_df = self.pipeline.fit_transform(df)

        processed_df.to_csv(
            self.config.processed_path,
            index=False
        )

        logger.info(
            f"Processed dataset saved at "
            f"{self.config.processed_path}"
        )

        return processed_df
