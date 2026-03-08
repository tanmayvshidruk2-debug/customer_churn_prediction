import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.logger.logger import get_logger
from src.schema.train_schema import TrainConfig
from src.training.evaluate import Evaluator
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


logger = get_logger(__name__)


class Trainer:
    """
    Industrial-grade Trainer with Cross Validation
    """

    def __init__(self, config: dict):

        self.config = TrainConfig(**config["training"])
        self.registry_path = self.config.model_registry_path
        os.makedirs(self.registry_path, exist_ok=True)

        logger.info("Trainer initialized successfully")

    # =========================================================
    # Pipeline Builder
    # =========================================================
    def _build_pipeline(self, X: pd.DataFrame) -> Tuple[Pipeline, Dict]:
        """
        Build preprocessing pipeline with LabelEncoder for categorical variables.
        Returns pipeline and mapping of categorical features to their encoders.
        """

        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")

        # Create label encoders for categorical features
        categorical_encoders = {}
        for cat_feature in categorical_features:
            le = LabelEncoder()
            le.fit(X[cat_feature].astype(str))
            categorical_encoders[cat_feature] = le

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # For categorical features, we'll use a custom transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
            ],
            remainder="drop"
        )

        model_config = self.config.model

        if model_config.type != "xgboost":
            raise ValueError(f"Only XGBoost is supported. Got: {model_config.type}")

        xgb_params = model_config.xgboost.dict()

        model = XGBClassifier(
            **xgb_params,
            eval_metric='logloss'
        )

        logger.info(f"Using model: {model_config.type}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        return pipeline, categorical_encoders

    # =========================================================
    # Extract Transformed Feature Names
    # =========================================================
    def _get_feature_names(self, pipeline: Pipeline) -> list:
        preprocessor = pipeline.named_steps["preprocessor"]
        return preprocessor.get_feature_names_out().tolist()

    # =========================================================
    # Metadata Extraction
    # =========================================================
    def _extract_model_metadata(
        self,
        pipeline: Pipeline,
        feature_names: list,
        categorical_encoders: Dict = None
    ) -> Dict[str, Any]:

        model = pipeline.named_steps["model"]

        metadata = {
            "model_type": model.__class__.__name__,
            "training_timestamp": datetime.now().isoformat(),
            "random_state": self.config.random_state,
            "test_size": self.config.test_size,
            "cross_validation_folds": self.config.cross_validation_folds,
            "n_features": len(feature_names),
            "features": feature_names,
            "hyperparameters": model.get_params(),
        }

        if hasattr(model, "feature_importances_"):
            metadata["feature_importance"] = dict(
                zip(feature_names, model.feature_importances_.tolist())
            )

        elif hasattr(model, "coef_"):
            metadata["coefficients"] = dict(
                zip(feature_names, model.coef_[0].tolist())
            )

        if categorical_encoders:
            metadata["categorical_features"] = list(categorical_encoders.keys())

        return metadata

    #cross validation
    def model_cross_validation(self,pipeline, X, y) -> Dict[str, float]:

        skf = StratifiedKFold(
            n_splits=self.config.cross_validation_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        acc_scores = []
        f1_scores = []
        auc_scores = []

        folds = list(skf.split(X, y))

        for fold_idx, (train_idx, val_idx) in enumerate(
            tqdm(folds, desc="CV Progress", total=len(folds))
        ):

            logger.info("Training fold %d", fold_idx + 1)

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_val)

            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds)

            acc_scores.append(acc)
            f1_scores.append(f1)

            # ROC-AUC if available
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, probs)
                auc_scores.append(auc)

        metrics = {
            "cv_accuracy_mean": float(np.mean(acc_scores)),
            "cv_accuracy_std": float(np.std(acc_scores)),
            "cv_f1_mean": float(np.mean(f1_scores)),
            "cv_f1_std": float(np.std(f1_scores)),
        }

        if auc_scores:
            metrics["cv_auc_mean"] = float(np.mean(auc_scores))
            metrics["cv_auc_std"] = float(np.std(auc_scores))

        return metrics
    
    # =========================================================
    # Training Execution
    # =========================================================
    def train(self, df: pd.DataFrame) -> Tuple[str, Dict]:

        logger.info("Training started")

        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' not found"
            )

        # Exclude specified columns
        exclude_cols = self.config.exclude_columns or []
        exclude_cols = exclude_cols + [self.config.target_column]
        
        logger.info(f"Excluding columns: {exclude_cols}")
        
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[self.config.target_column]

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        X_encoded = X.copy()
        
        categorical_encoders = {}
        for cat_col in categorical_cols:
            le = LabelEncoder()
            X_encoded[cat_col] = le.fit_transform(X_encoded[cat_col].astype(str))
            categorical_encoders[cat_col] = le
        
        logger.info(f"Encoded categorical columns: {categorical_cols}")

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if len(y.unique()) <= 10 else None
        )

        logger.info("Train-test split completed")

        # Apply SMOTE if enabled
        if self.config.apply_smote:
            logger.info("Applying SMOTE for class balancing")
            smote = SMOTE(
                k_neighbors=self.config.smote_k_neighbors,
                random_state=self.config.random_state
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE applied. New training set shape: {X_train.shape}")

        pipeline, cat_encoders = self._build_pipeline(X_train)

        # Store encoders for later use during prediction
        self.categorical_encoders = categorical_encoders

        # Stratified K-Fold Cross Validation
        logger.info("Starting Stratified K-Fold Cross Validation")
        cv_metrics = self.model_cross_validation(
            pipeline,
            X_train,
            y_train
        )
        logger.info("Cross validation completed: %s", cv_metrics)

        # Final Training on Full Training Data
        pipeline.fit(X_train, y_train)

        logger.info("Final model training completed")

        # Evaluation on Hold-out Test Set
        evaluator = Evaluator()

        predictions = pipeline.predict(X_test)

        metrics = evaluator.evaluate(y_test, predictions)

        # Versioned Model Saving
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_path = os.path.join(
            self.registry_path,
            f"model_{version}"
        )

        os.makedirs(version_path, exist_ok=True)

        model_path = os.path.join(version_path, "model.pkl")
        joblib.dump(pipeline, model_path)

        logger.info(f"Model saved at {model_path}")

        # Save categorical encoders
        encoders_path = os.path.join(version_path, "encoders.pkl")
        joblib.dump(categorical_encoders, encoders_path)
        logger.info(f"Categorical encoders saved at {encoders_path}")

        # Save predictions
        predictions_df = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": predictions
        })
        predictions_df.to_csv(
            os.path.join(version_path, "predictions.csv"),
            index=False
        )

        evaluator.save_report(metrics, version_path)

        # Metadata Saving
        feature_names = X_train.columns.tolist()

        metadata = self._extract_model_metadata(
            pipeline,
            feature_names,
            categorical_encoders
        )

        metadata["test_metrics"] = metrics.dict()
        metadata["cross_validation_metrics"] = cv_metrics
        metadata["excluded_columns"] = self.config.exclude_columns or []
        metadata["smote_applied"] = self.config.apply_smote

        metadata_path = os.path.join(version_path, "model.yaml")

        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, sort_keys=False)

        logger.info(f"Model metadata saved at {metadata_path}")
        logger.info("Training pipeline completed successfully")

        return model_path, metrics
