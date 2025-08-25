#!/usr/bin/env python3
"""
Advanced ML Training Pipeline for Visual Anomaly Detection
Incorporates AutoML, hyperparameter optimization, and model versioning
"""

import os
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Hyperparameter Optimization
import optuna
from optuna.integration import MLflowCallback

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

# Data validation
import great_expectations as gx

# Active learning
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

# Vector database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
import click
import yaml
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Training configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.data_dir = Path('/app/data')
        self.training_data_dir = Path('/app/training_data')
        self.models_dir = Path('/app/models')
        self.experiments_dir = Path('/app/experiments')
        
        # Load custom config if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self.__dict__.update(custom_config)
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.training_data_dir, self.models_dir, self.experiments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

class DataCollector:
    """Collect and prepare training data from Qdrant and local storage."""
    
    def __init__(self, config: Config):
        self.config = config
        self.qdrant_client = QdrantClient(url=config.qdrant_url)
        
    def collect_from_qdrant(self, collection_name: str = "ui_diffs_v2", limit: int = 1000) -> pd.DataFrame:
        """Collect feature vectors and metadata from Qdrant."""
        try:
            # Scroll through all points in collection
            records = []
            offset = None
            
            while True:
                result = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not result[0]:  # No more points
                    break
                
                for point in result[0]:
                    # Extract features from ML result metrics
                    payload = point.payload
                    ml_result = payload.get('ml_result', {})
                    metrics = ml_result.get('metrics', {})
                    
                    if metrics:
                        record = {
                            'point_id': point.id,
                            'test_id': payload.get('test_id'),
                            'run_id': payload.get('run_id'),
                            'timestamp': payload.get('timestamp'),
                            **metrics,  # All ML metrics as features
                            'clip_embedding': point.vector,
                            # Label (needs manual annotation in production)
                            'is_anomaly': metrics.get('is_anomaly', 0),
                            'anomaly_score': metrics.get('anomaly_score', 0.0)
                        }
                        records.append(record)
                
                offset = result[1]  # Next offset
                if len(records) >= limit:
                    break
            
            df = pd.DataFrame(records)
            logger.info(f"Collected {len(df)} records from Qdrant")
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect from Qdrant: {e}")
            return pd.DataFrame()
    
    def load_labeled_data(self) -> pd.DataFrame:
        """Load manually labeled training data."""
        labeled_files = list(self.config.training_data_dir.glob("*.csv"))
        
        if not labeled_files:
            logger.warning("No labeled training data found")
            return pd.DataFrame()
        
        dfs = []
        for file_path in labeled_files:
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} samples from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total labeled samples: {len(combined_df)}")
            return combined_df
        
        return pd.DataFrame()

class FeatureEngineer:
    """Advanced feature engineering for visual anomaly detection."""
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = None
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features from raw metrics."""
        if df.empty:
            return df
        
        df_features = df.copy()
        
        # Ratio features
        if 'pixel_diff_ratio' in df_features.columns and 'anomaly_score' in df_features.columns:
            df_features['pixel_anomaly_ratio'] = (
                df_features['pixel_diff_ratio'] / (df_features['anomaly_score'] + 1e-8)
            )
        
        # Interaction features
        if all(col in df_features.columns for col in ['ssim', 'lpips']):
            df_features['ssim_lpips_interaction'] = df_features['ssim'] * (1 - df_features['lpips'])
        
        # Composite scores
        perceptual_cols = ['lpips', 'vit_cosine_distance']
        structural_cols = ['ssim', 'pixel_diff_ratio']
        
        if all(col in df_features.columns for col in perceptual_cols):
            df_features['perceptual_score'] = df_features[perceptual_cols].mean(axis=1)
        
        if all(col in df_features.columns for col in structural_cols):
            df_features['structural_score'] = (
                (1 - df_features['ssim']) + df_features['pixel_diff_ratio']
            ) / 2
        
        # Time-based features (if timestamp available)
        if 'timestamp' in df_features.columns:
            df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            df_features['is_weekend'] = df_features['day_of_week'] >= 5
        
        # Statistical features from CLIP embeddings
        if 'clip_embedding' in df_features.columns:
            embedding_stats = df_features['clip_embedding'].apply(
                lambda x: {
                    'embedding_mean': np.mean(x) if isinstance(x, list) else 0,
                    'embedding_std': np.std(x) if isinstance(x, list) else 0,
                    'embedding_max': np.max(x) if isinstance(x, list) else 0,
                    'embedding_min': np.min(x) if isinstance(x, list) else 0
                } if x else {'embedding_mean': 0, 'embedding_std': 0, 'embedding_max': 0, 'embedding_min': 0}
            )
            
            for stat_name in ['embedding_mean', 'embedding_std', 'embedding_max', 'embedding_min']:
                df_features[stat_name] = [stats[stat_name] for stats in embedding_stats]
        
        # Feature selection (numerical columns only)
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        exclude_cols = ['is_anomaly', 'anomaly_score', 'point_id', 'run_id', 'test_id']
        self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Engineered {len(self.feature_columns)} features")
        return df_features
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale features using robust scaling."""
        if fit or self.scaler is None:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X[self.feature_columns])
        else:
            X_scaled = self.scaler.transform(X[self.feature_columns])
        
        return X_scaled

class ModelTrainer:
    """Advanced model training with hyperparameter optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.feature_engineer = FeatureEngineer()
        
        # Set MLflow tracking
        mlflow.set_tracking_uri(config.mlflow_uri)
        
    def create_optuna_study(self, study_name: str) -> optuna.Study:
        """Create Optuna study for hyperparameter optimization."""
        mlflc = MLflowCallback(
            tracking_uri=self.config.mlflow_uri,
            metric_name="roc_auc"
        )
        
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage="sqlite:///optuna_study.db",
            load_if_exists=True
        )
        study.add_callback(mlflc)
        
        return study
    
    def objective_xgboost(self, trial: optuna.Trial, X_train: np.ndarray, 
                         y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Objective function for XGBoost hyperparameter optimization."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'eval_metric': 'auc',
            'use_label_encoder': False
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)
    
    def objective_lightgbm(self, trial: optuna.Trial, X_train: np.ndarray, 
                          y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Objective function for LightGBM hyperparameter optimization."""
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
            'random_state': 42,
            'verbosity': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        return roc_auc_score(y_val, y_pred_proba)
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train multiple models with hyperparameter optimization."""
        results = {}
        
        with mlflow.start_run(run_name="visual_anomaly_training") as parent_run:
            
            # 1. XGBoost with Optuna
            logger.info("Training XGBoost with hyperparameter optimization...")
            study_xgb = self.create_optuna_study("xgboost_optimization")
            
            with mlflow.start_run(run_name="xgboost_tuning", nested=True):
                study_xgb.optimize(
                    lambda trial: self.objective_xgboost(trial, X_train, y_train, X_val, y_val),
                    n_trials=50
                )
                
                best_xgb_params = study_xgb.best_params
                best_xgb_model = xgb.XGBClassifier(**best_xgb_params, random_state=42)
                best_xgb_model.fit(X_train, y_train)
                
                y_pred_proba = best_xgb_model.predict_proba(X_val)[:, 1]
                xgb_auc = roc_auc_score(y_val, y_pred_proba)
                
                mlflow.log_params(best_xgb_params)
                mlflow.log_metric("val_auc", xgb_auc)
                mlflow.xgboost.log_model(best_xgb_model, "model")
                
                results['xgboost'] = {
                    'model': best_xgb_model,
                    'auc': xgb_auc,
                    'params': best_xgb_params
                }
            
            # 2. LightGBM with Optuna
            logger.info("Training LightGBM with hyperparameter optimization...")
            study_lgb = self.create_optuna_study("lightgbm_optimization")
            
            with mlflow.start_run(run_name="lightgbm_tuning", nested=True):
                study_lgb.optimize(
                    lambda trial: self.objective_lightgbm(trial, X_train, y_train, X_val, y_val),
                    n_trials=50
                )
                
                best_lgb_params = study_lgb.best_params
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val)
                
                best_lgb_model = lgb.train(
                    best_lgb_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
                
                y_pred_proba = best_lgb_model.predict(X_val, num_iteration=best_lgb_model.best_iteration)
                lgb_auc = roc_auc_score(y_val, y_pred_proba)
                
                mlflow.log_params(best_lgb_params)
                mlflow.log_metric("val_auc", lgb_auc)
                mlflow.lightgbm.log_model(best_lgb_model, "model")
                
                results['lightgbm'] = {
                    'model': best_lgb_model,
                    'auc': lgb_auc,
                    'params': best_lgb_params
                }
            
            # 3. CatBoost (auto hyperparameter tuning)
            logger.info("Training CatBoost...")
            with mlflow.start_run(run_name="catboost_training", nested=True):
                catboost_model = CatBoostClassifier(
                    iterations=1000,
                    random_seed=42,
                    verbose=False,
                    early_stopping_rounds=20,
                    eval_metric='AUC'
                )
                
                catboost_model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    plot=False
                )
                
                y_pred_proba = catboost_model.predict_proba(X_val)[:, 1]
                catboost_auc = roc_auc_score(y_val, y_pred_proba)
                
                mlflow.log_metric("val_auc", catboost_auc)
                mlflow.sklearn.log_model(catboost_model, "model")
                
                results['catboost'] = {
                    'model': catboost_model,
                    'auc': catboost_auc,
                    'params': catboost_model.get_all_params()
                }
            
            # 4. Ensemble model
            logger.info("Creating ensemble model...")
            with mlflow.start_run(run_name="ensemble", nested=True):
                # Simple average ensemble
                xgb_pred = results['xgboost']['model'].predict_proba(X_val)[:, 1]
                lgb_pred = results['lightgbm']['model'].predict(X_val, 
                                                              num_iteration=results['lightgbm']['model'].best_iteration)
                cat_pred = results['catboost']['model'].predict_proba(X_val)[:, 1]
                
                ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3
                ensemble_auc = roc_auc_score(y_val, ensemble_pred)
                
                mlflow.log_metric("val_auc", ensemble_auc)
                
                results['ensemble'] = {
                    'auc': ensemble_auc,
                    'weights': [1/3, 1/3, 1/3]
                }
            
            # Select best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
            self.best_model = results[best_model_name]['model']
            
            logger.info(f"Best model: {best_model_name} with AUC: {results[best_model_name]['auc']:.4f}")
            
            # Log best model info
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_auc", results[best_model_name]['auc'])
        
        return results
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:  # LightGBM
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0].plot([0, 1], [0, 1], 'k--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        
        # Confusion Matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[1], cmap='Blues')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(self.config.experiments_dir / 'model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'auc': auc_score,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred_proba.tolist()
        }

class ActiveLearningPipeline:
    """Active learning for continuous model improvement."""
    
    def __init__(self, config: Config):
        self.config = config
        self.learner = None
        
    def initialize_learner(self, X_initial: np.ndarray, y_initial: np.ndarray):
        """Initialize active learner with initial labeled data."""
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.learner = ActiveLearner(
            estimator=classifier,
            query_strategy=uncertainty_sampling,
            X_training=X_initial,
            y_training=y_initial
        )
        
        logger.info(f"Initialized active learner with {len(X_initial)} samples")
    
    def query_next_samples(self, X_pool: np.ndarray, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Query next most informative samples for labeling."""
        if self.learner is None:
            raise ValueError("Active learner not initialized")
        
        query_idx, query_instances = self.learner.query(X_pool, n_instances=n_samples)
        
        return query_idx, query_instances
    
    def teach_learner(self, X_new: np.ndarray, y_new: np.ndarray):
        """Teach the learner with newly labeled samples."""
        if self.learner is None:
            raise ValueError("Active learner not initialized")
        
        self.learner.teach(X_new, y_new)
        logger.info(f"Taught learner with {len(X_new)} new samples")

@click.command()
@click.option('--config', type=str, help='Path to configuration file')
@click.option('--experiment-name', default='visual-anomaly-detection', help='MLflow experiment name')
@click.option('--max-samples', default=10000, help='Maximum samples to use for training')
def main(config: Optional[str], experiment_name: str, max_samples: int):
    """Main training pipeline."""
    
    logger.info("Starting advanced visual anomaly detection training pipeline")
    
    # Initialize configuration
    config_obj = Config(config)
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    try:
        # 1. Data Collection
        logger.info("=== Phase 1: Data Collection ===")
        data_collector = DataCollector(config_obj)
        
        # Collect from Qdrant
        qdrant_df = data_collector.collect_from_qdrant(limit=max_samples)
        
        # Load labeled data
        labeled_df = data_collector.load_labeled_data()
        
        # Combine datasets
        if not qdrant_df.empty and not labeled_df.empty:
            # Align columns
            common_cols = list(set(qdrant_df.columns) & set(labeled_df.columns))
            df = pd.concat([
                qdrant_df[common_cols],
                labeled_df[common_cols]
            ], ignore_index=True)
        elif not qdrant_df.empty:
            df = qdrant_df
        elif not labeled_df.empty:
            df = labeled_df
        else:
            logger.error("No training data available")
            return
        
        logger.info(f"Total training samples: {len(df)}")
        
        # 2. Feature Engineering
        logger.info("=== Phase 2: Feature Engineering ===")
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_features(df)
        
        # Check if we have target labels
        if 'is_anomaly' not in df_features.columns:
            logger.warning("No target labels found. Creating synthetic labels based on anomaly_score")
            # Use anomaly score threshold for synthetic labels
            df_features['is_anomaly'] = (df_features.get('anomaly_score', 0) > 0.3).astype(int)
        
        # Prepare training data
        X = df_features[feature_engineer.feature_columns]
        y = df_features['is_anomaly'].astype(int)
        
        # Handle class imbalance
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            logger.error("Need both positive and negative samples for training")
            return
        
        # Data validation
        logger.info("Validating data quality...")
        # Basic validation - extend with Great Expectations in production
        assert not X.isnull().any().any(), "Features contain null values"
        assert len(X) == len(y), "Feature and target lengths don't match"
        
        # 3. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Scale features
        X_train_scaled = feature_engineer.scale_features(X_train, fit=True)
        X_val_scaled = feature_engineer.scale_features(X_val, fit=False)
        X_test_scaled = feature_engineer.scale_features(X_test, fit=False)
        
        logger.info(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # 4. Model Training
        logger.info("=== Phase 3: Model Training ===")
        trainer = ModelTrainer(config_obj)
        trainer.feature_engineer = feature_engineer
        
        training_results = trainer.train_models(
            X_train_scaled, y_train,
            X_val_scaled, y_val
        )
        
        # 5. Model Evaluation
        logger.info("=== Phase 4: Model Evaluation ===")
        evaluation_results = trainer.evaluate_model(
            trainer.best_model,
            X_test_scaled, y_test
        )
        
        logger.info(f"Final Test AUC: {evaluation_results['auc']:.4f}")
        
        # 6. Save Models
        logger.info("=== Phase 5: Model Saving ===")
        model_artifacts = {
            'best_model': trainer.best_model,
            'feature_engineer': feature_engineer,
            'feature_columns': feature_engineer.feature_columns,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
        
        # Save with pickle
        model_path = config_obj.models_dir / f'visual_anomaly_model_{experiment_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_artifacts, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # 7. Active Learning Setup (optional)
        if len(df) > 1000:  # Only if we have enough data
            logger.info("=== Phase 6: Active Learning Setup ===")
            al_pipeline = ActiveLearningPipeline(config_obj)
            
            # Use a subset for initial training
            n_initial = min(100, len(X_train) // 2)
            al_pipeline.initialize_learner(
                X_train_scaled[:n_initial],
                y_train[:n_initial]
            )
            
            # Query next samples for labeling
            query_idx, query_instances = al_pipeline.query_next_samples(
                X_train_scaled[n_initial:],
                n_samples=20
            )
            
            # Save query instances for manual labeling
            query_df = pd.DataFrame(query_instances, columns=feature_engineer.feature_columns)
            query_path = config_obj.training_data_dir / 'active_learning_queries.csv'
            query_df.to_csv(query_path, index=False)
            
            logger.info(f"Active learning queries saved to {query_path}")
        
        # 8. Generate Training Report
        logger.info("=== Phase 7: Training Report ===")
        report = {
            'experiment_name': experiment_name,
            'training_date': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'total_samples': len(df),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'feature_count': len(feature_engineer.feature_columns),
                'class_distribution': dict(y.value_counts())
            },
            'model_performance': {
                'best_model': max(training_results.keys(), key=lambda k: training_results[k]['auc']),
                'validation_auc': max(result['auc'] for result in training_results.values()),
                'test_auc': evaluation_results['auc'],
                'test_accuracy': evaluation_results['classification_report']['accuracy']
            },
            'feature_importance': feature_engineer.feature_columns[:10]  # Top 10 features
        }
        
        report_path = config_obj.experiments_dir / f'training_report_{experiment_name}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {report_path}")
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()