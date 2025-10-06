import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel

# Configure reproducibility and logging
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
np.random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarineSpeciesPredictor:
    """Enhanced ML pipeline for marine species abundance prediction."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.pipeline = None
        self.feature_selector = None
        self.best_params = None
        self.metrics = {}
        
    def create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific derived features."""
        X = X.copy()
        
        # Environmental interaction features
        if all(col in X.columns for col in ['mean_sst', 'biodiversity_index']):
            X['temp_biodiv_interaction'] = X['mean_sst'] * X['biodiversity_index']
        
        # Genetic diversity features
        if 'genetic_diversity' in X.columns:
            X['genetic_stress'] = 1 - X['genetic_diversity']
            
        # Ecosystem health indicators
        if all(col in X.columns for col in ['species_richness', 'biodiversity_index']):
            X['ecosystem_health_score'] = np.sqrt(X['species_richness'] * X['biodiversity_index'])
            
        return X
    
    def build_pipeline(self) -> Pipeline:
        """Build the ML pipeline with preprocessing and model."""
        base_model = GradientBoostingRegressor(
            random_state=self.random_state,
            n_estimators=200
        )
        
        pipeline = Pipeline([
            ('scaler', RobustScaler(quantile_range=(5, 95))),
            ('feature_selector', SelectFromModel(estimator=base_model, prefit=False)),
            ('model', base_model)
        ])
        
        return pipeline
    
    def get_param_grid(self) -> Dict:
        """Define hyperparameter search space."""
        return {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__min_samples_split': [2, 5, 10],
            'feature_selector__threshold': ['mean', '0.5*mean', '1.5*mean']
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Tuple[Pipeline, Dict]:
        """Train the model with cross-validation and hyperparameter tuning."""
        logger.info("Creating derived features...")
        X = self.create_derived_features(X)
        
        logger.info("Building and tuning pipeline...")
        self.pipeline = self.build_pipeline()
        param_grid = self.get_param_grid()
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Calculate metrics
        y_pred = self.pipeline.predict(X)
        self.metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'cv_scores': cross_val_score(self.pipeline, X, y, cv=cv_folds, scoring='r2').tolist()
        }
        
        logger.info(f"Training metrics: {self.metrics}")
        return self.pipeline, self.metrics
    
    def save_model(self, out_dir: str, version: str = "v1.0.0") -> str:
        """Save the model, feature selector, and metadata."""
        os.makedirs(out_dir, exist_ok=True)
        
        # Save pipeline
        pipeline_path = os.path.join(out_dir, f"species_abundance_model_{version}.pkl")
        joblib.dump(self.pipeline, pipeline_path)
        
        # Save feature selector
        selector_path = os.path.join(out_dir, "feature_selector.pkl")
        joblib.dump(self.pipeline.named_steps['feature_selector'], selector_path)
        
        # Save metrics
        metrics_path = os.path.join(out_dir, "model_metrics.txt")
        with open(metrics_path, 'w') as f:
            for metric, value in self.metrics.items():
                f.write(f"{metric}: {value}\n")
        
        # Save metadata
        metadata = {
            "model_name": "marine_species_abundance_predictor",
            "version": version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "random_seed": self.random_state,
            "best_parameters": self.best_params,
            "metrics": self.metrics,
            "feature_importance": {
                name: float(importance) 
                for name, importance in zip(
                    self.pipeline.named_steps['feature_selector'].get_feature_names_out(),
                    self.pipeline.named_steps['model'].feature_importances_
                )
            }
        }
        
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        return pipeline_path

def train_from_csv(csv_path: str, out_dir: str, target_col: str, version: str = "v1.0.0"):
    """Train model from CSV file."""
    logger.info(f"Training model from {csv_path}")
    
    # Load and validate data
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV")
        
    # Handle date columns
    date_columns = df.select_dtypes(include=['object']).columns
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
            # Extract useful features from dates
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            # Drop original date column
            df = df.drop(columns=[col])
        except:
            continue
            
    # Drop any remaining non-numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df = df[numeric_cols]
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not available after preprocessing")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Training with features: {list(X.columns)}")
    
    # Train model
    predictor = MarineSpeciesPredictor()
    predictor.train(X, y)
    
    # Save model and artifacts
    pipeline_path = predictor.save_model(out_dir, version)
    logger.info(f"Saved model artifacts to {out_dir}")
    return pipeline_path

def parse_args():
    parser = argparse.ArgumentParser(description="Train marine species abundance prediction model")
    parser.add_argument("--data-csv", required=True, help="Path to training CSV file")
    parser.add_argument("--target", default="species_abundance", help="Target column name")
    parser.add_argument("--out-dir", default="models/saved_models", help="Output directory for model artifacts")
    parser.add_argument("--version", default="v1.0.0", help="Model version tag")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline_path = train_from_csv(args.data_csv, args.out_dir, args.target, args.version)
    logger.info(f"Training complete. Model saved to {pipeline_path}")