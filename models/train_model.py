import os
import json
import logging
import argparse
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor

# reproducibility
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
np.random.seed(RANDOM_SEED)

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_and_train_pipeline(X_train: pd.DataFrame, y_train: pd.Series):
    """Builds a pipeline with scaler + model and fits it."""
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler(quantile_range=(5, 95))),
            ("model", GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=200)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def save_pipeline_and_metadata(pipeline, X_train: pd.DataFrame, out_dir: str, version: str = "v1.0.0"):
    os.makedirs(out_dir, exist_ok=True)
    pipeline_path = os.path.join(out_dir, f"species_abundance_pipeline_{version}.pkl")
    latest_path = os.path.join(out_dir, "species_abundance_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(pipeline, latest_path)

    metadata = {
        "model_name": "species_abundance_gbr",
        "version": version,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": RANDOM_SEED,
        "features": list(X_train.columns),
        "hyperparameters": pipeline.named_steps["model"].get_params(),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    return pipeline_path


def train_from_csv(csv_path: str, out_dir: str, target_col: str, version: str = "v1.0.0"):
    logger.info(f"Training pipeline from {csv_path}")
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in CSV")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    pipeline = build_and_train_pipeline(X, y)
    pipeline_path = save_pipeline_and_metadata(pipeline, X, out_dir, version)
    logger.info(f"Saved pipeline to {pipeline_path}")
    return pipeline_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", help="Path to training CSV file", required=False)
    parser.add_argument("--target", help="Target column name", required=False, default="species_count")
    parser.add_argument("--out-dir", help="Output model dir", required=False, default="models/saved_models")
    parser.add_argument("--version", help="Model version tag", required=False, default="v1.0.0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.data_csv:
        logger.info("No data CSV provided. Exiting. Use --data-csv to provide training data.")
        exit(0)
    pipeline_path = train_from_csv(args.data_csv, args.out_dir, args.target, args.version)
    logger.info(f"Saved pipeline to {pipeline_path}")
    logger.info(f"Saved pipeline to {pipeline_path}")