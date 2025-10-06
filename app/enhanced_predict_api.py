import os
import json
import logging
import numpy as np
from functools import lru_cache
from typing import Tuple, Dict, Any, List, Optional

import joblib
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("marine_predict_api")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Configuration
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join("models", "saved_models"))
PIPELINE_PATH = os.path.join(MODEL_DIR, "species_abundance_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
FEATURE_SELECTOR_PATH = os.path.join(MODEL_DIR, "feature_selector.pkl")

class ModelService:
    """Service class for model operations and predictions."""
    
    def __init__(self):
        self.pipeline = None
        self.metadata = None
        self.feature_selector = None
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load model artifacts from disk."""
        try:
            self.pipeline = joblib.load(PIPELINE_PATH)
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            if os.path.exists(FEATURE_SELECTOR_PATH):
                self.feature_selector = joblib.load(FEATURE_SELECTOR_PATH)
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise
            
    def validate_input(self, data: Dict) -> Tuple[bool, Optional[str]]:
        """Validate input data against model requirements."""
        required_features = self.metadata.get('feature_importance', {}).keys()
        if not required_features:
            return True, None
            
        missing = [f for f in required_features if f not in data]
        if missing:
            return False, f"Missing required features: {', '.join(missing)}"
            
        try:
            # Validate numeric values
            for feature in required_features:
                value = float(data[feature])
                if np.isnan(value) or np.isinf(value):
                    return False, f"Invalid value for feature {feature}"
        except ValueError:
            return False, "All features must be numeric"
            
        return True, None
        
    def create_derived_features(self, data: Dict) -> Dict:
        """Create derived features for prediction."""
        data = data.copy()
        
        # Add derived features based on available inputs
        if all(k in data for k in ['mean_sst', 'biodiversity_index']):
            data['temp_biodiv_interaction'] = data['mean_sst'] * data['biodiversity_index']
            
        if 'genetic_diversity' in data:
            data['genetic_stress'] = 1 - data['genetic_diversity']
            
        if all(k in data for k in ['species_richness', 'biodiversity_index']):
            data['ecosystem_health_score'] = np.sqrt(
                data['species_richness'] * data['biodiversity_index']
            )
            
        return data
        
    def predict(self, data: Dict) -> Tuple[float, Dict]:
        """Make prediction and return confidence metrics."""
        # Validate input
        valid, error = self.validate_input(data)
        if not valid:
            raise ValueError(error)
            
        # Create derived features
        data = self.create_derived_features(data)
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = self.pipeline.predict(df)[0]
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(self.pipeline, 'feature_importances_'):
            importance = self.pipeline.feature_importances_
            features = df.columns
            feature_importance = dict(zip(features, importance))
            
        return prediction, feature_importance

# Initialize model service
model_service = ModelService()

@app.route("/health", methods=["GET"])
def health() -> Response:
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "model_version": model_service.metadata.get("version", "unknown"),
        "model_timestamp": model_service.metadata.get("created_at", "unknown")
    }
    return jsonify(status), 200

@app.route("/metadata", methods=["GET"])
def metadata() -> Response:
    """Return model metadata."""
    return jsonify(model_service.metadata), 200

@app.route("/predict", methods=["POST"])
def predict() -> Response:
    """Make predictions with confidence metrics."""
    try:
        input_data = request.get_json(force=True)
    except Exception as e:
        logger.warning(f"Invalid JSON input: {e}")
        return jsonify({"error": "Invalid JSON input"}), 400
        
    if not isinstance(input_data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400
        
    try:
        prediction, feature_importance = model_service.predict(input_data)
        
        response = {
            "prediction": float(prediction),
            "input_features": input_data,
            "feature_importance": feature_importance,
            "model_version": model_service.metadata.get("version", "unknown")
        }
        
        return jsonify(response), 200
        
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict() -> Response:
    """Batch prediction endpoint."""
    try:
        input_data = request.get_json(force=True)
    except Exception as e:
        logger.warning(f"Invalid JSON input: {e}")
        return jsonify({"error": "Invalid JSON input"}), 400
        
    if not isinstance(input_data, list):
        return jsonify({"error": "Expected a JSON array of prediction requests"}), 400
        
    results = []
    errors = []
    
    for idx, item in enumerate(input_data):
        try:
            prediction, feature_importance = model_service.predict(item)
            results.append({
                "id": idx,
                "prediction": float(prediction),
                "input_features": item,
                "feature_importance": feature_importance
            })
        except Exception as e:
            errors.append({
                "id": idx,
                "error": str(e)
            })
            
    response = {
        "predictions": results,
        "errors": errors if errors else None,
        "model_version": model_service.metadata.get("version", "unknown")
    }
    
    return jsonify(response), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)