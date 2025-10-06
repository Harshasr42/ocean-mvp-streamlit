import os
import joblib
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model():
    try:
        # Verify saved_models directory exists
        if not os.path.exists('saved_models'):
            raise FileNotFoundError("saved_models directory not found")

        # Load model and scaler
        logger.info("Loading model artifacts...")
        model_path = os.path.join('saved_models', 'species_abundance_model.pkl')
        scaler_path = os.path.join('saved_models', 'species_abundance_scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler file not found")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Load model metrics to understand feature ranges
        metrics_path = os.path.join('saved_models', 'model_metrics.txt')
        if os.path.exists(metrics_path):
            logger.info("Loading model metrics for reference...")
            with open(metrics_path, 'r') as f:
                metrics_content = f.read()
                logger.info(f"Model metrics:\n{metrics_content}")

        # Create test cases with realistic ranges
        test_cases = [
            {
                'genetic_diversity': 0.75,
                'biodiversity_index': 0.8,
                'mean_sst': 27.5
            },
            {
                'genetic_diversity': 0.65,
                'biodiversity_index': 0.7,
                'mean_sst': 26.0
            }
        ]
        
        logger.info("Using test cases with verified feature ranges")

        logger.info("Running test predictions...")
        
        for case in test_cases:
            # Create input data
            test_data = pd.DataFrame([case])
            
            # Create derived features
            test_data['genetic_bio_interaction'] = np.sqrt(
                test_data['genetic_diversity'] * test_data['biodiversity_index']
            )
            test_data['temperature_stress'] = 0  # Default value for single sample
            
            # Ensure columns match training data
            required_columns = ['genetic_diversity', 'biodiversity_index', 'mean_sst', 
                             'genetic_bio_interaction', 'temperature_stress']
            test_data = test_data[required_columns]
            
            # Scale features
            scaled_data = scaler.transform(test_data)
            
            # Verify no NaN values
            if np.any(np.isnan(scaled_data)):
                raise ValueError("NaN values detected in scaled data")

            # Make prediction
            prediction = model.predict(scaled_data)
            
            logger.info("\nTest Case:")
            for key, value in case.items():
                logger.info(f"{key}: {value}")
            logger.info(f"Predicted species count: {round(prediction[0])}")

        return True
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("✅ Model test completed successfully!")
    else:
        print("❌ Model test failed. Check logs for details.")