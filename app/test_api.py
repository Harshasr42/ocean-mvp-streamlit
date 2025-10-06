import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api():
    try:
        # API endpoint
        url = 'http://localhost:5000/predict'
        
        # Test cases
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
        
        logger.info("Testing API predictions...")
        
        for case in test_cases:
            # Make API request
            response = requests.post(url, json=case)
            
            if response.status_code == 200:
                result = response.json()
                logger.info("\nTest Case:")
                for key, value in case.items():
                    logger.info(f"{key}: {value}")
                logger.info(f"Predicted species count: {result['predicted_species_count']}")
            else:
                logger.error(f"Error response: {response.text}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing API: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    if success:
        print("✅ API test completed successfully!")
    else:
        print("❌ API test failed. Check logs for details.")