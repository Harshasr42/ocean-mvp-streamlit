from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = joblib.load("../models/species_abundance_model.pkl")
encoder = joblib.load("../models/species_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        sst = float(data["sst"])
        
        # Create feature array
        features = np.array([[latitude, longitude, sst]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get species name
        species_idx = int(round(prediction))
        if 0 <= species_idx < len(encoder.classes_):
            species_name = encoder.classes_[species_idx]
        else:
            species_name = "Unknown"
        
        return jsonify({
            "predicted_species": species_name,
            "confidence": float(prediction),
            "latitude": latitude,
            "longitude": longitude,
            "sst": sst
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8507, debug=True)