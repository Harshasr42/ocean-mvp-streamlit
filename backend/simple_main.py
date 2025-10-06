"""
Simplified FastAPI Backend for Ocean Data Integration Platform
Minimal setup for demo purposes
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

# Mock OceanDataCollector class to avoid import errors
class OceanDataCollector:
    def get_marine_weather(self, lat, lon):
        return {"sea_surface_temp": 28.0, "wave_height": 1.2}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Ocean Data Integration Platform API",
    description="REST API for marine biodiversity, fisheries, and ocean data management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - Fixed for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:8502", "http://127.0.0.1:8502"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SpeciesOccurrenceCreate(BaseModel):
    species: str
    latitude: float
    longitude: float
    event_date: str
    individual_count: int
    phylum: str
    class_name: str
    order_name: str
    family: str
    genus: str

class SpeciesOccurrenceResponse(BaseModel):
    id: int
    species: str
    latitude: float
    longitude: float
    event_date: str
    individual_count: int
    sst_at_point: Optional[float] = None
    created_at: str

class PredictionRequest(BaseModel):
    mean_sst: float
    biodiversity_index: float
    genetic_diversity: float
    species_richness: int
    season: str
    sst_category: str
    biodiversity_category: str

class PredictionResponse(BaseModel):
    predicted_species_count: float
    confidence: float
    model_version: str
    prediction_timestamp: str

class CatchReportCreate(BaseModel):
    species: str
    latitude: float
    longitude: float
    catch_weight: float
    individual_count: int
    gear_type: str
    vessel_type: str
    fishing_depth: int
    timestamp: str

class CatchReportResponse(BaseModel):
    id: int
    species: str
    latitude: float
    longitude: float
    catch_weight: float
    individual_count: int
    gear_type: str
    vessel_type: str
    fishing_depth: int
    timestamp: str
    created_at: str

# Mock data storage
species_data = []
vessels_data = []
edna_data = []
catch_reports = []

# Initialize ocean data collector
ocean_collector = OceanDataCollector()

# Load sample data
def load_sample_data():
    """Load sample data from CSV files."""
    global species_data, vessels_data, edna_data
    
    try:
        # Load species data
        if os.path.exists("../data/obis_occurrences.csv"):
            df = pd.read_csv("../data/obis_occurrences.csv")
            species_data = df.to_dict('records')
            logger.info(f"Loaded {len(species_data)} species records")
        
        # Load vessels data
        if os.path.exists("../data/vessels_demo.csv"):
            df = pd.read_csv("../data/vessels_demo.csv")
            vessels_data = df.to_dict('records')
            logger.info(f"Loaded {len(vessels_data)} vessel records")
        
        # Load eDNA data
        if os.path.exists("../data/edna_demo.csv"):
            df = pd.read_csv("../data/edna_demo.csv")
            edna_data = df.to_dict('records')
            logger.info(f"Loaded {len(edna_data)} eDNA records")
            
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        # Create mock data if files don't exist
        create_mock_data()

def create_mock_data():
    """Create mock data for demonstration."""
    global species_data, vessels_data, edna_data
    
    # Mock species data
    species_data = [
        {
            "id": 1,
            "species": "Thunnus albacares",
            "latitude": 12.5,
            "longitude": 77.2,
            "event_date": "2023-01-15",
            "individual_count": 3,
            "sst_at_point": 28.5,
            "created_at": datetime.now().isoformat()
        },
        {
            "id": 2,
            "species": "Scomberomorus commerson",
            "latitude": 12.8,
            "longitude": 77.5,
            "event_date": "2023-01-20",
            "individual_count": 2,
            "sst_at_point": 29.1,
            "created_at": datetime.now().isoformat()
        }
    ]
    
    # Mock vessels data
    vessels_data = [
        {
            "id": 1,
            "vessel_id": "V001",
            "latitude": 12.5,
            "longitude": 77.2,
            "timestamp": "2023-01-15T08:30:00",
            "catch_kg": 150.5,
            "gear_type": "longline",
            "vessel_type": "commercial",
            "created_at": datetime.now().isoformat()
        }
    ]
    
    # Mock eDNA data
    edna_data = [
        {
            "id": 1,
            "sample_id": "EDNA001",
            "latitude": 12.5,
            "longitude": 77.2,
            "sample_date": "2023-01-15",
            "biodiversity_index": 0.75,
            "species_richness": 12,
            "genetic_diversity": 0.68,
            "dominant_species": "Thunnus albacares",
            "created_at": datetime.now().isoformat()
        }
    ]

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    load_sample_data()
    logger.info("Ocean Data API started successfully")
    print("✅ FastAPI running on http://127.0.0.1:8000")
    print("📊 API Documentation: http://127.0.0.1:8000/docs")
    print("🔧 Health Check: http://127.0.0.1:8000/health")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ocean Data Integration Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "mock",
        "ml_model": "mock"
    }

# Species Data Endpoints

@app.post("/api/species", response_model=SpeciesOccurrenceResponse)
async def create_species_occurrence(species_data_input: SpeciesOccurrenceCreate):
    """Create a new species occurrence record."""
    try:
        new_species = {
            "id": len(species_data) + 1,
            "species": species_data_input.species,
            "latitude": species_data_input.latitude,
            "longitude": species_data_input.longitude,
            "event_date": species_data_input.event_date,
            "individual_count": species_data_input.individual_count,
            "sst_at_point": 28.0 + np.random.normal(0, 1),
            "created_at": datetime.now().isoformat()
        }
        
        species_data.append(new_species)
        logger.info(f"Created species occurrence: {new_species['id']}")
        return new_species
    except Exception as e:
        logger.error(f"Error creating species occurrence: {e}")
        raise HTTPException(status_code=500, detail="Failed to create species occurrence")

@app.get("/api/species")
async def get_species_occurrences(
    skip: int = 0,
    limit: int = 100,
    species: Optional[str] = None
):
    """Get species occurrence records with filtering."""
    try:
        filtered_data = species_data
        
        if species:
            filtered_data = [s for s in species_data if species.lower() in s["species"].lower()]
        
        return filtered_data[skip:skip + limit]
    except Exception as e:
        logger.error(f"Error fetching species occurrences: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch species occurrences")

@app.get("/api/species/stats")
async def get_species_stats():
    """Get species occurrence statistics."""
    try:
        total_records = len(species_data)
        unique_species = len(set([s["species"] for s in species_data]))
        
        return {
            "total_records": total_records,
            "unique_species": unique_species,
            "date_range": {
                "start": min([s["event_date"] for s in species_data]) if species_data else None,
                "end": max([s["event_date"] for s in species_data]) if species_data else None
            }
        }
    except Exception as e:
        logger.error(f"Error fetching species stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch species statistics")

# Vessel Data Endpoints

@app.get("/api/vessels")
async def get_vessel_records(skip: int = 0, limit: int = 100):
    """Get vessel tracking records."""
    try:
        return vessels_data[skip:skip + limit]
    except Exception as e:
        logger.error(f"Error fetching vessel records: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vessel records")

# eDNA Data Endpoints

@app.get("/api/edna")
async def get_edna_samples(skip: int = 0, limit: int = 100):
    """Get eDNA sample records."""
    try:
        return edna_data[skip:skip + limit]
    except Exception as e:
        logger.error(f"Error fetching eDNA samples: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch eDNA samples")

# AI/ML Prediction Endpoints

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_species_abundance(prediction_request: PredictionRequest):
    """Predict species abundance based on environmental parameters."""
    try:
        # Mock prediction logic
        base_prediction = 15
        sst_factor = (prediction_request.mean_sst - 28) * 2
        biodiversity_factor = prediction_request.biodiversity_index * 10
        season_factor = {"Winter": -2, "Spring": 0, "Summer": 3, "Autumn": 1}[prediction_request.season]
        
        prediction = base_prediction + sst_factor + biodiversity_factor + season_factor
        prediction = max(0, prediction)  # Ensure non-negative
        
        return PredictionResponse(
            predicted_species_count=float(prediction),
            confidence=0.85,
            model_version="1.0.0",
            prediction_timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to make prediction")

@app.get("/api/trends")
async def get_environmental_trends():
    """Get environmental trend analysis."""
    try:
        # Mock trend data
        trends = {
            "sst_trend": {
                "current": 28.5,
                "change": 0.3,
                "trend": "increasing"
            },
            "biodiversity_trend": {
                "current": 0.75,
                "change": 0.05,
                "trend": "stable"
            },
            "species_count": {
                "current": 18,
                "change": 2,
                "trend": "increasing"
            }
        }
        
        return trends
    except Exception as e:
        logger.error(f"Error fetching trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch trends")

# Analytics Endpoints

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    """Get dashboard analytics data."""
    try:
        return {
            "species": {
                "total_records": len(species_data),
                "unique_species": len(set([s["species"] for s in species_data]))
            },
            "vessels": {
                "total_records": len(vessels_data),
                "total_catch_kg": sum([v.get("catch_kg", 0) for v in vessels_data])
            },
            "edna": {
                "total_samples": len(edna_data),
                "avg_biodiversity_index": np.mean([e.get("biodiversity_index", 0) for e in edna_data]) if edna_data else 0
            },
            "catch_reports": {
                "total_reports": len(catch_reports),
                "total_weight": sum([c.get("catch_weight", 0) for c in catch_reports])
            }
        }
    except Exception as e:
        logger.error(f"Error fetching dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard analytics")

# Catch Report Endpoints

@app.post("/api/catch-reports")
async def create_catch_report(catch_data: dict):
    """Create a new catch report - accepts any JSON data."""
    try:
        # Add required fields if not present
        catch_data["id"] = len(catch_reports) + 1
        catch_data["created_at"] = datetime.now().isoformat()
        
        # Add timestamp if not provided
        if "timestamp" not in catch_data:
            catch_data["timestamp"] = datetime.now().isoformat()
        
        # Ensure required fields have defaults
        if "species" not in catch_data:
            catch_data["species"] = "Unknown"
        if "latitude" not in catch_data:
            catch_data["latitude"] = 0.0
        if "longitude" not in catch_data:
            catch_data["longitude"] = 0.0
        if "catch_weight" not in catch_data:
            catch_data["catch_weight"] = 0.0
        if "individual_count" not in catch_data:
            catch_data["individual_count"] = 1
        if "gear_type" not in catch_data:
            catch_data["gear_type"] = "Unknown"
        if "vessel_type" not in catch_data:
            catch_data["vessel_type"] = "Unknown"
        if "fishing_depth" not in catch_data:
            catch_data["fishing_depth"] = 0
            
        # Add to catch reports
        catch_reports.append(catch_data)
        
        logger.info(f"Created catch report: {catch_data['id']} - Species: {catch_data.get('species', 'Unknown')}")
        return {
            "id": catch_data["id"],
            "message": "Catch report submitted successfully",
            "status": "success",
            "data": catch_data
        }
    except Exception as e:
        logger.error(f"Error creating catch report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create catch report: {str(e)}")

@app.get("/api/catch-reports")
async def get_catch_reports():
    """Get all catch reports."""
    try:
        return catch_reports
    except Exception as e:
        logger.error(f"Error fetching catch reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch catch reports")

# Ocean Data Endpoints

@app.get("/api/ocean-data")
async def get_ocean_data(lat: float, lon: float):
    """Get real-time ocean data for a location."""
    try:
        marine_data = ocean_collector.get_marine_weather(lat, lon)
        if marine_data:
            return marine_data
        else:
            # Return mock data if API fails
            return {
                'sea_surface_temp': 28.0 + (lat - 12) * 0.1,
                'wave_height': 1.2,
                'wave_period': 8.5,
                'wind_speed': 12.0,
                'wind_direction': 135.0,
                'humidity': 75.0,
                'pressure': 1013.0,
                'visibility': 10.0,
                'sea_state': 'Moderate'
            }
    except Exception as e:
        logger.error(f"Error fetching ocean data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ocean data")

# Vessel Endpoints
@app.post("/api/vessels")
async def create_vessel_position(vessel_data: dict):
    """Create a new vessel position."""
    try:
        vessel_data["id"] = len(vessels_data) + 1
        vessel_data["created_at"] = datetime.now().isoformat()
        vessels_data.append(vessel_data)
        
        logger.info(f"Created vessel position: {vessel_data['id']}")
        return vessel_data
    except Exception as e:
        logger.error(f"Error creating vessel position: {e}")
        raise HTTPException(status_code=500, detail="Failed to create vessel position")

# eDNA Endpoints
@app.post("/api/edna")
async def create_edna_sample(edna_sample: dict):
    """Create a new eDNA sample."""
    try:
        edna_sample["id"] = len(edna_data) + 1
        edna_sample["created_at"] = datetime.now().isoformat()
        edna_data.append(edna_sample)
        
        logger.info(f"Created eDNA sample: {edna_sample['id']}")
        return edna_sample
    except Exception as e:
        logger.error(f"Error creating eDNA sample: {e}")
        raise HTTPException(status_code=500, detail="Failed to create eDNA sample")

# Initialize catch reports storage
catch_reports = []

import uvicorn

if __name__ == "__main__":
    uvicorn.run("simple_main:app", host="127.0.0.1", port=8000, reload=False)
