"""
Enhanced FastAPI Backend for Ocean Data Integration Platform (SQLite Version)
Complete integration with SQLite, external APIs, and real-time data
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime, timedelta
import asyncio
import json
import redis
from contextlib import asynccontextmanager

# Import our modules
from backend.database_schema_sqlite import Base, User, Vessel, VesselPosition, CatchReport, SpeciesOccurrence, EDNASample, OceanographicData, WeatherData, SatelliteData
from backend.external_apis import data_sync_manager, NOAAAPI, OpenWeatherAPI
from backend.nasa_earthdata_api import nasa_earthdata_api
from backend.mapping_integration import map_visualization_api
from backend.config_sqlite import config
from backend.migration_scripts import DatabaseMigrator

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Redis connection (optional for SQLite)
redis_client = None
try:
    if config.REDIS_PASSWORD:
        redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            db=config.REDIS_DB,
            decode_responses=True
        )
    else:
        redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Continuing without caching.")
    redis_client = None

# Database setup
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Ocean Data Integration Platform (SQLite)...")
    
    # Initialize database
    try:
        migrator = DatabaseMigrator()
        migrator.create_tables()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Load ML models
    load_ml_models()
    
    # Start background sync task
    asyncio.create_task(background_data_sync())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ocean Data Integration Platform...")

app = FastAPI(
    title="Ocean Data Integration Platform API (SQLite)",
    description="Complete marine data management with real-time integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# ML Models
ml_models = {}

def load_ml_models():
    """Load ML models for predictions."""
    global ml_models
    try:
        model_path = "../models/species_sst_rf.pkl"
        scaler_path = "../models/species_sst_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            ml_models["species_model"] = joblib.load(model_path)
            ml_models["scaler"] = joblib.load(scaler_path)
            logger.info("ML models loaded successfully")
        else:
            logger.warning("ML model files not found, using mock models")
            ml_models["species_model"] = None
            ml_models["scaler"] = None
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")
        ml_models["species_model"] = None
        ml_models["scaler"] = None

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication dependency
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token."""
    # In production, implement proper JWT validation
    # For demo purposes, return a mock user
    return {"user_id": 1, "email": "admin@oceandata.in", "role": "admin"}

# Cache functions
def get_cache_key(prefix: str, **kwargs) -> str:
    """Generate cache key."""
    key_parts = [prefix]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    return ":".join(key_parts)

async def get_cached_data(key: str) -> Optional[Dict]:
    """Get data from cache."""
    try:
        if redis_client:
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)
    except Exception as e:
        logger.error(f"Cache get error: {e}")
    return None

async def set_cached_data(key: str, data: Dict, ttl: int = 3600):
    """Set data in cache."""
    try:
        if redis_client:
            redis_client.setex(key, ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.error(f"Cache set error: {e}")

# Background tasks
async def background_data_sync():
    """Background task for data synchronization."""
    while True:
        try:
            logger.info("Starting background data sync...")
            
            # Sync oceanographic data
            bbox = (-180, -90, 180, 90)  # Global
            start_date = datetime.utcnow() - timedelta(days=1)
            end_date = datetime.utcnow()
            
            sync_results = await data_sync_manager.sync_all_data(bbox, start_date, end_date)
            logger.info(f"Background sync completed: {sync_results}")
            
        except Exception as e:
            logger.error(f"Background sync error: {e}")
        
        # Wait for next sync
        await asyncio.sleep(config.SYNC_INTERVAL_MINUTES * 60)

# Pydantic models
class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

class SpeciesOccurrenceCreate(BaseModel):
    species: str
    latitude: float
    longitude: float
    event_date: datetime
    individual_count: int
    phylum: str
    class_name: str
    order_name: str
    family: str
    genus: str
    scientific_name: Optional[str] = None
    depth: Optional[float] = None
    observation_method: Optional[str] = None

class VesselPositionCreate(BaseModel):
    vessel_id: str
    latitude: float
    longitude: float
    timestamp: datetime
    heading: Optional[float] = None
    speed: Optional[float] = None
    course: Optional[float] = None
    status: Optional[str] = None

class CatchReportCreate(BaseModel):
    vessel_id: str
    fishing_date: datetime
    latitude: float
    longitude: float
    depth: Optional[float] = None
    gear_type: str
    gear_specifications: Optional[Dict[str, Any]] = None
    fishing_effort_hours: float
    total_catch_kg: float
    species_composition: Dict[str, Any]
    environmental_conditions: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    date: datetime
    depth: Optional[float] = None
    gear_type: Optional[str] = None

class PredictionResponse(BaseModel):
    predicted_species_count: float
    confidence: float
    recommended_species: List[str]
    environmental_conditions: Dict[str, Any]
    model_version: str
    prediction_timestamp: datetime

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ocean Data Integration Platform API (SQLite)",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "operational",
        "features": [
            "SQLite Database",
            "Real-time data sync",
            "External API integration",
            "Spatial data processing",
            "ML predictions",
            "Redis caching (optional)"
        ]
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "database": "connected",
        "redis": "connected" if redis_client else "not_configured",
        "ml_models": "loaded" if ml_models.get("species_model") else "mock",
        "external_apis": {
            "noaa": "configured" if config.NOAA_API_KEY else "not_configured",
            "nasa_earthdata": "configured" if config.NASA_API_KEY else "not_configured",
            "marine_traffic": "mock" if config.USE_MOCK_DATA["marine_traffic"] else "configured",
            "weather": "configured" if config.NOAA_API_KEY else "mock",
            "satellite": "configured" if config.NASA_API_KEY else "mock"
        }
    }
    
    # Test database connection
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Test Redis connection
    if redis_client:
        try:
            redis_client.ping()
        except Exception as e:
            health_status["redis"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
    
    return health_status

# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Enhanced login with JWT tokens."""
    # In production, implement proper authentication
    # For demo, use mock authentication
    if request.email == "admin@oceandata.in" and request.password == "admin123":
        return LoginResponse(
            access_token="demo_jwt_token",
            token_type="bearer",
            user={
                "email": request.email,
                "role": "admin",
                "name": "System Administrator"
            }
        )
    elif request.email == "fisherman@oceandata.in" and request.password == "fisher123":
        return LoginResponse(
            access_token="demo_jwt_token",
            token_type="bearer",
            user={
                "email": request.email,
                "role": "fisherman",
                "name": "Commercial Fisherman"
            }
        )
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Species data endpoints
@app.post("/api/species", response_model=Dict[str, Any])
async def create_species_occurrence(
    species_data: SpeciesOccurrenceCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new species occurrence record."""
    try:
        # Create database record
        db_species = SpeciesOccurrence(
            species=species_data.species,
            scientific_name=species_data.scientific_name,
            latitude=species_data.latitude,
            longitude=species_data.longitude,
            event_date=species_data.event_date,
            individual_count=species_data.individual_count,
            phylum=species_data.phylum,
            class_name=species_data.class_name,
            order_name=species_data.order_name,
            family=species_data.family,
            genus=species_data.genus,
            depth=species_data.depth,
            observation_method=species_data.observation_method,
            data_source="manual",
            is_verified=False
        )
        
        db.add(db_species)
        db.commit()
        db.refresh(db_species)
        
        logger.info(f"Created species occurrence: {db_species.id}")
        return {
            "id": db_species.id,
            "uuid": str(db_species.uuid),
            "species": db_species.species,
            "latitude": db_species.latitude,
            "longitude": db_species.longitude,
            "event_date": db_species.event_date,
            "created_at": db_species.created_at
        }
    except Exception as e:
        logger.error(f"Error creating species occurrence: {e}")
        raise HTTPException(status_code=500, detail="Failed to create species occurrence")

@app.get("/api/species", response_model=List[Dict[str, Any]])
async def get_species_occurrences(
    skip: int = 0,
    limit: int = 100,
    species: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    bbox: Optional[str] = None,  # "min_lat,min_lon,max_lat,max_lon"
    db: Session = Depends(get_db)
):
    """Get species occurrence records with spatial filtering."""
    try:
        # Check cache first
        cache_key = get_cache_key("species", skip=skip, limit=limit, species=species, 
                                 start_date=start_date, end_date=end_date, bbox=bbox)
        cached_data = await get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        # Build query
        query = db.query(SpeciesOccurrence)
        
        if species:
            query = query.filter(SpeciesOccurrence.species.ilike(f"%{species}%"))
        if start_date:
            query = query.filter(SpeciesOccurrence.event_date >= start_date)
        if end_date:
            query = query.filter(SpeciesOccurrence.event_date <= end_date)
        if bbox:
            try:
                min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
                query = query.filter(
                    SpeciesOccurrence.latitude.between(min_lat, max_lat),
                    SpeciesOccurrence.longitude.between(min_lon, max_lon)
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid bbox format")
        
        species_list = query.offset(skip).limit(limit).all()
        
        # Format response
        result = []
        for species_record in species_list:
            result.append({
                "id": species_record.id,
                "uuid": str(species_record.uuid),
                "species": species_record.species,
                "scientific_name": species_record.scientific_name,
                "latitude": species_record.latitude,
                "longitude": species_record.longitude,
                "event_date": species_record.event_date,
                "individual_count": species_record.individual_count,
                "phylum": species_record.phylum,
                "class_name": species_record.class_name,
                "depth": species_record.depth,
                "sst_at_point": species_record.sst_at_point,
                "created_at": species_record.created_at
            })
        
        # Cache result
        await set_cached_data(cache_key, result, config.CACHE_TTL["weather_data"])
        
        return result
    except Exception as e:
        logger.error(f"Error fetching species occurrences: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch species occurrences")

# ML Prediction endpoints
@app.post("/api/predict", response_model=PredictionResponse)
async def predict_species_abundance(
    prediction_request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict species abundance using ML models."""
    try:
        # Get environmental data for the location
        environmental_data = await get_environmental_data(
            prediction_request.latitude,
            prediction_request.longitude,
            prediction_request.date
        )
        
        # Prepare features for ML model
        features = prepare_ml_features(prediction_request, environmental_data)
        
        if ml_models["species_model"] and ml_models["scaler"]:
            # Use trained model
            features_scaled = ml_models["scaler"].transform([features])
            prediction = ml_models["species_model"].predict(features_scaled)[0]
            confidence = 0.85
        else:
            # Mock prediction
            prediction = 15 + (environmental_data.get("sst", 28) - 28) * 2 + environmental_data.get("biodiversity", 0.5) * 10
            confidence = 0.75
        
        # Get recommended species based on location and season
        recommended_species = get_recommended_species(
            prediction_request.latitude,
            prediction_request.longitude,
            prediction_request.date
        )
        
        return PredictionResponse(
            predicted_species_count=float(prediction),
            confidence=confidence,
            recommended_species=recommended_species,
            environmental_conditions=environmental_data,
            model_version="2.0.0",
            prediction_timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to make prediction")

# Analytics endpoints
@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(db: Session = Depends(get_db)):
    """Get comprehensive dashboard analytics."""
    try:
        # Check cache first
        cache_key = "dashboard_analytics"
        cached_data = await get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        # Get analytics data
        analytics = {
            "species": {
                "total_records": db.query(SpeciesOccurrence).count(),
                "unique_species": db.query(SpeciesOccurrence.species).distinct().count(),
                "recent_observations": db.query(SpeciesOccurrence).filter(
                    SpeciesOccurrence.event_date >= datetime.utcnow() - timedelta(days=30)
                ).count()
            },
            "vessels": {
                "total_vessels": db.query(Vessel).count(),
                "active_vessels": db.query(Vessel).filter(Vessel.is_active == True).count(),
                "total_positions": db.query(VesselPosition).count(),
                "recent_positions": db.query(VesselPosition).filter(
                    VesselPosition.timestamp >= datetime.utcnow() - timedelta(hours=24)
                ).count()
            },
            "catch_reports": {
                "total_reports": db.query(CatchReport).count(),
                "total_catch_kg": db.query(db.func.sum(CatchReport.total_catch_kg)).scalar() or 0,
                "recent_reports": db.query(CatchReport).filter(
                    CatchReport.fishing_date >= datetime.utcnow() - timedelta(days=7)
                ).count()
            },
            "oceanographic": {
                "total_records": db.query(OceanographicData).count(),
                "recent_data": db.query(OceanographicData).filter(
                    OceanographicData.measurement_date >= datetime.utcnow() - timedelta(days=1)
                ).count()
            }
        }
        
        # Cache result
        await set_cached_data(cache_key, analytics, config.CACHE_TTL["weather_data"])
        
        return analytics
    except Exception as e:
        logger.error(f"Error fetching dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard analytics")

# Helper functions
async def get_environmental_data(lat: float, lon: float, date: datetime) -> Dict[str, Any]:
    """Get environmental data for a location and date."""
    try:
        # Try to get real data from NOAA
        if config.NOAA_API_KEY:
            async with NOAAAPI(config.NOAA_API_KEY) as noaa:
                sst_data = await noaa.get_sea_surface_temperature(lat, lon, date, date)
                if sst_data:
                    return {
                        "sst": sst_data[0].get("value", 28.0) if sst_data else 28.0,
                        "salinity": 35.0,
                        "biodiversity": 0.75,
                        "data_source": "noaa"
                    }
        
        # Return mock data
        return {
            "sst": 28.0 + np.random.normal(0, 2),
            "salinity": 35.0 + np.random.normal(0, 1),
            "biodiversity": 0.75 + np.random.normal(0, 0.1),
            "data_source": "mock"
        }
    except Exception as e:
        logger.error(f"Error getting environmental data: {e}")
        return {
            "sst": 28.0,
            "salinity": 35.0,
            "biodiversity": 0.75,
            "data_source": "default"
        }

def prepare_ml_features(prediction_request: PredictionRequest, environmental_data: Dict[str, Any]) -> List[float]:
    """Prepare features for ML model."""
    # Extract features from request and environmental data
    features = [
        environmental_data.get("sst", 28.0),
        environmental_data.get("biodiversity", 0.75),
        environmental_data.get("salinity", 35.0),
        prediction_request.depth or 0.0,
        1.0,  # season (simplified)
        1.0,  # sst_category
        1.0,  # biodiversity_category
        0.0,  # prev_month_sst
        0.0,  # prev_month_species_count
        0.0,  # sst_trend
        0.0   # species_trend
    ]
    return features

def get_recommended_species(lat: float, lon: float, date: datetime) -> List[str]:
    """Get recommended species for location and season."""
    # Simple recommendation based on location
    if 5 <= lat <= 25 and 70 <= lon <= 90:  # Indian Ocean
        return ["Yellowfin Tuna", "Skipjack Tuna", "Indian Mackerel", "Pomfret", "Prawn"]
    else:
        return ["Tuna", "Mackerel", "Sardine", "Anchovy"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
