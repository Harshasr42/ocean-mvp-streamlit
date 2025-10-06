"""
Enhanced FastAPI Backend for Ocean Data Integration Platform
Complete integration with PostgreSQL/PostGIS, external APIs, and real-time data
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
from database_schema import Base, User, Vessel, VesselPosition, CatchReport, SpeciesOccurrence, EDNASample, OceanographicData, WeatherData, SatelliteData
from external_apis import data_sync_manager, NOAAAPI, OpenWeatherAPI
from nasa_earthdata_api import nasa_earthdata_api
from mapping_integration import map_visualization_api
from config import config
from migration_scripts import DatabaseMigrator

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Redis connection
redis_client = None
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

# Database setup
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Ocean Data Integration Platform...")
    
    # Initialize database
    try:
        migrator = DatabaseMigrator()
        migrator.create_postgis_extension()
        migrator.create_tables()
        migrator.create_spatial_indexes()
        migrator.create_additional_indexes()
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
    title="Ocean Data Integration Platform API",
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
        "message": "Ocean Data Integration Platform API",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "operational",
        "features": [
            "PostgreSQL with PostGIS",
            "Real-time data sync",
            "External API integration",
            "Spatial data processing",
            "ML predictions",
            "Redis caching"
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
        # Create geometry from lat/lon
        geometry = f"POINT({species_data.longitude} {species_data.latitude})"
        
        # Create database record
        db_species = SpeciesOccurrence(
            species=species_data.species,
            scientific_name=species_data.scientific_name,
            latitude=species_data.latitude,
            longitude=species_data.longitude,
            geometry=geometry,
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

# Vessel tracking endpoints
@app.post("/api/vessels/positions", response_model=Dict[str, Any])
async def create_vessel_position(
    position_data: VesselPositionCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new vessel position record."""
    try:
        # Get vessel by vessel_id
        vessel = db.query(Vessel).filter(Vessel.vessel_id == position_data.vessel_id).first()
        if not vessel:
            raise HTTPException(status_code=404, detail="Vessel not found")
        
        # Create geometry
        geometry = f"POINT({position_data.longitude} {position_data.latitude})"
        
        # Create position record
        db_position = VesselPosition(
            vessel_id=vessel.id,
            timestamp=position_data.timestamp,
            latitude=position_data.latitude,
            longitude=position_data.longitude,
            geometry=geometry,
            heading=position_data.heading,
            speed=position_data.speed,
            course=position_data.course,
            status=position_data.status,
            ais_source="manual"
        )
        
        db.add(db_position)
        db.commit()
        db.refresh(db_position)
        
        logger.info(f"Created vessel position: {db_position.id}")
        return {
            "id": db_position.id,
            "vessel_id": position_data.vessel_id,
            "latitude": db_position.latitude,
            "longitude": db_position.longitude,
            "timestamp": db_position.timestamp,
            "heading": db_position.heading,
            "speed": db_position.speed,
            "created_at": db_position.created_at
        }
    except Exception as e:
        logger.error(f"Error creating vessel position: {e}")
        raise HTTPException(status_code=500, detail="Failed to create vessel position")

@app.get("/api/vessels/positions", response_model=List[Dict[str, Any]])
async def get_vessel_positions(
    vessel_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    bbox: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get vessel position records."""
    try:
        query = db.query(VesselPosition)
        
        if vessel_id:
            vessel = db.query(Vessel).filter(Vessel.vessel_id == vessel_id).first()
            if vessel:
                query = query.filter(VesselPosition.vessel_id == vessel.id)
        if start_date:
            query = query.filter(VesselPosition.timestamp >= start_date)
        if end_date:
            query = query.filter(VesselPosition.timestamp <= end_date)
        if bbox:
            try:
                min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
                query = query.filter(
                    VesselPosition.latitude.between(min_lat, max_lat),
                    VesselPosition.longitude.between(min_lon, max_lon)
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid bbox format")
        
        positions = query.order_by(VesselPosition.timestamp.desc()).limit(limit).all()
        
        result = []
        for position in positions:
            vessel = db.query(Vessel).filter(Vessel.id == position.vessel_id).first()
            result.append({
                "id": position.id,
                "vessel_id": vessel.vessel_id if vessel else None,
                "vessel_name": vessel.name if vessel else None,
                "latitude": position.latitude,
                "longitude": position.longitude,
                "timestamp": position.timestamp,
                "heading": position.heading,
                "speed": position.speed,
                "status": position.status
            })
        
        return result
    except Exception as e:
        logger.error(f"Error fetching vessel positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vessel positions")

# Catch report endpoints
@app.post("/api/catch-reports", response_model=Dict[str, Any])
async def create_catch_report(
    catch_data: CatchReportCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new catch report."""
    try:
        # Get vessel
        vessel = db.query(Vessel).filter(Vessel.vessel_id == catch_data.vessel_id).first()
        if not vessel:
            raise HTTPException(status_code=404, detail="Vessel not found")
        
        # Create geometry
        geometry = f"POINT({catch_data.longitude} {catch_data.latitude})"
        
        # Create catch report
        db_catch = CatchReport(
            user_id=current_user["user_id"],
            vessel_id=vessel.id,
            report_date=datetime.utcnow(),
            fishing_date=catch_data.fishing_date,
            latitude=catch_data.latitude,
            longitude=catch_data.longitude,
            geometry=geometry,
            depth=catch_data.depth,
            gear_type=catch_data.gear_type,
            gear_specifications=catch_data.gear_specifications,
            fishing_effort_hours=catch_data.fishing_effort_hours,
            total_catch_kg=catch_data.total_catch_kg,
            species_composition=catch_data.species_composition,
            environmental_conditions=catch_data.environmental_conditions,
            is_verified=False
        )
        
        db.add(db_catch)
        db.commit()
        db.refresh(db_catch)
        
        logger.info(f"Created catch report: {db_catch.id}")
        return {
            "id": db_catch.id,
            "uuid": str(db_catch.uuid),
            "vessel_id": catch_data.vessel_id,
            "fishing_date": db_catch.fishing_date,
            "latitude": db_catch.latitude,
            "longitude": db_catch.longitude,
            "total_catch_kg": db_catch.total_catch_kg,
            "gear_type": db_catch.gear_type,
            "created_at": db_catch.created_at
        }
    except Exception as e:
        logger.error(f"Error creating catch report: {e}")
        raise HTTPException(status_code=500, detail="Failed to create catch report")

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

# External data sync endpoints
@app.post("/api/sync/oceanographic")
async def sync_oceanographic_data(
    bbox: str,  # "min_lat,min_lon,max_lat,max_lon"
    start_date: datetime,
    end_date: datetime,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Trigger oceanographic data synchronization."""
    try:
        # Parse bounding box
        min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
        bbox_tuple = (min_lat, min_lon, max_lat, max_lon)
        
        # Start background sync
        background_tasks.add_task(
            data_sync_manager.sync_oceanographic_data,
            bbox_tuple,
            start_date,
            end_date
        )
        
        return {
            "message": "Oceanographic data sync started",
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error starting oceanographic sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to start data sync")

@app.post("/api/sync/vessels")
async def sync_vessel_data(
    bbox: str,
    background_tasks: BackgroundTasks,
    vessel_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Trigger vessel data synchronization."""
    try:
        min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
        bbox_tuple = (min_lat, min_lon, max_lat, max_lon)
        
        background_tasks.add_task(
            data_sync_manager.sync_vessel_data,
            bbox_tuple,
            vessel_type
        )
        
        return {
            "message": "Vessel data sync started",
            "bbox": bbox,
            "vessel_type": vessel_type,
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error starting vessel sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to start vessel sync")

# Mapping endpoints
@app.get("/api/map/config")
async def get_map_config():
    """Get map configuration for frontend."""
    try:
        config_data = map_visualization_api.get_map_config()
        return config_data
    except Exception as e:
        logger.error(f"Error getting map config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get map configuration")

@app.get("/api/map/species")
async def get_species_map_data(
    bbox: Optional[str] = None,
    species: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get species data formatted for mapping."""
    try:
        # Get species data from database
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
        
        species_data = query.limit(1000).all()
        
        # Convert to dict format
        species_list = []
        for record in species_data:
            species_list.append({
                "species": record.species,
                "scientific_name": record.scientific_name,
                "latitude": record.latitude,
                "longitude": record.longitude,
                "event_date": record.event_date,
                "individual_count": record.individual_count,
                "depth": record.depth,
                "sst_at_point": record.sst_at_point,
                "phylum": record.phylum,
                "class_name": record.class_name
            })
        
        # Process for mapping
        map_data = map_visualization_api.get_species_map_data(species_list)
        return map_data
        
    except Exception as e:
        logger.error(f"Error getting species map data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get species map data")

@app.get("/api/map/vessels")
async def get_vessel_map_data(
    bbox: Optional[str] = None,
    vessel_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get vessel data formatted for mapping."""
    try:
        # Get vessel positions from database
        query = db.query(VesselPosition)
        
        if start_date:
            query = query.filter(VesselPosition.timestamp >= start_date)
        if end_date:
            query = query.filter(VesselPosition.timestamp <= end_date)
        if bbox:
            try:
                min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
                query = query.filter(
                    VesselPosition.latitude.between(min_lat, max_lat),
                    VesselPosition.longitude.between(min_lon, max_lon)
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid bbox format")
        
        positions = query.order_by(VesselPosition.timestamp.desc()).limit(1000).all()
        
        # Convert to dict format
        vessel_data = []
        for position in positions:
            vessel = db.query(Vessel).filter(Vessel.id == position.vessel_id).first()
            vessel_data.append({
                "vessel_id": vessel.vessel_id if vessel else None,
                "vessel_name": vessel.name if vessel else None,
                "vessel_type": vessel.vessel_type if vessel else None,
                "latitude": position.latitude,
                "longitude": position.longitude,
                "timestamp": position.timestamp,
                "heading": position.heading,
                "speed": position.speed,
                "status": position.status
            })
        
        # Process for mapping
        map_data = map_visualization_api.get_vessel_map_data(vessel_data)
        return map_data
        
    except Exception as e:
        logger.error(f"Error getting vessel map data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get vessel map data")

@app.get("/api/map/combined")
async def get_combined_map_data(
    bbox: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    include_species: bool = True,
    include_vessels: bool = True,
    include_oceanographic: bool = True,
    db: Session = Depends(get_db)
):
    """Get combined map data from all sources."""
    try:
        species_data = None
        vessel_data = None
        ocean_data = None
        
        if include_species:
            # Get species data
            species_query = db.query(SpeciesOccurrence)
            if start_date:
                species_query = species_query.filter(SpeciesOccurrence.event_date >= start_date)
            if end_date:
                species_query = species_query.filter(SpeciesOccurrence.event_date <= end_date)
            if bbox:
                min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
                species_query = species_query.filter(
                    SpeciesOccurrence.latitude.between(min_lat, max_lat),
                    SpeciesOccurrence.longitude.between(min_lon, max_lon)
                )
            
            species_records = species_query.limit(500).all()
            species_data = []
            for record in species_records:
                species_data.append({
                    "species": record.species,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "event_date": record.event_date,
                    "individual_count": record.individual_count,
                    "sst_at_point": record.sst_at_point
                })
        
        if include_vessels:
            # Get vessel data
            vessel_query = db.query(VesselPosition)
            if start_date:
                vessel_query = vessel_query.filter(VesselPosition.timestamp >= start_date)
            if end_date:
                vessel_query = vessel_query.filter(VesselPosition.timestamp <= end_date)
            if bbox:
                min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
                vessel_query = vessel_query.filter(
                    VesselPosition.latitude.between(min_lat, max_lat),
                    VesselPosition.longitude.between(min_lon, max_lon)
                )
            
            vessel_records = vessel_query.limit(500).all()
            vessel_data = []
            for position in vessel_records:
                vessel = db.query(Vessel).filter(Vessel.id == position.vessel_id).first()
                vessel_data.append({
                    "vessel_id": vessel.vessel_id if vessel else None,
                    "latitude": position.latitude,
                    "longitude": position.longitude,
                    "timestamp": position.timestamp,
                    "heading": position.heading,
                    "speed": position.speed
                })
        
        if include_oceanographic:
            # Get oceanographic data
            ocean_query = db.query(OceanographicData)
            if start_date:
                ocean_query = ocean_query.filter(OceanographicData.measurement_date >= start_date)
            if end_date:
                ocean_query = ocean_query.filter(OceanographicData.measurement_date <= end_date)
            if bbox:
                min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
                ocean_query = ocean_query.filter(
                    OceanographicData.latitude.between(min_lat, max_lat),
                    OceanographicData.longitude.between(min_lon, max_lon)
                )
            
            ocean_records = ocean_query.limit(500).all()
            ocean_data = []
            for record in ocean_records:
                ocean_data.append({
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "measurement_date": record.measurement_date,
                    "sea_surface_temperature": record.sea_surface_temperature,
                    "sea_surface_salinity": record.sea_surface_salinity,
                    "wave_height": record.wave_height,
                    "wind_speed": record.wind_speed
                })
        
        # Get combined map data
        combined_data = map_visualization_api.get_combined_map_data(
            species_data, vessel_data, ocean_data
        )
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Error getting combined map data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get combined map data")

# Satellite data endpoints
@app.get("/api/satellite/search")
async def search_satellite_data(
    bbox: str,  # "min_lat,min_lon,max_lat,max_lon"
    start_date: datetime,
    end_date: datetime,
    product_types: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Search for satellite data using NASA Earthdata."""
    try:
        # Parse bounding box
        min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
        bbox_tuple = (min_lat, min_lon, max_lat, max_lon)
        
        # Parse product types
        products = None
        if product_types:
            products = [p.strip() for p in product_types.split(',')]
        
        # Search satellite data
        async with nasa_earthdata_api as nasa:
            results = await nasa.search_satellite_data(
                bbox_tuple, start_date, end_date, products
            )
        
        return {
            "results": results,
            "count": len(results),
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "search_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error searching satellite data: {e}")
        raise HTTPException(status_code=500, detail="Failed to search satellite data")

@app.get("/api/satellite/sst")
async def get_satellite_sst(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    current_user: dict = Depends(get_current_user)
):
    """Get sea surface temperature from satellite data."""
    try:
        async with nasa_earthdata_api as nasa:
            sst_data = await nasa.get_sea_surface_temperature(
                lat, lon, start_date, end_date
            )
        
        return {
            "location": {"latitude": lat, "longitude": lon},
            "sst_data": sst_data,
            "count": len(sst_data),
            "date_range": {"start": start_date, "end": end_date}
        }
        
    except Exception as e:
        logger.error(f"Error getting satellite SST data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get satellite SST data")

@app.get("/api/satellite/ocean-color")
async def get_satellite_ocean_color(
    bbox: str,
    start_date: datetime,
    end_date: datetime,
    current_user: dict = Depends(get_current_user)
):
    """Get ocean color (chlorophyll) data from satellite."""
    try:
        min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
        bbox_tuple = (min_lat, min_lon, max_lat, max_lon)
        
        async with nasa_earthdata_api as nasa:
            color_data = await nasa.get_ocean_color_data(
                bbox_tuple, start_date, end_date
            )
        
        return {
            "bbox": bbox,
            "color_data": color_data,
            "count": len(color_data),
            "date_range": {"start": start_date, "end": end_date}
        }
        
    except Exception as e:
        logger.error(f"Error getting satellite ocean color data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get satellite ocean color data")

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

# =========================
# eDNA Endpoints
# =========================

class EDNASampleCreate(BaseModel):
    sample_id: str
    project_name: Optional[str] = None
    researcher: Optional[str] = None
    organization: Optional[str] = None
    latitude: float
    longitude: float
    sample_date: datetime
    depth: Optional[float] = None
    volume_filtered_l: Optional[float] = None
    filter_type: Optional[str] = None
    preservation_method: Optional[str] = None
    storage_temperature: Optional[float] = None
    dna_extraction_method: Optional[str] = None
    sequencing_platform: Optional[str] = None
    sequencing_depth: Optional[int] = None
    biodiversity_index: Optional[float] = None
    species_richness: Optional[int] = None
    genetic_diversity: Optional[float] = None
    dominant_species: Optional[str] = None
    species_detected: Optional[List[str]] = None
    abundance_data: Optional[Dict[str, Any]] = None
    environmental_parameters: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/edna", response_model=Dict[str, Any])
async def create_edna_sample(
    sample: EDNASampleCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new eDNA sample record."""
    try:
        geometry = f"POINT({sample.longitude} {sample.latitude})"

        db_sample = EDNASample(
            sample_id=sample.sample_id,
            project_name=sample.project_name,
            researcher=sample.researcher,
            organization=sample.organization,
            latitude=sample.latitude,
            longitude=sample.longitude,
            geometry=geometry,
            sample_date=sample.sample_date,
            depth=sample.depth,
            volume_filtered_l=sample.volume_filtered_l,
            filter_type=sample.filter_type,
            preservation_method=sample.preservation_method,
            storage_temperature=sample.storage_temperature,
            dna_extraction_method=sample.dna_extraction_method,
            sequencing_platform=sample.sequencing_platform,
            sequencing_depth=sample.sequencing_depth,
            biodiversity_index=sample.biodiversity_index,
            species_richness=sample.species_richness,
            genetic_diversity=sample.genetic_diversity,
            dominant_species=sample.dominant_species,
            species_detected=sample.species_detected,
            abundance_data=sample.abundance_data,
            environmental_parameters=sample.environmental_parameters,
            quality_metrics=sample.quality_metrics,
            metadata=sample.metadata,
            is_processed=False
        )

        db.add(db_sample)
        db.commit()
        db.refresh(db_sample)

        return {
            "id": db_sample.id,
            "uuid": str(db_sample.uuid),
            "sample_id": db_sample.sample_id,
            "latitude": db_sample.latitude,
            "longitude": db_sample.longitude,
            "sample_date": db_sample.sample_date,
            "created_at": db_sample.created_at
        }
    except Exception as e:
        logger.error(f"Error creating eDNA sample: {e}")
        raise HTTPException(status_code=500, detail="Failed to create eDNA sample")


@app.get("/api/edna", response_model=List[Dict[str, Any]])
async def get_edna_samples(
    skip: int = 0,
    limit: int = 100,
    project_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    bbox: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get eDNA sample records with spatial and temporal filters."""
    try:
        query = db.query(EDNASample)

        if project_name:
            query = query.filter(EDNASample.project_name.ilike(f"%{project_name}%"))
        if start_date:
            query = query.filter(EDNASample.sample_date >= start_date)
        if end_date:
            query = query.filter(EDNASample.sample_date <= end_date)
        if bbox:
            try:
                min_lat, min_lon, max_lat, max_lon = map(float, bbox.split(','))
                query = query.filter(
                    EDNASample.latitude.between(min_lat, max_lat),
                    EDNASample.longitude.between(min_lon, max_lon)
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid bbox format")

        samples = query.order_by(EDNASample.sample_date.desc()).offset(skip).limit(limit).all()

        result = []
        for s in samples:
            result.append({
                "id": s.id,
                "uuid": str(s.uuid),
                "sample_id": s.sample_id,
                "project_name": s.project_name,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "sample_date": s.sample_date,
                "depth": s.depth,
                "biodiversity_index": s.biodiversity_index,
                "species_richness": s.species_richness,
                "genetic_diversity": s.genetic_diversity,
                "dominant_species": s.dominant_species,
                "species_detected": s.species_detected
            })

        return result
    except Exception as e:
        logger.error(f"Error fetching eDNA samples: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch eDNA samples")


@app.get("/api/edna/stats", response_model=Dict[str, Any])
async def get_edna_stats(
    db: Session = Depends(get_db)
):
    """Get summary statistics for eDNA samples."""
    try:
        total_samples = db.query(EDNASample).count()
        recent_samples = db.query(EDNASample).filter(
            EDNASample.sample_date >= datetime.utcnow() - timedelta(days=30)
        ).count()

        richness_avg = db.query(db.func.avg(EDNASample.species_richness)).scalar()
        biodiversity_avg = db.query(db.func.avg(EDNASample.biodiversity_index)).scalar()

        return {
            "total_samples": total_samples,
            "recent_samples": recent_samples,
            "avg_species_richness": float(richness_avg or 0),
            "avg_biodiversity_index": float(biodiversity_avg or 0)
        }
    except Exception as e:
        logger.error(f"Error fetching eDNA stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch eDNA stats")


@app.post("/api/edna/ingest-csv", response_model=Dict[str, Any])
async def ingest_edna_csv(
    file: UploadFile = File(...),
    project_name: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Bulk-ingest eDNA samples from a CSV with columns: sample_id, latitude, longitude, sample_date, depth, biodiversity_index, species_richness, genetic_diversity, dominant_species."""
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))

        required_cols = {"sample_id", "latitude", "longitude", "sample_date"}
        if not required_cols.issubset(set(df.columns)):
            raise HTTPException(status_code=400, detail=f"Missing required columns: {required_cols - set(df.columns)}")

        inserted = 0
        for _, row in df.iterrows():
            try:
                geometry = f"POINT({float(row['longitude'])} {float(row['latitude'])})"
                db_sample = EDNASample(
                    sample_id=str(row["sample_id"]),
                    project_name=project_name or row.get("project_name"),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    geometry=geometry,
                    sample_date=pd.to_datetime(row["sample_date"]).to_pydatetime(),
                    depth=float(row["depth"]) if "depth" in df.columns and pd.notna(row["depth"]) else None,
                    biodiversity_index=float(row["biodiversity_index"]) if "biodiversity_index" in df.columns and pd.notna(row["biodiversity_index"]) else None,
                    species_richness=int(row["species_richness"]) if "species_richness" in df.columns and pd.notna(row["species_richness"]) else None,
                    genetic_diversity=float(row["genetic_diversity"]) if "genetic_diversity" in df.columns and pd.notna(row["genetic_diversity"]) else None,
                    dominant_species=row.get("dominant_species") if "dominant_species" in df.columns else None,
                )
                db.add(db_sample)
                inserted += 1
            except Exception as inner_e:
                logger.warning(f"Skipping row due to error: {inner_e}")
                continue

        db.commit()

        return {"inserted": inserted}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting eDNA CSV: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest eDNA CSV")

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
