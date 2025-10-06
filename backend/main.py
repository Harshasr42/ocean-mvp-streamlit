"""
FastAPI Backend for Ocean Data Integration Platform
REST API endpoints for marine data management and AI predictions
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime, timedelta
import asyncio
from supabase import create_client, Client
import json
import requests
import io

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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predict API Configuration
PREDICT_API_URL = "http://localhost:8507"  # Updated predict API URL

# Security
security = HTTPBearer()

# Auth models
class LoginRequest(BaseModel):
    email: str
    password: str

# Mock user database
MOCK_USERS = {
    "demo@oceandata.in": {
        "password": "demo123",
        "role": "fisherman",
        "name": "Demo User"
    }
}

# Auth endpoints
@app.post("/auth/login")
async def login(request: LoginRequest):
    """Login endpoint that accepts email and password"""
    user = MOCK_USERS.get(request.email)
    if user and user["password"] == request.password:
        return {
            "access_token": "demo_token",
            "token_type": "bearer",
            "user": {
                "email": request.email,
                "role": user["role"],
                "name": user["name"]
            }
        }
    raise HTTPException(
        status_code=401,
        detail="Invalid credentials"
    )

# Predict API proxy endpoints
@app.get("/api/predict/health")
async def predict_health():
    try:
        response = requests.get(f"{PREDICT_API_URL}/health")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/api/predict")
async def predict_proxy(data: Dict[str, Any]):
    try:
        response = requests.post(f"{PREDICT_API_URL}/predict", json=data)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# Database models
Base = declarative_base()

class SpeciesOccurrence(Base):
    __tablename__ = "species_occurrences"
    
    id = Column(Integer, primary_key=True, index=True)
    species = Column(String, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    event_date = Column(DateTime)
    individual_count = Column(Integer)
    phylum = Column(String)
    class_name = Column(String)
    order_name = Column(String)
    family = Column(String)
    genus = Column(String)
    sst_at_point = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Vessel(Base):
    __tablename__ = "vessels"
    
    id = Column(Integer, primary_key=True, index=True)
    vessel_id = Column(String, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    timestamp = Column(DateTime)
    catch_kg = Column(Float)
    gear_type = Column(String)
    vessel_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class EDNASample(Base):
    __tablename__ = "edna_samples"
    
    id = Column(Integer, primary_key=True, index=True)
    sample_id = Column(String, unique=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    sample_date = Column(DateTime)
    biodiversity_index = Column(Float)
    species_richness = Column(Integer)
    genetic_diversity = Column(Float)
    dominant_species = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    organization = Column(String)
    role = Column(String, default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic models
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

class SpeciesOccurrenceResponse(BaseModel):
    id: int
    species: str
    latitude: float
    longitude: float
    event_date: datetime
    individual_count: int
    sst_at_point: Optional[float]
    created_at: datetime

class VesselCreate(BaseModel):
    vessel_id: str
    latitude: float
    longitude: float
    timestamp: datetime
    catch_kg: float
    gear_type: str
    vessel_type: str

class VesselResponse(BaseModel):
    id: int
    vessel_id: str
    latitude: float
    longitude: float
    timestamp: datetime
    catch_kg: float
    gear_type: str
    vessel_type: str
    created_at: datetime

class EDNASampleCreate(BaseModel):
    sample_id: str
    latitude: float
    longitude: float
    sample_date: datetime
    biodiversity_index: float
    species_richness: int
    genetic_diversity: float
    dominant_species: str

class EDNASampleResponse(BaseModel):
    id: int
    sample_id: str
    latitude: float
    longitude: float
    sample_date: datetime
    biodiversity_index: float
    species_richness: int
    genetic_diversity: float
    dominant_species: str
    created_at: datetime

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
    prediction_timestamp: datetime

class UserCreate(BaseModel):
    email: str
    name: str
    organization: str
    role: str = "user"

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    organization: str
    role: str
    is_active: bool
    created_at: datetime

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ocean_data.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ML model loading
model = None
scaler = None

def load_ml_model():
    """Load the trained ML model and scaler."""
    global model, scaler
    try:
        model_path = "../models/species_sst_rf.pkl"
        scaler_path = "../models/species_sst_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info("ML model loaded successfully")
        else:
            logger.warning("ML model files not found, using mock model")
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token."""
    # In production, implement proper JWT validation
    # For demo purposes, return a mock user
    return {"user_id": 1, "email": "demo@oceandata.in", "role": "admin"}

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Load ML model
    load_ml_model()
    
    logger.info("Ocean Data API started successfully")

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
        "timestamp": datetime.utcnow(),
        "database": "connected",
        "ml_model": "loaded" if model is not None else "mock"
    }

# Species Data Endpoints

@app.post("/api/species", response_model=SpeciesOccurrenceResponse)
async def create_species_occurrence(
    species_data: SpeciesOccurrenceCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new species occurrence record."""
    try:
        db_species = SpeciesOccurrence(**species_data.dict())
        db.add(db_species)
        db.commit()
        db.refresh(db_species)
        
        logger.info(f"Created species occurrence: {db_species.id}")
        return db_species
    except Exception as e:
        logger.error(f"Error creating species occurrence: {e}")
        raise HTTPException(status_code=500, detail="Failed to create species occurrence")

@app.get("/api/species", response_model=List[SpeciesOccurrenceResponse])
async def get_species_occurrences(
    skip: int = 0,
    limit: int = 100,
    species: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get species occurrence records with filtering."""
    try:
        query = db.query(SpeciesOccurrence)
        
        if species:
            query = query.filter(SpeciesOccurrence.species.ilike(f"%{species}%"))
        if start_date:
            query = query.filter(SpeciesOccurrence.event_date >= start_date)
        if end_date:
            query = query.filter(SpeciesOccurrence.event_date <= end_date)
        
        species_list = query.offset(skip).limit(limit).all()
        return species_list
    except Exception as e:
        logger.error(f"Error fetching species occurrences: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch species occurrences")

@app.get("/api/species/stats")
async def get_species_stats(db: Session = Depends(get_db)):
    """Get species occurrence statistics."""
    try:
        total_records = db.query(SpeciesOccurrence).count()
        unique_species = db.query(SpeciesOccurrence.species).distinct().count()
        date_range = db.query(
            db.func.min(SpeciesOccurrence.event_date),
            db.func.max(SpeciesOccurrence.event_date)
        ).first()
        
        return {
            "total_records": total_records,
            "unique_species": unique_species,
            "date_range": {
                "start": date_range[0],
                "end": date_range[1]
            }
        }
    except Exception as e:
        logger.error(f"Error fetching species stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch species statistics")

# Vessel Data Endpoints

@app.post("/api/vessels", response_model=VesselResponse)
async def create_vessel_record(
    vessel_data: VesselCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new vessel tracking record."""
    try:
        db_vessel = Vessel(**vessel_data.dict())
        db.add(db_vessel)
        db.commit()
        db.refresh(db_vessel)
        
        logger.info(f"Created vessel record: {db_vessel.id}")
        return db_vessel
    except Exception as e:
        logger.error(f"Error creating vessel record: {e}")
        raise HTTPException(status_code=500, detail="Failed to create vessel record")

@app.get("/api/vessels", response_model=List[VesselResponse])
async def get_vessel_records(
    skip: int = 0,
    limit: int = 100,
    vessel_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get vessel tracking records with filtering."""
    try:
        query = db.query(Vessel)
        
        if vessel_id:
            query = query.filter(Vessel.vessel_id == vessel_id)
        if start_date:
            query = query.filter(Vessel.timestamp >= start_date)
        if end_date:
            query = query.filter(Vessel.timestamp <= end_date)
        
        vessels = query.offset(skip).limit(limit).all()
        return vessels
    except Exception as e:
        logger.error(f"Error fetching vessel records: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vessel records")

# eDNA Data Endpoints

@app.post("/api/edna", response_model=EDNASampleResponse)
async def create_edna_sample(
    edna_data: EDNASampleCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new eDNA sample record."""
    try:
        db_edna = EDNASample(**edna_data.dict())
        db.add(db_edna)
        db.commit()
        db.refresh(db_edna)
        
        logger.info(f"Created eDNA sample: {db_edna.id}")
        return db_edna
    except Exception as e:
        logger.error(f"Error creating eDNA sample: {e}")
        raise HTTPException(status_code=500, detail="Failed to create eDNA sample")

@app.get("/api/edna", response_model=List[EDNASampleResponse])
async def get_edna_samples(
    skip: int = 0,
    limit: int = 100,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get eDNA sample records with filtering."""
    try:
        query = db.query(EDNASample)
        
        if start_date:
            query = query.filter(EDNASample.sample_date >= start_date)
        if end_date:
            query = query.filter(EDNASample.sample_date <= end_date)
        
        samples = query.offset(skip).limit(limit).all()
        return samples
    except Exception as e:
        logger.error(f"Error fetching eDNA samples: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch eDNA samples")

# AI/ML Prediction Endpoints

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_species_abundance(
    prediction_request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict species abundance based on environmental parameters."""
    try:
        # Prepare features
        season_encoded = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}[prediction_request.season]
        sst_cat_encoded = {"Cool": 0, "Moderate": 1, "Warm": 2, "Hot": 3}[prediction_request.sst_category]
        bio_cat_encoded = {"Low": 0, "Medium": 1, "High": 2}[prediction_request.biodiversity_category]
        
        features = np.array([[
            prediction_request.mean_sst,
            prediction_request.biodiversity_index,
            prediction_request.genetic_diversity,
            prediction_request.species_richness,
            prediction_request.species_richness,
            prediction_request.mean_sst - 0.5,  # prev_month_sst
            20,  # prev_month_species_count
            0.2,  # sst_trend
            0,   # species_trend
            season_encoded,
            sst_cat_encoded,
            bio_cat_encoded
        ]])
        
        if model is not None and scaler is not None:
            # Scale features
            features_scaled = scaler.transform(features)
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            confidence = 0.85  # Mock confidence
        else:
            # Mock prediction
            prediction = 15 + (prediction_request.mean_sst - 28) * 2 + prediction_request.biodiversity_index * 10
            confidence = 0.75
        
        return PredictionResponse(
            predicted_species_count=float(prediction),
            confidence=confidence,
            model_version="1.0.0",
            prediction_timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to make prediction")

@app.get("/api/trends")
async def get_environmental_trends(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get environmental trend analysis."""
    try:
        # Mock trend data - in production, implement real trend analysis
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

# User Management Endpoints

@app.post("/api/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Create a new user."""
    try:
        db_user = User(**user_data.dict())
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"Created user: {db_user.id}")
        return db_user
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.get("/api/users", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all users."""
    try:
        users = db.query(User).offset(skip).limit(limit).all()
        return users
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch users")

# File Upload Endpoints

@app.post("/api/upload/species")
async def upload_species_data(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload species occurrence data from CSV file."""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate required columns
        required_columns = ['species', 'latitude', 'longitude', 'date', 'count']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")
        
        # Process and save data
        # In production, implement proper data processing and validation
        
        return {
            "message": f"Successfully processed {len(df)} records",
            "filename": file.filename,
            "records_processed": len(df)
        }
    except Exception as e:
        logger.error(f"Error uploading species data: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload species data")

# Analytics Endpoints

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(db: Session = Depends(get_db)):
    """Get dashboard analytics data."""
    try:
        # Species analytics
        species_count = db.query(SpeciesOccurrence).count()
        unique_species = db.query(SpeciesOccurrence.species).distinct().count()
        
        # Vessel analytics
        vessel_count = db.query(Vessel).count()
        total_catch = db.query(db.func.sum(Vessel.catch_kg)).scalar() or 0
        
        # eDNA analytics
        edna_count = db.query(EDNASample).count()
        avg_biodiversity = db.query(db.func.avg(EDNASample.biodiversity_index)).scalar() or 0
        
        return {
            "species": {
                "total_records": species_count,
                "unique_species": unique_species
            },
            "vessels": {
                "total_records": vessel_count,
                "total_catch_kg": float(total_catch)
            },
            "edna": {
                "total_samples": edna_count,
                "avg_biodiversity_index": float(avg_biodiversity)
            }
        }
    except Exception as e:
        logger.error(f"Error fetching dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard analytics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
