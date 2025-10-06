"""
Simple FastAPI Backend for Ocean Data Integration Platform
Minimal backend with catch-reports endpoint for testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Ocean Data Integration Platform API",
    description="Simple REST API for marine data management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - This is crucial for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CatchReportCreate(BaseModel):
    species: str
    latitude: float
    longitude: float
    catch_weight: float
    individual_count: int
    gear_type: str
    vessel_type: str
    fishing_depth: Optional[float] = None
    timestamp: Optional[datetime] = None

class CatchReportResponse(BaseModel):
    id: int
    species: str
    latitude: float
    longitude: float
    catch_weight: float
    individual_count: int
    gear_type: str
    vessel_type: str
    fishing_depth: Optional[float] = None
    timestamp: datetime
    created_at: datetime

# In-memory storage for demo
catch_reports = []

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ocean Data Integration Platform API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "catch_reports": "/api/catch-reports",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "database": "in_memory",
        "catch_reports_count": len(catch_reports)
    }

# Catch Reports Endpoints
@app.post("/api/catch-reports", response_model=CatchReportResponse)
async def create_catch_report(catch_data: CatchReportCreate):
    """Create a new catch report."""
    try:
        # Generate ID
        report_id = len(catch_reports) + 1
        
        # Create report
        new_report = {
            "id": report_id,
            "species": catch_data.species,
            "latitude": catch_data.latitude,
            "longitude": catch_data.longitude,
            "catch_weight": catch_data.catch_weight,
            "individual_count": catch_data.individual_count,
            "gear_type": catch_data.gear_type,
            "vessel_type": catch_data.vessel_type,
            "fishing_depth": catch_data.fishing_depth,
            "timestamp": catch_data.timestamp or datetime.utcnow(),
            "created_at": datetime.utcnow()
        }
        
        # Store in memory
        catch_reports.append(new_report)
        
        logger.info(f"Created catch report: {report_id} - Species: {catch_data.species}")
        
        return CatchReportResponse(**new_report)
        
    except Exception as e:
        logger.error(f"Error creating catch report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create catch report: {str(e)}")

@app.get("/api/catch-reports", response_model=List[CatchReportResponse])
async def get_catch_reports(skip: int = 0, limit: int = 100):
    """Get all catch reports."""
    try:
        # Return paginated results
        start_idx = skip
        end_idx = skip + limit
        reports = catch_reports[start_idx:end_idx]
        
        logger.info(f"Retrieved {len(reports)} catch reports")
        return [CatchReportResponse(**report) for report in reports]
        
    except Exception as e:
        logger.error(f"Error fetching catch reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch catch reports")

# Species endpoint for testing
@app.get("/api/species")
async def get_species():
    """Get species data."""
    return {
        "species": [
            {"name": "Yellowfin Tuna", "scientific_name": "Thunnus albacares"},
            {"name": "Skipjack Tuna", "scientific_name": "Katsuwonus pelamis"},
            {"name": "Indian Mackerel", "scientific_name": "Rastrelliger kanagurta"},
            {"name": "Pomfret", "scientific_name": "Pampus argenteus"}
        ]
    }

# Analytics endpoint
@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    """Get dashboard analytics."""
    try:
        total_catch_weight = sum(report["catch_weight"] for report in catch_reports)
        unique_species = len(set(report["species"] for report in catch_reports))
        
        return {
            "catch_reports": {
                "total_records": len(catch_reports),
                "total_catch_weight": total_catch_weight,
                "unique_species": unique_species
            },
            "species": {
                "total_records": len(catch_reports),
                "unique_species": unique_species
            },
            "vessels": {
                "total_vessels": 0,
                "active_vessels": 0
            }
        }
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
