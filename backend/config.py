"""
Configuration file for Ocean Data Integration Platform
Environment variables and API configurations
"""

import os
from typing import Optional

class Config:
    """Application configuration."""
    
    # Database Configuration
    DATABASE_URL = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/ocean_db"
    )
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://dcwhukoaermngvubwdwo.supabase.co")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRjd2h1a29hZXJtbmd2dWJ3ZHdvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg4NTc2MTgsImV4cCI6MjA3NDQzMzYxOH0.UfY7zxJ_zrC8Hq32gcKtFLR7FXHE7xuvb5EAHnhw-wg")
    
    # External API Keys
    NOAA_API_KEY = os.getenv("NOAA_API_KEY", "HvbIFXHkuuogtHJieGZivmHLmMcxZnCF")
    MARINE_TRAFFIC_API_KEY = os.getenv("MARINE_TRAFFIC_API_KEY", "")  # Empty - using mock data
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")  # Empty - using NOAA weather
    NASA_API_KEY = os.getenv("NASA_API_KEY", "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImhhcnNoYXd0MTIzNCIsImV4cCI6MTc2NDExNTE5OSwiaWF0IjoxNzU4ODU4MTk0LCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhY3IiOiJlZGwiLCJhc3N1cmFuY2VfbGV2ZWwiOjN9.TEnV_0Z6Q5EMWxOC0iFR82xASIyEwT_RmNw6RHITjS0tlJHejXm8yNuKssh81VCWM4URWnKOcwM55uHfSJg4rX-tW9nqGlyHQcskrI30WKBUCHpZFsMFZqsLly7mSoeZ38CPf4E5_YFqDdhufSreCruWk4K74U4yboFxSAFxxrbz6fgVrRiwxsKEqp0vWDix76RKL6NVu0Aro4UdROmnJQfxpCRePmH_PTHfYlNxzuFaM-BkSUCd-FVFg6tmYUhOO-ALESAHRRIQfGjGUMQgR08sCSNj-M2gPYd5zYpKk717O2Rn0d2TviM4YrLqDQa2ZN2cKgVr5Tc4LG29txting")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "AIzaSyB74feW22WeyFCHpRzbNhB-gaXOkWgPz-w")
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24
    
    # API Configuration
    API_V1_PREFIX = "/api/v1"
    PROJECT_NAME = "Ocean Data Integration Platform"
    VERSION = "1.0.0"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = 100
    
    # Data Sync Configuration
    SYNC_INTERVAL_MINUTES = 30
    MAX_RECORDS_PER_SYNC = 1000
    
    # File Upload
    MAX_FILE_SIZE_MB = 100
    ALLOWED_FILE_TYPES = [".csv", ".json", ".xlsx", ".nc"]
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "ocean_platform.log")
    
    # Development/Production
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # CORS
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8502",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8502",
    ]
    
    # External API URLs
    NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
    MARINE_TRAFFIC_BASE_URL = "https://services.marinetraffic.com/api"
    OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    NASA_BASE_URL = "https://api.nasa.gov"
    
    # Default Bounding Box (Global)
    DEFAULT_BBOX = (-180, -90, 180, 90)  # min_lon, min_lat, max_lon, max_lat
    
    # Data Quality Thresholds
    MIN_LATITUDE = -90
    MAX_LATITUDE = 90
    MIN_LONGITUDE = -180
    MAX_LONGITUDE = 180
    MIN_TEMPERATURE = -2  # Celsius
    MAX_TEMPERATURE = 40  # Celsius
    MIN_SALINITY = 0
    MAX_SALINITY = 50  # PSU
    
    # Cache TTL (Time To Live) in seconds
    CACHE_TTL = {
        "weather_data": 3600,  # 1 hour
        "oceanographic_data": 1800,  # 30 minutes
        "vessel_data": 300,  # 5 minutes
        "satellite_data": 7200,  # 2 hours
        "user_data": 3600,  # 1 hour
    }
    
    # Mock Data Configuration
    USE_MOCK_DATA = {
        "marine_traffic": True,  # Use mock vessel data (Marine Traffic is paid)
        "weather": False,  # Use NOAA weather data
        "satellite": False,  # Use NASA satellite data (now available)
        "ocean_currents": False,  # Use NOAA current data
    }
    
    # Data Sources Priority
    DATA_SOURCES = {
        "sst": ["noaa", "satellite"],
        "weather": ["noaa", "openweather"],
        "vessels": ["marine_traffic", "mock"],
        "currents": ["noaa", "mock"],
        "satellite": ["nasa", "mock"]
    }

# Global config instance
config = Config()

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    # Add production-specific settings
    CORS_ORIGINS = [
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ]

class TestingConfig(Config):
    """Testing configuration."""
    DATABASE_URL = "sqlite:///./test_ocean.db"
    TESTING = True
    LOG_LEVEL = "ERROR"

# Configuration factory
def get_config(environment: str = None) -> Config:
    """Get configuration based on environment."""
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()
