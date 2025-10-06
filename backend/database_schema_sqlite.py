"""
SQLite-Compatible Database Schema for Ocean Data Integration Platform
Simplified schema without PostGIS for testing and development
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
import json

Base = declarative_base()

class User(Base):
    """Enhanced user management with roles and permissions."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    full_name = Column(String(255))
    organization = Column(String(255))
    role = Column(String(50), default="user", index=True)  # admin, researcher, fisherman, observer
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    phone = Column(String(20))
    country = Column(String(100))
    timezone = Column(String(50), default="UTC")
    preferences = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    catch_reports = relationship("CatchReport", back_populates="user")
    vessel_ownerships = relationship("VesselOwnership", back_populates="user")

class Vessel(Base):
    """Enhanced vessel tracking."""
    __tablename__ = "vessels"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    vessel_id = Column(String(100), unique=True, index=True)  # AIS MMSI or custom ID
    name = Column(String(255))
    imo_number = Column(String(20), unique=True, index=True)
    call_sign = Column(String(20))
    vessel_type = Column(String(100), index=True)  # fishing, cargo, passenger, etc.
    flag_state = Column(String(100))
    port_of_registry = Column(String(255))
    length_overall = Column(Float)  # meters
    gross_tonnage = Column(Float)
    engine_power = Column(Float)  # kW
    gear_types = Column(Text)  # JSON as text for SQLite
    target_species = Column(Text)  # JSON as text for SQLite
    is_active = Column(Boolean, default=True)
    vessel_metadata = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    positions = relationship("VesselPosition", back_populates="vessel")
    catch_reports = relationship("CatchReport", back_populates="vessel")
    ownerships = relationship("VesselOwnership", back_populates="vessel")

class VesselPosition(Base):
    """Real-time vessel position tracking."""
    __tablename__ = "vessel_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    vessel_id = Column(Integer, ForeignKey("vessels.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    heading = Column(Float)  # degrees
    speed = Column(Float)  # knots
    course = Column(Float)  # degrees
    status = Column(String(50))  # underway, anchored, moored, etc.
    navigational_status = Column(String(50))
    ais_source = Column(String(50))  # AIS, manual, satellite
    accuracy = Column(Float)  # position accuracy in meters
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    vessel = relationship("Vessel", back_populates="positions")
    
    # Indexes for queries
    __table_args__ = (
        Index('idx_vessel_positions_vessel_timestamp', 'vessel_id', 'timestamp'),
        Index('idx_vessel_positions_timestamp', 'timestamp'),
    )

class CatchReport(Base):
    """Enhanced catch reporting with spatial and temporal data."""
    __tablename__ = "catch_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    vessel_id = Column(Integer, ForeignKey("vessels.id"), nullable=False, index=True)
    report_date = Column(DateTime, nullable=False, index=True)
    fishing_date = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth = Column(Float)  # fishing depth in meters
    gear_type = Column(String(100), nullable=False, index=True)
    gear_specifications = Column(Text)  # JSON as text for SQLite
    fishing_effort_hours = Column(Float)  # hours of fishing effort
    total_catch_kg = Column(Float, nullable=False)
    total_catch_count = Column(Integer)
    species_composition = Column(Text)  # JSON as text for SQLite
    environmental_conditions = Column(Text)  # JSON as text for SQLite
    economic_data = Column(Text)  # JSON as text for SQLite
    quality_indicators = Column(Text)  # JSON as text for SQLite
    is_verified = Column(Boolean, default=False)
    verification_notes = Column(Text)
    record_metadata = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="catch_reports")
    vessel = relationship("Vessel", back_populates="catch_reports")
    species_details = relationship("CatchSpeciesDetail", back_populates="catch_report")
    
    # Indexes
    __table_args__ = (
        Index('idx_catch_reports_date', 'fishing_date'),
        Index('idx_catch_reports_vessel_date', 'vessel_id', 'fishing_date'),
    )

class CatchSpeciesDetail(Base):
    """Detailed species composition in catch reports."""
    __tablename__ = "catch_species_details"
    
    id = Column(Integer, primary_key=True, index=True)
    catch_report_id = Column(Integer, ForeignKey("catch_reports.id"), nullable=False, index=True)
    species_name = Column(String(255), nullable=False, index=True)
    scientific_name = Column(String(255), index=True)
    common_name = Column(String(255))
    weight_kg = Column(Float, nullable=False)
    count = Column(Integer)
    average_length_cm = Column(Float)
    average_weight_kg = Column(Float)
    size_distribution = Column(Text)  # JSON as text for SQLite
    condition = Column(String(50))  # fresh, frozen, processed
    market_value = Column(Float)  # estimated market value
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    catch_report = relationship("CatchReport", back_populates="species_details")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('weight_kg >= 0', name='check_weight_positive'),
        CheckConstraint('count >= 0', name='check_count_positive'),
    )

class SpeciesOccurrence(Base):
    """Enhanced species occurrence."""
    __tablename__ = "species_occurrences"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    species = Column(String(255), nullable=False, index=True)
    scientific_name = Column(String(255), index=True)
    common_name = Column(String(255))
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    event_date = Column(DateTime, nullable=False, index=True)
    individual_count = Column(Integer)
    abundance_estimate = Column(Float)
    observation_method = Column(String(100))  # visual, acoustic, eDNA, etc.
    observer = Column(String(255))
    organization = Column(String(255))
    data_source = Column(String(100), index=True)  # OBIS, GBIF, manual, etc.
    phylum = Column(String(100), index=True)
    class_name = Column(String(100), index=True)
    order_name = Column(String(100), index=True)
    family = Column(String(100), index=True)
    genus = Column(String(100), index=True)
    species_authority = Column(String(255))
    depth = Column(Float)  # observation depth
    sst_at_point = Column(Float)  # sea surface temperature
    salinity = Column(Float)
    dissolved_oxygen = Column(Float)
    ph = Column(Float)
    turbidity = Column(Float)
    environmental_data = Column(Text)  # JSON as text for SQLite
    quality_flags = Column(Text)  # JSON as text for SQLite
    is_verified = Column(Boolean, default=False)
    verification_notes = Column(Text)
    record_metadata = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_species_occurrences_species_date', 'species', 'event_date'),
        Index('idx_species_occurrences_taxonomy', 'phylum', 'class_name', 'order_name'),
    )

class EDNASample(Base):
    """Enhanced eDNA sampling."""
    __tablename__ = "edna_samples"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    sample_id = Column(String(100), unique=True, index=True)
    project_name = Column(String(255))
    researcher = Column(String(255))
    organization = Column(String(255))
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    sample_date = Column(DateTime, nullable=False, index=True)
    depth = Column(Float)  # sampling depth
    volume_filtered_l = Column(Float)  # volume of water filtered
    filter_type = Column(String(100))
    preservation_method = Column(String(100))
    storage_temperature = Column(Float)
    dna_extraction_method = Column(String(100))
    sequencing_platform = Column(String(100))
    sequencing_depth = Column(Integer)
    biodiversity_index = Column(Float)
    species_richness = Column(Integer)
    genetic_diversity = Column(Float)
    dominant_species = Column(String(255))
    species_detected = Column(Text)  # JSON as text for SQLite
    abundance_data = Column(Text)  # JSON as text for SQLite
    environmental_parameters = Column(Text)  # JSON as text for SQLite
    quality_metrics = Column(Text)  # JSON as text for SQLite
    is_processed = Column(Boolean, default=False)
    processing_notes = Column(Text)
    record_metadata = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_edna_samples_date', 'sample_date'),
        Index('idx_edna_samples_project', 'project_name'),
    )

class OceanographicData(Base):
    """Real-time oceanographic data from various sources."""
    __tablename__ = "oceanographic_data"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    data_source = Column(String(100), nullable=False, index=True)  # NOAA, satellite, buoy, etc.
    station_id = Column(String(100), index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    measurement_date = Column(DateTime, nullable=False, index=True)
    sea_surface_temperature = Column(Float)
    sea_surface_salinity = Column(Float)
    sea_surface_height = Column(Float)
    wave_height = Column(Float)
    wave_period = Column(Float)
    wave_direction = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    atmospheric_pressure = Column(Float)
    air_temperature = Column(Float)
    humidity = Column(Float)
    precipitation = Column(Float)
    dissolved_oxygen = Column(Float)
    ph = Column(Float)
    turbidity = Column(Float)
    chlorophyll_a = Column(Float)
    nutrients = Column(Text)  # JSON as text for SQLite
    currents = Column(Text)  # JSON as text for SQLite
    additional_parameters = Column(Text)  # JSON as text for SQLite
    quality_flags = Column(Text)  # JSON as text for SQLite
    record_metadata = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_oceanographic_date', 'measurement_date'),
        Index('idx_oceanographic_source', 'data_source'),
    )

class WeatherData(Base):
    """Weather data integration."""
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    data_source = Column(String(100), nullable=False, index=True)
    station_id = Column(String(100), index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    measurement_date = Column(DateTime, nullable=False, index=True)
    temperature = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    precipitation = Column(Float)
    visibility = Column(Float)
    cloud_cover = Column(Float)
    uv_index = Column(Float)
    weather_conditions = Column(String(255))
    forecast_data = Column(Text)  # JSON as text for SQLite
    record_metadata = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_weather_date', 'measurement_date'),
    )

class SatelliteData(Base):
    """Satellite imagery and derived products."""
    __tablename__ = "satellite_data"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    satellite = Column(String(100), nullable=False, index=True)  # Sentinel, Landsat, MODIS, etc.
    sensor = Column(String(100), index=True)
    product_type = Column(String(100), index=True)  # SST, chlorophyll, etc.
    acquisition_date = Column(DateTime, nullable=False, index=True)
    spatial_resolution = Column(Float)  # meters
    temporal_resolution = Column(Float)  # hours
    data_url = Column(String(500))
    file_size_mb = Column(Float)
    processing_level = Column(String(50))
    cloud_cover_percentage = Column(Float)
    quality_flags = Column(Text)  # JSON as text for SQLite
    record_metadata = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_satellite_date', 'acquisition_date'),
        Index('idx_satellite_product', 'product_type'),
    )

class VesselOwnership(Base):
    """Vessel ownership and management."""
    __tablename__ = "vessel_ownership"
    
    id = Column(Integer, primary_key=True, index=True)
    vessel_id = Column(Integer, ForeignKey("vessels.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ownership_type = Column(String(50), nullable=False)  # owner, operator, crew, observer
    ownership_percentage = Column(Float)  # for partial ownership
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    permissions = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    vessel = relationship("Vessel", back_populates="ownerships")
    user = relationship("User", back_populates="vessel_ownerships")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('vessel_id', 'user_id', 'start_date', name='unique_ownership'),
        CheckConstraint('ownership_percentage >= 0 AND ownership_percentage <= 100', name='check_ownership_percentage'),
    )

class DataSyncLog(Base):
    """Log for external data synchronization."""
    __tablename__ = "data_sync_log"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(100), nullable=False, index=True)  # NOAA, Marine Traffic, etc.
    sync_type = Column(String(50), nullable=False)  # full, incremental, real-time
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    status = Column(String(50), nullable=False)  # success, failed, partial
    records_processed = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(Text)
    record_metadata = Column(Text)  # JSON as text for SQLite
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_sync_log_source_status', 'source', 'status'),
        Index('idx_sync_log_start_time', 'start_time'),
    )
