"""
Database Migration Scripts for Ocean Data Integration Platform
Creates tables, indexes, and initial data
"""

import asyncio
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import Geometry
from database_schema import Base
from config import config
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handles database migrations and setup."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or config.DATABASE_URL
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def create_postgis_extension(self):
        """Create PostGIS extension if using PostgreSQL."""
        try:
            with self.engine.connect() as conn:
                # Enable PostGIS extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology;"))
                conn.commit()
                logger.info("PostGIS extensions created successfully")
        except Exception as e:
            logger.warning(f"PostGIS extension creation failed (might not be PostgreSQL): {e}")
    
    def create_spatial_indexes(self):
        """Create spatial indexes for better performance."""
        try:
            with self.engine.connect() as conn:
                # Create spatial indexes
                spatial_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_vessel_positions_geometry ON vessel_positions USING GIST (geometry);",
                    "CREATE INDEX IF NOT EXISTS idx_catch_reports_geometry ON catch_reports USING GIST (geometry);",
                    "CREATE INDEX IF NOT EXISTS idx_species_occurrences_geometry ON species_occurrences USING GIST (geometry);",
                    "CREATE INDEX IF NOT EXISTS idx_edna_samples_geometry ON edna_samples USING GIST (geometry);",
                    "CREATE INDEX IF NOT EXISTS idx_oceanographic_geometry ON oceanographic_data USING GIST (geometry);",
                    "CREATE INDEX IF NOT EXISTS idx_weather_geometry ON weather_data USING GIST (geometry);",
                    "CREATE INDEX IF NOT EXISTS idx_satellite_coverage ON satellite_data USING GIST (coverage_area);"
                ]
                
                for index_sql in spatial_indexes:
                    conn.execute(text(index_sql))
                
                conn.commit()
                logger.info("Spatial indexes created successfully")
        except Exception as e:
            logger.warning(f"Spatial index creation failed: {e}")
    
    def create_additional_indexes(self):
        """Create additional performance indexes."""
        try:
            with self.engine.connect() as conn:
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);",
                    "CREATE INDEX IF NOT EXISTS idx_users_role ON users (role);",
                    "CREATE INDEX IF NOT EXISTS idx_vessels_vessel_id ON vessels (vessel_id);",
                    "CREATE INDEX IF NOT EXISTS idx_vessels_type ON vessels (vessel_type);",
                    "CREATE INDEX IF NOT EXISTS idx_vessel_positions_timestamp ON vessel_positions (timestamp);",
                    "CREATE INDEX IF NOT EXISTS idx_vessel_positions_vessel_timestamp ON vessel_positions (vessel_id, timestamp);",
                    "CREATE INDEX IF NOT EXISTS idx_catch_reports_date ON catch_reports (fishing_date);",
                    "CREATE INDEX IF NOT EXISTS idx_catch_reports_vessel_date ON catch_reports (vessel_id, fishing_date);",
                    "CREATE INDEX IF NOT EXISTS idx_species_occurrences_species ON species_occurrences (species);",
                    "CREATE INDEX IF NOT EXISTS idx_species_occurrences_date ON species_occurrences (event_date);",
                    "CREATE INDEX IF NOT EXISTS idx_species_occurrences_taxonomy ON species_occurrences (phylum, class_name, order_name);",
                    "CREATE INDEX IF NOT EXISTS idx_edna_samples_date ON edna_samples (sample_date);",
                    "CREATE INDEX IF NOT EXISTS idx_edna_samples_project ON edna_samples (project_name);",
                    "CREATE INDEX IF NOT EXISTS idx_oceanographic_date ON oceanographic_data (measurement_date);",
                    "CREATE INDEX IF NOT EXISTS idx_oceanographic_source ON oceanographic_data (data_source);",
                    "CREATE INDEX IF NOT EXISTS idx_weather_date ON weather_data (measurement_date);",
                    "CREATE INDEX IF NOT EXISTS idx_satellite_date ON satellite_data (acquisition_date);",
                    "CREATE INDEX IF NOT EXISTS idx_satellite_product ON satellite_data (product_type);",
                    "CREATE INDEX IF NOT EXISTS idx_sync_log_source ON data_sync_log (source);",
                    "CREATE INDEX IF NOT EXISTS idx_sync_log_status ON data_sync_log (status);"
                ]
                
                for index_sql in indexes:
                    conn.execute(text(index_sql))
                
                conn.commit()
                logger.info("Additional indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating additional indexes: {e}")
    
    def create_initial_data(self):
        """Create initial sample data for testing."""
        try:
            session = self.SessionLocal()
            
            # Create sample users
            from database_schema import User, Vessel, VesselOwnership
            
            # Check if users already exist
            if session.query(User).count() == 0:
                sample_users = [
                    User(
                        email="admin@oceandata.in",
                        username="admin",
                        full_name="System Administrator",
                        organization="Ocean Data Platform",
                        role="admin",
                        is_active=True,
                        is_verified=True
                    ),
                    User(
                        email="researcher@oceandata.in",
                        username="researcher",
                        full_name="Marine Researcher",
                        organization="Marine Research Institute",
                        role="researcher",
                        is_active=True,
                        is_verified=True
                    ),
                    User(
                        email="fisherman@oceandata.in",
                        username="fisherman",
                        full_name="Commercial Fisherman",
                        organization="Local Fishing Cooperative",
                        role="fisherman",
                        is_active=True,
                        is_verified=True
                    )
                ]
                
                for user in sample_users:
                    session.add(user)
                
                session.commit()
                logger.info("Sample users created")
            
            # Create sample vessels
            if session.query(Vessel).count() == 0:
                sample_vessels = [
                    Vessel(
                        vessel_id="VESSEL001",
                        name="Ocean Explorer",
                        imo_number="1234567",
                        call_sign="OE001",
                        vessel_type="fishing",
                        flag_state="India",
                        port_of_registry="Mumbai",
                        length_overall=25.5,
                        gross_tonnage=150.0,
                        engine_power=500.0,
                        gear_types=["trawl", "gillnet"],
                        target_species=["tuna", "mackerel", "prawn"],
                        is_active=True
                    ),
                    Vessel(
                        vessel_id="VESSEL002",
                        name="Marine Surveyor",
                        imo_number="2345678",
                        call_sign="MS002",
                        vessel_type="research",
                        flag_state="India",
                        port_of_registry="Chennai",
                        length_overall=30.0,
                        gross_tonnage=200.0,
                        engine_power=400.0,
                        gear_types=["research_equipment"],
                        target_species=["all_species"],
                        is_active=True
                    )
                ]
                
                for vessel in sample_vessels:
                    session.add(vessel)
                
                session.commit()
                logger.info("Sample vessels created")
            
            # Create vessel ownership relationships
            if session.query(VesselOwnership).count() == 0:
                admin_user = session.query(User).filter(User.email == "admin@oceandata.in").first()
                researcher_user = session.query(User).filter(User.email == "researcher@oceandata.in").first()
                fisherman_user = session.query(User).filter(User.email == "fisherman@oceandata.in").first()
                
                vessel1 = session.query(Vessel).filter(Vessel.vessel_id == "VESSEL001").first()
                vessel2 = session.query(Vessel).filter(Vessel.vessel_id == "VESSEL002").first()
                
                ownerships = [
                    VesselOwnership(
                        vessel_id=vessel1.id,
                        user_id=fisherman_user.id,
                        ownership_type="owner",
                        ownership_percentage=100.0,
                        start_date=datetime.utcnow() - timedelta(days=365),
                        is_active=True
                    ),
                    VesselOwnership(
                        vessel_id=vessel2.id,
                        user_id=researcher_user.id,
                        ownership_type="operator",
                        ownership_percentage=100.0,
                        start_date=datetime.utcnow() - timedelta(days=180),
                        is_active=True
                    )
                ]
                
                for ownership in ownerships:
                    session.add(ownership)
                
                session.commit()
                logger.info("Vessel ownership relationships created")
            
            session.close()
            
        except Exception as e:
            logger.error(f"Error creating initial data: {e}")
    
    def create_sample_oceanographic_data(self):
        """Create sample oceanographic data for testing."""
        try:
            session = self.SessionLocal()
            
            from database_schema import OceanographicData
            
            # Check if data already exists
            if session.query(OceanographicData).count() == 0:
                # Generate sample oceanographic data
                np.random.seed(42)  # For reproducible data
                
                # Indian Ocean region
                latitudes = np.random.uniform(5, 25, 50)  # 5°N to 25°N
                longitudes = np.random.uniform(70, 90, 50)  # 70°E to 90°E
                
                for i in range(50):
                    lat = latitudes[i]
                    lon = longitudes[i]
                    
                    # Generate realistic oceanographic data
                    sst = np.random.normal(28, 2)  # Sea surface temperature
                    salinity = np.random.normal(35, 1)  # Salinity
                    wave_height = np.random.exponential(1.5)  # Wave height
                    wind_speed = np.random.exponential(5)  # Wind speed
                    
                    ocean_data = OceanographicData(
                        data_source="noaa",
                        station_id=f"STATION_{i:03d}",
                        latitude=lat,
                        longitude=lon,
                        geometry=f"POINT({lon} {lat})",
                        measurement_date=datetime.utcnow() - timedelta(days=np.random.randint(0, 30)),
                        sea_surface_temperature=sst,
                        sea_surface_salinity=salinity,
                        wave_height=wave_height,
                        wind_speed=wind_speed,
                        wind_direction=np.random.uniform(0, 360),
                        atmospheric_pressure=np.random.normal(1013, 10),
                        air_temperature=sst + np.random.normal(0, 2),
                        humidity=np.random.uniform(60, 90),
                        dissolved_oxygen=np.random.uniform(4, 8),
                        ph=np.random.uniform(7.8, 8.4),
                        turbidity=np.random.exponential(2),
                        chlorophyll_a=np.random.exponential(1),
                        metadata={
                            "quality_flags": ["good", "validated"],
                            "instrument": "CTD",
                            "depth": 0.5
                        }
                    )
                    
                    session.add(ocean_data)
                
                session.commit()
                logger.info("Sample oceanographic data created")
            
            session.close()
            
        except Exception as e:
            logger.error(f"Error creating sample oceanographic data: {e}")
    
    def create_sample_species_data(self):
        """Create sample species occurrence data."""
        try:
            session = self.SessionLocal()
            
            from database_schema import SpeciesOccurrence
            
            # Check if data already exists
            if session.query(SpeciesOccurrence).count() == 0:
                # Common marine species in Indian Ocean
                species_list = [
                    {"name": "Yellowfin Tuna", "scientific": "Thunnus albacares", "phylum": "Chordata", "class": "Actinopterygii"},
                    {"name": "Skipjack Tuna", "scientific": "Katsuwonus pelamis", "phylum": "Chordata", "class": "Actinopterygii"},
                    {"name": "Indian Mackerel", "scientific": "Rastrelliger kanagurta", "phylum": "Chordata", "class": "Actinopterygii"},
                    {"name": "Pomfret", "scientific": "Pampus argenteus", "phylum": "Chordata", "class": "Actinopterygii"},
                    {"name": "Prawn", "scientific": "Penaeus monodon", "phylum": "Arthropoda", "class": "Malacostraca"},
                    {"name": "Crab", "scientific": "Portunus pelagicus", "phylum": "Arthropoda", "class": "Malacostraca"},
                    {"name": "Squid", "scientific": "Loligo duvauceli", "phylum": "Mollusca", "class": "Cephalopoda"},
                    {"name": "Dolphin", "scientific": "Delphinus delphis", "phylum": "Chordata", "class": "Mammalia"}
                ]
                
                np.random.seed(42)
                
                # Generate sample occurrences
                for i in range(100):
                    species = np.random.choice(species_list)
                    lat = np.random.uniform(5, 25)
                    lon = np.random.uniform(70, 90)
                    
                    occurrence = SpeciesOccurrence(
                        species=species["name"],
                        scientific_name=species["scientific"],
                        latitude=lat,
                        longitude=lon,
                        geometry=f"POINT({lon} {lat})",
                        event_date=datetime.utcnow() - timedelta(days=np.random.randint(0, 365)),
                        individual_count=np.random.randint(1, 100),
                        abundance_estimate=np.random.uniform(0.1, 10.0),
                        observation_method=np.random.choice(["visual", "acoustic", "eDNA", "fishing"]),
                        observer="Sample Observer",
                        organization="Marine Research Institute",
                        data_source="manual",
                        phylum=species["phylum"],
                        class_name=species["class"],
                        order_name="Unknown",
                        family="Unknown",
                        genus=species["scientific"].split()[0],
                        depth=np.random.uniform(0, 200),
                        sst_at_point=np.random.normal(28, 2),
                        salinity=np.random.normal(35, 1),
                        dissolved_oxygen=np.random.uniform(4, 8),
                        ph=np.random.uniform(7.8, 8.4),
                        turbidity=np.random.exponential(2),
                        environmental_data={
                            "temperature": np.random.normal(28, 2),
                            "salinity": np.random.normal(35, 1),
                            "depth": np.random.uniform(0, 200)
                        },
                        quality_flags=["good", "validated"],
                        is_verified=True
                    )
                    
                    session.add(occurrence)
                
                session.commit()
                logger.info("Sample species occurrence data created")
            
            session.close()
            
        except Exception as e:
            logger.error(f"Error creating sample species data: {e}")
    
    def run_full_migration(self):
        """Run complete database migration."""
        try:
            logger.info("Starting full database migration...")
            
            # Create PostGIS extension
            self.create_postgis_extension()
            
            # Create tables
            self.create_tables()
            
            # Create indexes
            self.create_spatial_indexes()
            self.create_additional_indexes()
            
            # Create initial data
            self.create_initial_data()
            self.create_sample_oceanographic_data()
            self.create_sample_species_data()
            
            logger.info("Database migration completed successfully")
            
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            raise

def run_migration():
    """Run database migration."""
    migrator = DatabaseMigrator()
    migrator.run_full_migration()

if __name__ == "__main__":
    run_migration()
