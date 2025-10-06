"""
Database connection and setup for Ocean Data Integration Platform
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection and management class."""
    
    def __init__(self):
        """Initialize database connection."""
        self.connection = None
        self.engine = None
        self.session = None
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'ocean_data'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'ocean123')
        }
    
    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            # Direct psycopg2 connection
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = True
            logger.info("Connected to PostgreSQL database")
            
            # SQLAlchemy engine
            connection_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            self.engine = create_engine(connection_string)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            logger.info("SQLAlchemy engine created")
            
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database."""
        try:
            if self.connection:
                self.connection.close()
            if self.session:
                self.session.close()
            logger.info("Disconnected from database")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    def execute_query(self, query, params=None):
        """Execute a SQL query and return results."""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
    
    def execute_insert(self, query, params=None):
        """Execute an INSERT query."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Insert execution failed: {e}")
            return False
    
    def create_tables(self):
        """Create database tables from schema."""
        try:
            # Read schema file
            schema_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'schema.sql')
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema
            cursor = self.connection.cursor()
            cursor.execute(schema_sql)
            self.connection.commit()
            cursor.close()
            
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            return False
    
    def insert_species_occurrence(self, species, latitude, longitude, event_date, individual_count, sst_at_point=None):
        """Insert a species occurrence record."""
        query = """
        INSERT INTO species_occurrences (species, latitude, longitude, event_date, individual_count, sst_at_point)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        params = (species, latitude, longitude, event_date, individual_count, sst_at_point)
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            self.connection.commit()
            cursor.close()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Species occurrence insert failed: {e}")
            return None
    
    def insert_catch_report(self, species, latitude, longitude, catch_weight, individual_count, 
                          gear_type, vessel_type, fishing_depth, timestamp):
        """Insert a catch report record."""
        query = """
        INSERT INTO catch_reports (species, latitude, longitude, catch_weight, individual_count, 
                                 gear_type, vessel_type, fishing_depth, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        params = (species, latitude, longitude, catch_weight, individual_count, 
                 gear_type, vessel_type, fishing_depth, timestamp)
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            self.connection.commit()
            cursor.close()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Catch report insert failed: {e}")
            return None
    
    def get_species_occurrences(self, limit=100, offset=0):
        """Get species occurrence records."""
        query = """
        SELECT * FROM species_occurrences 
        ORDER BY created_at DESC 
        LIMIT %s OFFSET %s
        """
        return self.execute_query(query, (limit, offset))
    
    def get_catch_reports(self, limit=100, offset=0):
        """Get catch report records."""
        query = """
        SELECT * FROM catch_reports 
        ORDER BY created_at DESC 
        LIMIT %s OFFSET %s
        """
        return self.execute_query(query, (limit, offset))
    
    def get_environmental_data(self, latitude, longitude, radius=0.1):
        """Get environmental data for a location."""
        query = """
        SELECT * FROM environmental_data 
        WHERE latitude BETWEEN %s AND %s 
        AND longitude BETWEEN %s AND %s
        ORDER BY timestamp DESC
        LIMIT 10
        """
        params = (latitude - radius, latitude + radius, 
                 longitude - radius, longitude + radius)
        return self.execute_query(query, params)
    
    def get_analytics_data(self):
        """Get analytics data for dashboard."""
        try:
            # Species statistics
            species_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT species) as unique_species,
                AVG(individual_count) as avg_individual_count
            FROM species_occurrences
            """
            species_stats = self.execute_query(species_query)
            
            # Catch statistics
            catch_query = """
            SELECT 
                COUNT(*) as total_catches,
                SUM(catch_weight) as total_weight,
                AVG(catch_weight) as avg_weight
            FROM catch_reports
            """
            catch_stats = self.execute_query(catch_query)
            
            # Environmental statistics
            env_query = """
            SELECT 
                AVG(sst) as avg_sst,
                AVG(wind_speed) as avg_wind_speed,
                AVG(wave_height) as avg_wave_height
            FROM environmental_data
            """
            env_stats = self.execute_query(env_query)
            
            return {
                'species': species_stats[0] if species_stats else {},
                'catch': catch_stats[0] if catch_stats else {},
                'environmental': env_stats[0] if env_stats else {}
            }
        except Exception as e:
            logger.error(f"Analytics query failed: {e}")
            return {}
    
    def test_connection(self):
        """Test database connection."""
        try:
            result = self.execute_query("SELECT 1 as test")
            return result is not None and len(result) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

def get_database():
    """Get database manager instance."""
    return db_manager

def init_database():
    """Initialize database connection and tables."""
    if db_manager.connect():
        if db_manager.test_connection():
            logger.info("Database connection successful")
            # Uncomment to create tables (only run once)
            # db_manager.create_tables()
            return True
        else:
            logger.error("Database connection test failed")
            return False
    else:
        logger.error("Failed to connect to database")
        return False

if __name__ == "__main__":
    # Test database connection
    if init_database():
        print("✅ Database connection successful!")
        
        # Test queries
        species_data = db_manager.get_species_occurrences(limit=5)
        print(f"📊 Found {len(species_data)} species records")
        
        catch_data = db_manager.get_catch_reports(limit=5)
        print(f"🐟 Found {len(catch_data)} catch reports")
        
        analytics = db_manager.get_analytics_data()
        print(f"📈 Analytics data: {analytics}")
        
        db_manager.disconnect()
    else:
        print("❌ Database connection failed!")
