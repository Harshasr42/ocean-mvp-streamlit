"""
Supabase Configuration for Ocean Data Integration Platform
Cloud database integration for production deployment
"""

import os
from supabase import create_client, Client
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class SupabaseManager:
    """Manages Supabase database operations for the Ocean Data Platform."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.client: Optional[Client] = None
        
        if self.url and self.key:
            self.client = create_client(self.url, self.key)
            logger.info("Supabase client initialized successfully")
        else:
            logger.warning("Supabase credentials not found, using local database")
    
    def is_connected(self) -> bool:
        """Check if Supabase is connected."""
        return self.client is not None
    
    # Species Data Operations
    
    async def create_species_occurrence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new species occurrence record in Supabase."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            result = self.client.table("species_occurrences").insert(data).execute()
            logger.info(f"Created species occurrence: {result.data[0]['id']}")
            return result.data[0]
        except Exception as e:
            logger.error(f"Error creating species occurrence: {e}")
            raise
    
    async def get_species_occurrences(
        self, 
        limit: int = 100, 
        offset: int = 0,
        species_filter: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get species occurrence records with filtering."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            query = self.client.table("species_occurrences").select("*")
            
            if species_filter:
                query = query.ilike("species", f"%{species_filter}%")
            if start_date:
                query = query.gte("event_date", start_date)
            if end_date:
                query = query.lte("event_date", end_date)
            
            result = query.range(offset, offset + limit - 1).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching species occurrences: {e}")
            raise
    
    async def get_species_stats(self) -> Dict[str, Any]:
        """Get species occurrence statistics."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            # Get total count
            total_result = self.client.table("species_occurrences").select("id", count="exact").execute()
            total_count = total_result.count
            
            # Get unique species count
            unique_result = self.client.table("species_occurrences").select("species").execute()
            unique_species = len(set([record["species"] for record in unique_result.data]))
            
            # Get date range
            date_result = self.client.table("species_occurrences").select("event_date").order("event_date").execute()
            if date_result.data:
                min_date = min([record["event_date"] for record in date_result.data])
                max_date = max([record["event_date"] for record in date_result.data])
            else:
                min_date = max_date = None
            
            return {
                "total_records": total_count,
                "unique_species": unique_species,
                "date_range": {
                    "start": min_date,
                    "end": max_date
                }
            }
        except Exception as e:
            logger.error(f"Error fetching species stats: {e}")
            raise
    
    # Vessel Data Operations
    
    async def create_vessel_record(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new vessel tracking record in Supabase."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            result = self.client.table("vessels").insert(data).execute()
            logger.info(f"Created vessel record: {result.data[0]['id']}")
            return result.data[0]
        except Exception as e:
            logger.error(f"Error creating vessel record: {e}")
            raise
    
    async def get_vessel_records(
        self,
        limit: int = 100,
        offset: int = 0,
        vessel_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get vessel tracking records with filtering."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            query = self.client.table("vessels").select("*")
            
            if vessel_id:
                query = query.eq("vessel_id", vessel_id)
            if start_date:
                query = query.gte("timestamp", start_date)
            if end_date:
                query = query.lte("timestamp", end_date)
            
            result = query.range(offset, offset + limit - 1).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching vessel records: {e}")
            raise
    
    # eDNA Data Operations
    
    async def create_edna_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new eDNA sample record in Supabase."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            result = self.client.table("edna_samples").insert(data).execute()
            logger.info(f"Created eDNA sample: {result.data[0]['id']}")
            return result.data[0]
        except Exception as e:
            logger.error(f"Error creating eDNA sample: {e}")
            raise
    
    async def get_edna_samples(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get eDNA sample records with filtering."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            query = self.client.table("edna_samples").select("*")
            
            if start_date:
                query = query.gte("sample_date", start_date)
            if end_date:
                query = query.lte("sample_date", end_date)
            
            result = query.range(offset, offset + limit - 1).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching eDNA samples: {e}")
            raise
    
    # Analytics Operations
    
    async def get_dashboard_analytics(self) -> Dict[str, Any]:
        """Get dashboard analytics data from Supabase."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            # Species analytics
            species_result = self.client.table("species_occurrences").select("id", count="exact").execute()
            species_count = species_result.count
            
            unique_species_result = self.client.table("species_occurrences").select("species").execute()
            unique_species = len(set([record["species"] for record in unique_species_result.data]))
            
            # Vessel analytics
            vessel_result = self.client.table("vessels").select("id", count="exact").execute()
            vessel_count = vessel_result.count
            
            catch_result = self.client.table("vessels").select("catch_kg").execute()
            total_catch = sum([record["catch_kg"] for record in catch_result.data if record["catch_kg"]])
            
            # eDNA analytics
            edna_result = self.client.table("edna_samples").select("id", count="exact").execute()
            edna_count = edna_result.count
            
            biodiversity_result = self.client.table("edna_samples").select("biodiversity_index").execute()
            biodiversity_values = [record["biodiversity_index"] for record in biodiversity_result.data if record["biodiversity_index"]]
            avg_biodiversity = sum(biodiversity_values) / len(biodiversity_values) if biodiversity_values else 0
            
            return {
                "species": {
                    "total_records": species_count,
                    "unique_species": unique_species
                },
                "vessels": {
                    "total_records": vessel_count,
                    "total_catch_kg": total_catch
                },
                "edna": {
                    "total_samples": edna_count,
                    "avg_biodiversity_index": avg_biodiversity
                }
            }
        except Exception as e:
            logger.error(f"Error fetching dashboard analytics: {e}")
            raise
    
    # User Management Operations
    
    async def create_user(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user in Supabase."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            result = self.client.table("users").insert(data).execute()
            logger.info(f"Created user: {result.data[0]['id']}")
            return result.data[0]
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    async def get_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all users from Supabase."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            result = self.client.table("users").select("*").range(offset, offset + limit - 1).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            raise
    
    # Real-time Subscriptions
    
    async def subscribe_to_species_changes(self, callback):
        """Subscribe to real-time species occurrence changes."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            subscription = self.client.table("species_occurrences").on("INSERT", callback).subscribe()
            return subscription
        except Exception as e:
            logger.error(f"Error subscribing to species changes: {e}")
            raise
    
    async def subscribe_to_vessel_changes(self, callback):
        """Subscribe to real-time vessel tracking changes."""
        if not self.client:
            raise Exception("Supabase not connected")
        
        try:
            subscription = self.client.table("vessels").on("INSERT", callback).subscribe()
            return subscription
        except Exception as e:
            logger.error(f"Error subscribing to vessel changes: {e}")
            raise

# Global Supabase manager instance
supabase_manager = SupabaseManager()

# Helper functions for easy access
async def create_species_occurrence(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a species occurrence record."""
    return await supabase_manager.create_species_occurrence(data)

async def get_species_occurrences(**kwargs) -> List[Dict[str, Any]]:
    """Get species occurrence records."""
    return await supabase_manager.get_species_occurrences(**kwargs)

async def get_species_stats() -> Dict[str, Any]:
    """Get species occurrence statistics."""
    return await supabase_manager.get_species_stats()

async def create_vessel_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a vessel tracking record."""
    return await supabase_manager.create_vessel_record(data)

async def get_vessel_records(**kwargs) -> List[Dict[str, Any]]:
    """Get vessel tracking records."""
    return await supabase_manager.get_vessel_records(**kwargs)

async def create_edna_sample(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create an eDNA sample record."""
    return await supabase_manager.create_edna_sample(data)

async def get_edna_samples(**kwargs) -> List[Dict[str, Any]]:
    """Get eDNA sample records."""
    return await supabase_manager.get_edna_samples(**kwargs)

async def get_dashboard_analytics() -> Dict[str, Any]:
    """Get dashboard analytics data."""
    return await supabase_manager.get_dashboard_analytics()

async def create_user(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new user."""
    return await supabase_manager.create_user(data)

async def get_users(**kwargs) -> List[Dict[str, Any]]:
    """Get all users."""
    return await supabase_manager.get_users(**kwargs)
