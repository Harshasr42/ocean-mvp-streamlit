"""
External API Integration for Ocean Data Platform
NOAA, Marine Traffic, Weather APIs, and Satellite Data
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for external APIs."""
    base_url: str
    api_key: str
    rate_limit: int = 100  # requests per minute
    timeout: int = 30
    retry_attempts: int = 3

class NOAAAPI:
    """NOAA API integration for oceanographic data."""
    
    def __init__(self, api_key: str):
        self.config = APIConfig(
            base_url="https://www.ncdc.noaa.gov/cdo-web/api/v2",
            api_key=api_key,
            rate_limit=1000
        )
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"token": self.config.api_key},
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_sea_surface_temperature(
        self, 
        lat: float, 
        lon: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get sea surface temperature data from NOAA."""
        try:
            params = {
                "datasetid": "GHCND",  # Global Historical Climatology Network Daily
                "datatypeid": "SST",  # Sea Surface Temperature
                "locationid": f"FIPS:{lat},{lon}",
                "startdate": start_date.strftime("%Y-%m-%d"),
                "enddate": end_date.strftime("%Y-%m-%d"),
                "limit": 1000
            }
            
            async with self.session.get(
                f"{self.config.base_url}/data",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("results", [])
                else:
                    logger.error(f"NOAA API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching SST data: {e}")
            return []
    
    async def get_ocean_currents(
        self, 
        lat: float, 
        lon: float, 
        depth: float = 0
    ) -> Dict[str, Any]:
        """Get ocean current data from NOAA."""
        try:
            # This would use NOAA's ocean current APIs
            # For now, return mock data structure
            return {
                "latitude": lat,
                "longitude": lon,
                "depth": depth,
                "u_velocity": np.random.uniform(-0.5, 0.5),  # m/s
                "v_velocity": np.random.uniform(-0.5, 0.5),  # m/s
                "speed": np.random.uniform(0, 1.0),  # m/s
                "direction": np.random.uniform(0, 360),  # degrees
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching ocean currents: {e}")
            return {}
    
    async def get_weather_data(
        self, 
        lat: float, 
        lon: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get weather data from NOAA."""
        try:
            params = {
                "datasetid": "GHCND",
                "datatypeid": "TMAX,TMIN,PRCP,WSFG,WSFI",  # Temperature, precipitation, wind
                "locationid": f"FIPS:{lat},{lon}",
                "startdate": start_date.strftime("%Y-%m-%d"),
                "enddate": end_date.strftime("%Y-%m-%d"),
                "limit": 1000
            }
            
            async with self.session.get(
                f"{self.config.base_url}/data",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("results", [])
                else:
                    logger.error(f"NOAA Weather API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return []

class MarineTrafficAPI:
    """Marine Traffic API integration for vessel tracking."""
    
    def __init__(self, api_key: str):
        self.config = APIConfig(
            base_url="https://services.marinetraffic.com/api",
            api_key=api_key,
            rate_limit=100
        )
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"apikey": self.config.api_key},
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_vessel_positions(
        self, 
        bbox: Tuple[float, float, float, float],  # min_lat, min_lon, max_lat, max_lon
        vessel_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get vessel positions within bounding box."""
        try:
            params = {
                "protocol": "json",
                "timespan": 10,  # minutes
                "minlat": bbox[0],
                "minlon": bbox[1],
                "maxlat": bbox[2],
                "maxlon": bbox[3]
            }
            
            if vessel_type:
                params["vesseltype"] = vessel_type
            
            async with self.session.get(
                f"{self.config.base_url}/exportvessels/v:8",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data if isinstance(data, list) else []
                else:
                    logger.error(f"Marine Traffic API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching vessel positions: {e}")
            return []
    
    async def get_vessel_details(self, mmsi: str) -> Dict[str, Any]:
        """Get detailed vessel information."""
        try:
            params = {
                "protocol": "json",
                "mmsi": mmsi
            }
            
            async with self.session.get(
                f"{self.config.base_url}/exportvesseltrack/v:8",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Marine Traffic vessel details error: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching vessel details: {e}")
            return {}
    
    async def get_vessel_track(
        self, 
        mmsi: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get vessel track history."""
        try:
            params = {
                "protocol": "json",
                "mmsi": mmsi,
                "fromdate": start_date.strftime("%Y-%m-%d"),
                "todate": end_date.strftime("%Y-%m-%d")
            }
            
            async with self.session.get(
                f"{self.config.base_url}/exportvesseltrack/v:8",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data if isinstance(data, list) else []
                else:
                    logger.error(f"Marine Traffic track error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching vessel track: {e}")
            return []

class OpenWeatherAPI:
    """OpenWeatherMap API integration."""
    
    def __init__(self, api_key: str):
        self.config = APIConfig(
            base_url="https://api.openweathermap.org/data/2.5",
            api_key=api_key,
            rate_limit=1000
        )
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_current_weather(
        self, 
        lat: float, 
        lon: float
    ) -> Dict[str, Any]:
        """Get current weather data."""
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.config.api_key,
                "units": "metric"
            }
            
            async with self.session.get(
                f"{self.config.base_url}/weather",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "pressure": data["main"]["pressure"],
                        "wind_speed": data["wind"]["speed"],
                        "wind_direction": data["wind"]["deg"],
                        "weather_description": data["weather"][0]["description"],
                        "cloud_cover": data["clouds"]["all"],
                        "visibility": data.get("visibility", 0),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    logger.error(f"OpenWeather API error: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching current weather: {e}")
            return {}
    
    async def get_weather_forecast(
        self, 
        lat: float, 
        lon: float, 
        days: int = 5
    ) -> List[Dict[str, Any]]:
        """Get weather forecast."""
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.config.api_key,
                "units": "metric",
                "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            async with self.session.get(
                f"{self.config.base_url}/forecast",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("list", [])
                else:
                    logger.error(f"OpenWeather forecast error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return []

class SatelliteDataAPI:
    """Satellite data integration (NASA, ESA, etc.)."""
    
    def __init__(self, nasa_api_key: str):
        self.config = APIConfig(
            base_url="https://api.nasa.gov",
            api_key=nasa_api_key,
            rate_limit=1000
        )
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_satellite_imagery(
        self, 
        lat: float, 
        lon: float, 
        date: datetime,
        satellite: str = "MODIS"
    ) -> Dict[str, Any]:
        """Get satellite imagery data."""
        try:
            # This is a simplified example - real implementation would use
            # NASA's Earthdata API or similar services
            params = {
                "lat": lat,
                "lon": lon,
                "date": date.strftime("%Y-%m-%d"),
                "api_key": self.config.api_key
            }
            
            # Mock satellite data for demonstration
            return {
                "satellite": satellite,
                "latitude": lat,
                "longitude": lon,
                "acquisition_date": date.isoformat(),
                "sea_surface_temperature": np.random.uniform(15, 30),
                "chlorophyll_a": np.random.uniform(0.1, 5.0),
                "turbidity": np.random.uniform(0, 10),
                "cloud_cover": np.random.uniform(0, 100),
                "spatial_resolution": 1000,  # meters
                "data_url": f"https://example.com/satellite/{satellite}_{date.strftime('%Y%m%d')}.nc",
                "metadata": {
                    "sensor": "MODIS",
                    "processing_level": "L3",
                    "quality_flags": ["good", "validated"]
                }
            }
        except Exception as e:
            logger.error(f"Error fetching satellite data: {e}")
            return {}
    
    async def get_ocean_color_data(
        self, 
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get ocean color (chlorophyll) data."""
        try:
            # Mock ocean color data
            data_points = []
            for i in range(10):  # Generate sample points
                lat = np.random.uniform(bbox[0], bbox[2])
                lon = np.random.uniform(bbox[1], bbox[3])
                data_points.append({
                    "latitude": lat,
                    "longitude": lon,
                    "chlorophyll_a": np.random.uniform(0.1, 5.0),
                    "sea_surface_temperature": np.random.uniform(15, 30),
                    "turbidity": np.random.uniform(0, 10),
                    "date": start_date.isoformat()
                })
            
            return data_points
        except Exception as e:
            logger.error(f"Error fetching ocean color data: {e}")
            return []

class DataSyncManager:
    """Manages synchronization of external data sources."""
    
    def __init__(self):
        self.noaa_api = NOAAAPI(os.getenv("NOAA_API_KEY", ""))
        self.marine_traffic_api = MarineTrafficAPI(os.getenv("MARINE_TRAFFIC_API_KEY", ""))
        self.weather_api = OpenWeatherAPI(os.getenv("OPENWEATHER_API_KEY", ""))
        self.satellite_api = SatelliteDataAPI(os.getenv("NASA_API_KEY", ""))
    
    async def sync_oceanographic_data(
        self, 
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Sync oceanographic data from multiple sources."""
        results = {
            "sst_data": [],
            "currents_data": [],
            "weather_data": [],
            "satellite_data": [],
            "sync_timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Sync SST data from NOAA
            async with self.noaa_api as noaa:
                for lat in np.linspace(bbox[0], bbox[2], 5):
                    for lon in np.linspace(bbox[1], bbox[3], 5):
                        sst_data = await noaa.get_sea_surface_temperature(
                            lat, lon, start_date, end_date
                        )
                        results["sst_data"].extend(sst_data)
            
            # Sync weather data
            async with self.weather_api as weather:
                for lat in np.linspace(bbox[0], bbox[2], 3):
                    for lon in np.linspace(bbox[1], bbox[3], 3):
                        weather_data = await weather.get_current_weather(lat, lon)
                        if weather_data:
                            results["weather_data"].append(weather_data)
            
            # Sync satellite data
            async with self.satellite_api as satellite:
                for date in pd.date_range(start_date, end_date, freq='D'):
                    satellite_data = await satellite.get_satellite_imagery(
                        (bbox[0] + bbox[2]) / 2,  # center lat
                        (bbox[1] + bbox[3]) / 2,  # center lon
                        date
                    )
                    if satellite_data:
                        results["satellite_data"].append(satellite_data)
            
            logger.info(f"Synced oceanographic data: {len(results['sst_data'])} SST records, "
                       f"{len(results['weather_data'])} weather records, "
                       f"{len(results['satellite_data'])} satellite records")
            
        except Exception as e:
            logger.error(f"Error syncing oceanographic data: {e}")
        
        return results
    
    async def sync_vessel_data(
        self, 
        bbox: Tuple[float, float, float, float],
        vessel_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Sync vessel tracking data."""
        results = {
            "vessel_positions": [],
            "vessel_details": [],
            "sync_timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            async with self.marine_traffic_api as mt:
                # Get vessel positions
                positions = await mt.get_vessel_positions(bbox, vessel_type)
                results["vessel_positions"] = positions
                
                # Get details for each vessel
                for position in positions[:10]:  # Limit to first 10 for demo
                    if "MMSI" in position:
                        details = await mt.get_vessel_details(position["MMSI"])
                        if details:
                            results["vessel_details"].append(details)
            
            logger.info(f"Synced vessel data: {len(results['vessel_positions'])} positions, "
                       f"{len(results['vessel_details'])} vessel details")
            
        except Exception as e:
            logger.error(f"Error syncing vessel data: {e}")
        
        return results
    
    async def sync_all_data(
        self, 
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Sync all external data sources."""
        logger.info("Starting comprehensive data sync...")
        
        # Run all sync operations concurrently
        oceanographic_task = self.sync_oceanographic_data(bbox, start_date, end_date)
        vessel_task = self.sync_vessel_data(bbox)
        
        oceanographic_results, vessel_results = await asyncio.gather(
            oceanographic_task, vessel_task, return_exceptions=True
        )
        
        return {
            "oceanographic": oceanographic_results if not isinstance(oceanographic_results, Exception) else {},
            "vessel": vessel_results if not isinstance(vessel_results, Exception) else {},
            "sync_completed": datetime.utcnow().isoformat()
        }

# Global data sync manager instance
data_sync_manager = DataSyncManager()
