"""
Free Mapping Integration for Ocean Data Platform
OpenStreetMap + Leaflet.js integration (No API keys required)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MapConfig:
    """Configuration for mapping services."""
    default_center: Tuple[float, float] = (15.0, 80.0)  # Indian Ocean
    default_zoom: int = 6
    max_zoom: int = 18
    min_zoom: int = 2
    tile_server: str = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
    attribution: str = "© OpenStreetMap contributors"

class OceanMapLayer:
    """Ocean-specific map layers for marine data visualization."""
    
    def __init__(self):
        self.config = MapConfig()
    
    def get_base_layers(self) -> Dict[str, Any]:
        """Get base map layers configuration."""
        from config import config
        
        layers = {
            "openstreetmap": {
                "name": "OpenStreetMap",
                "url": self.config.tile_server,
                "attribution": self.config.attribution,
                "max_zoom": self.config.max_zoom,
                "min_zoom": self.config.min_zoom,
                "type": "tile"
            },
            "satellite": {
                "name": "Satellite",
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "attribution": "© Esri",
                "max_zoom": 18,
                "min_zoom": 2,
                "type": "tile"
            }
        }
        
        # Add Google Maps layers if API key is available
        if config.GOOGLE_MAPS_API_KEY:
            layers.update({
                "google_roadmap": {
                    "name": "Google Roadmap",
                    "api_key": config.GOOGLE_MAPS_API_KEY,
                    "type": "google",
                    "map_type": "roadmap",
                    "max_zoom": 20,
                    "min_zoom": 1
                },
                "google_satellite": {
                    "name": "Google Satellite",
                    "api_key": config.GOOGLE_MAPS_API_KEY,
                    "type": "google",
                    "map_type": "satellite",
                    "max_zoom": 20,
                    "min_zoom": 1
                },
                "google_hybrid": {
                    "name": "Google Hybrid",
                    "api_key": config.GOOGLE_MAPS_API_KEY,
                    "type": "google",
                    "map_type": "hybrid",
                    "max_zoom": 20,
                    "min_zoom": 1
                },
                "google_terrain": {
                    "name": "Google Terrain",
                    "api_key": config.GOOGLE_MAPS_API_KEY,
                    "type": "google",
                    "map_type": "terrain",
                    "max_zoom": 20,
                    "min_zoom": 1
                }
            })
        
        return layers
    
    def get_overlay_layers(self) -> Dict[str, Any]:
        """Get overlay layers for marine data."""
        return {
            "bathymetry": {
                "name": "Bathymetry",
                "url": "https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
                "attribution": "© OpenSeaMap",
                "opacity": 0.7,
                "visible": False
            },
            "marine_traffic": {
                "name": "Marine Traffic",
                "url": "https://tiles.marinetraffic.com/v1/overlay/{z}/{x}/{y}.png",
                "attribution": "© MarineTraffic",
                "opacity": 0.8,
                "visible": False
            },
            "weather": {
                "name": "Weather",
                "url": "https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png",
                "attribution": "© OpenWeatherMap",
                "opacity": 0.6,
                "visible": False
            }
        }
    
    def get_marine_boundaries(self) -> List[Dict[str, Any]]:
        """Get marine boundaries and protected areas."""
        return [
            {
                "name": "Indian Ocean",
                "type": "ocean",
                "bounds": [[-60, 20], [-60, 120], [30, 120], [30, 20]],
                "color": "#0066cc",
                "fillOpacity": 0.1
            },
            {
                "name": "Arabian Sea",
                "type": "sea",
                "bounds": [[5, 50], [5, 80], [30, 80], [30, 50]],
                "color": "#0099cc",
                "fillOpacity": 0.2
            },
            {
                "name": "Bay of Bengal",
                "type": "sea", 
                "bounds": [[5, 80], [5, 100], [25, 100], [25, 80]],
                "color": "#00ccff",
                "fillOpacity": 0.2
            }
        ]
    
    def get_fishing_zones(self) -> List[Dict[str, Any]]:
        """Get fishing zones and regulations."""
        return [
            {
                "name": "EEZ India",
                "type": "eez",
                "bounds": [[6, 68], [6, 97], [24, 97], [24, 68]],
                "color": "#ff6600",
                "fillOpacity": 0.3,
                "description": "Indian Exclusive Economic Zone"
            },
            {
                "name": "Protected Marine Area",
                "type": "protected",
                "bounds": [[8, 77], [8, 79], [10, 79], [10, 77]],
                "color": "#ff0000",
                "fillOpacity": 0.5,
                "description": "No fishing zone"
            }
        ]

class SpatialDataProcessor:
    """Process spatial data for mapping visualization."""
    
    def __init__(self):
        self.map_layer = OceanMapLayer()
    
    def process_species_data(self, species_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process species occurrence data for mapping."""
        try:
            # Group by species
            species_groups = {}
            for record in species_data:
                species = record.get("species", "Unknown")
                if species not in species_groups:
                    species_groups[species] = []
                
                species_groups[species].append({
                    "lat": record.get("latitude"),
                    "lon": record.get("longitude"),
                    "count": record.get("individual_count", 1),
                    "date": record.get("event_date"),
                    "depth": record.get("depth"),
                    "sst": record.get("sst_at_point")
                })
            
            # Create map markers
            markers = []
            for species, records in species_groups.items():
                for record in records:
                    markers.append({
                        "type": "species",
                        "species": species,
                        "lat": record["lat"],
                        "lon": record["lon"],
                        "count": record["count"],
                        "date": record["date"],
                        "icon": self._get_species_icon(species),
                        "color": self._get_species_color(species)
                    })
            
            return {
                "markers": markers,
                "species_count": len(species_groups),
                "total_records": len(species_data),
                "bounds": self._calculate_bounds(species_data)
            }
            
        except Exception as e:
            logger.error(f"Error processing species data: {e}")
            return {"markers": [], "species_count": 0, "total_records": 0, "bounds": None}
    
    def process_vessel_data(self, vessel_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process vessel tracking data for mapping."""
        try:
            # Group by vessel
            vessel_groups = {}
            for record in vessel_data:
                vessel_id = record.get("vessel_id", "Unknown")
                if vessel_id not in vessel_groups:
                    vessel_groups[vessel_id] = []
                
                vessel_groups[vessel_id].append({
                    "lat": record.get("latitude"),
                    "lon": record.get("longitude"),
                    "timestamp": record.get("timestamp"),
                    "heading": record.get("heading"),
                    "speed": record.get("speed"),
                    "status": record.get("status")
                })
            
            # Create vessel tracks and markers
            tracks = []
            markers = []
            
            for vessel_id, positions in vessel_groups.items():
                # Create track line
                track_coords = [[pos["lat"], pos["lon"]] for pos in positions]
                tracks.append({
                    "vessel_id": vessel_id,
                    "coordinates": track_coords,
                    "color": self._get_vessel_color(vessel_id),
                    "weight": 3
                })
                
                # Create current position marker
                if positions:
                    latest = positions[-1]
                    markers.append({
                        "type": "vessel",
                        "vessel_id": vessel_id,
                        "lat": latest["lat"],
                        "lon": latest["lon"],
                        "heading": latest["heading"],
                        "speed": latest["speed"],
                        "status": latest["status"],
                        "icon": "vessel",
                        "color": self._get_vessel_color(vessel_id)
                    })
            
            return {
                "tracks": tracks,
                "markers": markers,
                "vessel_count": len(vessel_groups),
                "bounds": self._calculate_bounds(vessel_data)
            }
            
        except Exception as e:
            logger.error(f"Error processing vessel data: {e}")
            return {"tracks": [], "markers": [], "vessel_count": 0, "bounds": None}
    
    def process_oceanographic_data(self, ocean_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process oceanographic data for mapping."""
        try:
            # Create heatmap data
            heatmap_data = []
            for record in ocean_data:
                heatmap_data.append({
                    "lat": record.get("latitude"),
                    "lon": record.get("longitude"),
                    "intensity": record.get("sea_surface_temperature", 28),
                    "salinity": record.get("sea_surface_salinity", 35),
                    "date": record.get("measurement_date")
                })
            
            # Create contour data
            contours = self._create_contours(heatmap_data)
            
            return {
                "heatmap": heatmap_data,
                "contours": contours,
                "bounds": self._calculate_bounds(ocean_data)
            }
            
        except Exception as e:
            logger.error(f"Error processing oceanographic data: {e}")
            return {"heatmap": [], "contours": [], "bounds": None}
    
    def _get_species_icon(self, species: str) -> str:
        """Get icon for species type."""
        species_icons = {
            "Tuna": "🐟",
            "Mackerel": "🐟", 
            "Prawn": "🦐",
            "Crab": "🦀",
            "Squid": "🦑",
            "Dolphin": "🐬",
            "Whale": "🐋"
        }
        
        for key, icon in species_icons.items():
            if key.lower() in species.lower():
                return icon
        
        return "🐟"  # Default fish icon
    
    def _get_species_color(self, species: str) -> str:
        """Get color for species type."""
        species_colors = {
            "Tuna": "#ff6b6b",
            "Mackerel": "#4ecdc4",
            "Prawn": "#45b7d1",
            "Crab": "#96ceb4",
            "Squid": "#feca57",
            "Dolphin": "#ff9ff3",
            "Whale": "#54a0ff"
        }
        
        for key, color in species_colors.items():
            if key.lower() in species.lower():
                return color
        
        return "#74b9ff"  # Default blue
    
    def _get_vessel_color(self, vessel_id: str) -> str:
        """Get color for vessel."""
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        hash_val = hash(vessel_id) % len(colors)
        return colors[hash_val]
    
    def _calculate_bounds(self, data: List[Dict[str, Any]]) -> Optional[List[List[float]]]:
        """Calculate bounding box for data."""
        if not data:
            return None
        
        lats = [record.get("latitude", 0) for record in data if record.get("latitude")]
        lons = [record.get("longitude", 0) for record in data if record.get("longitude")]
        
        if not lats or not lons:
            return None
        
        return [
            [min(lats), min(lons)],  # Southwest
            [max(lats), max(lons)]   # Northeast
        ]
    
    def _create_contours(self, heatmap_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create contour lines from heatmap data."""
        # Simplified contour creation
        contours = []
        
        # Group data by temperature ranges
        temp_ranges = [(20, 25), (25, 28), (28, 30), (30, 35)]
        colors = ["#0066cc", "#0099cc", "#00ccff", "#ff6600"]
        
        for i, (min_temp, max_temp) in enumerate(temp_ranges):
            contour_points = [
                point for point in heatmap_data 
                if min_temp <= point["intensity"] < max_temp
            ]
            
            if contour_points:
                contours.append({
                    "level": f"{min_temp}-{max_temp}°C",
                    "color": colors[i],
                    "points": contour_points
                })
        
        return contours

class MapVisualizationAPI:
    """API for map visualization data."""
    
    def __init__(self):
        self.spatial_processor = SpatialDataProcessor()
        self.map_layer = OceanMapLayer()
    
    def get_map_config(self) -> Dict[str, Any]:
        """Get complete map configuration."""
        return {
            "center": self.map_layer.config.default_center,
            "zoom": self.map_layer.config.default_zoom,
            "base_layers": self.map_layer.get_base_layers(),
            "overlay_layers": self.map_layer.get_overlay_layers(),
            "marine_boundaries": self.map_layer.get_marine_boundaries(),
            "fishing_zones": self.map_layer.get_fishing_zones()
        }
    
    def get_species_map_data(self, species_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get species data formatted for mapping."""
        return self.spatial_processor.process_species_data(species_data)
    
    def get_vessel_map_data(self, vessel_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get vessel data formatted for mapping."""
        return self.spatial_processor.process_vessel_data(vessel_data)
    
    def get_oceanographic_map_data(self, ocean_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get oceanographic data formatted for mapping."""
        return self.spatial_processor.process_oceanographic_data(ocean_data)
    
    def get_combined_map_data(
        self,
        species_data: List[Dict[str, Any]] = None,
        vessel_data: List[Dict[str, Any]] = None,
        ocean_data: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get combined map data from all sources."""
        combined_data = {
            "config": self.get_map_config(),
            "species": {},
            "vessels": {},
            "oceanographic": {},
            "bounds": None
        }
        
        if species_data:
            combined_data["species"] = self.get_species_map_data(species_data)
        
        if vessel_data:
            combined_data["vessels"] = self.get_vessel_map_data(vessel_data)
        
        if ocean_data:
            combined_data["oceanographic"] = self.get_oceanographic_map_data(ocean_data)
        
        # Calculate overall bounds
        all_bounds = []
        for data_type in ["species", "vessels", "oceanographic"]:
            if combined_data[data_type].get("bounds"):
                all_bounds.append(combined_data[data_type]["bounds"])
        
        if all_bounds:
            # Calculate overall bounding box
            min_lats = [bounds[0][0] for bounds in all_bounds]
            min_lons = [bounds[0][1] for bounds in all_bounds]
            max_lats = [bounds[1][0] for bounds in all_bounds]
            max_lons = [bounds[1][1] for bounds in all_bounds]
            
            combined_data["bounds"] = [
                [min(min_lats), min(min_lons)],
                [max(max_lats), max(max_lons)]
            ]
        
        return combined_data

# Global map visualization API instance
map_visualization_api = MapVisualizationAPI()
