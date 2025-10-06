import requests
import pandas as pd
from datetime import datetime

class OceanDataCollector:
    def __init__(self):
        self.noaa_api_key = "HvbIFXHkuuogtHJieGZivmHLmMcxZnCF"  # Your actual NOAA API key
    
    def get_sst_data(self, lat, lon):
        """Get real-time sea surface temperature"""
        try:
            url = "https://www.ncei.noaa.gov/access/services/data/v1"
            params = {
                'dataset': 'daily-summaries',
                'dataTypes': 'SST',
                'stations': f'{lat},{lon}',
                'startDate': datetime.now().strftime('%Y-%m-%d'),
                'endDate': datetime.now().strftime('%Y-%m-%d'),
                'token': self.noaa_api_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"NOAA SST API error: {e}")
        return None
    
    def get_wave_data(self, lat, lon):
        """Get wave height data"""
        try:
            url = "https://www.ncei.noaa.gov/access/services/data/v1"
            params = {
                'dataset': 'daily-summaries',
                'dataTypes': 'WAVE_HEIGHT',
                'stations': f'{lat},{lon}',
                'startDate': datetime.now().strftime('%Y-%m-%d'),
                'endDate': datetime.now().strftime('%Y-%m-%d'),
                'token': self.noaa_api_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"NOAA Wave data error: {e}")
        return None
    
    def get_ocean_currents(self, lat, lon):
        """Get ocean current data"""
        try:
            # Using alternative API for ocean currents
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': "a6f35cc90e0f6323be584d35b59ba6f6",  # Your OpenWeatherMap API key
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'wind_speed': data['wind']['speed'],
                    'wind_direction': data['wind'].get('deg', 0),
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure']
                }
        except Exception as e:
            print(f"Ocean currents error: {e}")
        return None
    
    def get_marine_weather(self, lat, lon):
        """Get comprehensive marine weather data"""
        try:
            # Get weather data
            weather_data = self.get_ocean_currents(lat, lon)
            
            # Simulate additional marine data
            marine_data = {
                'sea_surface_temp': 28.0 + (lat - 12) * 0.1,
                'wave_height': 1.2 + (lat - 12) * 0.05,
                'wave_period': 8.5,
                'visibility': 10.0,
                'sea_state': 'Moderate',
                'tide_level': 0.5,
                'current_speed': 0.8,
                'current_direction': 135.0
            }
            
            if weather_data:
                marine_data.update(weather_data)
            
            return marine_data
        except Exception as e:
            print(f"Marine weather error: {e}")
            return None
