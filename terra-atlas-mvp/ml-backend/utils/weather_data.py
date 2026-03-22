"""
Weather Data Provider - Utilities for fetching and processing weather data
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

class WeatherDataProvider:
    """
    Provides weather data for energy predictions
    """
    
    def __init__(self):
        self.providers = {
            "openweather": "https://api.openweathermap.org/",
            "noaa": "https://api.weather.gov/",
            "era5": "https://cds.climate.copernicus.eu/"
        }
        self.cache = {}
        
    async def get_forecast(self, lat: float, lon: float, days: int = 7) -> Dict:
        """
        Get weather forecast for a location
        """
        cache_key = f"{lat:.2f}_{lon:.2f}_{days}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if (datetime.now() - cached_data["timestamp"]).hours < 6:
                return cached_data["data"]
        
        # Simulate fetching weather data
        weather_data = await self._simulate_weather_fetch(lat, lon, days)
        
        # Cache the result
        self.cache[cache_key] = {
            "data": weather_data,
            "timestamp": datetime.now()
        }
        
        return weather_data
    
    async def _simulate_weather_fetch(self, lat: float, lon: float, days: int) -> Dict:
        """
        Simulate fetching weather data
        """
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Generate realistic weather patterns
        base_temp = 15 + 10 * np.cos(np.radians(lat))  # Temperature based on latitude
        base_wind = 5 + 3 * np.abs(np.sin(np.radians(lat)))  # Wind based on latitude
        base_solar = 5.5 - 0.04 * np.abs(lat)  # Solar irradiance based on latitude
        
        # Generate time series
        temperatures = []
        wind_speeds = []
        solar_irradiances = []
        precipitations = []
        cloud_covers = []
        
        for day in range(days):
            # Daily variations
            temp_variation = np.random.normal(0, 5)
            wind_variation = np.random.exponential(2)
            
            # Seasonal adjustment (simplified)
            season_factor = np.sin(2 * np.pi * (day % 365) / 365)
            
            temperatures.append(base_temp + temp_variation + 10 * season_factor)
            wind_speeds.append(max(0, base_wind + wind_variation))
            
            # Cloud cover affects solar irradiance
            cloud = np.random.beta(2, 5)  # Skewed towards less clouds
            cloud_covers.append(cloud)
            solar_irradiances.append(base_solar * (1 - cloud * 0.7))
            
            # Precipitation (correlated with clouds)
            if cloud > 0.6:
                precipitations.append(np.random.exponential(5))
            else:
                precipitations.append(0)
        
        return {
            "temperature": temperatures,
            "wind_speed": wind_speeds,
            "solar_irradiance": solar_irradiances,
            "precipitation": precipitations,
            "cloud_cover": cloud_covers,
            "humidity": [50 + 30 * c for c in cloud_covers],
            "pressure": [1013 + np.random.normal(0, 10) for _ in range(days)]
        }
    
    def get_historical_weather(self, lat: float, lon: float, start_date: datetime, end_date: datetime) -> Dict:
        """
        Get historical weather data
        """
        days = (end_date - start_date).days
        
        # Generate synthetic historical data (in production, use actual historical APIs)
        return {
            "temperature": self._generate_historical_temperature(lat, days, start_date),
            "wind_speed": self._generate_historical_wind(lat, days),
            "solar_irradiance": self._generate_historical_solar(lat, days, start_date),
            "precipitation": self._generate_historical_precipitation(days)
        }
    
    def _generate_historical_temperature(self, lat: float, days: int, start_date: datetime) -> List[float]:
        """
        Generate historical temperature data
        """
        base_temp = 15 + 10 * np.cos(np.radians(lat))
        temperatures = []
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            day_of_year = current_date.timetuple().tm_yday
            
            # Seasonal variation
            seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
            
            # Daily variation
            daily = np.random.normal(0, 3)
            
            temperatures.append(base_temp + seasonal + daily)
        
        return temperatures
    
    def _generate_historical_wind(self, lat: float, days: int) -> List[float]:
        """
        Generate historical wind speed data
        """
        # Trade winds are stronger at certain latitudes
        if 10 <= abs(lat) <= 30:
            base_wind = 7
        elif 40 <= abs(lat) <= 60:
            base_wind = 8
        else:
            base_wind = 5
        
        # Weibull distribution for wind speeds
        wind_speeds = np.random.weibull(2, days) * base_wind
        
        return wind_speeds.tolist()
    
    def _generate_historical_solar(self, lat: float, days: int, start_date: datetime) -> List[float]:
        """
        Generate historical solar irradiance data
        """
        solar_irradiances = []
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            day_of_year = current_date.timetuple().tm_yday
            
            # Solar declination
            declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
            
            # Day length calculation
            if abs(lat) < 66.5:  # Not in polar region
                sunset_hour_angle = np.arccos(-np.tan(np.radians(lat)) * np.tan(np.radians(declination)))
                day_length = 2 * sunset_hour_angle * 12 / np.pi
            else:
                # Polar day/night
                day_length = 24 if (lat > 0 and declination > 0) or (lat < 0 and declination < 0) else 0
            
            # Base solar irradiance
            base_solar = 5.5 - 0.04 * abs(lat)
            
            # Adjust for day length and clouds
            cloud_factor = np.random.beta(2, 5)  # Skewed towards clear sky
            daily_solar = base_solar * (day_length / 12) * (1 - cloud_factor * 0.7)
            
            solar_irradiances.append(max(0, daily_solar))
        
        return solar_irradiances
    
    def _generate_historical_precipitation(self, days: int) -> List[float]:
        """
        Generate historical precipitation data
        """
        precipitation = []
        
        for _ in range(days):
            # Most days are dry
            if np.random.random() < 0.7:
                precipitation.append(0)
            else:
                # Exponential distribution for rain amount
                precipitation.append(np.random.exponential(5))
        
        return precipitation
    
    def calculate_heating_degree_days(self, temperatures: List[float], base_temp: float = 18) -> float:
        """
        Calculate heating degree days
        """
        hdd = sum(max(0, base_temp - temp) for temp in temperatures)
        return hdd
    
    def calculate_cooling_degree_days(self, temperatures: List[float], base_temp: float = 18) -> float:
        """
        Calculate cooling degree days
        """
        cdd = sum(max(0, temp - base_temp) for temp in temperatures)
        return cdd