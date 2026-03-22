"""
Energy Predictor - ML models for predicting energy generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import torch
import torch.nn as nn

class EnergyPredictor:
    """
    Predicts energy generation for different renewable technologies
    """
    
    def __init__(self):
        self.solar_model = SolarPredictor()
        self.wind_model = WindPredictor()
        self.hydro_model = HydroPredictor()
        
    def calculate_potential(self, lat: float, lon: float, site_type: str, site_features: Dict) -> float:
        """
        Calculate energy generation potential for a site
        """
        if site_type == "solar":
            return self.solar_model.calculate_potential(lat, lon, site_features)
        elif site_type == "wind":
            return self.wind_model.calculate_potential(lat, lon, site_features)
        elif site_type == "hydro":
            return self.hydro_model.calculate_potential(lat, lon, site_features)
        elif site_type == "nuclear":
            # Nuclear has consistent output
            return 8760 * 1000 * 0.9  # hours * MW * capacity_factor
        else:  # storage
            return 0  # Storage doesn't generate, it stores
            
    def predict_solar_output(self, lat: float, lon: float, capacity_mw: float, weather_data: Dict) -> Dict:
        """
        Predict solar energy output
        """
        return self.solar_model.predict(lat, lon, capacity_mw, weather_data)
    
    def predict_wind_output(self, lat: float, lon: float, capacity_mw: float, weather_data: Dict) -> Dict:
        """
        Predict wind energy output
        """
        return self.wind_model.predict(lat, lon, capacity_mw, weather_data)
    
    def predict_hydro_output(self, lat: float, lon: float, capacity_mw: float) -> Dict:
        """
        Predict hydro energy output
        """
        return self.hydro_model.predict(lat, lon, capacity_mw)


class SolarPredictor:
    """
    Solar energy prediction model
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        
    def _build_model(self):
        """
        LSTM model for solar prediction
        """
        model = nn.Sequential(
            nn.LSTM(input_size=10, hidden_size=64, num_layers=2, batch_first=True),
        )
        return model.to(self.device)
    
    def calculate_potential(self, lat: float, lon: float, site_features: Dict) -> float:
        """
        Calculate solar potential in MWh/year
        """
        # Solar irradiance based on latitude
        base_irradiance = 5.5 - abs(lat) * 0.04  # kWh/m²/day
        
        # Adjust for site features
        cloud_factor = 1 - site_features.get("cloud_coverage", 0.2)
        terrain_factor = 1 - (site_features.get("terrain_slope", 0) / 100)
        
        # Annual generation potential (assuming 1 MW installation)
        annual_kwh = base_irradiance * 365 * 1000 * cloud_factor * terrain_factor * 0.18  # 18% efficiency
        
        return float(annual_kwh / 1000)  # Convert to MWh
    
    def predict(self, lat: float, lon: float, capacity_mw: float, weather_data: Dict) -> Dict:
        """
        Predict solar generation with weather data
        """
        # Get solar irradiance data
        irradiance = weather_data.get("solar_irradiance", [5] * 365)
        temperature = weather_data.get("temperature", [25] * 365)
        
        # Temperature coefficient (panels less efficient when hot)
        temp_coefficient = -0.004  # -0.4% per degree C above 25C
        
        daily_generation = []
        for i in range(min(365, len(irradiance))):
            # Basic solar generation formula
            base_generation = capacity_mw * irradiance[i] * 0.18  # 18% panel efficiency
            
            # Temperature adjustment
            temp_loss = max(0, (temperature[i] - 25) * temp_coefficient)
            adjusted_generation = base_generation * (1 + temp_loss)
            
            daily_generation.append(adjusted_generation * 24)  # Convert to daily MWh
        
        annual_mwh = sum(daily_generation)
        capacity_factor = annual_mwh / (capacity_mw * 8760)
        
        # Monthly aggregation
        monthly = []
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_index = 0
        for days in days_in_month:
            if day_index < len(daily_generation):
                month_total = sum(daily_generation[day_index:day_index+days])
                monthly.append(month_total)
                day_index += days
        
        return {
            "annual_mwh": annual_mwh,
            "capacity_factor": capacity_factor,
            "monthly": monthly,
            "daily_average": annual_mwh / 365
        }


class WindPredictor:
    """
    Wind energy prediction model
    """
    
    def __init__(self):
        self.cut_in_speed = 3.5  # m/s
        self.rated_speed = 12  # m/s
        self.cut_out_speed = 25  # m/s
        
    def calculate_potential(self, lat: float, lon: float, site_features: Dict) -> float:
        """
        Calculate wind potential in MWh/year
        """
        # Base wind speed estimation
        base_wind_speed = 6.5 + np.random.uniform(-2, 2)
        
        # Height adjustment (80m hub height)
        height_factor = (80/10) ** 0.14  # Wind shear formula
        adjusted_wind_speed = base_wind_speed * height_factor
        
        # Capacity factor based on wind speed
        if adjusted_wind_speed < self.cut_in_speed:
            capacity_factor = 0
        elif adjusted_wind_speed < self.rated_speed:
            capacity_factor = (adjusted_wind_speed - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed) * 0.45
        elif adjusted_wind_speed < self.cut_out_speed:
            capacity_factor = 0.45
        else:
            capacity_factor = 0
        
        # Annual generation for 1 MW turbine
        annual_mwh = 8760 * capacity_factor
        
        return float(annual_mwh)
    
    def predict(self, lat: float, lon: float, capacity_mw: float, weather_data: Dict) -> Dict:
        """
        Predict wind generation with weather data
        """
        wind_speeds = weather_data.get("wind_speed", [7] * 365)
        
        daily_generation = []
        for wind_speed in wind_speeds:
            # Wind power curve
            if wind_speed < self.cut_in_speed:
                power_output = 0
            elif wind_speed < self.rated_speed:
                # Cubic relationship between wind speed and power
                power_output = capacity_mw * ((wind_speed - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)) ** 3
            elif wind_speed < self.cut_out_speed:
                power_output = capacity_mw
            else:
                power_output = 0  # Shutdown for safety
            
            daily_generation.append(power_output * 24)  # Daily MWh
        
        annual_mwh = sum(daily_generation)
        capacity_factor = annual_mwh / (capacity_mw * 8760)
        
        # Monthly aggregation
        monthly = []
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_index = 0
        for days in days_in_month:
            if day_index < len(daily_generation):
                month_total = sum(daily_generation[day_index:day_index+days])
                monthly.append(month_total)
                day_index += days
        
        return {
            "annual_mwh": annual_mwh,
            "capacity_factor": capacity_factor,
            "monthly": monthly,
            "daily_average": annual_mwh / 365,
            "avg_wind_speed": np.mean(wind_speeds)
        }


class HydroPredictor:
    """
    Hydro energy prediction model
    """
    
    def calculate_potential(self, lat: float, lon: float, site_features: Dict) -> float:
        """
        Calculate hydro potential in MWh/year
        """
        # Hydro has high capacity factors
        base_capacity_factor = 0.5 + np.random.uniform(-0.1, 0.2)
        
        # Water availability factor
        water_factor = site_features.get("water_flow", 1.0)
        
        # Annual generation for 1 MW installation
        annual_mwh = 8760 * base_capacity_factor * water_factor
        
        return float(annual_mwh)
    
    def predict(self, lat: float, lon: float, capacity_mw: float) -> Dict:
        """
        Predict hydro generation
        """
        # Hydro is more predictable, with seasonal variations
        capacity_factors_monthly = [
            0.6, 0.65, 0.7, 0.75,  # Winter/Spring (high flow)
            0.55, 0.45, 0.4, 0.35,  # Summer (low flow)
            0.4, 0.45, 0.5, 0.55    # Fall (recovering flow)
        ]
        
        monthly = []
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        for i, (days, cf) in enumerate(zip(days_in_month, capacity_factors_monthly)):
            monthly_mwh = capacity_mw * 24 * days * cf
            monthly.append(monthly_mwh)
        
        annual_mwh = sum(monthly)
        capacity_factor = annual_mwh / (capacity_mw * 8760)
        
        return {
            "annual_mwh": annual_mwh,
            "capacity_factor": capacity_factor,
            "monthly": monthly,
            "daily_average": annual_mwh / 365
        }