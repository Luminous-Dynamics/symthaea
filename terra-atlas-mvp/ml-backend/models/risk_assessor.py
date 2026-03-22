"""
Risk Assessor - ML model for project risk analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime

class RiskAssessor:
    """
    Comprehensive risk assessment for energy projects
    """
    
    def __init__(self):
        self.risk_factors = {
            "environmental": EnvironmentalRiskAnalyzer(),
            "regulatory": RegulatoryRiskAnalyzer(),
            "technical": TechnicalRiskAnalyzer(),
            "market": MarketRiskAnalyzer(),
            "climate": ClimateRiskAnalyzer()
        }
        
    def analyze(self, lat: float, lon: float, project_type: str, investment: float) -> Dict:
        """
        Perform comprehensive risk analysis
        """
        risks = {}
        
        # Analyze each risk category
        for category, analyzer in self.risk_factors.items():
            risks[category] = analyzer.assess(lat, lon, project_type, investment)
        
        # Calculate overall risk score (weighted average)
        weights = {
            "environmental": 0.2,
            "regulatory": 0.25,
            "technical": 0.15,
            "market": 0.25,
            "climate": 0.15
        }
        
        overall_risk = sum(risks[cat] * weights[cat] for cat in risks)
        risks["overall"] = float(np.clip(overall_risk, 0, 1))
        
        return risks


class EnvironmentalRiskAnalyzer:
    """
    Analyzes environmental risks
    """
    
    def assess(self, lat: float, lon: float, project_type: str, investment: float) -> float:
        """
        Assess environmental risk factors
        """
        base_risk = 0.3
        
        # Protected areas risk
        protected_area_risk = np.random.uniform(0, 0.3)
        
        # Biodiversity risk
        biodiversity_risk = np.random.uniform(0, 0.2)
        
        # Water resource risk
        water_risk = 0.1 if project_type != "hydro" else 0.3
        
        # Habitat disruption risk
        habitat_risk = {
            "solar": 0.2,  # Land use change
            "wind": 0.15,  # Bird/bat mortality
            "hydro": 0.4,  # River ecosystem
            "nuclear": 0.1,  # Minimal land use
            "storage": 0.05  # Minimal impact
        }.get(project_type, 0.2)
        
        total_risk = base_risk + protected_area_risk + biodiversity_risk + water_risk + habitat_risk
        
        return float(np.clip(total_risk / 2, 0, 1))


class RegulatoryRiskAnalyzer:
    """
    Analyzes regulatory and permitting risks
    """
    
    def assess(self, lat: float, lon: float, project_type: str, investment: float) -> float:
        """
        Assess regulatory risk factors
        """
        # Base regulatory complexity
        base_risk = {
            "solar": 0.2,
            "wind": 0.3,
            "hydro": 0.5,
            "nuclear": 0.8,
            "storage": 0.25
        }.get(project_type, 0.3)
        
        # Permitting timeline risk
        permit_risk = np.random.uniform(0.1, 0.3)
        
        # Policy stability risk
        policy_risk = np.random.uniform(0.05, 0.25)
        
        # Local opposition risk
        nimby_risk = np.random.uniform(0, 0.3)
        
        # Grid connection regulations
        grid_reg_risk = np.random.uniform(0.05, 0.15)
        
        total_risk = base_risk + permit_risk + policy_risk + nimby_risk + grid_reg_risk
        
        return float(np.clip(total_risk / 2.5, 0, 1))


class TechnicalRiskAnalyzer:
    """
    Analyzes technical and operational risks
    """
    
    def assess(self, lat: float, lon: float, project_type: str, investment: float) -> float:
        """
        Assess technical risk factors
        """
        # Technology maturity risk
        maturity_risk = {
            "solar": 0.1,  # Mature technology
            "wind": 0.1,   # Mature technology
            "hydro": 0.05, # Very mature
            "nuclear": 0.3, # Complex technology
            "storage": 0.25 # Evolving technology
        }.get(project_type, 0.2)
        
        # Equipment reliability risk
        reliability_risk = np.random.uniform(0.05, 0.15)
        
        # Grid integration risk
        grid_risk = np.random.uniform(0.05, 0.2)
        
        # Maintenance complexity
        maintenance_risk = {
            "solar": 0.05,
            "wind": 0.15,
            "hydro": 0.1,
            "nuclear": 0.3,
            "storage": 0.1
        }.get(project_type, 0.1)
        
        # Performance degradation risk
        degradation_risk = np.random.uniform(0.02, 0.08)
        
        total_risk = maturity_risk + reliability_risk + grid_risk + maintenance_risk + degradation_risk
        
        return float(np.clip(total_risk, 0, 1))


class MarketRiskAnalyzer:
    """
    Analyzes market and financial risks
    """
    
    def assess(self, lat: float, lon: float, project_type: str, investment: float) -> float:
        """
        Assess market risk factors
        """
        # Electricity price volatility
        price_risk = np.random.uniform(0.15, 0.35)
        
        # Competition risk
        competition_risk = np.random.uniform(0.1, 0.25)
        
        # PPA availability risk
        ppa_risk = np.random.uniform(0.05, 0.2)
        
        # Currency/inflation risk
        macro_risk = np.random.uniform(0.05, 0.15)
        
        # Technology cost evolution risk
        tech_cost_risk = {
            "solar": 0.05,  # Costs still declining
            "wind": 0.05,   # Costs still declining
            "hydro": 0.02,  # Stable costs
            "nuclear": 0.15, # Cost overruns common
            "storage": 0.2   # Rapidly evolving costs
        }.get(project_type, 0.1)
        
        # Investment size risk (larger = more risk)
        size_risk = min(investment / 1e9, 0.3)  # Cap at 0.3 for very large projects
        
        total_risk = price_risk + competition_risk + ppa_risk + macro_risk + tech_cost_risk + size_risk
        
        return float(np.clip(total_risk / 2.5, 0, 1))


class ClimateRiskAnalyzer:
    """
    Analyzes climate and weather-related risks
    """
    
    def assess(self, lat: float, lon: float, project_type: str, investment: float) -> float:
        """
        Assess climate risk factors
        """
        # Extreme weather risk (location-dependent)
        extreme_weather_risk = 0.1 + abs(lat) / 180  # Higher risk at extreme latitudes
        
        # Resource variability risk
        variability_risk = {
            "solar": 0.15,  # Cloud cover variability
            "wind": 0.2,    # Wind speed variability
            "hydro": 0.25,  # Precipitation variability
            "nuclear": 0.02, # Not weather dependent
            "storage": 0.05  # Minimal weather impact
        }.get(project_type, 0.15)
        
        # Long-term climate change risk
        climate_change_risk = np.random.uniform(0.05, 0.2)
        
        # Natural disaster risk
        disaster_risk = np.random.uniform(0.02, 0.15)
        
        # Seasonal variation risk
        seasonal_risk = {
            "solar": 0.2,   # High seasonal variation
            "wind": 0.15,   # Moderate seasonal variation
            "hydro": 0.25,  # High seasonal variation
            "nuclear": 0,    # No seasonal impact
            "storage": 0.05  # Minimal seasonal impact
        }.get(project_type, 0.1)
        
        total_risk = extreme_weather_risk + variability_risk + climate_change_risk + disaster_risk + seasonal_risk
        
        return float(np.clip(total_risk / 2, 0, 1))