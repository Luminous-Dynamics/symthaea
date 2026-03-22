#!/usr/bin/env python3
"""
Terra Atlas Site Analysis Engine
Advanced Python backend for energy site analysis and optimization
"""

import os
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import aiohttp
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# ========================================
# CONFIGURATION
# ========================================

# Database connection (from environment or BWS)
DB_URL = os.environ.get('DATABASE_URL', 'postgresql://user:password@localhost/terra_atlas')
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://fyyszjyixenujgbjaqkd.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_ANON_KEY', '')

# Redis for caching
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')

# API Keys
PLANET_API_KEY = os.environ.get('PLANET_API_KEY', '')
MAPBOX_TOKEN = os.environ.get('MAPBOX_TOKEN', '')
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
NREL_API_KEY = os.environ.get('NREL_API_KEY', '')

# ========================================
# DATA MODELS
# ========================================

@dataclass
class SiteAnalysis:
    """Complete site analysis result"""
    site_id: str
    latitude: float
    longitude: float
    solar_potential_kwh_m2_day: float
    wind_speed_avg_ms: float
    hydro_potential_mw: Optional[float]
    grid_distance_km: float
    land_cost_usd_acre: float
    environmental_score: float
    social_acceptance_score: float
    roi_estimate_percent: float
    optimal_technology: str
    capacity_factor: float
    lcoe_usd_mwh: float
    carbon_offset_tons_year: float
    investment_required_usd: float
    payback_years: float
    irr_percent: float
    risks: List[str]
    opportunities: List[str]
    timestamp: datetime

@dataclass
class GridConnection:
    """Grid connection analysis"""
    nearest_substation_km: float
    transmission_voltage_kv: int
    available_capacity_mw: float
    interconnection_cost_usd: float
    queue_position: Optional[int]
    estimated_approval_date: Optional[datetime]

# ========================================
# SITE ANALYSIS ENGINE
# ========================================

class SiteAnalysisEngine:
    """
    Advanced site analysis using multiple data sources and ML
    """
    
    def __init__(self):
        self.db = self._connect_db()
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.session = None
        self.ml_model = self._load_ml_model()
        self.scaler = StandardScaler()
    
    def _connect_db(self):
        """Connect to PostgreSQL database"""
        return psycopg2.connect(DB_URL)
    
    def _load_ml_model(self):
        """Load or create ML model for site scoring"""
        model_path = 'models/site_scorer.joblib'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            # Create new model with default parameters
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    async def analyze_site(self, lat: float, lon: float, 
                          project_type: str = 'auto') -> SiteAnalysis:
        """
        Comprehensive site analysis combining multiple data sources
        """
        
        # Check cache first
        cache_key = f"site_analysis:{lat:.4f},{lon:.4f}:{project_type}"
        cached = self.redis.get(cache_key)
        if cached:
            return SiteAnalysis(**json.loads(cached))
        
        # Gather data from multiple sources
        async with aiohttp.ClientSession() as self.session:
            # Parallel data fetching
            solar_task = self.fetch_solar_data(lat, lon)
            wind_task = self.fetch_wind_data(lat, lon)
            hydro_task = self.fetch_hydro_potential(lat, lon)
            grid_task = self.analyze_grid_connection(lat, lon)
            land_task = self.fetch_land_data(lat, lon)
            environmental_task = self.assess_environmental_impact(lat, lon)
            social_task = self.assess_social_factors(lat, lon)
            satellite_task = self.fetch_satellite_imagery(lat, lon)
            
            results = await asyncio.gather(
                solar_task, wind_task, hydro_task, grid_task,
                land_task, environmental_task, social_task, satellite_task
            )
        
        solar, wind, hydro, grid, land, env, social, satellite = results
        
        # Determine optimal technology if auto
        if project_type == 'auto':
            project_type = self.determine_optimal_technology(
                solar, wind, hydro, grid
            )
        
        # Calculate financial metrics
        financial = self.calculate_financial_metrics(
            project_type, solar, wind, hydro, grid, land
        )
        
        # ML-based site scoring
        site_score = self.score_site_ml({
            'solar_potential': solar['daily_kwh_m2'],
            'wind_speed': wind['avg_speed_ms'],
            'grid_distance': grid.nearest_substation_km,
            'land_cost': land['cost_per_acre'],
            'environmental_score': env['score'],
            'social_score': social['acceptance_score']
        })
        
        # Identify risks and opportunities
        risks = self.identify_risks(env, social, grid, financial)
        opportunities = self.identify_opportunities(
            solar, wind, hydro, grid, land
        )
        
        analysis = SiteAnalysis(
            site_id=f"{lat:.4f},{lon:.4f}",
            latitude=lat,
            longitude=lon,
            solar_potential_kwh_m2_day=solar['daily_kwh_m2'],
            wind_speed_avg_ms=wind['avg_speed_ms'],
            hydro_potential_mw=hydro.get('potential_mw'),
            grid_distance_km=grid.nearest_substation_km,
            land_cost_usd_acre=land['cost_per_acre'],
            environmental_score=env['score'],
            social_acceptance_score=social['acceptance_score'],
            roi_estimate_percent=financial['roi_percent'],
            optimal_technology=project_type,
            capacity_factor=financial['capacity_factor'],
            lcoe_usd_mwh=financial['lcoe'],
            carbon_offset_tons_year=financial['carbon_offset'],
            investment_required_usd=financial['total_investment'],
            payback_years=financial['payback_years'],
            irr_percent=financial['irr'],
            risks=risks,
            opportunities=opportunities,
            timestamp=datetime.now()
        )
        
        # Cache the result
        self.redis.setex(
            cache_key, 
            3600,  # 1 hour TTL
            json.dumps(asdict(analysis), default=str)
        )
        
        return analysis
    
    async def fetch_solar_data(self, lat: float, lon: float) -> Dict:
        """Fetch solar irradiance data from NREL"""
        if not NREL_API_KEY:
            # Use estimated values
            return {
                'daily_kwh_m2': 4.5 + np.random.random(),  # Rough average
                'dni': 5.0,
                'ghi': 4.5
            }
        
        url = f"https://developer.nrel.gov/api/solar/solar_resource/v1.json"
        params = {
            'api_key': NREL_API_KEY,
            'lat': lat,
            'lon': lon
        }
        
        async with self.session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    'daily_kwh_m2': data['outputs']['avg_dni']['annual'],
                    'dni': data['outputs']['avg_dni']['annual'],
                    'ghi': data['outputs']['avg_ghi']['annual']
                }
            else:
                return {'daily_kwh_m2': 4.5, 'dni': 5.0, 'ghi': 4.5}
    
    async def fetch_wind_data(self, lat: float, lon: float) -> Dict:
        """Fetch wind speed data"""
        # OpenWeather API for basic wind data
        if OPENWEATHER_API_KEY:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': OPENWEATHER_API_KEY
            }
            
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'avg_speed_ms': data['wind']['speed'],
                        'direction': data['wind'].get('deg', 0)
                    }
        
        # Estimate based on location
        return {
            'avg_speed_ms': 5.5 + np.random.random() * 3,
            'direction': np.random.randint(0, 360)
        }
    
    async def fetch_hydro_potential(self, lat: float, lon: float) -> Dict:
        """Analyze hydro potential from nearby water bodies"""
        # Query database for nearby dams/water bodies
        with self.db.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    dam_name,
                    nid_height,
                    drainage_area,
                    ST_Distance(
                        ST_MakePoint(%s, %s)::geography,
                        ST_MakePoint(longitude, latitude)::geography
                    ) / 1000 as distance_km
                FROM usace_dams
                WHERE ST_DWithin(
                    ST_MakePoint(%s, %s)::geography,
                    ST_MakePoint(longitude, latitude)::geography,
                    50000  -- 50km radius
                )
                AND purposes NOT LIKE '%%H%%'  -- Not already hydro
                ORDER BY distance_km
                LIMIT 1
            """, (lon, lat, lon, lat))
            
            dam = cur.fetchone()
            if dam:
                # Estimate potential based on height and drainage
                height_m = float(dam['nid_height']) * 0.3048
                drainage_sqmi = float(dam['drainage_area'])
                potential_mw = height_m * 0.3 * (drainage_sqmi / 100)
                
                return {
                    'potential_mw': potential_mw,
                    'dam_name': dam['dam_name'],
                    'distance_km': dam['distance_km']
                }
        
        return {'potential_mw': None}
    
    async def analyze_grid_connection(self, lat: float, lon: float) -> GridConnection:
        """Analyze grid connection feasibility"""
        # In production, would query actual transmission line databases
        # For now, estimate based on location
        
        # Simulate distance to nearest substation
        distance = np.random.uniform(1, 50)  # km
        
        # Estimate interconnection cost
        base_cost = 100000  # $100k base
        distance_cost = distance * 50000  # $50k per km
        
        return GridConnection(
            nearest_substation_km=distance,
            transmission_voltage_kv=115 if distance < 20 else 230,
            available_capacity_mw=np.random.uniform(50, 500),
            interconnection_cost_usd=base_cost + distance_cost,
            queue_position=np.random.randint(1, 100),
            estimated_approval_date=datetime.now() + timedelta(days=365*2)
        )
    
    async def fetch_land_data(self, lat: float, lon: float) -> Dict:
        """Fetch land cost and availability data"""
        # In production, would use real estate APIs or government data
        # Estimate based on rough US averages by region
        
        if -125 < lon < -115:  # West Coast
            cost_per_acre = np.random.uniform(5000, 15000)
        elif -100 < lon < -90:  # Central
            cost_per_acre = np.random.uniform(2000, 8000)
        else:  # East Coast
            cost_per_acre = np.random.uniform(3000, 10000)
        
        return {
            'cost_per_acre': cost_per_acre,
            'available_acres': np.random.uniform(10, 1000),
            'zoning': 'agricultural',
            'ownership': 'private'
        }
    
    async def assess_environmental_impact(self, lat: float, lon: float) -> Dict:
        """Environmental impact assessment"""
        # Check for protected areas, endangered species, etc.
        # In production, would query EPA, FWS databases
        
        score = np.random.uniform(60, 95)  # 0-100 scale
        
        issues = []
        if score < 70:
            issues.append("Wetlands present")
        if score < 80:
            issues.append("Migratory bird pathway")
        
        return {
            'score': score,
            'issues': issues,
            'permits_required': ['NEPA', 'State Environmental Review']
        }
    
    async def assess_social_factors(self, lat: float, lon: float) -> Dict:
        """Social acceptance and community factors"""
        # In production, would analyze demographics, voting patterns, etc.
        
        acceptance_score = np.random.uniform(50, 90)
        
        factors = {
            'population_density': np.random.uniform(10, 1000),
            'median_income': np.random.uniform(30000, 80000),
            'renewable_support': acceptance_score,
            'local_employment_impact': np.random.randint(10, 100)
        }
        
        return {
            'acceptance_score': acceptance_score,
            'factors': factors
        }
    
    async def fetch_satellite_imagery(self, lat: float, lon: float) -> Dict:
        """Fetch satellite imagery for site assessment"""
        if not PLANET_API_KEY:
            return {'imagery_available': False}
        
        # Planet API for satellite imagery
        url = "https://api.planet.com/data/v1/quick-search"
        headers = {"Authorization": f"api-key {PLANET_API_KEY}"}
        
        payload = {
            "filter": {
                "type": "AndFilter",
                "config": [
                    {
                        "type": "GeometryFilter",
                        "field_name": "geometry",
                        "config": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        }
                    },
                    {
                        "type": "DateRangeFilter",
                        "field_name": "acquired",
                        "config": {
                            "gte": (datetime.now() - timedelta(days=30)).isoformat(),
                            "lte": datetime.now().isoformat()
                        }
                    }
                ]
            },
            "item_types": ["PSScene4Band"]
        }
        
        async with self.session.post(url, json=payload, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    'imagery_available': True,
                    'recent_images': len(data.get('features', [])),
                    'cloud_coverage': np.mean([
                        f['properties'].get('cloud_cover', 0) 
                        for f in data.get('features', [])[:5]
                    ])
                }
        
        return {'imagery_available': False}
    
    def determine_optimal_technology(self, solar, wind, hydro, grid) -> str:
        """Determine optimal renewable technology for site"""
        scores = {}
        
        # Solar scoring
        if solar['daily_kwh_m2'] > 5:
            scores['solar'] = 90
        elif solar['daily_kwh_m2'] > 4:
            scores['solar'] = 70
        else:
            scores['solar'] = 50
        
        # Wind scoring
        if wind['avg_speed_ms'] > 7:
            scores['wind'] = 90
        elif wind['avg_speed_ms'] > 5.5:
            scores['wind'] = 70
        else:
            scores['wind'] = 40
        
        # Hydro scoring
        if hydro.get('potential_mw'):
            if hydro['potential_mw'] > 10:
                scores['hydro'] = 95
            elif hydro['potential_mw'] > 1:
                scores['hydro'] = 80
            else:
                scores['hydro'] = 60
        
        # Consider hybrid if multiple are good
        good_technologies = [k for k, v in scores.items() if v > 70]
        if len(good_technologies) > 1:
            return f"hybrid_{'+'.join(good_technologies)}"
        
        # Return best single technology
        return max(scores.items(), key=lambda x: x[1])[0] if scores else 'solar'
    
    def calculate_financial_metrics(self, project_type, solar, wind, 
                                   hydro, grid, land) -> Dict:
        """Calculate comprehensive financial metrics"""
        
        # Base assumptions
        capacity_mw = 100  # Standard 100MW project
        
        # Technology-specific parameters
        if 'solar' in project_type:
            capex_per_mw = 1000000  # $1M/MW
            capacity_factor = min(solar['daily_kwh_m2'] / 8.0, 0.35)
            opex_per_mw_year = 20000
            lifetime_years = 25
        elif 'wind' in project_type:
            capex_per_mw = 1300000
            capacity_factor = min(wind['avg_speed_ms'] / 15.0, 0.45)
            opex_per_mw_year = 30000
            lifetime_years = 20
        elif 'hydro' in project_type:
            capex_per_mw = 2000000
            capacity_factor = 0.5
            opex_per_mw_year = 15000
            lifetime_years = 50
        else:
            capex_per_mw = 1200000
            capacity_factor = 0.3
            opex_per_mw_year = 25000
            lifetime_years = 25
        
        # Calculate totals
        total_capex = capacity_mw * capex_per_mw
        land_cost = land['cost_per_acre'] * capacity_mw * 5  # 5 acres/MW avg
        interconnection_cost = grid.interconnection_cost_usd
        total_investment = total_capex + land_cost + interconnection_cost
        
        # Energy production
        annual_mwh = capacity_mw * capacity_factor * 8760
        
        # Revenue (assume $50/MWh PPA)
        ppa_price = 50
        annual_revenue = annual_mwh * ppa_price
        
        # OPEX
        annual_opex = capacity_mw * opex_per_mw_year
        
        # Simple financial metrics
        annual_cashflow = annual_revenue - annual_opex
        payback_years = total_investment / annual_cashflow if annual_cashflow > 0 else 999
        
        # LCOE calculation
        lcoe = (total_capex + (annual_opex * lifetime_years)) / (annual_mwh * lifetime_years)
        
        # IRR approximation (simplified)
        irr = ((annual_cashflow * lifetime_years - total_investment) / total_investment) / lifetime_years * 100
        
        # ROI
        total_profit = (annual_cashflow * lifetime_years) - total_investment
        roi_percent = (total_profit / total_investment) * 100
        
        # Carbon offset (0.4 tons CO2/MWh avoided)
        carbon_offset = annual_mwh * 0.4
        
        return {
            'total_investment': total_investment,
            'capacity_factor': capacity_factor,
            'annual_revenue': annual_revenue,
            'annual_opex': annual_opex,
            'annual_cashflow': annual_cashflow,
            'payback_years': payback_years,
            'lcoe': lcoe,
            'irr': irr,
            'roi_percent': roi_percent,
            'carbon_offset': carbon_offset
        }
    
    def score_site_ml(self, features: Dict) -> float:
        """ML-based site scoring"""
        # Prepare features for ML model
        X = np.array([[
            features['solar_potential'],
            features['wind_speed'],
            features['grid_distance'],
            features['land_cost'],
            features['environmental_score'],
            features['social_score']
        ]])
        
        # Scale features
        if hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.transform(X)
        else:
            # Fit scaler with dummy data if not fitted
            dummy_data = np.random.rand(100, 6)
            self.scaler.fit(dummy_data)
            X_scaled = self.scaler.transform(X)
        
        # Predict score (would be trained on historical project success)
        # For now, return weighted average
        weights = [0.2, 0.2, -0.15, -0.1, 0.2, 0.15]
        score = sum(X[0][i] * weights[i] for i in range(len(weights)))
        
        return min(max(score, 0), 100)  # Normalize to 0-100
    
    def identify_risks(self, env, social, grid, financial) -> List[str]:
        """Identify project risks"""
        risks = []
        
        if env['score'] < 70:
            risks.append("Environmental permitting challenges")
        
        if social['acceptance_score'] < 60:
            risks.append("Community opposition likely")
        
        if grid.nearest_substation_km > 30:
            risks.append("High transmission costs")
        
        if financial['payback_years'] > 15:
            risks.append("Long payback period")
        
        if grid.queue_position and grid.queue_position > 50:
            risks.append("Long interconnection queue")
        
        return risks
    
    def identify_opportunities(self, solar, wind, hydro, grid, land) -> List[str]:
        """Identify project opportunities"""
        opportunities = []
        
        if solar['daily_kwh_m2'] > 5.5:
            opportunities.append("Excellent solar resource")
        
        if wind['avg_speed_ms'] > 7:
            opportunities.append("Strong wind resource")
        
        if hydro.get('potential_mw', 0) > 5:
            opportunities.append("Significant hydro potential")
        
        if grid.nearest_substation_km < 5:
            opportunities.append("Close grid connection")
        
        if land['cost_per_acre'] < 3000:
            opportunities.append("Low land costs")
        
        if grid.available_capacity_mw > 200:
            opportunities.append("Ample grid capacity")
        
        return opportunities

# ========================================
# API ENDPOINTS
# ========================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Terra Atlas Site Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SiteRequest(BaseModel):
    latitude: float
    longitude: float
    project_type: str = "auto"

engine = SiteAnalysisEngine()

@app.post("/api/analyze-site")
async def analyze_site(request: SiteRequest):
    """Analyze a potential energy site"""
    try:
        analysis = await engine.analyze_site(
            request.latitude,
            request.longitude,
            request.project_type
        )
        return asdict(analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/batch-analyze")
async def batch_analyze(state: str, limit: int = 100):
    """Batch analyze all sites in a state"""
    # Implementation for batch processing
    pass

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "site-analysis-engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)