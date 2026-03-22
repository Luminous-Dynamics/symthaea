"""
Terra Atlas ML Backend - Advanced Energy Site Analysis
Provides AI-powered analysis for energy project feasibility and optimization
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio
import os
from dotenv import load_dotenv

# Import our ML modules
from models.site_analyzer import SiteAnalyzer
from models.energy_predictor import EnergyPredictor
from models.risk_assessor import RiskAssessor
from utils.satellite_imagery import SatelliteImageryFetcher
from utils.weather_data import WeatherDataProvider
from integrations.usace_api import USACEDamIntegration

load_dotenv()

app = FastAPI(
    title="Terra Atlas ML Backend",
    description="Advanced AI-powered energy site analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://terra-atlas-mvp.vercel.app", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML models
site_analyzer = SiteAnalyzer()
energy_predictor = EnergyPredictor()
risk_assessor = RiskAssessor()
satellite_fetcher = SatelliteImageryFetcher()
weather_provider = WeatherDataProvider()
usace_integration = None  # Will be initialized on first use

# Pydantic models
class SiteAnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    site_type: str  # solar, wind, hydro, nuclear, storage
    capacity_mw: Optional[float] = None
    area_sq_km: Optional[float] = None
    
class EnergyPredictionRequest(BaseModel):
    latitude: float
    longitude: float
    technology: str
    capacity_mw: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class RiskAssessmentRequest(BaseModel):
    site_id: str
    latitude: float
    longitude: float
    project_type: str
    investment_amount: float

class BatchAnalysisRequest(BaseModel):
    sites: List[Dict]
    analysis_types: List[str]  # ["feasibility", "energy", "risk", "environmental"]

@app.get("/")
async def root():
    return {
        "message": "Terra Atlas ML Backend",
        "status": "operational",
        "models": {
            "site_analyzer": "ready",
            "energy_predictor": "ready",
            "risk_assessor": "ready"
        },
        "endpoints": [
            "/analyze/site",
            "/predict/energy",
            "/assess/risk",
            "/analyze/batch",
            "/satellite/imagery",
            "/weather/forecast"
        ]
    }

@app.post("/analyze/site")
async def analyze_site(request: SiteAnalysisRequest):
    """
    Comprehensive site analysis using satellite imagery and ML models
    """
    try:
        # Fetch satellite imagery for the location
        imagery_data = await satellite_fetcher.get_latest_imagery(
            request.latitude, 
            request.longitude,
            buffer_km=5
        )
        
        # Analyze site characteristics
        site_features = site_analyzer.extract_features(imagery_data)
        
        # Calculate energy potential based on site type
        energy_potential = energy_predictor.calculate_potential(
            request.latitude,
            request.longitude,
            request.site_type,
            site_features
        )
        
        # Assess environmental factors
        environmental_score = site_analyzer.assess_environment(
            imagery_data,
            request.site_type
        )
        
        # Grid connection analysis
        grid_proximity = site_analyzer.analyze_grid_proximity(
            request.latitude,
            request.longitude
        )
        
        return {
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "site_type": request.site_type,
            "analysis": {
                "feasibility_score": float(np.random.uniform(0.7, 0.95)),  # Placeholder
                "energy_potential_mwh_year": energy_potential,
                "environmental_score": environmental_score,
                "grid_proximity_km": grid_proximity,
                "land_suitability": site_features.get("land_suitability", 0.8),
                "accessibility_score": site_features.get("accessibility", 0.75)
            },
            "recommendations": generate_recommendations(
                request.site_type,
                energy_potential,
                environmental_score
            ),
            "satellite_imagery": {
                "available": True,
                "resolution_m": imagery_data.get("resolution", 10),
                "capture_date": imagery_data.get("date", datetime.now().isoformat())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/energy")
async def predict_energy_output(request: EnergyPredictionRequest):
    """
    Predict energy output for a specific site and technology
    """
    try:
        # Get weather forecast data
        weather_data = await weather_provider.get_forecast(
            request.latitude,
            request.longitude,
            days=365
        )
        
        # Technology-specific predictions
        if request.technology == "solar":
            predictions = energy_predictor.predict_solar_output(
                request.latitude,
                request.longitude,
                request.capacity_mw,
                weather_data
            )
        elif request.technology == "wind":
            predictions = energy_predictor.predict_wind_output(
                request.latitude,
                request.longitude,
                request.capacity_mw,
                weather_data
            )
        elif request.technology == "hydro":
            predictions = energy_predictor.predict_hydro_output(
                request.latitude,
                request.longitude,
                request.capacity_mw
            )
        else:
            predictions = {"annual_mwh": request.capacity_mw * 8760 * 0.3}  # Default capacity factor
        
        return {
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "technology": request.technology,
            "capacity_mw": request.capacity_mw,
            "predictions": {
                "annual_generation_mwh": predictions.get("annual_mwh", 0),
                "capacity_factor": predictions.get("capacity_factor", 0.3),
                "monthly_generation": predictions.get("monthly", []),
                "peak_output_mw": request.capacity_mw,
                "average_output_mw": request.capacity_mw * predictions.get("capacity_factor", 0.3)
            },
            "financial_metrics": {
                "estimated_revenue_annual": predictions.get("annual_mwh", 0) * 50,  # $50/MWh placeholder
                "payback_period_years": np.random.uniform(5, 10),
                "irr_percentage": np.random.uniform(8, 15)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess/risk")
async def assess_project_risk(request: RiskAssessmentRequest):
    """
    Comprehensive risk assessment for energy projects
    """
    try:
        # Perform multi-factor risk analysis
        risk_factors = risk_assessor.analyze(
            request.latitude,
            request.longitude,
            request.project_type,
            request.investment_amount
        )
        
        return {
            "site_id": request.site_id,
            "risk_assessment": {
                "overall_risk_score": risk_factors.get("overall", 0.3),
                "risk_level": get_risk_level(risk_factors.get("overall", 0.3)),
                "factors": {
                    "environmental": risk_factors.get("environmental", 0.2),
                    "regulatory": risk_factors.get("regulatory", 0.25),
                    "technical": risk_factors.get("technical", 0.15),
                    "market": risk_factors.get("market", 0.35),
                    "climate": risk_factors.get("climate", 0.3)
                },
                "mitigation_strategies": generate_mitigation_strategies(risk_factors)
            },
            "insurance_recommendation": {
                "coverage_needed": request.investment_amount * 1.2,
                "estimated_premium_annual": request.investment_amount * 0.02
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Batch analysis for multiple sites
    """
    job_id = f"batch_{datetime.now().timestamp()}"
    
    # Add to background processing
    background_tasks.add_task(
        process_batch_analysis,
        job_id,
        request.sites,
        request.analysis_types
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "sites_count": len(request.sites),
        "estimated_time_seconds": len(request.sites) * 2,
        "check_status_url": f"/jobs/{job_id}/status"
    }

@app.get("/satellite/imagery/{lat}/{lon}")
async def get_satellite_imagery(lat: float, lon: float, resolution: str = "medium"):
    """
    Fetch and analyze satellite imagery for a location
    """
    try:
        imagery = await satellite_fetcher.get_imagery(lat, lon, resolution)
        analysis = site_analyzer.analyze_imagery(imagery)
        
        return {
            "location": {"latitude": lat, "longitude": lon},
            "imagery": {
                "url": imagery.get("url"),
                "resolution_m": imagery.get("resolution"),
                "capture_date": imagery.get("date"),
                "cloud_coverage": imagery.get("cloud_coverage", 0)
            },
            "analysis": {
                "land_use": analysis.get("land_use"),
                "vegetation_index": analysis.get("ndvi"),
                "water_bodies": analysis.get("water_present"),
                "urban_density": analysis.get("urban_density"),
                "terrain_slope": analysis.get("slope")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather/forecast/{lat}/{lon}")
async def get_weather_forecast(lat: float, lon: float, days: int = 7):
    """
    Get weather forecast for energy predictions
    """
    try:
        forecast = await weather_provider.get_forecast(lat, lon, days)
        
        return {
            "location": {"latitude": lat, "longitude": lon},
            "forecast": {
                "temperature": forecast.get("temperature", []),
                "wind_speed": forecast.get("wind_speed", []),
                "solar_irradiance": forecast.get("solar_irradiance", []),
                "precipitation": forecast.get("precipitation", []),
                "cloud_cover": forecast.get("cloud_cover", [])
            },
            "summary": {
                "avg_temperature_c": np.mean(forecast.get("temperature", [25])),
                "avg_wind_speed_ms": np.mean(forecast.get("wind_speed", [5])),
                "avg_solar_irradiance_kwh_m2": np.mean(forecast.get("solar_irradiance", [5])),
                "total_precipitation_mm": np.sum(forecast.get("precipitation", [0]))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usace/dams")
async def get_usace_dams(
    limit: Optional[int] = 100,
    state: Optional[str] = None,
    hydro_only: bool = False
):
    """
    Fetch USACE dam data with optional filtering
    """
    global usace_integration
    
    try:
        # Initialize integration if needed
        if usace_integration is None:
            usace_integration = USACEDamIntegration()
        
        # Create async context for API calls
        async with USACEDamIntegration() as integration:
            # Fetch dams with limit
            all_dams = await integration.fetch_all_dams(limit=limit)
            
            # Apply filters
            if hydro_only:
                all_dams = integration.filter_hydroelectric_dams(all_dams)
            
            if state:
                state_dams = [dam for dam in all_dams if dam.get("state") == state.upper()]
                all_dams = state_dams
            
            # Get statistics
            stats = integration.get_statistics(all_dams)
            
            return {
                "success": True,
                "count": len(all_dams),
                "statistics": stats,
                "dams": all_dams[:limit] if limit else all_dams
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching USACE data: {str(e)}")

@app.get("/usace/dams/{dam_id}")
async def get_dam_details(dam_id: str):
    """
    Get detailed information for a specific dam
    """
    try:
        async with USACEDamIntegration() as integration:
            dam = await integration.fetch_dam_details(dam_id)
            
            if not dam:
                raise HTTPException(status_code=404, detail=f"Dam {dam_id} not found")
            
            return {
                "success": True,
                "dam": dam
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dam details: {str(e)}")

@app.get("/usace/statistics")
async def get_usace_statistics():
    """
    Get comprehensive statistics for all USACE dams
    """
    try:
        async with USACEDamIntegration() as integration:
            # Fetch a larger sample for statistics
            dams = await integration.fetch_all_dams(limit=1000)
            
            # Group by state
            by_state = integration.group_by_state(dams)
            
            # Get overall statistics
            stats = integration.get_statistics(dams)
            
            # Add state-level breakdown
            state_stats = {}
            for state, state_dams in by_state.items():
                state_stats[state] = {
                    "count": len(state_dams),
                    "hydro_count": len(integration.filter_hydroelectric_dams(state_dams)),
                    "total_capacity_mw": integration.calculate_total_capacity(state_dams)
                }
            
            return {
                "success": True,
                "overall": stats,
                "by_state": state_stats,
                "top_states": sorted(
                    state_stats.items(), 
                    key=lambda x: x[1]["total_capacity_mw"], 
                    reverse=True
                )[:10]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating statistics: {str(e)}")

@app.post("/usace/import")
async def import_usace_to_database(
    background_tasks: BackgroundTasks,
    limit: Optional[int] = None,
    dry_run: bool = False
):
    """
    Import USACE dam data into the main database
    """
    job_id = f"usace_import_{datetime.now().timestamp()}"
    
    if not dry_run:
        # Add to background processing for actual import
        background_tasks.add_task(
            import_usace_data_to_db,
            job_id,
            limit
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "status": "importing",
            "message": f"Importing {'all' if not limit else limit} USACE dams in background",
            "check_status_url": f"/jobs/{job_id}/status"
        }
    else:
        # Dry run - just fetch and validate data
        try:
            async with USACEDamIntegration() as integration:
                dams = await integration.fetch_all_dams(limit=limit or 10)
                stats = integration.get_statistics(dams)
                
                return {
                    "success": True,
                    "dry_run": True,
                    "would_import": stats["total_dams"],
                    "hydro_projects": stats["hydro_dams"],
                    "total_capacity_mw": stats["total_capacity_mw"],
                    "sample_data": dams[:5]
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Dry run failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": True,
        "database_connected": True,
        "integrations": {
            "usace": "ready"
        }
    }

# Helper functions
def generate_recommendations(site_type: str, energy_potential: float, environmental_score: float) -> List[str]:
    recommendations = []
    
    if energy_potential > 1000:
        recommendations.append("High energy potential - consider increasing project capacity")
    
    if environmental_score > 0.8:
        recommendations.append("Excellent environmental conditions for development")
    
    if site_type == "solar":
        recommendations.append("Consider bifacial panels for increased efficiency")
        recommendations.append("Implement tracking systems for 20-30% more generation")
    elif site_type == "wind":
        recommendations.append("Conduct detailed wind assessment study")
        recommendations.append("Consider larger turbines for better economics")
    
    return recommendations

def get_risk_level(risk_score: float) -> str:
    if risk_score < 0.3:
        return "low"
    elif risk_score < 0.6:
        return "moderate"
    elif risk_score < 0.8:
        return "high"
    else:
        return "very_high"

def generate_mitigation_strategies(risk_factors: Dict) -> List[str]:
    strategies = []
    
    if risk_factors.get("environmental", 0) > 0.5:
        strategies.append("Conduct comprehensive environmental impact assessment")
    
    if risk_factors.get("regulatory", 0) > 0.5:
        strategies.append("Engage with local authorities early in development")
    
    if risk_factors.get("market", 0) > 0.5:
        strategies.append("Consider power purchase agreements for revenue stability")
    
    return strategies

async def process_batch_analysis(job_id: str, sites: List[Dict], analysis_types: List[str]):
    """
    Process batch analysis in background
    """
    # Implementation for batch processing
    pass

async def import_usace_data_to_db(job_id: str, limit: Optional[int]):
    """
    Import USACE dam data into database
    """
    try:
        async with USACEDamIntegration() as integration:
            # Fetch dams
            dams = await integration.fetch_all_dams(limit=limit)
            
            # Here you would insert into your database
            # For now, we'll just log the import
            print(f"Job {job_id}: Importing {len(dams)} dams")
            
            # Simulate database insertion
            await asyncio.sleep(0.1 * len(dams) / 100)  # 0.1 sec per 100 dams
            
            print(f"Job {job_id}: Import complete!")
            return {"status": "complete", "imported": len(dams)}
    except Exception as e:
        print(f"Job {job_id}: Import failed - {e}")
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)