#!/usr/bin/env python3
"""
Terra Atlas Real Data Fetcher
Fetches actual data from multiple environmental APIs
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv('.env.local')

# API Keys
FIRMS_API_KEY = os.getenv('FIRMS_API_KEY', '')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
CARBON_API_KEY = os.getenv('CARBON_API_KEY', '')

# API Endpoints
FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov"
USGS_API_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0"
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5"

def fetch_nasa_fires(days_back: int = 1) -> Dict[str, Any]:
    """
    Fetch active fire data from NASA FIRMS
    """
    if not FIRMS_API_KEY:
        print("⚠️  NASA FIRMS API key not found, using demo data")
        return load_demo_data('nasa-firms.json')
    
    try:
        # FIRMS API URL for global MODIS/VIIRS data
        url = f"{FIRMS_API_BASE}/api/area/csv/{FIRMS_API_KEY}/MODIS_NRT/world/{days_back}"
        
        print(f"🔥 Fetching NASA FIRMS data...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse CSV response
        lines = response.text.strip().split('\n')
        headers = lines[0].split(',')
        
        features = []
        for line in lines[1:]:
            values = line.split(',')
            if len(values) >= 3:
                try:
                    lat = float(values[headers.index('latitude')])
                    lon = float(values[headers.index('longitude')])
                    brightness = float(values[headers.index('brightness')] if 'brightness' in headers else '300')
                    confidence = int(values[headers.index('confidence')] if 'confidence' in headers else '50')
                    
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "type": "fire",
                            "source": "NASA FIRMS",
                            "confidence": confidence,
                            "brightness": brightness,
                            "detection_time": values[headers.index('acq_date')] if 'acq_date' in headers else datetime.now().isoformat(),
                            "quality_score": confidence / 100.0,
                            "data_lineage": ["NASA", "MODIS/VIIRS", "Real-time"]
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        }
                    })
                except (ValueError, IndexError):
                    continue
        
        print(f"✅ Fetched {len(features)} active fires from NASA FIRMS")
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "source": "NASA FIRMS",
                "timestamp": datetime.utcnow().isoformat(),
                "count": len(features)
            }
        }
        
    except Exception as e:
        print(f"❌ Error fetching NASA FIRMS data: {e}")
        return load_demo_data('nasa-firms.json')

def fetch_usgs_earthquakes(days_back: int = 7) -> Dict[str, Any]:
    """
    Fetch recent earthquake data from USGS
    No API key required
    """
    try:
        # USGS provides different feeds based on time and magnitude
        # Using "all" earthquakes from past week
        if days_back == 1:
            url = f"{USGS_API_BASE}/summary/all_day.geojson"
        elif days_back <= 7:
            url = f"{USGS_API_BASE}/summary/all_week.geojson"
        else:
            url = f"{USGS_API_BASE}/summary/all_month.geojson"
        
        print(f"🌍 Fetching USGS earthquake data...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Transform USGS format to our standard format
        features = []
        for feature in data.get('features', []):
            props = feature.get('properties', {})
            magnitude = props.get('mag', 0)
            
            # Only include earthquakes with magnitude > 2.5 for relevance
            if magnitude > 2.5:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "type": "earthquake",
                        "source": "USGS",
                        "magnitude": magnitude,
                        "depth": feature.get('geometry', {}).get('coordinates', [0, 0, 0])[2],
                        "place": props.get('place', 'Unknown'),
                        "time": datetime.fromtimestamp(props.get('time', 0) / 1000).isoformat() if props.get('time') else datetime.now().isoformat(),
                        "quality_score": min(magnitude / 10.0, 1.0),  # Normalize magnitude to 0-1
                        "data_lineage": ["USGS", "Seismic Network", "Real-time"],
                        "alert": props.get('alert', None),
                        "tsunami": props.get('tsunami', 0)
                    },
                    "geometry": feature.get('geometry')
                })
        
        print(f"✅ Fetched {len(features)} earthquakes from USGS")
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "source": "USGS",
                "timestamp": datetime.utcnow().isoformat(),
                "count": len(features)
            }
        }
        
    except Exception as e:
        print(f"❌ Error fetching USGS data: {e}")
        return load_demo_data('usgs-earthquakes.json')

def fetch_weather_data(cities: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fetch weather data from OpenWeatherMap
    """
    if not OPENWEATHER_API_KEY:
        print("⚠️  OpenWeatherMap API key not found, using demo data")
        return load_demo_data('openweather.json')
    
    # Default major cities if none provided
    if not cities:
        cities = [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
            {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"name": "Cairo", "lat": 30.0444, "lon": 31.2357},
            {"name": "São Paulo", "lat": -23.5505, "lon": -46.6333},
            {"name": "Moscow", "lat": 55.7558, "lon": 37.6173},
            {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
            {"name": "Lagos", "lat": 6.5244, "lon": 3.3792}
        ]
    
    features = []
    
    for city in cities:
        try:
            url = f"{OPENWEATHER_API_BASE}/weather"
            params = {
                "lat": city["lat"],
                "lon": city["lon"],
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            features.append({
                "type": "Feature",
                "properties": {
                    "type": "weather",
                    "source": "OpenWeatherMap",
                    "city": city["name"],
                    "temperature": data.get('main', {}).get('temp', 0),
                    "feels_like": data.get('main', {}).get('feels_like', 0),
                    "humidity": data.get('main', {}).get('humidity', 0),
                    "pressure": data.get('main', {}).get('pressure', 0),
                    "wind_speed": data.get('wind', {}).get('speed', 0),
                    "weather": data.get('weather', [{}])[0].get('main', 'Unknown'),
                    "description": data.get('weather', [{}])[0].get('description', ''),
                    "quality_score": 0.95,  # OpenWeatherMap data is generally reliable
                    "data_lineage": ["OpenWeatherMap", "Weather Stations", "Real-time"],
                    "timestamp": datetime.utcnow().isoformat()
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [city["lon"], city["lat"]]
                }
            })
            
            # Rate limiting to avoid hitting API limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"⚠️  Error fetching weather for {city['name']}: {e}")
            continue
    
    print(f"☁️  Fetched weather data for {len(features)} cities")
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": "OpenWeatherMap",
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(features)
        }
    }

def fetch_carbon_emissions() -> Dict[str, Any]:
    """
    Fetch carbon emissions data
    Note: Carbon Monitor requires special access
    """
    if not CARBON_API_KEY:
        print("⚠️  Carbon Monitor API key not found, using synthetic data")
        # Generate synthetic carbon data for major industrial regions
        regions = [
            {"name": "Eastern US", "lat": 40.0, "lon": -75.0, "emissions": 450},
            {"name": "Western Europe", "lat": 50.0, "lon": 10.0, "emissions": 380},
            {"name": "Eastern China", "lat": 35.0, "lon": 115.0, "emissions": 720},
            {"name": "India", "lat": 25.0, "lon": 78.0, "emissions": 320},
            {"name": "Japan", "lat": 36.0, "lon": 138.0, "emissions": 280},
            {"name": "Middle East", "lat": 26.0, "lon": 50.0, "emissions": 240},
            {"name": "Russia", "lat": 60.0, "lon": 90.0, "emissions": 310},
            {"name": "Brazil", "lat": -10.0, "lon": -55.0, "emissions": 180},
            {"name": "Australia", "lat": -25.0, "lon": 135.0, "emissions": 210},
            {"name": "South Africa", "lat": -29.0, "lon": 24.0, "emissions": 190}
        ]
        
        features = []
        for region in regions:
            features.append({
                "type": "Feature",
                "properties": {
                    "type": "emissions",
                    "source": "Synthetic Data",
                    "region": region["name"],
                    "co2_mt": region["emissions"],  # Million tons CO2
                    "trend": "stable",
                    "quality_score": 0.7,  # Lower score for synthetic data
                    "data_lineage": ["Estimated", "Regional Averages", "Synthetic"],
                    "timestamp": datetime.utcnow().isoformat()
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [region["lon"], region["lat"]]
                }
            })
        
        print(f"📊 Generated {len(features)} synthetic carbon emission points")
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "source": "Synthetic Estimates",
                "timestamp": datetime.utcnow().isoformat(),
                "count": len(features)
            }
        }
    
    # TODO: Implement real Carbon Monitor API when key is available
    return load_demo_data('carbon-monitor.json')

def load_demo_data(filename: str) -> Dict[str, Any]:
    """
    Load demo data as fallback
    """
    try:
        filepath = f"public/data/{filename}"
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return {"type": "FeatureCollection", "features": []}

def save_data(data: Dict[str, Any], filename: str):
    """
    Save data to public/data directory
    """
    os.makedirs('public/data', exist_ok=True)
    filepath = f"public/data/{filename}"
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    if 'features' in data:
        print(f"💾 Saved {filename}: {len(data['features'])} features")
    else:
        print(f"💾 Saved {filename}")

def main():
    """
    Main function to fetch all data sources
    """
    print("\n🌍 Terra Atlas Real Data Fetcher")
    print("=" * 50)
    
    # Check for API keys
    api_keys_status = {
        "NASA FIRMS": "✅" if FIRMS_API_KEY else "❌",
        "OpenWeatherMap": "✅" if OPENWEATHER_API_KEY else "❌",
        "Carbon Monitor": "✅" if CARBON_API_KEY else "❌",
        "USGS": "✅ (No key required)"
    }
    
    print("\n📋 API Key Status:")
    for api, status in api_keys_status.items():
        print(f"  {status} {api}")
    
    print("\n🔄 Fetching data from all sources...")
    
    # Fetch data from each source
    data_sources = {
        'nasa-firms.json': fetch_nasa_fires(1),
        'usgs-earthquakes.json': fetch_usgs_earthquakes(7),
        'openweather.json': fetch_weather_data(),
        'carbon-monitor.json': fetch_carbon_emissions()
    }
    
    # Save all data
    print("\n💾 Saving data files...")
    total_features = 0
    for filename, data in data_sources.items():
        save_data(data, filename)
        if 'features' in data:
            total_features += len(data['features'])
    
    # Create summary
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_features": total_features,
        "sources": {
            name.replace('.json', ''): {
                "feature_count": len(data.get('features', [])),
                "has_real_data": not data.get('metadata', {}).get('source', '').startswith('Synthetic')
            }
            for name, data in data_sources.items()
        }
    }
    
    save_data(summary, 'data-summary.json')
    
    print("\n✅ Data fetch complete!")
    print(f"📊 Total features: {total_features}")
    print("\n📝 To use real data:")
    print("1. Copy .env.example to .env.local")
    print("2. Add your API keys")
    print("3. Run this script again")
    print("4. Restart the Next.js dev server")

if __name__ == "__main__":
    main()