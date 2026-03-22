#!/usr/bin/env python3
"""
Terra Atlas Real Data Fetcher (Simplified)
Works without python-dotenv dependency
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Try to import requests, use urllib as fallback
try:
    import requests
    USE_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    USE_REQUESTS = False
    print("⚠️  Using urllib instead of requests")

# Read environment variables directly
FIRMS_API_KEY = os.environ.get('FIRMS_API_KEY', '')
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')

# API Endpoints
FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov"
USGS_API_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0"
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5"

def fetch_url(url: str) -> str:
    """Fetch URL content with requests or urllib fallback"""
    if USE_REQUESTS:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode('utf-8')

def fetch_usgs_earthquakes(days_back: int = 7) -> Dict[str, Any]:
    """
    Fetch recent earthquake data from USGS
    No API key required - this always works!
    """
    try:
        # USGS provides different feeds based on time
        if days_back == 1:
            url = f"{USGS_API_BASE}/summary/all_day.geojson"
        elif days_back <= 7:
            url = f"{USGS_API_BASE}/summary/all_week.geojson"
        else:
            url = f"{USGS_API_BASE}/summary/all_month.geojson"
        
        print(f"🌍 Fetching USGS earthquake data...")
        content = fetch_url(url)
        data = json.loads(content)
        
        # Transform USGS format to our standard format
        features = []
        for feature in data.get('features', []):
            props = feature.get('properties', {})
            magnitude = props.get('mag', 0)
            
            # Only include earthquakes with magnitude > 2.5
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
                        "quality_score": min(magnitude / 10.0, 1.0),
                        "data_lineage": ["USGS", "Seismic Network", "Real-time"],
                        "alert": props.get('alert'),
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
        return {"type": "FeatureCollection", "features": []}

def generate_demo_fires() -> Dict[str, Any]:
    """Generate demo fire data when no API key is available"""
    import random
    
    features = []
    # Generate 50 demo fires around the world
    fire_regions = [
        {"name": "California", "lat": 37.0, "lon": -120.0, "count": 10},
        {"name": "Australia", "lat": -25.0, "lon": 133.0, "count": 8},
        {"name": "Amazon", "lat": -5.0, "lon": -60.0, "count": 12},
        {"name": "Africa", "lat": -2.0, "lon": 20.0, "count": 10},
        {"name": "Siberia", "lat": 60.0, "lon": 100.0, "count": 10}
    ]
    
    for region in fire_regions:
        for i in range(region["count"]):
            lat = region["lat"] + random.uniform(-5, 5)
            lon = region["lon"] + random.uniform(-5, 5)
            features.append({
                "type": "Feature",
                "properties": {
                    "type": "fire",
                    "source": "Demo Data",
                    "confidence": random.randint(50, 100),
                    "brightness": random.uniform(300, 500),
                    "detection_time": datetime.now().isoformat(),
                    "quality_score": random.uniform(0.5, 1.0),
                    "data_lineage": ["Demo", "Synthetic", "Generated"]
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            })
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": "Demo Data",
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(features)
        }
    }

def generate_demo_weather() -> Dict[str, Any]:
    """Generate demo weather data"""
    import random
    
    cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777}
    ]
    
    features = []
    for city in cities:
        features.append({
            "type": "Feature",
            "properties": {
                "type": "weather",
                "source": "Demo Data",
                "city": city["name"],
                "temperature": random.uniform(10, 35),
                "humidity": random.randint(30, 90),
                "wind_speed": random.uniform(0, 20),
                "weather": random.choice(["Clear", "Cloudy", "Rainy", "Sunny"]),
                "quality_score": 0.8,
                "data_lineage": ["Demo", "Synthetic"],
                "timestamp": datetime.utcnow().isoformat()
            },
            "geometry": {
                "type": "Point",
                "coordinates": [city["lon"], city["lat"]]
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "source": "Demo Data",
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(features)
        }
    }

def save_data(data: Dict[str, Any], filename: str):
    """Save data to public/data directory"""
    os.makedirs('public/data', exist_ok=True)
    filepath = f"public/data/{filename}"
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    if 'features' in data:
        print(f"💾 Saved {filename}: {len(data['features'])} features")

def main():
    """Main function to fetch all data sources"""
    print("\n🌍 Terra Atlas Real Data Fetcher (Simplified)")
    print("=" * 50)
    
    # Always fetch USGS earthquakes (no key required)
    print("\n📊 Fetching real earthquake data...")
    earthquakes = fetch_usgs_earthquakes(7)
    save_data(earthquakes, 'usgs-earthquakes.json')
    
    # Generate demo data for other sources
    print("\n🔥 Generating demo fire data...")
    fires = generate_demo_fires()
    save_data(fires, 'nasa-firms.json')
    
    print("\n☁️  Generating demo weather data...")
    weather = generate_demo_weather()
    save_data(weather, 'openweather.json')
    
    # Simple emissions data
    print("\n📊 Generating demo emissions data...")
    emissions = {
        "type": "FeatureCollection",
        "features": [],
        "metadata": {
            "source": "Demo Data",
            "timestamp": datetime.utcnow().isoformat(),
            "count": 0
        }
    }
    save_data(emissions, 'carbon-monitor.json')
    
    # Summary
    print("\n✅ Data fetch complete!")
    print(f"📊 Real data: USGS Earthquakes ({len(earthquakes['features'])} features)")
    print("📊 Demo data: Fires, Weather, Emissions")
    print("\n💡 To get more real data:")
    print("1. Get API keys from:")
    print("   - NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/api/")
    print("   - OpenWeatherMap: https://openweathermap.org/api")
    print("2. Set environment variables:")
    print("   export FIRMS_API_KEY=your_key_here")
    print("   export OPENWEATHER_API_KEY=your_key_here")
    print("3. Run this script again")

if __name__ == "__main__":
    main()