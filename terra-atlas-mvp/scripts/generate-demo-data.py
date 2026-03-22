#!/usr/bin/env python3
"""
Generate realistic demo data for Terra Atlas MVP
"""

import json
import random
from datetime import datetime, timedelta
import os

def generate_fire_data(count=100):
    """Generate realistic fire detection data"""
    features = []
    
    # Fire hotspots around the world
    hotspots = [
        {"name": "California", "lat": 36.7783, "lng": -119.4179, "radius": 5},
        {"name": "Amazon", "lat": -3.4653, "lng": -62.2159, "radius": 8},
        {"name": "Australia", "lat": -25.2744, "lng": 133.7751, "radius": 10},
        {"name": "Siberia", "lat": 60.0, "lng": 100.0, "radius": 12},
        {"name": "Central Africa", "lat": 0.0, "lng": 20.0, "radius": 7}
    ]
    
    for _ in range(count):
        hotspot = random.choice(hotspots)
        
        # Generate location near hotspot
        lat = hotspot["lat"] + random.uniform(-hotspot["radius"], hotspot["radius"])
        lng = hotspot["lng"] + random.uniform(-hotspot["radius"], hotspot["radius"])
        
        # Generate timestamp within last 48 hours
        hours_ago = random.uniform(0, 48)
        timestamp = datetime.now() - timedelta(hours=hours_ago)
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lng, lat]
            },
            "properties": {
                "id": f"fire_{lat:.4f}_{lng:.4f}_{timestamp.isoformat()}",
                "source": "NASA_FIRMS_DEMO",
                "type": "active_fire",
                "timestamp": timestamp.isoformat() + "Z",
                "confidence": random.randint(30, 100),
                "brightness": random.uniform(300, 500),
                "frp": random.uniform(10, 1000),  # Fire Radiative Power
                "satellite": random.choice(["Terra", "Aqua", "SNPP", "NOAA20"]),
                "quality_score": random.randint(60, 100),
                "area": hotspot["name"]
            }
        }
        
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "metadata": {
            "source": "NASA FIRMS (Demo Data)",
            "description": "Simulated active fire detections for demonstration",
            "last_updated": datetime.now().isoformat(),
            "update_frequency": "3 hours",
            "license": "Demo Data - Not for Production Use",
            "attribution": "Terra Atlas Demo",
            "total_features": len(features),
            "quality_score": 85
        },
        "features": features
    }

def generate_earthquake_data(count=50):
    """Generate realistic earthquake data"""
    features = []
    
    # Seismic zones
    zones = [
        {"name": "Ring of Fire", "lat": 35.0, "lng": 140.0, "radius": 15},
        {"name": "Mediterranean", "lat": 38.0, "lng": 15.0, "radius": 8},
        {"name": "Himalayas", "lat": 28.0, "lng": 85.0, "radius": 10},
        {"name": "California", "lat": 36.0, "lng": -120.0, "radius": 5},
        {"name": "Indonesia", "lat": -2.0, "lng": 120.0, "radius": 12}
    ]
    
    for _ in range(count):
        zone = random.choice(zones)
        
        lat = zone["lat"] + random.uniform(-zone["radius"], zone["radius"])
        lng = zone["lng"] + random.uniform(-zone["radius"], zone["radius"])
        
        # Generate timestamp within last 7 days
        days_ago = random.uniform(0, 7)
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        # Magnitude distribution (more small quakes)
        magnitude = random.triangular(2.0, 4.0, 7.5)
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lng, lat]
            },
            "properties": {
                "id": f"eq_{timestamp.timestamp()}",
                "source": "USGS_DEMO",
                "type": "earthquake",
                "timestamp": timestamp.isoformat() + "Z",
                "magnitude": round(magnitude, 1),
                "depth": random.uniform(0, 700),  # km
                "confidence": 95,
                "quality_score": random.randint(70, 100),
                "area": zone["name"]
            }
        }
        
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "metadata": {
            "source": "USGS (Demo Data)",
            "description": "Simulated earthquake data for demonstration",
            "last_updated": datetime.now().isoformat(),
            "update_frequency": "Real-time",
            "total_features": len(features),
            "quality_score": 90
        },
        "features": features
    }

def generate_weather_data(count=200):
    """Generate weather event data"""
    features = []
    
    # Generate grid of weather stations
    for _ in range(count):
        lat = random.uniform(-60, 70)
        lng = random.uniform(-180, 180)
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lng, lat]
            },
            "properties": {
                "id": f"weather_{lat:.2f}_{lng:.2f}",
                "source": "OpenWeather_DEMO",
                "type": "weather_station",
                "timestamp": datetime.now().isoformat() + "Z",
                "temperature": random.uniform(-30, 45),  # Celsius
                "humidity": random.uniform(20, 100),  # Percentage
                "wind_speed": random.uniform(0, 120),  # km/h
                "precipitation": random.uniform(0, 50),  # mm
                "confidence": 99,
                "quality_score": 95
            }
        }
        
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "metadata": {
            "source": "OpenWeatherMap (Demo Data)",
            "description": "Simulated weather data for demonstration",
            "last_updated": datetime.now().isoformat(),
            "update_frequency": "Hourly",
            "total_features": len(features),
            "quality_score": 95
        },
        "features": features
    }

def save_to_file(data, filename):
    """Save data to JSON file"""
    output_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'data', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Only print feature count if data has features key
    if 'features' in data:
        print(f"✅ Generated {filename}: {len(data['features'])} features")
    else:
        print(f"✅ Generated {filename}")
    return output_path

def main():
    """Generate all demo datasets"""
    print("🌍 Generating Terra Atlas demo data...")
    
    # Generate datasets
    fire_data = generate_fire_data(150)
    earthquake_data = generate_earthquake_data(75)
    weather_data = generate_weather_data(100)
    
    # Save to files
    save_to_file(fire_data, 'nasa-firms.json')
    save_to_file(earthquake_data, 'usgs-earthquakes.json')
    save_to_file(weather_data, 'openweather.json')
    
    # Create empty carbon emissions placeholder
    carbon_data = {
        "type": "FeatureCollection",
        "metadata": {
            "source": "Carbon Monitor (Demo)",
            "description": "Carbon emissions data coming soon",
            "last_updated": datetime.now().isoformat(),
            "total_features": 0,
            "quality_score": 0
        },
        "features": []
    }
    save_to_file(carbon_data, 'carbon-monitor.json')
    
    # Create summary
    summary = {
        "last_updated": datetime.now().isoformat(),
        "datasets": {
            "fires": {
                "count": len(fire_data["features"]),
                "high_confidence": len([f for f in fire_data["features"] if f["properties"]["confidence"] >= 70])
            },
            "earthquakes": {
                "count": len(earthquake_data["features"]),
                "significant": len([f for f in earthquake_data["features"] if f["properties"]["magnitude"] >= 4.5])
            },
            "weather": {
                "count": len(weather_data["features"]),
                "extreme": len([f for f in weather_data["features"] if f["properties"]["wind_speed"] >= 60])
            }
        },
        "total_features": len(fire_data["features"]) + len(earthquake_data["features"]) + len(weather_data["features"]),
        "demo_mode": True
    }
    
    save_to_file(summary, 'summary.json')
    
    print("\n✨ Demo data generation complete!")
    print(f"   Total features: {summary['total_features']}")
    print("   Ready for Terra Atlas MVP!")

if __name__ == "__main__":
    main()