#!/usr/bin/env python3
"""
NASA FIRMS (Fire Information for Resource Management System) Data Fetcher
Fetches near real-time active fire data from NASA's MODIS and VIIRS satellites
"""

import json
import requests
from datetime import datetime, timedelta
import os
import sys

# NASA FIRMS API endpoints
FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov"
FIRMS_MAP_KEY = "da1c20a8b5ff92cc45077f6c3f60d5cc"  # Public demo key - replace with your own for production

def fetch_active_fires(days_back=1):
    """
    Fetch active fire data from NASA FIRMS
    
    Args:
        days_back: Number of days of historical data to fetch (max 10 for public key)
    
    Returns:
        Normalized GeoJSON feature collection
    """
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # FIRMS API URL for global MODIS + VIIRS data
    # Format: area/csv/map_key/source/days
    url = f"{FIRMS_API_BASE}/api/area/csv/{FIRMS_MAP_KEY}/MODIS_NRT/world/{days_back}"
    
    print(f"Fetching NASA FIRMS data for last {days_back} day(s)...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse CSV response
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            print("No fire data available")
            return create_empty_geojson()
        
        # Parse header and data
        headers = lines[0].split(',')
        features = []
        
        for line in lines[1:]:
            values = line.split(',')
            if len(values) != len(headers):
                continue
                
            # Create a dictionary from CSV row
            fire_data = dict(zip(headers, values))
            
            # Convert to GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(fire_data.get('longitude', 0)),
                        float(fire_data.get('latitude', 0))
                    ]
                },
                "properties": {
                    "id": f"firms_{fire_data.get('latitude')}_{fire_data.get('longitude')}_{fire_data.get('acq_date')}_{fire_data.get('acq_time')}",
                    "source": "NASA_FIRMS_MODIS",
                    "type": "active_fire",
                    "timestamp": parse_firms_datetime(
                        fire_data.get('acq_date', ''),
                        fire_data.get('acq_time', '')
                    ),
                    "confidence": parse_confidence(fire_data.get('confidence', '')),
                    "brightness": float(fire_data.get('bright_t31', 0)),
                    "frp": float(fire_data.get('frp', 0)),  # Fire Radiative Power (MW)
                    "satellite": fire_data.get('satellite', 'Unknown'),
                    "quality_score": calculate_quality_score(fire_data)
                }
            }
            
            features.append(feature)
        
        print(f"Successfully fetched {len(features)} active fire detections")
        
        # Create GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "source": "NASA FIRMS",
                "description": "Active fire detections from MODIS and VIIRS satellites",
                "last_updated": datetime.now().isoformat(),
                "update_frequency": "3 hours",
                "license": "Public Domain (US Government Work)",
                "attribution": "NASA Fire Information for Resource Management System (FIRMS)",
                "total_features": len(features)
            },
            "features": features
        }
        
        return geojson
        
    except requests.RequestException as e:
        print(f"Error fetching NASA FIRMS data: {e}")
        return create_empty_geojson()

def parse_firms_datetime(date_str, time_str):
    """Convert FIRMS date and time to ISO format"""
    try:
        # FIRMS date format: YYYY-MM-DD, time format: HHMM
        if date_str and time_str:
            # Pad time with zeros if needed
            time_str = str(time_str).zfill(4)
            hour = time_str[:2]
            minute = time_str[2:4]
            dt = datetime.strptime(f"{date_str} {hour}:{minute}", "%Y-%m-%d %H:%M")
            return dt.isoformat() + "Z"
    except:
        pass
    return datetime.now().isoformat() + "Z"

def parse_confidence(confidence_value):
    """Normalize confidence values to 0-100 scale"""
    try:
        # MODIS confidence is 0-100%, VIIRS is low/nominal/high
        if confidence_value.lower() in ['low', 'l']:
            return 30
        elif confidence_value.lower() in ['nominal', 'n']:
            return 60
        elif confidence_value.lower() in ['high', 'h']:
            return 90
        else:
            return int(float(confidence_value))
    except:
        return 50  # Default medium confidence

def calculate_quality_score(fire_data):
    """Calculate a quality score for the detection (0-100)"""
    score = 50  # Base score
    
    # Adjust based on confidence
    confidence = parse_confidence(fire_data.get('confidence', ''))
    score = (score + confidence) / 2
    
    # Adjust based on FRP (Fire Radiative Power)
    frp = float(fire_data.get('frp', 0))
    if frp > 100:
        score += 10
    elif frp > 50:
        score += 5
    
    # Cap at 100
    return min(int(score), 100)

def create_empty_geojson():
    """Create an empty GeoJSON structure"""
    return {
        "type": "FeatureCollection",
        "metadata": {
            "source": "NASA FIRMS",
            "description": "No active fire data available",
            "last_updated": datetime.now().isoformat(),
            "total_features": 0
        },
        "features": []
    }

def save_to_file(data, filename):
    """Save data to JSON file"""
    output_path = os.path.join(os.path.dirname(__file__), '..', 'public', 'data', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data saved to {output_path}")
    return output_path

def main():
    """Main execution function"""
    # Fetch last 2 days of fire data (for better coverage)
    fire_data = fetch_active_fires(days_back=2)
    
    # Save to public/data directory
    output_file = save_to_file(fire_data, 'nasa-firms.json')
    
    # Also create a summary file for quick stats
    summary = {
        "last_updated": datetime.now().isoformat(),
        "total_fires": len(fire_data['features']),
        "high_confidence_fires": len([
            f for f in fire_data['features'] 
            if f['properties']['confidence'] >= 70
        ]),
        "data_source": "NASA FIRMS",
        "update_frequency": "Every 3 hours"
    }
    
    save_to_file(summary, 'nasa-firms-summary.json')
    
    print(f"✅ NASA FIRMS data pipeline complete!")
    print(f"   Total fires detected: {summary['total_fires']}")
    print(f"   High confidence fires: {summary['high_confidence_fires']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())