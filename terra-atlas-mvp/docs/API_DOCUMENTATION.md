# 🌍 Terra Atlas API Documentation

## Base URL
- **Development**: `http://localhost:3002/api`
- **Production**: `https://terra-atlas.earth/api`

## Authentication
Currently, the API is **public and open**. No authentication required for MVP.

## Response Format
All endpoints return GeoJSON format with consistent metadata structure:

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "source": "string",
    "timestamp": "ISO 8601 datetime",
    "count": "number",
    "isRealData": "boolean",
    "dataMode": "live | demo | empty",
    "apiKeyPresent": "yes | no | not_required"
  },
  "features": [
    {
      "type": "Feature",
      "properties": { ... },
      "geometry": { ... }
    }
  ]
}
```

## Endpoints

### 🔥 Active Fires
```http
GET /api/data/fires
```

Returns active fire detections from NASA FIRMS satellites.

**Response Properties**:
- `type`: "fire"
- `source`: "NASA FIRMS" or "Demo Data"
- `confidence`: 0-100 (detection confidence)
- `brightness`: Temperature in Kelvin
- `detection_time`: ISO 8601 timestamp
- `quality_score`: 0-1 (data quality indicator)
- `data_lineage`: Array of data sources

**Example Response**:
```json
{
  "type": "FeatureCollection",
  "metadata": {
    "source": "NASA FIRMS",
    "timestamp": "2024-01-20T12:00:00Z",
    "count": 150,
    "isRealData": true,
    "dataMode": "live"
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "type": "fire",
        "source": "NASA FIRMS",
        "confidence": 85,
        "brightness": 350.5,
        "detection_time": "2024-01-20T11:30:00Z",
        "quality_score": 0.85
      },
      "geometry": {
        "type": "Point",
        "coordinates": [-120.5, 38.7]
      }
    }
  ]
}
```

### 🌍 Earthquakes
```http
GET /api/data/earthquakes
```

Returns recent earthquake data from USGS. **No API key required!**

**Response Properties**:
- `type`: "earthquake"
- `source`: "USGS"
- `magnitude`: Richter scale magnitude
- `depth`: Depth in kilometers
- `place`: Location description
- `time`: ISO 8601 timestamp
- `alert`: Alert level (green, yellow, orange, red)
- `tsunami`: 0 or 1 (tsunami warning)

**Example Response**:
```json
{
  "type": "FeatureCollection",
  "metadata": {
    "source": "USGS",
    "timestamp": "2024-01-20T12:00:00Z",
    "count": 75,
    "isRealData": true,
    "dataMode": "live"
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "type": "earthquake",
        "source": "USGS",
        "magnitude": 4.2,
        "depth": 10.5,
        "place": "10km N of San Francisco, CA",
        "time": "2024-01-20T10:15:00Z",
        "alert": "yellow",
        "tsunami": 0
      },
      "geometry": {
        "type": "Point",
        "coordinates": [-122.4194, 37.8749, 10.5]
      }
    }
  ]
}
```

### ☁️ Weather
```http
GET /api/data/weather
```

Returns current weather conditions from OpenWeatherMap.

**Response Properties**:
- `type`: "weather"
- `source`: "OpenWeatherMap" or "Demo Data"
- `city`: City name
- `temperature`: Temperature in Celsius
- `humidity`: Percentage
- `wind_speed`: Speed in m/s
- `weather`: Main weather condition
- `description`: Detailed description

**Example Response**:
```json
{
  "type": "FeatureCollection",
  "metadata": {
    "source": "OpenWeatherMap",
    "timestamp": "2024-01-20T12:00:00Z",
    "count": 10,
    "isRealData": true,
    "dataMode": "live"
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "type": "weather",
        "source": "OpenWeatherMap",
        "city": "New York",
        "temperature": 22.5,
        "humidity": 65,
        "wind_speed": 5.2,
        "weather": "Clouds",
        "description": "scattered clouds"
      },
      "geometry": {
        "type": "Point",
        "coordinates": [-74.006, 40.7128]
      }
    }
  ]
}
```

### 📊 Carbon Emissions
```http
GET /api/data/emissions
```

Returns carbon emission estimates by region.

**Response Properties**:
- `type`: "emissions"
- `source`: "Carbon Monitor" or "Synthetic Data"
- `region`: Region name
- `co2_mt`: CO2 in million tons
- `trend`: "increasing", "stable", or "decreasing"

## Error Handling

### Error Response Format
```json
{
  "error": "string",
  "message": "string",
  "status": "number"
}
```

### Status Codes
- `200 OK`: Request successful
- `400 Bad Request`: Invalid layer type
- `404 Not Found`: Data file not found
- `500 Internal Server Error`: Server error

## Rate Limiting
- **Development**: No rate limiting
- **Production**: 1000 requests per hour per IP

## CORS Headers
All endpoints include CORS headers for cross-origin requests:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, OPTIONS
```

## Caching
Responses include cache headers:
```
Cache-Control: s-maxage=900, stale-while-revalidate
```

Data is cached for 15 minutes with stale-while-revalidate for performance.

## WebSocket Support (Coming Soon)
Real-time updates via WebSocket:
```javascript
const ws = new WebSocket('wss://terra-atlas.earth/ws')
ws.on('message', (data) => {
  // Handle real-time updates
})
```

## SDKs and Libraries

### JavaScript/TypeScript
```bash
npm install @terra-atlas/sdk
```

```javascript
import { TerraAtlasClient } from '@terra-atlas/sdk'

const client = new TerraAtlasClient({
  baseURL: 'https://terra-atlas.earth/api'
})

const fires = await client.getFires()
const earthquakes = await client.getEarthquakes()
```

### Python
```bash
pip install terra-atlas
```

```python
from terra_atlas import TerraAtlasClient

client = TerraAtlasClient(base_url='https://terra-atlas.earth/api')

fires = client.get_fires()
earthquakes = client.get_earthquakes()
```

## Example Integration

### Fetch and Display Earthquakes
```javascript
async function displayEarthquakes() {
  const response = await fetch('http://localhost:3002/api/data/earthquakes')
  const data = await response.json()
  
  console.log(`Found ${data.metadata.count} earthquakes`)
  
  data.features.forEach(feature => {
    const { magnitude, place } = feature.properties
    console.log(`M${magnitude} - ${place}`)
  })
}
```

### Monitor All Layers
```javascript
const layers = ['fires', 'earthquakes', 'weather', 'emissions']

async function monitorPlanet() {
  for (const layer of layers) {
    const response = await fetch(`/api/data/${layer}`)
    const data = await response.json()
    
    console.log(`${layer}: ${data.metadata.count} features`)
    console.log(`Data mode: ${data.metadata.dataMode}`)
  }
}

// Update every 5 minutes
setInterval(monitorPlanet, 5 * 60 * 1000)
```

## Data Quality & Trust Layer

Each feature includes quality indicators:

### Quality Score (0-1)
- `1.0`: Highest quality, verified data
- `0.7-0.9`: Good quality, reliable
- `0.5-0.7`: Moderate quality, use with caution
- `<0.5`: Low quality, demonstration only

### Data Lineage
Array showing the path of data:
```json
"data_lineage": ["NASA", "MODIS/VIIRS", "Real-time"]
```

## Roadmap

### Q1 2024
- ✅ Basic API endpoints
- ✅ Real USGS earthquake data
- 🚧 NASA FIRMS integration
- 🚧 OpenWeatherMap integration

### Q2 2024
- WebSocket real-time updates
- Authentication & API keys
- Rate limiting
- GraphQL endpoint

### Q3 2024
- Machine learning predictions
- Historical data queries
- Advanced filtering
- Batch operations

## Support

- **GitHub Issues**: [github.com/terra-atlas/mvp/issues](https://github.com/terra-atlas/mvp/issues)
- **Discord**: [discord.gg/terra-atlas](https://discord.gg/terra-atlas)
- **Email**: api@terra-atlas.earth

## License

MIT License - Free to use for any purpose.

---

**Making planetary intelligence accessible to everyone** 🌍