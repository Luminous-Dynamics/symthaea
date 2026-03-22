# 🚀 Terra Atlas Discovery API

## For Developers with Stuck FERC Projects

If you're a renewable energy developer stuck in the FERC interconnection queue, this API helps you find:
- Similar projects that succeeded
- Available transmission capacity
- Shared infrastructure opportunities
- Cost-reduction strategies through corridors

## The Problem We Solve

**72% of renewable projects fail** due to:
- 5+ year interconnection queues
- Unexpected upgrade costs ($100M+ surprises)
- Lack of visibility into grid capacity
- No coordination between developers

## Quick Start

```bash
# Find similar successful projects
GET /api/discovery/similar?type=solar&capacity=100&state=TX

# Find available transmission
GET /api/discovery/transmission?lat=31.9686&lng=-99.9018&radius=50

# Find corridor opportunities
GET /api/discovery/corridors?state=TX&capacity=500
```

## API Endpoints

### 1. Find Similar Projects
```http
GET /api/discovery/similar
```

**Parameters:**
- `type` - Project type (solar, wind, storage, hybrid)
- `capacity` - Capacity in MW
- `state` - US state code
- `status` - Filter by status (operational, construction, permitted)

**Example Response:**
```json
{
  "similarProjects": [
    {
      "name": "West Texas Solar Ranch",
      "capacity_mw": 500,
      "status": "operational",
      "interconnection_cost": 45000000,
      "time_to_connect": "3.5 years",
      "developer": "NextEra Energy",
      "lessons_learned": "Shared substation reduced costs by 40%"
    }
  ],
  "insights": {
    "average_interconnection_cost": 52000000,
    "average_time": "4.2 years",
    "success_rate": "68%",
    "key_factors": ["Shared infrastructure", "Phased approach", "Early TSR"]
  }
}
```

### 2. Find Transmission Capacity
```http
GET /api/discovery/transmission
```

**Parameters:**
- `lat` - Latitude
- `lng` - Longitude  
- `radius` - Search radius in miles

**Example Response:**
```json
{
  "nearbyTransmission": [
    {
      "line": "West Texas 345kV",
      "distance_miles": 8.2,
      "available_capacity_mw": 230,
      "voltage_kv": 345,
      "owner": "Oncor",
      "upgrade_status": "Planned 2026"
    }
  ],
  "substations": [
    {
      "name": "Sweetwater Substation",
      "distance_miles": 12.5,
      "available_bays": 2,
      "recent_connections": 3
    }
  ]
}
```

### 3. Find Corridor Opportunities
```http
GET /api/discovery/corridors
```

**What are corridors?** Multiple projects sharing infrastructure to reduce costs by 74%.

**Parameters:**
- `state` - US state code
- `capacity` - Your project capacity in MW

**Example Response:**
```json
{
  "corridors": [
    {
      "name": "West Texas Renewable Corridor",
      "total_projects": 8,
      "total_capacity_mw": 2400,
      "shared_infrastructure": {
        "transmission_line": "New 500kV line",
        "substations": 2,
        "cost_per_project": 28000000,
        "savings_vs_standalone": "74%"
      },
      "open_capacity_mw": 600,
      "contact": "corridor-coordinator@terraatlas.com"
    }
  ]
}
```

### 4. FERC Queue Intelligence
```http
GET /api/discovery/queue-intelligence
```

**Parameters:**
- `region` - ISO/RTO (ERCOT, PJM, CAISO, etc.)
- `position` - Your queue position

**Example Response:**
```json
{
  "queue_analysis": {
    "your_position": 847,
    "projects_ahead": 846,
    "typical_dropout_rate": "72%",
    "expected_survivors_ahead": 237,
    "estimated_time_to_process": "4.5 years"
  },
  "acceleration_opportunities": [
    {
      "strategy": "Join West Texas Corridor",
      "time_savings": "2 years",
      "cost_savings": "$45M",
      "probability_of_success": "85%"
    }
  ]
}
```

## Real Success Stories

### Case Study: West Texas Corridor
- **8 projects, 2.4 GW total**
- **Individual cost**: $120M interconnection each
- **Corridor cost**: $28M per project (74% reduction)
- **Time saved**: 2.5 years
- **Status**: Under construction

## Integration Guide

### JavaScript/TypeScript
```javascript
const TERRA_ATLAS_API = 'https://api.terraatlas.com/discovery';

async function findTransmissionCapacity(lat, lng) {
  const response = await fetch(
    `${TERRA_ATLAS_API}/transmission?lat=${lat}&lng=${lng}&radius=50`
  );
  const data = await response.json();
  
  // Find best option
  const bestOption = data.nearbyTransmission
    .sort((a, b) => b.available_capacity_mw - a.available_capacity_mw)[0];
  
  return bestOption;
}
```

### Python
```python
import requests

TERRA_ATLAS_API = 'https://api.terraatlas.com/discovery'

def find_similar_projects(project_type, capacity_mw, state):
    response = requests.get(
        f'{TERRA_ATLAS_API}/similar',
        params={
            'type': project_type,
            'capacity': capacity_mw,
            'state': state
        }
    )
    
    data = response.json()
    return data['similarProjects']
```

## Authentication

Currently in beta - no authentication required.

Production will use API keys:
```http
Authorization: Bearer YOUR_API_KEY
```

## Rate Limits

- **Beta**: 100 requests/hour
- **Free tier**: 1,000 requests/day
- **Pro**: 10,000 requests/day
- **Enterprise**: Unlimited

## Database Schema

We track:
- 10,000+ FERC queue projects
- 87,000 USACE dams (4,350 feasible for retrofit)
- 1,000+ transmission lines
- 500+ substations
- Real interconnection costs
- Actual development timelines

## Contact

**For developers with stuck projects:**
- Email: developers@terraatlas.com
- Discord: discord.gg/terraatlas
- Schedule call: calendly.com/terraatlas

**We can help you:**
1. Find corridor opportunities
2. Identify available transmission
3. Connect with other developers
4. Reduce interconnection costs by 74%
5. Accelerate your timeline by 2+ years

## The Vision

Terra Atlas is building the "Planetary Nervous System" - real-time truth about energy infrastructure that helps developers succeed instead of failing.

**Join us in solving the 72% failure rate.**

---

*"Making the invisible visible, turning failure into success."*