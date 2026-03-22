# Terra Atlas Discovery API Documentation

## Overview
The Terra Atlas Discovery API provides free, public access to comprehensive clean energy project data including FERC queue projects, USACE dam retrofit opportunities, and transmission corridor optimization insights.

Base URL: `https://atlas.terra-lumina.com/api/discovery`

## Authentication
**No authentication required!** All endpoints are public and free to use.

## Rate Limiting
- 1000 requests per hour per IP
- Bulk data downloads available via direct file access

## Endpoints

### 1. FERC Queue Projects
Get real-time data on 11,547+ renewable energy projects in the FERC interconnection queue.

#### Endpoint
```
GET /api/discovery/projects
```

#### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | `projects` | Data type to return (`projects`, `stats`, `corridors`, `dams`, `usace-stats`) |
| `limit` | integer | 100 | Number of results to return (max 1000) |
| `offset` | integer | 0 | Pagination offset |
| `state` | string | - | Filter by US state code (e.g., "TX", "CA") |
| `status` | string | - | Filter by status: `active`, `withdrawn`, `operational` |
| `source` | string | - | Filter by energy source: `solar`, `wind`, `battery`, `hybrid` |
| `min_capacity` | float | 0 | Minimum capacity in MW |
| `max_capacity` | float | 10000 | Maximum capacity in MW |
| `developer` | string | - | Filter by developer name (partial match) |
| `sort` | string | `queue_date` | Sort field |
| `order` | string | `desc` | Sort order: `asc` or `desc` |

#### Example Request
```bash
curl "https://atlas.terra-lumina.com/api/discovery/projects?state=TX&status=active&limit=10"
```

#### Example Response
```json
{
  "success": true,
  "data": [
    {
      "queue_id": "ERCOT-2024-001",
      "project_name": "West Texas Solar Farm",
      "developer": "NextEra Energy",
      "state": "TX",
      "county": "Pecos County",
      "capacity_mw": 500,
      "energy_source": "Solar",
      "queue_date": "2024-01-15",
      "in_service_date": "2026-06-01",
      "withdrawn": false,
      "withdrawal_reason": null,
      "interconnection_cost": 45000000,
      "network_upgrade_cost": 125000000,
      "total_cost": 170000000,
      "latitude": 31.4234,
      "longitude": -102.8456,
      "iso": "ERCOT",
      "voltage_kv": 345,
      "county_fips": "48371",
      "corridor_opportunity": true,
      "corridor_id": "TX-WEST-01",
      "corridor_savings": 89250000,
      "corridor_savings_percent": 52.5
    }
  ],
  "pagination": {
    "total": 423,
    "limit": 10,
    "offset": 0,
    "has_more": true
  },
  "aggregates": {
    "total_projects": 423,
    "total_capacity_mw": 45234,
    "total_capacity_gw": "45.2",
    "total_interconnection_cost": 12300000000,
    "total_cost_billions": "12.3",
    "states_represented": 1,
    "developers_represented": 87,
    "withdrawn_count": 290,
    "withdrawn_rate": "68.5"
  },
  "message": "Found 423 FERC queue projects totaling 45.2 GW"
}
```

### 2. USACE Dam Retrofit Opportunities
Access data on 87,000 existing dams analyzed for hydroelectric retrofit potential.

#### Endpoint
```
GET /api/discovery/projects?type=dams
```

#### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | - | Must be `dams` to access USACE data |
| `limit` | integer | 100 | Number of results to return |
| `offset` | integer | 0 | Pagination offset |
| `state` | string | - | Filter by US state code |
| `min_capacity` | float | 0 | Minimum retrofit potential in MW |
| `max_capacity` | float | 10000 | Maximum retrofit potential in MW |
| `sort` | string | `year_completed` | Sort field |
| `order` | string | `desc` | Sort order |

#### Example Request
```bash
curl "https://atlas.terra-lumina.com/api/discovery/projects?type=dams&state=TX&limit=5"
```

#### Example Response
```json
{
  "success": true,
  "data": [
    {
      "nid_id": "TX03690",
      "dam_name": "Lake Travis Dam",
      "state": "TX",
      "county": "Travis County",
      "river": "Colorado River",
      "owner_type": "Federal",
      "dam_type": "Gravity",
      "dam_height_ft": 266,
      "dam_length_ft": 2091,
      "year_completed": 1942,
      "latitude": 30.3951,
      "longitude": -97.9103,
      "has_existing_hydro": false,
      "retrofit_potential_mw": 45.2,
      "estimated_annual_generation_mwh": 198876,
      "capacity_factor": 0.50,
      "retrofit_cost": 135600000,
      "levelized_cost_per_mwh": 34,
      "payback_period_years": 11.4,
      "environmental_score": 85,
      "community_support_score": 92,
      "technical_feasibility_score": 88,
      "overall_viability_score": 88,
      "distance_to_transmission_mi": 2.3,
      "interconnection_cost": 230000,
      "permitting_complexity": "Medium",
      "estimated_construction_months": 36,
      "jobs_created": 226,
      "annual_revenue_potential": 11932560
    }
  ],
  "data_source": "usace",
  "pagination": {
    "total": 1641,
    "limit": 5,
    "offset": 0,
    "has_more": true
  },
  "aggregates": {
    "total_dams": 1641,
    "total_retrofit_potential_mw": 24012,
    "total_retrofit_potential_gw": "24.0",
    "total_investment_required": 70100000000,
    "total_investment_billions": "70.1",
    "total_annual_generation_gwh": "103219.3",
    "total_jobs_potential": 136088,
    "avg_payback_years": "11.5"
  },
  "message": "Found 1641 viable dam retrofit opportunities totaling 24.0 GW"
}
```

### 3. Corridor Opportunities
Discover transmission corridor sharing opportunities that can reduce costs by 74%.

#### Endpoint
```
GET /api/discovery/projects?type=corridors
```

#### Example Response
```json
{
  "success": true,
  "data": [
    {
      "corridor_id": "TX-WEST-SOLAR",
      "state": "TX",
      "region": "West Texas",
      "project_count": 23,
      "total_capacity_mw": 8945,
      "individual_transmission_cost": 2875000000,
      "shared_corridor_cost": 748500000,
      "savings": 2126500000,
      "savings_percent": 74,
      "anchor_projects": [
        "Prospero Solar (2000 MW)",
        "Titan Solar Park (1500 MW)",
        "Permian Battery Storage (500 MW)"
      ],
      "status": "Forming",
      "contact_email": "corridor-tx-west@terra-atlas.com"
    }
  ],
  "pagination": {
    "total": 238,
    "limit": 100,
    "offset": 0,
    "has_more": true
  },
  "message": "Found 238 corridor opportunities with $47.6B potential savings"
}
```

### 4. Statistics Endpoints

#### FERC Queue Statistics
```
GET /api/discovery/projects?type=stats
```

Returns aggregate statistics for entire FERC queue dataset.

#### USACE Dam Statistics
```
GET /api/discovery/projects?type=usace-stats
```

Returns aggregate statistics for dam retrofit opportunities.

#### Example Statistics Response
```json
{
  "success": true,
  "data": {
    "total_projects": 11547,
    "withdrawn": 8301,
    "active": 3061,
    "operational": 185,
    "total_capacity_gw": "2704.3",
    "total_stuck_investment": 1235000000000,
    "corridor_opportunities": 238,
    "total_corridor_savings": 47649310365,
    "avg_interconnection_cost": 107000000,
    "median_project_size_mw": 200,
    "success_rate": 28.1,
    "avg_queue_time_years": 4.2
  },
  "message": "FERC queue statistics loaded"
}
```

## Data Sources
- **FERC Queue Data**: Updated monthly from FERC and ISO/RTO public queue reports
- **USACE Dam Data**: National Inventory of Dams (NID) database with proprietary retrofit analysis
- **Transmission Costs**: Berkeley Lab Interconnection Cost Database
- **Corridor Analysis**: Proprietary algorithm based on geographic proximity and grid topology

## Response Codes
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Endpoint does not exist |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

## SDKs and Libraries
- **Python**: `pip install terra-atlas-sdk`
- **JavaScript/TypeScript**: `npm install @terra-atlas/sdk`
- **R**: `install.packages("terratlasR")`

## Example Use Cases

### Find Stuck Solar Projects in Texas
```python
import requests

response = requests.get(
    "https://atlas.terra-lumina.com/api/discovery/projects",
    params={
        "state": "TX",
        "source": "solar",
        "status": "withdrawn",
        "limit": 100
    }
)
projects = response.json()["data"]
print(f"Found {len(projects)} stuck solar projects in Texas")
```

### Identify Best Dam Retrofit Opportunities
```javascript
const fetch = require('node-fetch');

const response = await fetch(
    'https://atlas.terra-lumina.com/api/discovery/projects?type=dams&sort=overall_viability_score&order=desc&limit=10'
);
const dams = await response.json();
console.log(`Top opportunity: ${dams.data[0].dam_name} - ${dams.data[0].retrofit_potential_mw} MW`);
```

### Calculate Total Corridor Savings
```r
library(httr)
library(jsonlite)

response <- GET("https://atlas.terra-lumina.com/api/discovery/projects",
                query = list(type = "corridors"))
corridors <- fromJSON(content(response, "text"))
total_savings <- sum(corridors$data$savings)
print(paste("Total corridor savings opportunity: $", 
            format(total_savings/1e9, digits=2), "billion"))
```

## Support
- **Email**: api@terra-atlas.com
- **GitHub**: https://github.com/terra-atlas/api
- **Discord**: https://discord.gg/terra-atlas

## License
All data is provided under Creative Commons CC-BY 4.0. You are free to use, share, and build upon this data with attribution to Terra Atlas.

## Updates
- **v2.0** (Current): Added USACE dam retrofit data
- **v1.5**: Added corridor opportunity detection
- **v1.0**: Initial FERC queue data release

Last Updated: September 2024