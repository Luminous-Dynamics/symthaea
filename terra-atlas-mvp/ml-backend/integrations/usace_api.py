"""
USACE Dam Data Integration
Fetches and processes data from US Army Corps of Engineers National Inventory of Dams
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class USACEDamIntegration:
    """
    Integration with USACE National Inventory of Dams API
    """
    
    def __init__(self):
        # USACE NID API endpoints
        self.base_url = "https://nid.sec.usace.army.mil/api/nation"
        self.search_url = "https://nid.sec.usace.army.mil/api/search"
        
        # Additional public data sources
        self.alternative_sources = {
            "opendatasoft": "https://public.opendatasoft.com/api/records/1.0/search/?dataset=national-inventory-of-dams",
            "data_gov": "https://catalog.data.gov/api/3/action/package_show?id=national-inventory-of-dams-nid"
        }
        
        self.session = None
        self.cache = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def fetch_all_dams(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Fetch all dam data from USACE
        
        Args:
            limit: Optional limit on number of dams to fetch
            
        Returns:
            List of dam dictionaries
        """
        all_dams = []
        
        try:
            # Try primary USACE API first
            dams = await self._fetch_from_usace(limit)
            if dams:
                all_dams.extend(dams)
                logger.info(f"Fetched {len(dams)} dams from USACE API")
                
        except Exception as e:
            logger.warning(f"USACE API failed: {e}, trying alternative sources")
            
            # Fallback to OpenDataSoft
            try:
                dams = await self._fetch_from_opendatasoft(limit)
                if dams:
                    all_dams.extend(dams)
                    logger.info(f"Fetched {len(dams)} dams from OpenDataSoft")
            except Exception as e2:
                logger.error(f"All API sources failed: {e2}")
                
                # Use mock data for development
                all_dams = self._generate_mock_dams(limit or 100)
                logger.info(f"Generated {len(all_dams)} mock dams for development")
                
        return all_dams
    
    async def _fetch_from_usace(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Fetch directly from USACE NID API
        """
        dams = []
        
        # USACE requires authentication for full access
        # Using public endpoints where available
        params = {
            "format": "json",
            "limit": limit or 10000
        }
        
        async with self.session.get(self.base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                dams = self._parse_usace_response(data)
            else:
                raise Exception(f"USACE API returned status {response.status}")
                
        return dams
    
    async def _fetch_from_opendatasoft(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Fetch from OpenDataSoft public dataset
        """
        dams = []
        
        # OpenDataSoft provides free access to NID data
        params = {
            "rows": min(limit or 10000, 10000),  # Max 10k per request
            "facet": ["state", "purposes", "dam_type", "owner_type"],
            "refine.country": "US"
        }
        
        async with self.session.get(self.alternative_sources["opendatasoft"], params=params) as response:
            if response.status == 200:
                data = await response.json()
                dams = self._parse_opendatasoft_response(data)
            else:
                raise Exception(f"OpenDataSoft API returned status {response.status}")
                
        return dams
    
    def _parse_usace_response(self, data: Dict) -> List[Dict]:
        """
        Parse USACE API response into standardized format
        """
        dams = []
        
        for item in data.get("dams", []):
            dam = self._standardize_dam_data(item, "usace")
            if dam:
                dams.append(dam)
                
        return dams
    
    def _parse_opendatasoft_response(self, data: Dict) -> List[Dict]:
        """
        Parse OpenDataSoft response into standardized format
        """
        dams = []
        
        for record in data.get("records", []):
            fields = record.get("fields", {})
            dam = self._standardize_dam_data(fields, "opendatasoft")
            if dam:
                dams.append(dam)
                
        return dams
    
    def _standardize_dam_data(self, raw_data: Dict, source: str) -> Optional[Dict]:
        """
        Standardize dam data from various sources
        """
        try:
            if source == "usace":
                return {
                    "id": f"USACE-{raw_data.get('nid_id', '')}",
                    "name": raw_data.get("dam_name", "Unknown Dam"),
                    "type": "hydro",
                    "subtype": "dam",
                    "latitude": float(raw_data.get("latitude", 0)),
                    "longitude": float(raw_data.get("longitude", 0)),
                    "capacity_mw": self._calculate_hydro_capacity(raw_data),
                    "status": raw_data.get("condition_assessment", "operational"),
                    "owner": raw_data.get("owner_name", "Unknown"),
                    "state": raw_data.get("state", ""),
                    "county": raw_data.get("county", ""),
                    "river": raw_data.get("river", ""),
                    "year_built": raw_data.get("year_completed", None),
                    "height_ft": raw_data.get("dam_height", 0),
                    "length_ft": raw_data.get("dam_length", 0),
                    "storage_af": raw_data.get("normal_storage", 0),
                    "drainage_area_sqmi": raw_data.get("drainage_area", 0),
                    "purposes": raw_data.get("purposes", "").split(","),
                    "hazard_potential": raw_data.get("hazard_potential", "low"),
                    "eha_rating": raw_data.get("eha_rating", None),
                    "source": "USACE NID"
                }
                
            elif source == "opendatasoft":
                # Map OpenDataSoft field names
                return {
                    "id": f"USACE-{raw_data.get('recordid', '')}",
                    "name": raw_data.get("dam_name", "Unknown Dam"),
                    "type": "hydro",
                    "subtype": "dam",
                    "latitude": float(raw_data.get("latitude", 0)),
                    "longitude": float(raw_data.get("longitude", 0)),
                    "capacity_mw": self._calculate_hydro_capacity(raw_data),
                    "status": "operational" if raw_data.get("inspection_date") else "unknown",
                    "owner": raw_data.get("owner_name", "Unknown"),
                    "state": raw_data.get("state", ""),
                    "county": raw_data.get("county", ""),
                    "river": raw_data.get("river_or_stream", ""),
                    "year_built": raw_data.get("year_completed", None),
                    "height_ft": raw_data.get("nid_height_ft", 0),
                    "length_ft": raw_data.get("dam_length_ft", 0),
                    "storage_af": raw_data.get("nid_storage_af", 0),
                    "drainage_area_sqmi": raw_data.get("drainage_area_sq_miles", 0),
                    "purposes": str(raw_data.get("purposes", "")).split(","),
                    "hazard_potential": raw_data.get("hazard_potential_classification", "low"),
                    "source": "OpenDataSoft NID"
                }
                
            elif source == "mock":
                # Direct return for mock data
                return raw_data
                
        except Exception as e:
            logger.error(f"Error standardizing dam data: {e}")
            return None
    
    def _calculate_hydro_capacity(self, dam_data: Dict) -> float:
        """
        Estimate hydroelectric capacity based on dam characteristics
        """
        # Check if dam has hydroelectric purpose
        purposes = str(dam_data.get("purposes", "")).upper()
        if "H" not in purposes and "HYDROELECTRIC" not in purposes.upper():
            return 0.0
            
        # Simple capacity estimation based on height and storage
        height = float(dam_data.get("dam_height", 0) or dam_data.get("nid_height_ft", 0))
        storage = float(dam_data.get("normal_storage", 0) or dam_data.get("nid_storage_af", 0))
        
        if height > 0 and storage > 0:
            # Rough estimate: capacity_mw = (height * storage)^0.5 / 50
            # This is a simplified formula for demonstration
            estimated_capacity = ((height * storage) ** 0.5) / 50
            
            # Cap at reasonable values
            return min(estimated_capacity, 5000)  # Max 5000 MW
        
        return 0.0
    
    def _generate_mock_dams(self, count: int = 100) -> List[Dict]:
        """
        Generate mock dam data for development/testing
        """
        import random
        
        dams = []
        
        states = ["CA", "WA", "OR", "ID", "MT", "WY", "CO", "AZ", "NV", "UT", "NY", "PA", "TN", "KY", "WV"]
        purposes = ["Hydroelectric", "Flood Control", "Water Supply", "Recreation", "Irrigation"]
        owners = ["USBR", "USACE", "TVA", "Private", "State", "Municipal"]
        
        for i in range(count):
            # Generate realistic US coordinates
            lat = random.uniform(30, 48)  # Continental US latitude range
            lon = random.uniform(-125, -70)  # Continental US longitude range
            
            # Generate dam characteristics
            height = random.uniform(10, 700)  # feet
            storage = random.uniform(100, 10000000)  # acre-feet
            
            dam = {
                "id": f"USACE-MOCK-{i+1:05d}",
                "name": f"Mock Dam {i+1}",
                "type": "hydro",
                "subtype": "dam",
                "latitude": lat,
                "longitude": lon,
                "capacity_mw": random.uniform(0, 2000) if random.random() > 0.5 else 0,
                "status": random.choice(["operational", "under_construction", "planned"]),
                "owner": random.choice(owners),
                "state": random.choice(states),
                "county": f"County {random.randint(1, 100)}",
                "river": f"River {random.randint(1, 50)}",
                "year_built": random.randint(1900, 2023) if random.random() > 0.2 else None,
                "height_ft": height,
                "length_ft": random.uniform(100, 10000),
                "storage_af": storage,
                "drainage_area_sqmi": random.uniform(10, 100000),
                "purposes": random.sample(purposes, random.randint(1, 3)),
                "hazard_potential": random.choice(["low", "significant", "high"]),
                "source": "Mock Data"
            }
            
            dams.append(dam)
            
        return dams
    
    async def fetch_dam_details(self, dam_id: str) -> Optional[Dict]:
        """
        Fetch detailed information for a specific dam
        """
        # Check cache first
        if dam_id in self.cache:
            return self.cache[dam_id]
            
        # In production, would query specific dam endpoint
        # For now, return from mock data
        all_dams = await self.fetch_all_dams(limit=1000)
        for dam in all_dams:
            if dam["id"] == dam_id:
                self.cache[dam_id] = dam
                return dam
                
        return None
    
    def filter_hydroelectric_dams(self, dams: List[Dict]) -> List[Dict]:
        """
        Filter only dams with hydroelectric capacity
        """
        return [dam for dam in dams if dam.get("capacity_mw", 0) > 0]
    
    def group_by_state(self, dams: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group dams by state
        """
        grouped = {}
        for dam in dams:
            state = dam.get("state", "Unknown")
            if state not in grouped:
                grouped[state] = []
            grouped[state].append(dam)
        return grouped
    
    def calculate_total_capacity(self, dams: List[Dict]) -> float:
        """
        Calculate total hydroelectric capacity
        """
        return sum(dam.get("capacity_mw", 0) for dam in dams)
    
    def get_statistics(self, dams: List[Dict]) -> Dict:
        """
        Generate statistics for dam dataset
        """
        hydro_dams = self.filter_hydroelectric_dams(dams)
        
        return {
            "total_dams": len(dams),
            "hydro_dams": len(hydro_dams),
            "total_capacity_mw": self.calculate_total_capacity(dams),
            "states_covered": len(set(dam.get("state") for dam in dams)),
            "average_capacity_mw": self.calculate_total_capacity(hydro_dams) / len(hydro_dams) if hydro_dams else 0,
            "hazard_breakdown": {
                "high": len([d for d in dams if d.get("hazard_potential") == "high"]),
                "significant": len([d for d in dams if d.get("hazard_potential") == "significant"]),
                "low": len([d for d in dams if d.get("hazard_potential") == "low"])
            }
        }