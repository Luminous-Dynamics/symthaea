#!/usr/bin/env python3
"""
Test script for USACE Dam Integration
"""

import asyncio
import aiohttp
import json
from integrations.usace_api import USACEDamIntegration

async def test_usace_integration():
    """
    Test the USACE integration functionality
    """
    print("🏗️ Testing USACE Dam Integration...")
    print("=" * 50)
    
    async with USACEDamIntegration() as integration:
        # Test 1: Fetch dams
        print("\n📊 Test 1: Fetching dams...")
        dams = await integration.fetch_all_dams(limit=10)
        print(f"✅ Fetched {len(dams)} dams")
        
        if dams:
            # Show sample dam
            print("\n🏗️ Sample dam data:")
            sample = dams[0]
            print(json.dumps(sample, indent=2))
        
        # Test 2: Filter hydroelectric dams
        print("\n⚡ Test 2: Filtering hydroelectric dams...")
        hydro_dams = integration.filter_hydroelectric_dams(dams)
        print(f"✅ Found {len(hydro_dams)} hydroelectric dams out of {len(dams)}")
        
        # Test 3: Group by state
        print("\n🗺️ Test 3: Grouping by state...")
        by_state = integration.group_by_state(dams)
        print(f"✅ Dams found in {len(by_state)} states:")
        for state, state_dams in by_state.items():
            print(f"  - {state}: {len(state_dams)} dams")
        
        # Test 4: Calculate statistics
        print("\n📈 Test 4: Calculating statistics...")
        stats = integration.get_statistics(dams)
        print(json.dumps(stats, indent=2))
        
        # Test 5: Test API endpoint
        print("\n🌐 Test 5: Testing FastAPI endpoint...")
        async with aiohttp.ClientSession() as session:
            try:
                # Assuming the FastAPI server is running
                async with session.get("http://localhost:8001/usace/dams?limit=5") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ API returned {data['count']} dams")
                        print(f"   Statistics: {data['statistics']}")
                    else:
                        print(f"⚠️ API returned status {response.status}")
            except aiohttp.ClientConnectorError:
                print("⚠️ Could not connect to API (server not running)")
            except Exception as e:
                print(f"❌ API test failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ USACE Integration tests complete!")

async def test_api_endpoints():
    """
    Test the API endpoints if server is running
    """
    print("\n🔄 Testing API endpoints...")
    
    base_url = "http://localhost:8001"
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test health endpoint
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    print("✅ Server is healthy")
                    
            # Test USACE endpoints
            endpoints = [
                "/usace/dams?limit=3",
                "/usace/statistics",
                "/usace/dams?state=CA&hydro_only=true",
                "/usace/import?dry_run=true&limit=5"
            ]
            
            for endpoint in endpoints:
                async with session.get(f"{base_url}{endpoint}") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ {endpoint}: Success")
                        if "count" in data:
                            print(f"   Returned {data['count']} items")
                        if "statistics" in data:
                            print(f"   Stats: {data['statistics'].get('total_dams', 'N/A')} total dams")
                    else:
                        print(f"❌ {endpoint}: Status {response.status}")
                        
        except aiohttp.ClientConnectorError:
            print("⚠️ Server not running. Start with: ./start-ml-backend.sh")
        except Exception as e:
            print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    print("🚀 USACE Dam Integration Test Suite")
    print("=====================================\n")
    
    # Run tests
    asyncio.run(test_usace_integration())
    
    # Test API if server is running
    asyncio.run(test_api_endpoints())