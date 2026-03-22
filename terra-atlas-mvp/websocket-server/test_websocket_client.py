#!/usr/bin/env python3
"""
Test WebSocket Client for Terra Atlas
"""

import asyncio
import json
import websockets
from datetime import datetime

async def test_websocket_connection():
    """Test WebSocket connection and functionality"""
    uri = "ws://localhost:8002/ws/test-client-123"
    
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            print(f"✅ Connected! Server says: {welcome_data['message']}")
            
            # Test 1: Subscribe to prices channel
            print("\n📊 Test 1: Subscribing to price updates...")
            await websocket.send(json.dumps({
                "type": "subscribe",
                "channel": "prices"
            }))
            response = await websocket.recv()
            print(f"   Response: {json.loads(response)}")
            
            # Test 2: Subscribe to updates channel  
            print("\n📢 Test 2: Subscribing to general updates...")
            await websocket.send(json.dumps({
                "type": "subscribe",
                "channel": "updates"
            }))
            response = await websocket.recv()
            print(f"   Response: {json.loads(response)}")
            
            # Test 3: Send a ping
            print("\n🏓 Test 3: Sending ping...")
            await websocket.send(json.dumps({
                "type": "ping"
            }))
            response = await websocket.recv()
            print(f"   Response: {json.loads(response)}")
            
            # Test 4: Send user activity
            print("\n👤 Test 4: Sending user activity...")
            await websocket.send(json.dumps({
                "type": "user_activity",
                "action": "viewing",
                "project_id": "solar-1234",
                "data": {"page": "details"}
            }))
            
            # Test 5: Listen for real-time updates
            print("\n📡 Test 5: Listening for real-time updates (15 seconds)...")
            print("   Waiting for price updates and status changes...")
            
            # Listen for messages for 15 seconds
            end_time = asyncio.get_event_loop().time() + 15
            update_count = 0
            
            while asyncio.get_event_loop().time() < end_time:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    update_count += 1
                    
                    if data['type'] == 'price_update':
                        price_data = data['data']
                        print(f"   💰 Price Update #{update_count}: {price_data['project_type']} - ${price_data['current_price']:.2f} ({price_data['change_24h']:+.2f}%)")
                    elif data['type'] == 'status_change':
                        status_data = data['data']
                        print(f"   🔄 Status Change #{update_count}: {status_data['project_id']} - {status_data['old_status']} → {status_data['new_status']}")
                    else:
                        print(f"   📨 Message #{update_count}: {data['type']}")
                        
                except asyncio.TimeoutError:
                    continue
            
            print(f"\n✅ Test complete! Received {update_count} updates in 15 seconds")
            
            # Test 6: Unsubscribe
            print("\n🚪 Test 6: Unsubscribing from channels...")
            await websocket.send(json.dumps({
                "type": "unsubscribe",
                "channel": "prices"
            }))
            response = await websocket.recv()
            print(f"   Response: {json.loads(response)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

async def test_multiple_clients():
    """Test multiple concurrent WebSocket connections"""
    print("\n🔥 Testing multiple concurrent connections...")
    
    async def connect_client(client_id):
        uri = f"ws://localhost:8002/ws/client-{client_id}"
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps({"type": "ping"}))
                response = await websocket.recv()
                return True
        except:
            return False
    
    # Create 5 concurrent connections
    tasks = [connect_client(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    successful = sum(results)
    print(f"✅ {successful}/5 clients connected successfully")
    
    return all(results)

async def main():
    """Run all tests"""
    print("🚀 Terra Atlas WebSocket Test Suite")
    print("====================================\n")
    
    # Test basic connection
    success = await test_websocket_connection()
    
    if success:
        # Test multiple connections
        await test_multiple_clients()
    
    print("\n====================================")
    print("✨ All tests complete!")

if __name__ == "__main__":
    asyncio.run(main())