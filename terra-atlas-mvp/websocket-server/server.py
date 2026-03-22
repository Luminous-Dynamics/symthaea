#!/usr/bin/env python3
"""
Terra Atlas WebSocket Server
Handles real-time updates for:
- Live project updates
- Price streaming
- User collaboration
- System notifications
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, asdict
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from contextlib import asynccontextmanager
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProjectUpdate:
    """Real-time project update"""
    project_id: str
    type: str  # 'status', 'price', 'investment', 'milestone'
    data: Dict
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

@dataclass
class UserActivity:
    """User collaboration activity"""
    user_id: str
    action: str  # 'viewing', 'investing', 'commenting'
    project_id: Optional[str]
    data: Dict
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

class ConnectionManager:
    """Manage WebSocket connections and subscriptions"""
    
    def __init__(self):
        # Store active connections by client_id
        self.active_connections: Dict[str, WebSocket] = {}
        # Store subscriptions (channel -> set of client_ids)
        self.subscriptions: Dict[str, Set[str]] = {}
        # Store client metadata
        self.client_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Dict = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        if metadata:
            self.client_metadata[client_id] = metadata
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove disconnected client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from all subscriptions
        for channel in self.subscriptions.values():
            channel.discard(client_id)
        
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]
        
        logger.info(f"Client {client_id} disconnected")
    
    async def subscribe(self, client_id: str, channel: str):
        """Subscribe client to a channel"""
        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()
        self.subscriptions[channel].add(client_id)
        logger.info(f"Client {client_id} subscribed to {channel}")
    
    async def unsubscribe(self, client_id: str, channel: str):
        """Unsubscribe client from a channel"""
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(client_id)
            logger.info(f"Client {client_id} unsubscribed from {channel}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_channel(self, message: dict, channel: str, exclude_client: str = None):
        """Broadcast message to all subscribers of a channel"""
        if channel not in self.subscriptions:
            return
        
        disconnected_clients = []
        for client_id in self.subscriptions[channel]:
            if exclude_client and client_id == exclude_client:
                continue
            
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

# Global connection manager
manager = ConnectionManager()

# Redis client for pub/sub
redis_client: Optional[redis.Redis] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global redis_client
    
    # Initialize Redis connection
    try:
        redis_client = await redis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
        await redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis not available: {e}. Running without Redis.")
        redis_client = None
    
    # Start background tasks
    asyncio.create_task(price_update_generator())
    asyncio.create_task(project_status_monitor())
    
    yield
    
    # Cleanup
    if redis_client:
        await redis_client.close()

# Create FastAPI app
app = FastAPI(title="Terra Atlas WebSocket Server", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Terra Atlas WebSocket Server",
        "connections": len(manager.active_connections),
        "channels": len(manager.subscriptions)
    }

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "total_connections": len(manager.active_connections),
        "channels": {
            channel: len(clients) 
            for channel, clients in manager.subscriptions.items()
        },
        "redis_connected": redis_client is not None
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Main WebSocket endpoint"""
    await manager.connect(websocket, client_id)
    
    # Send welcome message
    await manager.send_personal_message({
        "type": "welcome",
        "message": "Connected to Terra Atlas Real-time Server",
        "client_id": client_id,
        "timestamp": datetime.utcnow().isoformat()
    }, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "subscribe":
                # Subscribe to a channel
                channel = data.get("channel")
                if channel:
                    await manager.subscribe(client_id, channel)
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat()
                    }, client_id)
            
            elif message_type == "unsubscribe":
                # Unsubscribe from a channel
                channel = data.get("channel")
                if channel:
                    await manager.unsubscribe(client_id, channel)
                    await manager.send_personal_message({
                        "type": "unsubscribed",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat()
                    }, client_id)
            
            elif message_type == "project_update":
                # Broadcast project update to channel
                project_id = data.get("project_id")
                if project_id:
                    update = ProjectUpdate(
                        project_id=project_id,
                        type=data.get("update_type", "general"),
                        data=data.get("data", {})
                    )
                    await manager.broadcast_to_channel({
                        "type": "project_update",
                        "update": asdict(update)
                    }, f"project:{project_id}", exclude_client=client_id)
            
            elif message_type == "user_activity":
                # Broadcast user activity
                activity = UserActivity(
                    user_id=data.get("user_id", client_id),
                    action=data.get("action", "unknown"),
                    project_id=data.get("project_id"),
                    data=data.get("data", {})
                )
                
                # Broadcast to relevant channel
                if activity.project_id:
                    await manager.broadcast_to_channel({
                        "type": "user_activity",
                        "activity": asdict(activity)
                    }, f"project:{activity.project_id}", exclude_client=client_id)
            
            elif message_type == "ping":
                # Respond to ping
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, client_id)
            
            else:
                # Echo unknown messages back
                await manager.send_personal_message({
                    "type": "echo",
                    "original": data,
                    "timestamp": datetime.utcnow().isoformat()
                }, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

async def price_update_generator():
    """Generate simulated price updates for demo"""
    project_types = ["solar", "wind", "hydro", "nuclear", "storage"]
    
    while True:
        await asyncio.sleep(5)  # Update every 5 seconds
        
        # Generate random price update
        project_type = random.choice(project_types)
        project_id = f"{project_type}-{random.randint(1000, 9999)}"
        
        price_update = {
            "type": "price_update",
            "data": {
                "project_id": project_id,
                "project_type": project_type,
                "current_price": round(random.uniform(10, 1000), 2),
                "change_24h": round(random.uniform(-5, 5), 2),
                "volume_24h": round(random.uniform(10000, 1000000), 2),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Broadcast to all subscribers of the price channel
        await manager.broadcast_to_channel(price_update, "prices")

async def project_status_monitor():
    """Monitor project status changes"""
    statuses = ["planning", "funding", "construction", "operational", "maintenance"]
    
    while True:
        await asyncio.sleep(10)  # Check every 10 seconds
        
        # Simulate status change
        project_id = f"project-{random.randint(1, 100)}"
        old_status = random.choice(statuses)
        new_status = random.choice([s for s in statuses if s != old_status])
        
        status_update = {
            "type": "status_change",
            "data": {
                "project_id": project_id,
                "old_status": old_status,
                "new_status": new_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Broadcast to project channel and general updates
        await manager.broadcast_to_channel(status_update, f"project:{project_id}")
        await manager.broadcast_to_channel(status_update, "updates")

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )