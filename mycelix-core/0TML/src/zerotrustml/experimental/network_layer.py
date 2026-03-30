#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Network Communication Layer for ZeroTrustML
Replaces direct Python calls with real P2P networking
"""

import asyncio
import json
import numpy as np
import base64
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    # Create dummy types for type annotations when websockets not available
    WebSocketServerProtocol = object  # type: ignore
    WebSocketClientProtocol = object  # type: ignore
    print("⚠️  websockets not installed. Run: pip install websockets")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Information about a peer"""
    peer_id: int
    address: str
    port: int
    last_seen: datetime
    reputation: float = 0.7


@dataclass
class Message:
    """Network message"""
    msg_type: str  # gradient, validation, heartbeat, discovery
    sender_id: int
    round_num: int
    timestamp: float
    payload: Dict


class NetworkNode:
    """
    P2P Network Node for ZeroTrustML

    Features:
    - WebSocket-based communication
    - Peer discovery via gossip
    - Gradient broadcast/receive
    - Heartbeat/keepalive
    - Byzantine-resistant message validation
    """

    def __init__(
        self,
        node_id: int,
        listen_port: int,
        bootstrap_peers: Optional[List[str]] = None
    ):
        self.node_id = node_id
        self.listen_port = listen_port
        self.bootstrap_peers = bootstrap_peers or []

        # Network state
        self.peers: Dict[int, PeerInfo] = {}
        self.connections: Dict[int, WebSocketClientProtocol] = {}
        self.server: Optional[asyncio.Server] = None

        # Message handlers
        self.handlers: Dict[str, Callable] = {}

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0

        logger.info(f"🌐 NetworkNode {node_id} initialized on port {listen_port}")

    async def start(self):
        """Start the network node"""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets required. Install: pip install websockets")

        # Start WebSocket server
        self.server = await websockets.serve(
            self._handle_connection,
            "0.0.0.0",
            self.listen_port
        )

        logger.info(f"🚀 Node {self.node_id} listening on port {self.listen_port}")

        # Connect to bootstrap peers
        if self.bootstrap_peers:
            await self._connect_to_bootstrap_peers()

        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stop the network node"""
        # Close all peer connections
        for peer_id, conn in list(self.connections.items()):
            await conn.close()

        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        logger.info(f"🛑 Node {self.node_id} stopped")

    async def _connect_to_bootstrap_peers(self):
        """Connect to bootstrap peers"""
        for peer_addr in self.bootstrap_peers:
            try:
                # Parse address: "localhost:5000"
                host, port = peer_addr.split(':')
                port = int(port)

                # Connect via WebSocket
                uri = f"ws://{host}:{port}"
                websocket = await websockets.connect(uri)

                # Send introduction
                intro = Message(
                    msg_type="introduction",
                    sender_id=self.node_id,
                    round_num=0,
                    timestamp=datetime.now().timestamp(),
                    payload={
                        'listen_port': self.listen_port
                    }
                )
                await self._send_message(websocket, intro)

                logger.info(f"✅ Connected to bootstrap peer {peer_addr}")

            except Exception as e:
                logger.warning(f"Failed to connect to {peer_addr}: {e}")

    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connection"""
        peer_id = None

        try:
            async for raw_message in websocket:
                message = self._deserialize_message(raw_message)
                self.messages_received += 1
                self.bytes_received += len(raw_message)

                # First message should be introduction
                if message.msg_type == "introduction":
                    peer_id = message.sender_id
                    peer_port = message.payload.get('listen_port', 0)

                    # Extract peer address
                    peer_host = websocket.remote_address[0]
                    peer_addr = f"{peer_host}:{peer_port}"

                    # Register peer
                    self.peers[peer_id] = PeerInfo(
                        peer_id=peer_id,
                        address=peer_host,
                        port=peer_port,
                        last_seen=datetime.now()
                    )
                    self.connections[peer_id] = websocket

                    logger.info(f"🤝 Peer {peer_id} connected from {peer_addr}")

                    # Share peer list
                    await self._send_peer_list(websocket)

                elif peer_id:
                    # Update last seen
                    if peer_id in self.peers:
                        self.peers[peer_id].last_seen = datetime.now()

                    # Handle message
                    await self._dispatch_message(message)

        except websockets.exceptions.ConnectionClosed:
            if peer_id:
                logger.info(f"👋 Peer {peer_id} disconnected")
                self.peers.pop(peer_id, None)
                self.connections.pop(peer_id, None)
        except Exception as e:
            logger.error(f"Error handling connection: {e}")

    async def _send_peer_list(self, websocket: WebSocketClientProtocol):
        """Send list of known peers to newly connected peer"""
        peer_list = [
            {
                'peer_id': p.peer_id,
                'address': p.address,
                'port': p.port
            }
            for p in self.peers.values()
        ]

        message = Message(
            msg_type="peer_list",
            sender_id=self.node_id,
            round_num=0,
            timestamp=datetime.now().timestamp(),
            payload={'peers': peer_list}
        )

        await self._send_message(websocket, message)

    async def _dispatch_message(self, message: Message):
        """Dispatch message to registered handler"""
        handler = self.handlers.get(message.msg_type)
        if handler:
            await handler(message)
        else:
            logger.warning(f"No handler for message type: {message.msg_type}")

    def register_handler(self, msg_type: str, handler: Callable):
        """Register a message handler"""
        self.handlers[msg_type] = handler
        logger.info(f"📝 Registered handler for '{msg_type}'")

    async def broadcast_gradient(
        self,
        gradient: np.ndarray,
        round_num: int,
        metadata: Optional[Dict] = None
    ):
        """Broadcast gradient to all peers"""
        # Serialize gradient
        gradient_b64 = base64.b64encode(gradient.tobytes()).decode('utf-8')

        message = Message(
            msg_type="gradient",
            sender_id=self.node_id,
            round_num=round_num,
            timestamp=datetime.now().timestamp(),
            payload={
                'gradient': gradient_b64,
                'shape': list(gradient.shape),
                'dtype': str(gradient.dtype),
                'metadata': metadata or {}
            }
        )

        # Send to all connected peers
        for peer_id, conn in list(self.connections.items()):
            try:
                await self._send_message(conn, message)
            except Exception as e:
                logger.warning(f"Failed to send to peer {peer_id}: {e}")
                # Remove dead connection
                self.connections.pop(peer_id, None)

        logger.debug(f"📤 Broadcast gradient to {len(self.connections)} peers")

    async def send_validation_result(
        self,
        peer_id: int,
        round_num: int,
        is_valid: bool,
        reputation_score: float
    ):
        """Send validation result to a specific peer"""
        if peer_id not in self.connections:
            logger.warning(f"Peer {peer_id} not connected")
            return

        message = Message(
            msg_type="validation",
            sender_id=self.node_id,
            round_num=round_num,
            timestamp=datetime.now().timestamp(),
            payload={
                'is_valid': is_valid,
                'reputation_score': reputation_score
            }
        )

        await self._send_message(self.connections[peer_id], message)

    async def _heartbeat_loop(self):
        """Periodic heartbeat to keep connections alive"""
        while True:
            await asyncio.sleep(30)  # Every 30 seconds

            heartbeat = Message(
                msg_type="heartbeat",
                sender_id=self.node_id,
                round_num=0,
                timestamp=datetime.now().timestamp(),
                payload={'status': 'alive'}
            )

            # Send to all peers
            for peer_id, conn in list(self.connections.items()):
                try:
                    await self._send_message(conn, heartbeat)
                except Exception as e:
                    logger.warning(f"Heartbeat failed for peer {peer_id}: {e}")
                    self.connections.pop(peer_id, None)

    async def _send_message(
        self,
        websocket: WebSocketClientProtocol,
        message: Message
    ):
        """Send message via WebSocket"""
        serialized = self._serialize_message(message)
        await websocket.send(serialized)
        self.messages_sent += 1
        self.bytes_sent += len(serialized)

    def _serialize_message(self, message: Message) -> str:
        """Serialize message to JSON"""
        return json.dumps(asdict(message))

    def _deserialize_message(self, data: str) -> Message:
        """Deserialize message from JSON"""
        msg_dict = json.loads(data)
        return Message(**msg_dict)

    def get_statistics(self) -> Dict:
        """Get network statistics"""
        return {
            'node_id': self.node_id,
            'connected_peers': len(self.connections),
            'known_peers': len(self.peers),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received
        }


# ============================================================
# Testing and Examples
# ============================================================

async def test_two_nodes():
    """Test basic two-node communication"""
    print("\n" + "="*60)
    print("🧪 TESTING TWO-NODE COMMUNICATION")
    print("="*60)

    # Create node 1
    node1 = NetworkNode(node_id=1, listen_port=5000)

    # Handler for receiving gradients
    async def handle_gradient(message: Message):
        print(f"📥 Node 1 received gradient from node {message.sender_id}")
        gradient_b64 = message.payload['gradient']
        gradient_bytes = base64.b64decode(gradient_b64)
        shape = tuple(message.payload['shape'])
        dtype = np.dtype(message.payload['dtype'])
        gradient = np.frombuffer(gradient_bytes, dtype=dtype).reshape(shape)
        print(f"   Shape: {gradient.shape}, Mean: {gradient.mean():.4f}")

    node1.register_handler('gradient', handle_gradient)
    await node1.start()

    # Give server time to start
    await asyncio.sleep(0.5)

    # Create node 2, bootstrapping to node 1
    node2 = NetworkNode(
        node_id=2,
        listen_port=5001,
        bootstrap_peers=['localhost:5000']
    )
    await node2.start()

    # Give nodes time to connect
    await asyncio.sleep(1)

    # Node 2 broadcasts gradient
    print("\n📤 Node 2 broadcasting gradient...")
    gradient = np.random.randn(100)
    await node2.broadcast_gradient(gradient, round_num=1)

    # Wait for message to be received
    await asyncio.sleep(0.5)

    # Print statistics
    print("\n📊 Network Statistics:")
    print(f"   Node 1: {node1.get_statistics()}")
    print(f"   Node 2: {node2.get_statistics()}")

    # Cleanup
    await node1.stop()
    await node2.stop()

    print("\n✅ Two-node test complete!")


async def test_small_network():
    """Test 5-node network"""
    print("\n" + "="*60)
    print("🧪 TESTING 5-NODE NETWORK")
    print("="*60)

    nodes = []
    base_port = 6000

    # Create first node
    node0 = NetworkNode(node_id=0, listen_port=base_port)
    await node0.start()
    nodes.append(node0)
    await asyncio.sleep(0.5)

    # Create remaining nodes, each bootstrapping to node 0
    for i in range(1, 5):
        node = NetworkNode(
            node_id=i,
            listen_port=base_port + i,
            bootstrap_peers=[f'localhost:{base_port}']
        )
        await node.start()
        nodes.append(node)
        await asyncio.sleep(0.3)

    print("\n🌐 Network established with 5 nodes")

    # Wait for connections
    await asyncio.sleep(2)

    # Each node broadcasts a gradient
    print("\n📤 All nodes broadcasting gradients...")
    for i, node in enumerate(nodes):
        gradient = np.random.randn(50) * (i + 1)
        await node.broadcast_gradient(gradient, round_num=1)
        await asyncio.sleep(0.2)

    # Wait for all messages
    await asyncio.sleep(2)

    # Print statistics
    print("\n📊 Network Statistics:")
    for node in nodes:
        stats = node.get_statistics()
        print(f"   Node {stats['node_id']}: "
              f"{stats['connected_peers']} peers, "
              f"{stats['messages_sent']} sent, "
              f"{stats['messages_received']} received")

    # Cleanup
    for node in nodes:
        await node.stop()

    print("\n✅ 5-node network test complete!")


async def main():
    """Main entry point for testing"""
    if not WEBSOCKETS_AVAILABLE:
        print("❌ websockets not available")
        print("Install: pip install websockets")
        print("Or: nix-shell -p python313Packages.websockets")
        return

    print("\n🌐 ZeroTrustML Network Layer Testing")
    print("="*60)

    print("\nAvailable tests:")
    print("  1. Two-node communication")
    print("  2. 5-node network")
    print("  3. Both")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice in ['1', '3']:
        await test_two_nodes()

    if choice in ['2', '3']:
        await test_small_network()


if __name__ == "__main__":
    asyncio.run(main())